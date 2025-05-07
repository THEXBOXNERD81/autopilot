import os
import csv
import numpy as np
import torch as th
from stable_baselines3.common.evaluation import evaluate_policy
from stable_baselines3.common.callbacks import BaseCallback, CheckpointCallback
import gym
from gym import spaces
import numpy as np
import socket
import threading
import time
from stable_baselines3 import TD3
from stable_baselines3.common.logger import configure

class RealTimeFlightEnv(gym.Env):
    def __init__(self, host='127.0.0.1', port=54000, episode_length=1000, target_speed=150, logging=True):
        super(RealTimeFlightEnv, self).__init__()
        
        self.observation_space = spaces.Box(
            low=-1.0,
            high=1.0,
            shape=(6,),
            dtype=np.float32
        )
        
        self.action_space = spaces.Box(
            low=np.array([-1.0, -1.0], dtype=np.float32),
            high=np.array([1.0, 1.0], dtype=np.float32),
            dtype=np.float32
        )

        # Target parameters and episode configuration remain the same
        self.target_speed = target_speed
        self.episode_length = episode_length
        self.current_step = 0

        # Socket setup: connect to simulator
        self.client_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        self.client_socket.connect((host, port))
        self.client_socket.settimeout(None)

        # State management
        self.state = None
        self._lock = threading.Lock()
        self._stop_thread = False
        self.recv_thread = threading.Thread(target=self._receive_state)
        self.recv_thread.start()

        # Open CSV log file for observations (raw and normalized)
        if logging:
            self.log_csv_file = open("observations.csv","w")
        else:
            self.log_csv_file = None
        header = "timestamp," + ",".join([f"raw_{i}" for i in range(7)]) + "," + ",".join([f"norm_{i}" for i in range(7)]) + "\n"
        self.log_csv_file.write(header)
        self.log_csv_file.flush()

        # Initialize previous action and error for reward calculations.
        self.prev_heading = None
        self.prev_action = None
        self.prev_error = None
        self.prev_state = None
        self.prev_timestamp = None
        self.state_timestamp = None

    def _receive_state(self):
        buffer = ""
        while not self._stop_thread:
            try:
                data = self.client_socket.recv(4096)
                if data:
                    buffer += data.decode('utf-8', errors='ignore')
                    while '\n' in buffer:
                        line_end = buffer.find('\n')
                        csv_line = buffer[:line_end].strip()
                        buffer = buffer[line_end+1:]
                        state = self._parse_state(csv_line)
                        if state is not None:
                            norm_state = self._normalize_state(state)
                            ts = time.time()
                            raw_str = ",".join([f"{val:.2f}" for val in state])
                            norm_str = ",".join([f"{val:.2f}" for val in norm_state])
                            log_line = f"{ts},{raw_str},{norm_str}\n"
                            self.log_csv_file.write(log_line)
                            self.log_csv_file.flush()
                            raw_pitch = state[1]
                            raw_bank  = state[2]
                            raw_heading = state[3]

                            full_norm = self._normalize_state((raw_pitch, raw_bank, raw_heading))
                            norm_pitch, norm_bank, norm_heading = full_norm[0], full_norm[1], full_norm[2]

                            ts = time.time()
                            with self._lock:
                                self.state = (norm_pitch, norm_bank, norm_heading, ts)
            except Exception as e:
                if not self._stop_thread:
                    print(f"Socket error: {str(e)}")
                break

    def _parse_state(self, csv_line):
        try:
            csv_line = csv_line.replace('\x00', '').strip()
            if not csv_line:
                return None
            # Expecting 7 values after timestamp:
            parts = list(map(float, csv_line.split(',')[1:]))
            if len(parts) != 7:
                print(f"Invalid data: {csv_line}")
                return None
            return np.array(parts, dtype=np.float32)
        except Exception as e:
            print(f"Parse error: {str(e)}")
            return None

    def _normalize_state(self, raw_state):
        """Normalize state to [-1,1] range using fixed ranges.
           Order: Pitch, Bank
        """
        norms = [
            (-30, 30),      # Pitch (degrees)
            (-30, 30),      # Bank (degrees)
            (-10, 10),      # heading (degrees)
        ]
        return np.array([
            2 * ((raw_state[i] - min_val) / (max_val - min_val)) - 1
            for i, (min_val, max_val) in enumerate(norms)
        ], dtype=np.float32)
    
    def _normalize_value(self, value, min_val, max_val):
        """Normalize a single value to [-1,1]."""
        return 2 * ((value - min_val) / (max_val - min_val)) - 1

    def _calculate_reward(self, obs, action):
        pitch = obs[0]
        bank = obs[1]
        heading = obs[2]
        pitch_rate = obs[3]
        bank_rate  = obs[4]
        heading_rate = obs[5]
    
        # Compute errors (using squared error)
        error_pitch = pitch**2
        error_bank = bank**2
        error_heading = (heading - self.prev_heading)**2
        self.prev_heading = heading

        # Define weights for each term
        w_pitch = 1.0
        w_bank = 1.0
        w_heading = 1.0
        w_rate = 0.5       


        pitch_reward = np.exp(-16.0 * (w_pitch * error_pitch))
        bank_reward = np.exp(-16.0 * (w_bank * error_bank))
        heading_reward = np.exp(-16.0 * (w_heading * error_heading))

        rate_penalty = w_rate * (pitch_rate**2 + bank_rate**2 + heading_rate**2)

        reward = pitch_reward + bank_reward - rate_penalty + heading_reward
        

        w_act    = 6.0      
        threshold = 0.2     # normalized units

        inside_amount = np.maximum(threshold - np.abs(action), 0.0)
        action_bonus = w_act * np.sum(inside_amount**2)
        reward += action_bonus

        # Smoothness Reward
        w_action = 0.5
        action_norm = np.linalg.norm(action)
        reward -= w_action * max(0.0, action_norm - 0.1)
        
        # Safety Boundaries Reward:
        safety_threshold = 0.1  # Adjust threshold (normalized)
        gamma = 5.0
        if abs(pitch) > safety_threshold or abs(bank) > safety_threshold:
            reward -= gamma


        return np.clip(reward, -10.0, 10.0).item()

    def step(self, action):
        action = np.clip(action, self.action_space.low, self.action_space.high)
        action = np.nan_to_num(action, nan=0.0)
        fixed_throttle = 0.8
        fixed_rudder = 0.0
        # Construct a full command with fixed values:
        command = (
            f"THROTTLE:{fixed_throttle:.2f},"
            f"ELEVATOR:{action[0]:.2f},"
            f"AILERON:{action[1]:.2f},"
            f"RUDDER:{fixed_rudder:.2f}\n"
        )
        try:
            self.client_socket.sendall(command.encode('utf-8'))
        except Exception as e:
            print(f"Command send failed: {str(e)}")
        
        obs = self._get_observation()
        reward = self._calculate_reward(obs, action)
        self.current_step += 1
        done = self.current_step >= self.episode_length
        self.prev_action = action.copy()

        return obs, reward, done, {}

    def _get_observation(self):
        timeout = time.time() + 1.0
        while time.time() < timeout:
            with self._lock:
                if self.state is not None:
                    pitch, bank, heading, ts = self.state
                    break
            time.sleep(0.01)
        else:
            # timed out without state → return zeros
            return np.zeros(6, dtype=np.float32)
        # if no previous, zero rates
        if not hasattr(self, 'prev_pitch'):
            d_pitch = 0.0
            d_bank  = 0.0
            d_heading = 0.0
        else:
            dt = ts - self.prev_ts if ts > self.prev_ts else 1e-6
            d_pitch = (pitch - self.prev_pitch) / dt
            d_bank  = (bank  - self.prev_bank ) / dt
            d_heading  = (heading  - self.prev_heading ) / dt

        d_pitch = np.clip(d_pitch, -1.0, 1.0)
        d_bank  = np.clip(d_bank,  -1.0, 1.0)
        d_heading  = np.clip(d_heading,  -1.0, 1.0)

        # save for next call
        self.prev_pitch, self.prev_bank, self.prev_heading, self.prev_ts = pitch, bank, heading, ts

        return np.array([pitch, bank, heading, d_pitch, d_bank, d_heading], dtype=np.float32)

    def reset(self):
        try:
            self.client_socket.sendall(b"RESET\n")
        except Exception as e:
            print(f"Reset failed: {str(e)}")
        
        command = (
            f"THROTTLE:{0.8:.2f},"
            f"ELEVATOR:{0.0:.2f},"
            f"AILERON:{0.0:.2f},"
            f"RUDDER:{0.0:.2f}\n"
        )
        try:
            self.client_socket.sendall(command.encode('utf-8'))
        except Exception as e:
            print(f"Command send failed: {str(e)}")
        self.state = None
        obs = self._get_observation()
        self.prev_heading = obs[2]
        self.prev_action = np.zeros(self.action_space.shape, dtype=np.float32)
        self.prev_error = None
        timeout = time.time() + 5.0
        while time.time() < timeout and self.state is None:
            time.sleep(0.1)
        self.current_step = 0
        time.sleep(2)
        return self._get_observation()

    def render(self, mode='human'):
        with self._lock:
            if self.state is not None:
                print("Current state:", self.state)
            else:
                print("No state available.")

    def close(self):
        self._stop_thread = True
        self.recv_thread.join()
        self.client_socket.close()
        self.log_csv_file.close()




class CSVLoggingCallback(BaseCallback):
    """
    A custom callback that logs training metrics to a CSV file.
    """
    def __init__(self, log_dir: str, verbose: int = 0):
        super(CSVLoggingCallback, self).__init__(verbose)
        self.log_dir = log_dir
        self.file_path = os.path.join(self.log_dir, "training_metrics.csv")
        self.file = None
        self.writer = None

    def _on_training_start(self) -> None:
        os.makedirs(self.log_dir, exist_ok=True)
        self.file = open(self.file_path, "w", newline="")
        self.writer = csv.writer(self.file)
        self.writer.writerow(["num_timesteps", "episode_reward", "episode_length"])
        if self.verbose > 0:
            print("CSVLoggingCallback: Logging to", self.file_path)

    def _on_step(self) -> bool:
        infos = self.locals.get("infos", [])
        for info in infos:
            if "episode" in info:
                episode_info = info["episode"]
                # Episode info typically contains 'r' for reward and 'l' for length.
                episode_reward = episode_info.get("r")
                episode_length = episode_info.get("l")
                self.writer.writerow([self.num_timesteps, episode_reward, episode_length])
                self.file.flush()
        return True

    def _on_training_end(self) -> None:
        if self.file is not None:
            self.file.close()


csv_callback = CSVLoggingCallback(log_dir="./logs", verbose=2)
checkpoint_callback = CheckpointCallback(
    save_freq=100_000,
    save_path="./checkpoints",
    name_prefix="PPO-flight"
)

eval_csv_path = "eval_results.csv"
eval_file = open(eval_csv_path, "w", newline="")
eval_writer = csv.writer(eval_file)
eval_writer.writerow([
    "timesteps",
    "mean_return", "std_return",
    "rmse_heading", "std_heading",
    "pct_within_5deg",
    "mean_pitch_deg", "std_pitch_deg",
    "mean_bank_deg",  "std_bank_deg",
    "mean_effort",    "std_effort",
])

# Helper to aggregate per-episode metrics
def compute_flight_metrics(model, env, n_episodes=5):
    ep_stats = []
    for _ in range(n_episodes):
        obs = env.reset()
        done = False
        heading_initial = obs[2] * 10.0  

        pitch_errs   = []
        bank_errs    = []
        heading_errs = []
        efforts      = []

        while not done:
            action, _ = model.predict(obs, deterministic=True)
            obs, _, done, _ = env.step(action)

            # back‐convert from normalized to degrees
            pitch_deg   = abs(obs[0]) * 30.0
            bank_deg    = abs(obs[1]) * 30.0
            heading_deg = obs[2] * 10.0     


            heading_err = abs(heading_deg - heading_initial)

            pitch_errs.append(pitch_deg)
            bank_errs.append(bank_deg)
            heading_errs.append(heading_err)
            efforts.append(np.linalg.norm(action))

        ep_stats.append({
            "mean_pitch":   np.mean(pitch_errs),
            "mean_bank":    np.mean(bank_errs),
            "rmse_heading": np.sqrt(np.mean(np.square(heading_errs))),
            "pct_within5":  np.mean(np.array(heading_errs) < 5.0) * 100.0,
            "mean_effort":  np.mean(efforts),
        })

    # aggregate across episodes
    stats = {}
    stats["rmse_heading"]   = (
        np.mean([e["rmse_heading"] for e in ep_stats]),
        np.std ([e["rmse_heading"] for e in ep_stats])
    )
    stats["pct_within5"]    = (
        np.mean([e["pct_within5"]  for e in ep_stats]),
        np.std ([e["pct_within5"]  for e in ep_stats])
    )
    stats["mean_pitch_deg"] = (
        np.mean([e["mean_pitch"]   for e in ep_stats]),
        np.std ([e["mean_pitch"]   for e in ep_stats])
    )
    stats["mean_bank_deg"]  = (
        np.mean([e["mean_bank"]    for e in ep_stats]),
        np.std ([e["mean_bank"]    for e in ep_stats])
    )
    stats["mean_effort"]    = (
        np.mean([e["mean_effort"]  for e in ep_stats]),
        np.std ([e["mean_effort"]  for e in ep_stats])
    )
    return stats

def train_and_eval(model, eval_env, train_steps: int, n_eval: int = 5, cum_steps_holder=[0]):
    model.learn(
        total_timesteps=train_steps,
        reset_num_timesteps=False,
        callback=[csv_callback, checkpoint_callback],
    )
    cum_steps_holder[0] += train_steps

    mean_return, std_return = evaluate_policy(
        model,
        eval_env,
        n_eval_episodes=n_eval,
        deterministic=True,
    )

    fm = compute_flight_metrics(model, eval_env, n_episodes=n_eval)

    print(f"After {cum_steps_holder[0]:>7} steps: "
          f"return={mean_return:.2f}±{std_return:.2f}, "
          f"RMSE_hd={fm['rmse_heading'][0]:.2f}±{fm['rmse_heading'][1]:.2f}, "
          f"%≤5°={fm['pct_within5'][0]:.1f}±{fm['pct_within5'][1]:.1f}")

    eval_writer.writerow([
        cum_steps_holder[0],
        mean_return, std_return,
        fm["rmse_heading"][0], fm["rmse_heading"][1],
        fm["pct_within5"][0],
        fm["mean_pitch_deg"][0], fm["mean_pitch_deg"][1],
        fm["mean_bank_deg"][0],  fm["mean_bank_deg"][1],
        fm["mean_effort"][0],    fm["mean_effort"][1],
    ])
    eval_file.flush()

    return mean_return, std_return

train_env = RealTimeFlightEnv(host="127.0.0.1", port=54000, episode_length=1000)


eval_env = RealTimeFlightEnv(
    host="127.0.0.1",
    port=52000,            
    episode_length=6000,
)

new_logger = configure(
    "training_logs",                    
    ["stdout", "tensorboard", "csv"]    
)

policy_kwargs = dict(
    net_arch=[256, 256],    
    activation_fn=th.nn.ReLU
)

model = TD3(
    "MlpPolicy",
    train_env,
    learning_rate=1e-5,
    buffer_size=int(1e6),
    batch_size=512,
    gamma=0.99,
    tau=0.005, 
    gradient_steps=1,
    target_policy_noise=0.2,
    target_noise_clip=0.5,
    policy_delay=2, 
    verbose=2,
    tensorboard_log="./flight_logs",
    policy_kwargs=policy_kwargs
)
model.set_logger(new_logger)

train_chunks = [500_00] * 12
elav = []
for chunk in train_chunks:
    el = train_and_eval(model, eval_env, chunk, n_eval=20)
    elav.append(el)
eval_file.close()

model.save("ppo_flight_leveler")

for i, (mean, std) in enumerate(elav, 1):
    print(f" Block {i:>2}: {train_chunks[i-1]:>7} steps → "
          f"{mean:.2f} ± {std:.2f}")
    
