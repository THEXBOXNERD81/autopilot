import socket
import gym
from gym import spaces
import threading
import time
import numpy as np
from stable_baselines3 import PPO, TD3, SAC

# def parse_state(csv_line):
#     """
#     Parses a CSV string containing flight state data.
#     Expected CSV format: timestamp, airspeed, pitch, bank, heading, verticalSpeed, engineRPM, altitude
#     We ignore the timestamp and return a raw state vector with 7 elements.
#     """
#     parts = csv_line.strip().split(',')
#     if len(parts) != 8:
#         print("Invalid data length:", csv_line)
#         return None
#     try:
#         # Convert parts 1 to 7 (ignoring timestamp at index 0) to float
#         state = np.array([float(x) for x in parts[1:]], dtype=np.float32)
#         return state
#     except Exception as e:
#         print("Parsing error:", e)
#         return None

# def normalize_state(raw_state):
#     """
#     Normalize raw state values to [-1,1] using the same ranges as in training.
#     Order: Airspeed, Pitch, Bank, Heading, VerticalSpeed, EngineRPM, Altitude.
#     """
#     norms = [
#         (0, 250),       # Airspeed (knots)
#         (-60, 60),      # Pitch (degrees)
#         (-90, 90),      # Bank (degrees)
#         (0, 359),       # Heading (degrees true)
#         (-10000, 10000),# Vertical speed (ft/min)
#         (0, 3000),      # EngineRPM
#         (0, 10000)      # Altitude (feet)
#     ]
#     return np.array([2 * ((raw_state[i] - mn) / (mx - mn)) - 1 
#                      for i, (mn, mx) in enumerate(norms)], dtype=np.float32)

# def main():
#     host = 'localhost'
#     port = 54000

#     # Create a TCP/IP socket and connect to the C++ server
#     client_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
#     client_socket.connect(
#         (host, port))
#     print(f"Connected to server at {host}:{port}")

#     # Load the pre-trained RL model
#     model = SAC.load("best_model.zip")
    
#     # Set a timeout for recv so that we don't block indefinitely
#     client_socket.settimeout(1.0)

#     try:
#         while True:
#             try:
#                 # Receive state data
#                 data = client_socket.recv(4096)
#                 if not data:
#                     print("No data received. Connection might be closed.")
#                     break
#                 csv_line = data.decode('utf-8')
#                 print("Received state:", csv_line.strip())

#                 # Parse the CSV string into a raw state vector (7 elements)
#                 raw_state = parse_state(csv_line)
#                 if raw_state is None:
#                     print("Failed to parse state data.")
#                     continue

#                 # Normalize the raw state before feeding it into the model
#                 norm_state = normalize_state(raw_state)
#                 obs = norm_state.reshape(1, -1)

#                 # Get action from the model (model expects normalized observations)
#                 action, _ = model.predict(obs, deterministic=True)
#                 action = np.array(action).flatten()  # Ensure flat array

#                 # Format the action into a command string.
#                 # For example, if the command expects: THROTTLE, ELEVATOR, AILERON, RUDDER.
#                 command = f"THROTTLE:{0.70:.2f},ELEVATOR:{float(action[0]):.2f},AILERON:{float(action[1]):.2f},RUDDER:{0.00:.2f}\n"
#                 print("Sending command:", command.strip())

#                 # Send the command to the simulator
#                 client_socket.sendall(command.encode('utf-8'))

#             except socket.timeout:
#                 continue
#             except Exception as e:
#                 print("Error:", e)
#                 break

#             time.sleep(0.1)

#     except KeyboardInterrupt:
#         print("Interrupted by user.")

#     finally:
#         client_socket.close()
#         print("Connection closed.")

# if __name__ == '__main__':
#     main()


class RealTimeFlightEnv(gym.Env):
    def __init__(self, host='127.0.0.1', port=54000, episode_length=200, target_speed=150):
        super(RealTimeFlightEnv, self).__init__()
        
        # Observation space remains with 7 values
        self.observation_space = spaces.Box(
            low=-1.0,
            high=1.0,
            shape=(6,),
            dtype=np.float32
        )
        
        # Action space now has only 2 actions: Elevator (pitch) and Aileron (bank)
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
        self.client_socket.settimeout(3.0)

        # State management
        self.state = None
        self._lock = threading.Lock()
        self._stop_thread = False
        self.recv_thread = threading.Thread(target=self._receive_state)
        self.recv_thread.start()

        # Open CSV log file for observations (raw and normalized)
        self.log_csv_file = open("observations.csv", "w")
        header = "timestamp," + ",".join([f"raw_{i}" for i in range(7)]) + "," + ",".join([f"norm_{i}" for i in range(7)]) + "\n"
        self.log_csv_file.write(header)
        self.log_csv_file.flush()

        # Initialize previous action and error for reward calculations.
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
                            # extract only the raw pitch & bank from the 7-element state
                            raw_pitch = state[1]
                            raw_bank  = state[2]
                            raw_heading = state[3]

                            # normalize those two values
                            full_norm = self._normalize_state((raw_pitch, raw_bank, raw_heading))
                            # full_norm[0] is norm_pitch, full_norm[1] is norm_bank
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
        # For reward, we still compare the normalized pitch and bank against targets.
        # Here, we use a normalization range for pitch and bank appropriate for a stable flight.
        pitch = obs[0]
        bank = obs[1]
        heading = obs[2]
        pitch_rate = obs[3]
        bank_rate  = obs[4]
        heading_rate = obs[5]
        
        # Define target values (normalized)
        target_airspeed = self._normalize_value(self.target_speed, 0, 250)
        target_heading = self._normalize_value(270, 0, 359)
        target_vertical_speed = 0.0
        target_engineRPM = self._normalize_value(2000, 0, 3000)
        target_altitude = self._normalize_value(3000, 0, 10000)


        # Compute errors (using squared error)
        # error_airspeed = (self._normalize_value(obs[0], 0, 250) - target_airspeed) ** 2
        error_pitch = pitch**2
        error_bank = bank**2
        # error_heading = (heading - self.prev_heading)**2
        # self.prev_heading = heading.copy()

        # Define weights for each term
        w_speed = 0.0
        w_pitch = 1.0
        w_bank = 1.0
        w_heading = 1.0
        w_altitude = 0.0
        w_rate = 0.5       # tune this weight


        pitch_reward = np.exp(-16.0 * (w_pitch * error_pitch))
        bank_reward = np.exp(-16.0 * (w_bank * error_bank))
        # heading_reward = np.exp(-16.0 * (w_heading * error_heading))

        rate_penalty = w_rate * (pitch_rate**2 + bank_rate**2 + heading_rate**2)

        reward = pitch_reward + bank_reward - rate_penalty # + heading_reward
        # reward += (w_pitch * pitch_penalty + w_bank * bank_penalty)    
        
        # Progress Reward
        # progress_lambda = 0.5
        # if self.prev_error is not None and total_error < self.prev_error:
        #     reward += progress_lambda
        # self.prev_error = total_error

        # Smoothness Reward
        w_a = 0.1
        if self.prev_action is not None:
            delta_action = np.linalg.norm(action - self.prev_action)
            reward -= w_a * abs(delta_action)
        
        # Safety Boundaries Reward:
        # If pitch or bank deviate beyond a safe normalized threshold, penalize heavily.
        safety_threshold = 0.1  # Adjust threshold (normalized)
        gamma = 5.0
        if abs(pitch) > safety_threshold or abs(bank) > safety_threshold:
            reward -= gamma


        return np.clip(reward, -10.0, 10.0).item()

    def step(self, action):
        # Clip and prepare action (only two controls: elevator and aileron)
        action = np.clip(action, self.action_space.low, self.action_space.high)
        action = np.nan_to_num(action, nan=0.0)
        # Since only pitch and bank are controlled, we set fixed throttle and rudder.
        fixed_throttle = 0.8
        fixed_rudder = 0.0
        fixed_aileron = 0.0
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
            return np.zeros(4, dtype=np.float32)
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

        # clip to [-1,1] or whatever range you choose
        d_pitch = np.clip(d_pitch, -1.0, 1.0)
        d_bank  = np.clip(d_bank,  -1.0, 1.0)
        d_heading  = np.clip(d_heading,  -1.0, 1.0)

        # save for next call
        self.prev_pitch, self.prev_bank, self.prev_heading, self.prev_ts = pitch, bank, heading, ts

        # return a 4-vector
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



# 1) instantiate your env exactly as in training
env = RealTimeFlightEnv(host="127.0.0.1", port=54000, episode_length=1000)

# 2) reset to get a correct 4-dim obs
obs = env.reset()

model = TD3.load("checkpoints\PPO-flight_170000_steps.zip", env=env)

while True:
    # 3) ask the model for an action
    action, _ = model.predict(obs, deterministic=True)

    # 4) send the action to sim and get the next obs
    obs, reward, done, info = env.step(action)

    # 5) if the episode ended (unlikely in a real sim loop), reset
    if done:
        obs = env.reset()