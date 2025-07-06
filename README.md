# Autopilot: Reinforcement Learning for Level Flight

## Project Overview

**Autopilot** is a reinforcement learning project aimed at developing an intelligent flight control system that can maintain stable, level flight in a simulated aircraft. Using Microsoft Flight Simulator 2024 and a custom C++–Python integration, a TD3-based RL agent learns to control pitch and bank to keep the aircraft on a straight course — within ±5° of deviation — in real time.

The project demonstrates deep integration between flight dynamics, reinforcement learning, simulation control, and real-world inspired reward design. It serves as a practical exploration of how AI can perform fundamental autopilot tasks without traditional rule-based systems.

---

## Objective

> Can a reinforcement learning model maintain level flight with a deviation of less than ±5° in roll and pitch?

The project’s core aim was to create an RL-based autopilot that could stabilize an aircraft’s orientation in real time using only pitch and roll inputs — a simplified but pedagogically rich control task foundational to aviation.

---

## How It Works

### Simulation Environment

- **Microsoft Flight Simulator 2024 (MSFS2024)** was used as the high-fidelity flight environment.
- A **custom C++ socket integration** using the SimConnect API streamed live aircraft data to a Python-based RL agent and sent control commands back.

### Reinforcement Learning

- Implemented using **Stable-Baselines3** with the **TD3 algorithm**
- **Observation space**: Aircraft pitch, bank, heading, and their respective angular velocities
- **Action space**: Continuous control over elevator (pitch) and aileron (roll)
- **Reward shaping**: Combined positive rewards for staying level and stable, and penalties for abrupt or unsafe maneuvers

---

## Results

| Metric                          | Value            |
|--------------------------------|------------------|
| **Mean Evaluation Reward**     | 18,047           |
| **RMSE (Heading Deviation)**   | 0.55°            |
| **% of time within ±5°**       | 100.0%           |
| **Mean Control Effort**        | 0.109            |

### Key Observations

- The agent **learned to stabilize the aircraft after ~200,000 timesteps**
- Control inputs became smoother and more energy-efficient over time
- The system **achieved full compliance** with the deviation goal — staying within ±5° throughout test episodes

---

## Learning Highlights

- Integration of a **real-time RL agent** with a commercial-grade flight simulator
- Design of nuanced **reward functions** to balance stability, responsiveness, and safety
- Evaluation using **RMSE**, control effort, overshoot, and cumulative reward

---

## Next Steps

- Add **throttle control** to improve vertical stabilization and energy efficiency
- Train under **dynamic weather conditions** for improved robustness
- Expand control outputs to include **yaw (rudder)** and automate full flight sequences

---

## Demo Videos

- [Zero control baseline](https://drive.google.com/file/d/1RCoV2nTDnZWZo9qo0PKu1jVgsRtC5ONj/view?usp=sharing)  
- [Early training episodes](https://drive.google.com/file/d/1t6HA8Yxx1Mq00I5v_7KqZOriwBrH0l-K/view?usp=sharing)  
- [Final trained agent](https://drive.google.com/file/d/1aIqokKocqac5I3YC1aaab4siPDW4YapG/view?usp=sharing)

---

## About the Author

I'm **Leonardo Sjöberg**, a data science graduate from EC Utbildning in Sweden. My focus is on real-world AI integration, with interests in reinforcement learning, simulation environments, and control systems.

[Portfolio](https://datascienceportfol.io/leonardo01)

---