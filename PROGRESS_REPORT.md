# UAV-Aided Multi-User Communication Progress Report

## 1. Problem Statement
The primary challenge addressed by this solution is the dynamic and optimal allocation of communication resources from a moving Unmanned Aerial Vehicle (UAV) to multiple ground users. As the UAV travels, the channel conditions (distance, path loss, angles) constantly change. Ground users have varying data needs that must be satisfied within a strict time limit (episode duration).

A key problem is efficiently deciding *which users* should be served by *which antenna arrays* at any given time step to:
- Maximize the throughput delivered to users with active needs.
- Maintain fairness so that distant or hard-to-reach users are not starved (minimizing delay).
- Ensure no resources are wasted on users whose data needs have already been fulfilled.
- Account for realistic physical layer constraints such as beamforming, interference, and SINR thresholds.

## 2. Proposed Solution and Algorithmic Novelties
The solution models the resource allocation problem as a Markov Decision Process (MDP) and employs Deep Reinforcement Learning (DRL) to find an optimal policy. The codebase has been refactored from prototype notebooks into a modular, production-ready environment compatible with the Gymnasium API.

**Algorithmic Novelties & Key Features:**
- **Maskable Proximal Policy Optimization (MaskablePPO):** The agent uses a MaskablePPO architecture to prevent the selection of invalid actions. An `ActionMasker` wraps the environment and dynamically masks out users who have already achieved their data needs, ensuring the agent focuses strictly on users requiring service.
- **Fairness-Aware Multi-Objective Reward Function:** Rather than solely optimizing for throughput, the reward function heavily penalizes unfairness. It integrates **Jain's Fairness Index** (JFI) for user delay and **Proportional Fairness** for progress. Furthermore, it incorporates specific milestone bonuses (e.g., serving users up to their "half-life" needs) and penalties for failing to complete tasks within the time limit.
- **High-Fidelity Physical Layer Integration:** The simulation does not use abstract connectivity graphs. Instead, it accurately calculates Line-of-Sight (LoS) path loss, dynamically simulates uniform planar antenna arrays, calculates precise beamforming phase shifts, and models the resulting Signal-to-Interference-plus-Noise Ratio (SINR).
- **MLOps Integration:** The system leverages MLflow to systematically log hyperparameter configurations (from `yaml` files) and track experiment performance metrics across runs, facilitating reproducibility.

## 3. Operating Modes and Baselines
The environment is built to support various complexities of the simulation:
- **Single-User Mode:** Serves as a fundamental testing ground where the UAV serves one user at a time, allowing for the verification of path loss, tracking, and basic reward mechanics without interference complexities.
- **Multi-User Mode:** The primary operating mode where the UAV is equipped with multiple antenna sub-arrays, allowing it to serve multiple distinct users simultaneously. It requires resolving interference and ensuring equitable multi-user scheduling.

**Baselines for Comparison:**
To prove the efficacy of the RL approach, the agent is benchmarked against established heuristic scheduling algorithms:
1. **Multi-Greedy (Needs-based):** A heuristic that consistently allocates antenna arrays to the users with the highest remaining data needs. While it minimizes overall data deficit, it may struggle with rapidly changing channels or isolated users.
2. **Multi-FCFS (First-Come, First-Served / Proximity-based):** A heuristic that allocates resources to the closest active users first. This maximizes instantaneous SINR but often leads to severe fairness issues, completely starving edge users.
3. **Reinforcement Learning Agent:** The MaskablePPO agent, which aims to balance the throughput efficiency of proximity-based allocation with the equitable distribution required by the fairness metrics.

## 4. Real-World Applications and Benefits
The underlying architecture of this UAV-aided dynamic resource allocation scheme can be directly applied to several pressing technological domains:

- **Internet of Things (IoT) Data Harvesting:** In massive IoT deployments (like precision agriculture or remote industrial monitoring), sensors often lack the power to transmit data over long distances. UAVs acting as mobile data sinks can fly over these fields. This algorithm ensures that the UAV efficiently drains data from all sensors equitably before its flight battery depletes.
- **Emergency and Disaster Communications:** When terrestrial cellular infrastructure is destroyed by natural disasters, UAVs can act as temporary flying base stations. The fairness-aware algorithms guarantee that isolated individuals or separated rescue teams all receive a fair share of bandwidth for critical communications, rather than bandwidth being hogged by a dense cluster of users.
- **Smart Cities and Traffic Offloading:** In areas experiencing temporary massive crowds (e.g., stadiums, festivals), terrestrial networks become congested. UAVs can be deployed to offload traffic. The multi-array beamforming logic ensures that the UAVs can handle significant capacity by spatially multiplexing links to distinct sub-groups within the crowd without causing destructive interference.
- **Military and Tactical Operations:** Secure, rapid deployment of communication networks for moving ground units where minimizing connection delay across all units is mission-critical.

## 5. RL Performance Analysis and Challenges
Currently, the Reinforcement Learning agent (MaskablePPO) exhibits poorer performance compared to simpler heuristic baseline algorithms like Multi-Greedy and Multi-FCFS. Our investigation into the RL setup and reward engineering highlights the following critical reasons for this underperformance:

- **Action Masking Implementation for MultiDiscrete Spaces:** The environment uses a `MultiDiscrete` action space, where each antenna array can independently choose a user to serve. For `sb3-contrib`, the action mask for a `MultiDiscrete` space should ideally be a list of boolean arrays (one for each dimension) rather than a flat 1D boolean mask. If the mask is incorrectly structured, the agent may end up picking satisfied (invalid) users and receive repeated penalties.
- **Environment Wrapper Ordering and the Monitor:** If standard Gym wrappers like `Monitor` (necessary for logging episode metrics to MLflow) are placed incorrectly around the `ActionMasker`, the agent might fail to bypass the wrapper layers to correctly invoke `env.unwrapped.get_action_mask()`. This leads to silent failures where invalid actions aren't actually masked during training.
- **Reward Engineering Complexity:** The reward function has been heavily engineered and attempts to simultaneously optimize for urgency (LWDF-inspired throughput), max-min fairness (minimum progress ratio), equity (Jain's Fairness Index for delay), and episode completion bonuses. Such dense, multi-objective scalarized rewards are notoriously difficult for PPO to optimize effectively without extremely careful coefficient tuning and potentially much longer training time.
- **Insufficient Training Timesteps:** Training is currently configured for a limited number of timesteps (e.g. 1024 to 1M). Given the complex, continuous state space (users' locations, SINR, dynamic path-loss) and the large multi-discrete action space, the agent likely requires tens of millions of interactions to discover a policy that outperforms the highly specialized deterministic baselines.
