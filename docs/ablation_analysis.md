# Ablation Analysis: From Deterministic Masking to Soft Probabilistic Gambling

## 1. Introduction and Motivation
The original environment implemented a strict "Deterministic Repair Step." When arrays in the multi-user scenario selected conflicting users (users clustered too closely, causing severe interference based on the Gram matrix of their steering vectors), the environment forcefully re-assigned the conflicting array to a safe user.

While this guaranteed collision-free beamforming, it artificially boosted the performance of naive single-agent heuristic planners (like FCFS or Round Robin) because the environment acted as a "safety net." Furthermore, this centralized, instantaneous masking is physically unrealistic for real-world distributed telecom sites suffering from delayed message passing and distributed action-making.

This ablation removes the deterministic repair entirely. Agents (baselines and RL) are now forced to "gamble" on their joint actions. If they overlap beams, the physical simulation natively calculates the resulting low SINR, and the user receives 0 throughput.

## 2. Theoretical Expectations

### The Planners (Baselines)
**Theoretical Expectation:** Without the environment's forced repair step, simple heuristics like Multi-FCFS (M-FCFS) or Multi-Round-Robin (M-RR) should see a catastrophic drop in performance. Because they make decisions independently without accounting for the joint interference matrix, they will inevitably target spatially proximate users, triggering the interference penalty (SINR < threshold) constantly.
**Result:** As predicted, M-FCFS completion rates plummeted from ~74% (with masking) to ~18% (pure gambling). This confirms that the previous high performance of the baselines was an artifact of the environment's intervention, not the algorithms themselves.

### The Reinforcement Learning Agent
**Theoretical Expectation:** The RL agent should theoretically learn to avoid these collisions by navigating the 4096-action space (`MultiDiscrete([8,8,8,8])`). However, without the mask restricting the space, the positive reward signal becomes incredibly sparse. A random agent will almost always trigger a collision.
To aid this, we introduced a "Gambling Penalty" into the `train_reward` to give the agent a negative gradient to follow away from collisions, while preserving `true_reward` for pure telecom evaluation.
**Result:** Even with the gambling penalty, the RL agent currently struggles to outperform the plummeted baselines in short training windows (300,000 timesteps).

## 3. Analysis of Discrepancies and Bottlenecks
Why does the RL model still underperform in the gambling scenario?

1. **Dimensionality vs. Timesteps:** An action space of size $8^4$ coupled with continuous state spaces (locations, remaining needs, etc.) is massive. 300k timesteps is sufficient for fine-tuning a masked space, but grossly inadequate for mapping the complex, non-linear trigonometric boundaries of raw interference from scratch. Millions of timesteps (5M+) are typically required.
2. **Exploration Paralysis:** The introduced "Gambling Penalty" (-1.5 per failure) might be inducing conservative behavior. If the penalty for failing is too sharp relative to the reward of succeeding, the agent may learn that "doing nothing" (or finding an edge-case to avoid the penalty) yields a higher expected return than risking a multi-user transmission.
3. **Observation Space Deficit:** Currently, the agent must infer the interference geometry purely from user coordinates and azimuth angles. A shallow MLP (`pi: [256, 128]`) struggles to learn the underlying trigonometric relationships (e.g., $sin(\theta_1 - \theta_2)$) necessary to predict collisions.

## 4. Proposed Pathways for Next Iterations
To achieve a rigorous, adaptable telecom learning package that consistently outperforms heuristics in this unmasked environment, we recommend the following next steps:

* **Pathway 1: Curriculum Learning.** Re-introduce the deterministic mask, but decay its intervention probability from 1.0 to 0.0 over 2 million timesteps. This allows the RL to bootstrap off dense rewards before facing the harsh reality of the unmasked physics.
* **Pathway 2: Explicit State Representation.** Pass the estimated interference matrix (or a simplified local CSI representation) directly into the agent's observation space. Instead of forcing the MLP to deduce the geometry, we hand it the "Carrier Sense" metric, bridging the gap to real-world CSMA/CA protocols.
* **Pathway 3: Reward Reshaping.** Instead of penalizing failures, heavily scale the positive rewards for successful *simultaneous* non-conflicting transmissions, encouraging aggressive exploration rather than conservative safety.
## 5. Summary of Implementation Changes
- **`src/uav_comm/envs/core.py`**: Removed the deterministic "Repair Step." Agents now choose freely and face 0 throughput if interference is too high. Separated `raw_reward` into `train_reward` (which includes behavioral penalties for gambling failures) and `true_reward` (which is tracked in the `info` dict for unbiased evaluation). Fixed `evaluate_final.py` crash by storing the conflict matrix securely inside the environment state rather than passing it via the SB3 observation dictionary.
- **`evaluate_final.py`**: Updated to exclusively read `info['true_reward']` to report pure telecom performance.
- **`configs/env_config.yaml`**: Added the `enable_spatial_masking` toggle to control whether the environment provides strict safety masks or pure gambling logic.
- **`configs/train_config.yaml`**: Increased base timesteps to 300,000 to better accommodate the harder exploration landscape.
