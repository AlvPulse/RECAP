
import os
import sys
import numpy as np
import matplotlib.pyplot as plt
from sb3_contrib import MaskablePPO
from sb3_contrib.common.wrappers import ActionMasker
from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize

sys.path.append(os.getcwd())

from src.uav_comm.envs.core import UAVEnv
from src.uav_comm.agents.baselines import MultiUserBaselines

def mask_fn(env):
    return env.get_action_mask()

def evaluate_final():
    from src.uav_comm.utils.config_loader import load_config as load_env_config
    print("Starting Final Evaluation...")

    env_config = load_env_config()

    # Setup Env for Baselines (No masking needed for baselines logic, but physics matches)
    env = UAVEnv(config=env_config)
    baselines = MultiUserBaselines(env)

    # Load RL
    # We need the normalized env wrapper to load the stats
    # And we need to wrap with ActionMasker inside the DummyVecEnv
    def make_env():
        e = UAVEnv(config=env_config)
        e = ActionMasker(e, mask_fn)
        return e

    env_rl_wrapped = DummyVecEnv([make_env])
    env_rl_wrapped = VecNormalize.load("vec_normalize.pkl", env_rl_wrapped)
    env_rl_wrapped.training = False
    env_rl_wrapped.norm_reward = False

    model = MaskablePPO.load("ppo_multi_user_final", env=env_rl_wrapped)

    # Run episodes
    n_episodes = 1
    rewards_greedy = []
    rewards_fcfs = []
    rewards_rl = []

    print("\nEvaluating Multi-Greedy...")
    for i in range(n_episodes):
        obs, _ = env.reset(seed=i)
        done = False
        tot = 0
        while not done:
            action = baselines.multi_greedy()
            obs, reward, done, _, _ = env.step(action)
            tot += reward
        rewards_greedy.append(tot)

    print("\nEvaluating Multi-FCFS...")
    for i in range(n_episodes):
        obs, _ = env.reset(seed=i)
        done = False
        tot = 0
        while not done:
            action = baselines.multi_fcfs()
            obs, reward, done, _, _ = env.step(action)
            tot += reward
        rewards_fcfs.append(tot)

    print("\nEvaluating RL (MaskablePPO)...")
    for i in range(n_episodes):
        # Reset wrapped env
        env_rl_wrapped.seed(i)
        obs = env_rl_wrapped.reset()
        done = False
        tot = 0
        while not done:
            # Masking handled automatically by wrapped env + MaskablePPO if configured right,
            # or we pass mask explicitly.
            # With ActionMasker wrapped inside VecEnv, sb3_contrib usually handles it.
            # predict() will look for action_masks in the env.
            action, _ = model.predict(obs, deterministic=True)
            obs, reward, done, _ = env_rl_wrapped.step(action)
            tot += reward
        rewards_rl.append(tot) # Note: reward here is unnormalized because we set norm_reward=False

    print("\n--- Final Results ---")
    print(f"Multi-Greedy Avg Reward: {np.mean(rewards_greedy):.2f} +/- {np.std(rewards_greedy):.2f}")
    print(f"Multi-FCFS Avg Reward:   {np.mean(rewards_fcfs):.2f} +/- {np.std(rewards_fcfs):.2f}")
    print(f"Multi-User RL Avg Reward:{np.mean(rewards_rl):.2f} +/- {np.std(rewards_rl):.2f}")

    # Simple Plot
    plt.figure(figsize=(10, 6))
    means = [np.mean(rewards_greedy), np.mean(rewards_fcfs), np.mean(rewards_rl)]
    stds = [np.std(rewards_greedy), np.std(rewards_fcfs), np.std(rewards_rl)]
    labels = ['Multi-Greedy', 'Multi-FCFS', 'Multi-User RL']

    plt.bar(labels, means, yerr=stds, capsize=5)
    plt.ylabel("Average Episode Reward")
    plt.title("Multi-User UAV Communication: Algorithm Comparison")
    plt.savefig("evaluation_results.png")
    print("Plot saved to evaluation_results.png")

if __name__ == "__main__":
    evaluate_final()
