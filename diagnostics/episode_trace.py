"""
Trace a single episode to understand why users aren't completing.
Shows per-step SINR, throughput, and progress for baselines and (optionally) the RL.

Run: python diagnostics/episode_trace.py
     python diagnostics/episode_trace.py --rl models/checkpoints/ppo_uav_50000_steps.zip
"""

import sys, os, argparse
sys.path.append(os.getcwd())

import numpy as np
from src.uav_comm.utils.config_loader import load_config
from src.uav_comm.envs.core import UAVEnv
from src.uav_comm.agents.baselines import MultiUserBaselines

SEED = 42


def trace_episode(env, action_fn, label, n_steps=120):
    env.reset(seed=SEED)
    sinr_above_thr = 0
    unique_per_step = []

    for step in range(n_steps):
        action = action_fn()
        obs, r, done, trunc, info = env.step(action)

        n_unique = len(np.unique(action))
        unique_per_step.append(n_unique)

        n_above = np.sum(env.sinr > env.sinr_threshold_linear)
        sinr_above_thr += n_above
        if done or trunc:
            break

    avg_progress = np.mean(env.progress / np.maximum(env.needs, 1e-6))
    avg_unique   = np.mean(unique_per_step)
    sinr_pass_rate = sinr_above_thr / (env.num_arrays * n_steps)
    jfi = np.sum(env.progress)**2 / (env.num_users * np.sum(env.progress**2) + 1e-9)

    print(f"\n{'='*55}")
    print(f"  {label}")
    print(f"{'='*55}")
    print(f"  Steps run:             {n_steps}")
    print(f"  Users completing:      {np.sum(env.progress >= env.needs)}/{env.num_users}")
    print(f"  Avg progress ratio:    {avg_progress:.2f}  ({avg_progress*100:.0f}%)")
    print(f"  JFI:                   {jfi:.3f}")
    print(f"  SINR > threshold rate: {sinr_pass_rate:.2%}  ({sinr_above_thr}/{env.num_arrays*n_steps} user-slots)")
    print(f"  Avg unique users/step: {avg_unique:.2f} (out of {env.num_arrays} arrays)")
    print(f"  Actual progress/user:  {env.progress}")
    print(f"  Needs/user:            {env.needs}")
    print(f"  Progress ratio/user:   {env.progress / np.maximum(env.needs, 1e-6)}")

    return sinr_pass_rate, avg_unique


def trace_rl_episode(model, vec_env, label, n_steps=120):
    """Trace an RL episode with action masking. Uses inner env for stats."""
    from sb3_contrib import MaskablePPO

    obs = vec_env.reset()
    inner = vec_env.envs[0].env   # ActionMasker → UAVEnv
    inner.reset(seed=SEED)
    obs = vec_env.reset()

    sinr_above_thr = 0
    unique_per_step = []
    last_progress = np.zeros(inner.num_users)
    last_needs = np.ones(inner.num_users) * 10.0

    for step in range(n_steps):
        # Capture progress BEFORE step — DummyVecEnv auto-resets on done=True,
        # wiping inner.progress before we can read the terminal value.
        last_progress = inner.progress.copy()
        last_needs = inner.needs.copy()
        action_masks = np.array([e.env.get_action_mask() for e in vec_env.envs])
        action, _ = model.predict(obs, action_masks=action_masks, deterministic=True)
        obs, r, done_arr, _ = vec_env.step(action)

        chosen = action[0]  # first env
        n_unique = len(np.unique(chosen))
        unique_per_step.append(n_unique)

        n_above = np.sum(inner.sinr > inner.sinr_threshold_linear)
        sinr_above_thr += n_above
        if done_arr[0]:
            break

    avg_progress = np.mean(last_progress / np.maximum(last_needs, 1e-6))
    avg_unique   = np.mean(unique_per_step)
    sinr_pass_rate = sinr_above_thr / (inner.num_arrays * n_steps)
    jfi = np.sum(last_progress)**2 / (inner.num_users * np.sum(last_progress**2) + 1e-9)

    print(f"\n{'='*55}")
    print(f"  {label}")
    print(f"{'='*55}")
    print(f"  Steps run:             {n_steps}")
    print(f"  Users completing:      {np.sum(last_progress >= last_needs)}/{inner.num_users}")
    print(f"  Avg progress ratio:    {avg_progress:.2f}  ({avg_progress*100:.0f}%)")
    print(f"  JFI:                   {jfi:.3f}")
    print(f"  SINR > threshold rate: {sinr_pass_rate:.2%}  ({sinr_above_thr}/{inner.num_arrays*n_steps} user-slots)")
    print(f"  Avg unique users/step: {avg_unique:.2f} (out of {inner.num_arrays} arrays)")
    print(f"  Actual progress/user:  {last_progress}")
    print(f"  Needs/user:            {last_needs}")
    print(f"  Progress ratio/user:   {last_progress / np.maximum(last_needs, 1e-6)}")

    return sinr_pass_rate, avg_unique


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--rl', default=None, help='Path to RL checkpoint .zip')
    parser.add_argument('--seed', type=int, default=42, help='Episode seed')
    args = parser.parse_args()

    SEED = args.seed

    env = UAVEnv(config=load_config())
    bl  = MultiUserBaselines(env)

    print("\n--- FCFS baseline ---")
    bl.reset()
    fcfs_sinr, fcfs_unique = trace_episode(env, bl.multi_fcfs, "Multi-FCFS")

    print("\n--- Random baseline ---")
    bl.reset()
    rand_sinr, rand_unique = trace_episode(env, bl.multi_random, "Multi-Random")

    print("\n--- Greedy (by need) ---")
    bl.reset()
    greedy_sinr, greedy_unique = trace_episode(env, bl.multi_greedy, "Multi-Greedy")

    rl_sinr = rl_unique = None
    if args.rl:
        print(f"\n--- RL checkpoint: {args.rl} ---")
        from sb3_contrib import MaskablePPO
        from sb3_contrib.common.wrappers import ActionMasker
        from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize

        cfg = load_config()
        vec_env = DummyVecEnv([lambda: ActionMasker(UAVEnv(config=cfg), lambda e: e.get_action_mask())])

        vecnorm_path = args.rl.replace('_steps.zip', '_steps_vecnorm.pkl')
        if os.path.exists(vecnorm_path):
            print(f"  Using saved VecNormalize: {vecnorm_path}")
            vec_env = VecNormalize.load(vecnorm_path, vec_env)
            vec_env.training = False; vec_env.norm_reward = False
        else:
            print("  Fresh VecNormalize (approximate)")
            vec_env = VecNormalize(vec_env, norm_obs=True, norm_reward=False, clip_obs=10.0,
                norm_obs_keys=['needs','directions','distance','remaining_time','sinr_obs'])

        model = MaskablePPO.load(args.rl, env=vec_env)
        rl_sinr, rl_unique = trace_rl_episode(model, vec_env, f"RL ({os.path.basename(args.rl)})")

    print("\n\n--- Summary ---")
    print(f"  {'Algorithm':<20} {'SINR-pass%':>12} {'Avg-unique':>12}")
    print(f"  {'FCFS':<20} {fcfs_sinr:>11.1%} {fcfs_unique:>12.2f}")
    print(f"  {'Random':<20} {rand_sinr:>11.1%} {rand_unique:>12.2f}")
    print(f"  {'Greedy':<20} {greedy_sinr:>11.1%} {greedy_unique:>12.2f}")
    if rl_sinr is not None:
        print(f"  {'RL':<20} {rl_sinr:>11.1%} {rl_unique:>12.2f}")
    if rl_unique is not None:
        delta = rand_unique - rl_unique
        direction = "fewer" if delta > 0 else "more"
        print(f"\n  RL selects {abs(delta):.2f} {direction} unique users/step than Random.")
        if rl_unique < rand_unique:
            print("  --> RL has learned array concentration (good)!")
        else:
            print("  --> RL not yet concentrating arrays (still near-uniform).")
