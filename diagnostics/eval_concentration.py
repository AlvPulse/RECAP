"""
Measure avg unique users per step for the RL vs baselines (multi-episode).
This directly tracks whether RL is learning the array-concentration strategy.

Key threshold: Random gets 3.31 unique users/step with 21% SINR pass rate.
When RL drops below 3.31 unique/step, it has learned concentration.

Run: python diagnostics/eval_concentration.py --rl models/checkpoints/ppo_uav_50000_steps.zip
"""
import sys, os, argparse
sys.path.append(os.getcwd())

import numpy as np
from src.uav_comm.utils.config_loader import load_config
from src.uav_comm.envs.core import UAVEnv
from src.uav_comm.agents.baselines import MultiUserBaselines

N_EPISODES = 20
MAX_STEPS = 120
SEEDS = [i * 100 for i in range(N_EPISODES)]


def measure_episode(env, action_fn, seed, n_steps=MAX_STEPS):
    env.reset(seed=seed)
    sinr_passes = 0
    unique_per_step = []
    for step in range(n_steps):
        action = action_fn()
        obs, r, done, trunc, info = env.step(action)
        unique_per_step.append(len(np.unique(action)))
        sinr_passes += np.sum(env.sinr > env.sinr_threshold_linear)
        if done or trunc:
            break
    completion = float(np.mean(env.progress >= env.needs))
    sinr_rate = sinr_passes / (env.num_arrays * n_steps)
    return np.mean(unique_per_step), sinr_rate, completion


def measure_rl_episode(vec_env, model, seed, n_steps=MAX_STEPS):
    # Note: VecNormalize.reset() overrides inner.reset(seed), so RL episodes
    # run on effectively random seeds per episode. Aggregate stats are still
    # representative; we just can't do per-seed paired comparisons.
    obs = vec_env.reset()
    inner = vec_env.envs[0].env
    inner.reset(seed=seed)
    obs = vec_env.reset()
    sinr_passes = 0
    unique_per_step = []
    last_progress = np.zeros(inner.num_users)
    for step in range(n_steps):
        # Snapshot progress BEFORE step — DummyVecEnv auto-resets on done=True
        # which wipes inner.progress. We capture the pre-step value as best estimate.
        last_progress = inner.progress.copy()
        action_masks = np.array([e.env.get_action_mask() for e in vec_env.envs])
        action, _ = model.predict(obs, action_masks=action_masks, deterministic=True)
        obs, r, done_arr, _ = vec_env.step(action)
        unique_per_step.append(len(np.unique(action[0])))
        sinr_passes += np.sum(inner.sinr > inner.sinr_threshold_linear)
        if done_arr[0]:
            break
    # After done, inner.progress was reset by DummyVecEnv. Use last snapshot instead.
    needs = inner.needs if hasattr(inner, 'needs') else np.ones(inner.num_users) * 10
    completion = float(np.mean(last_progress >= needs))
    sinr_rate = sinr_passes / (inner.num_arrays * n_steps)
    return np.mean(unique_per_step), sinr_rate, completion


def print_row(label, uniq, sinr_rate, comp):
    note = ""
    if uniq < 3.31:
        note = "  << CONCENTRATION LEARNED"
    elif uniq < 3.5:
        note = "  (learning concentration)"
    print(f"  {label:<22} unique={uniq:.2f}  SINR-pass={sinr_rate:.1%}  complete={comp:.2f}{note}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--rl', default=None)
    args = parser.parse_args()

    cfg = load_config()
    env = UAVEnv(config=cfg)
    bl = MultiUserBaselines(env)

    print(f"\nConcentration analysis over {N_EPISODES} episodes")
    print("  Threshold: Random baseline gets 3.31 unique/step, 21% SINR pass rate")
    print("=" * 72)

    for label, fn in [('Multi-Random', bl.multi_random), ('Multi-FCFS', bl.multi_fcfs)]:
        uniq_list, sinr_list, comp_list = [], [], []
        for seed in SEEDS:
            bl.reset()
            u, s, c = measure_episode(env, fn, seed)
            uniq_list.append(u); sinr_list.append(s); comp_list.append(c)
        print_row(label, np.mean(uniq_list), np.mean(sinr_list), np.mean(comp_list))

    if args.rl and os.path.exists(args.rl):
        from sb3_contrib import MaskablePPO
        from sb3_contrib.common.wrappers import ActionMasker
        from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize

        vec_env = DummyVecEnv([lambda: ActionMasker(UAVEnv(config=cfg), lambda e: e.get_action_mask())])
        vecnorm_path = args.rl.replace('_steps.zip', '_steps_vecnorm.pkl')
        if os.path.exists(vecnorm_path):
            vec_env = VecNormalize.load(vecnorm_path, vec_env)
            vec_env.training = False; vec_env.norm_reward = False
            vn = "saved VecNorm"
        else:
            vec_env = VecNormalize(vec_env, norm_obs=True, norm_reward=False, clip_obs=10.0,
                norm_obs_keys=['needs','directions','distance','remaining_time','sinr_obs'])
            vn = "fresh VecNorm (approx)"

        model = MaskablePPO.load(args.rl, env=vec_env)
        uniq_list, sinr_list, comp_list = [], [], []
        for seed in SEEDS:
            u, s, c = measure_rl_episode(vec_env, model, seed)
            uniq_list.append(u); sinr_list.append(s); comp_list.append(c)
        steps = int(os.path.basename(args.rl).replace('ppo_uav_','').replace('_steps.zip',''))
        label = f"RL ({steps//1000}k, {vn[:5]})"
        print_row(label, np.mean(uniq_list), np.mean(sinr_list), np.mean(comp_list))

    print("=" * 72)
