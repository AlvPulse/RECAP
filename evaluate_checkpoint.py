"""
Mid-training evaluation: baselines vs RL checkpoint.
Evaluates without needing saved VecNormalize stats (checkpoint-only evaluation).
Run: python evaluate_checkpoint.py [--checkpoint models/checkpoints/ppo_uav_50000_steps.zip]
"""

import sys, os, argparse
sys.path.append(os.getcwd())

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

from sb3_contrib import MaskablePPO
from sb3_contrib.common.wrappers import ActionMasker
from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize

from src.uav_comm.utils.config_loader import load_config
from src.uav_comm.envs.core import UAVEnv
from src.uav_comm.agents.baselines import MultiUserBaselines

N_EPISODES = 20
MAX_STEPS  = 120

def make_masked_env(env_cfg):
    e = UAVEnv(config=env_cfg)
    return ActionMasker(e, lambda env: env.get_action_mask())

def _jfi(values):
    values = np.array(values, dtype=float)
    values = values[values > 0]
    if len(values) < 2: return 1.0
    return float(np.sum(values) ** 2 / (len(values) * np.sum(values ** 2)))

def run_baseline_episode(env, bl, baseline_fn, seed):
    obs, _ = env.reset(seed=seed)
    bl.reset()
    total_reward = 0
    for s in range(MAX_STEPS):
        action = baseline_fn()
        obs, r, done, trunc, info = env.step(action)
        total_reward += r
        if done or trunc:
            break
    completion = float(np.mean(env.progress >= env.needs))
    return total_reward, completion, _jfi(env.progress), s + 1

def run_rl_episode(vec_env, model, seed):
    obs = vec_env.reset()
    inner_env = vec_env.envs[0].env   # ActionMasker → UAVEnv
    inner_env.reset(seed=seed)
    obs = vec_env.reset()

    total_reward = 0
    last_progress = np.zeros(inner_env.num_users)
    last_needs = np.ones(inner_env.num_users) * 10.0
    for s in range(MAX_STEPS):
        # Snapshot BEFORE step — DummyVecEnv auto-resets on done=True, wiping inner_env.progress.
        last_progress = inner_env.progress.copy()
        last_needs = inner_env.needs.copy()
        action_masks = np.array([e.env.get_action_mask() for e in vec_env.envs])
        action, _ = model.predict(obs, action_masks=action_masks, deterministic=True)
        obs, r, done, infos = vec_env.step(action)
        total_reward += float(r[0])
        if done[0]:
            break
    completion = float(np.mean(last_progress >= last_needs))
    return total_reward, completion, _jfi(last_progress), s + 1

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--checkpoint', default=None)
    args = parser.parse_args()

    env_cfg = load_config()
    print(f"Evaluating over {N_EPISODES} episodes per algorithm ...\n")

    results = {}

    # ── Baselines ──────────────────────────────────────────────────────────────
    env = UAVEnv(config=env_cfg)
    bl  = MultiUserBaselines(env)

    for name, fn in [
        ('Multi-Random',   bl.multi_random),
        ('Multi-FCFS',     bl.multi_fcfs),
        ('Multi-Greedy',   bl.multi_greedy),
        ('Multi-RR',       bl.multi_round_robin),
        ('Multi-PF',       bl.multi_proportional_fair),
        ('Multi-Angular',  bl.multi_angular_greedy),
    ]:
        rews, comps, jfis, steps = [], [], [], []
        for ep in range(N_EPISODES):
            r, c, j, s = run_baseline_episode(env, bl, fn, seed=ep * 100)
            rews.append(r); comps.append(c); jfis.append(j); steps.append(s)
        results[name] = dict(reward=np.mean(rews), complete=np.mean(comps),
                             jfi=np.mean(jfis), steps=np.mean(steps))
        print(f"{name:20s}: reward={np.mean(rews):7.3f}  complete={np.mean(comps):.2f}"
              f"  jfi={np.mean(jfis):.3f}  steps={np.mean(steps):.0f}")

    # ── RL checkpoint ──────────────────────────────────────────────────────────
    ckpt = args.checkpoint or 'models/checkpoints/ppo_uav_50000_steps.zip'
    if not os.path.exists(ckpt):
        # Try largest available checkpoint
        ckpt_dir = 'models/checkpoints'
        today_files = [f for f in os.listdir(ckpt_dir) if f.endswith('.zip')]
        if today_files:
            ckpt = os.path.join(ckpt_dir, sorted(today_files)[-1])

    if os.path.exists(ckpt):
        steps_in_name = os.path.basename(ckpt).replace('ppo_uav_','').replace('_steps.zip','')
        rl_label = f'RL ({steps_in_name})'
        print(f"\nLoading RL checkpoint: {ckpt}")

        vec_env = DummyVecEnv([lambda: make_masked_env(env_cfg)])

        # Use companion VecNormalize if available (saved by VecNormalizeCheckpointCallback)
        vecnorm_path = ckpt.replace('_steps.zip', '_steps_vecnorm.pkl')
        if os.path.exists(vecnorm_path):
            print(f"  Using saved VecNormalize: {vecnorm_path}")
            vec_env = VecNormalize.load(vecnorm_path, vec_env)
            vec_env.training = False   # eval mode: don't update running stats
            vec_env.norm_reward = False
        else:
            print("  VecNormalize stats not available — using fresh normalisation (approximate)")
            vec_env = VecNormalize(
                vec_env, norm_obs=True, norm_reward=False, clip_obs=10.0,
                norm_obs_keys=['needs','directions','distance','remaining_time','sinr_obs']
            )
        try:
            model = MaskablePPO.load(ckpt, env=vec_env)
            rews, comps, jfis, steps = [], [], [], []
            for ep in range(N_EPISODES):
                r, c, j, s = run_rl_episode(vec_env, model, seed=ep * 100)
                rews.append(r); comps.append(c); jfis.append(j); steps.append(s)
            results[rl_label] = dict(reward=np.mean(rews), complete=np.mean(comps),
                                     jfi=np.mean(jfis), steps=np.mean(steps))
            print(f"\n{rl_label:20s}: reward={np.mean(rews):7.3f}  complete={np.mean(comps):.2f}"
                  f"  jfi={np.mean(jfis):.3f}  steps={np.mean(steps):.0f}")
        except Exception as e:
            print(f"[WARNING] Could not load RL checkpoint: {e}")
    else:
        print(f"\n[INFO] No RL checkpoint found at {ckpt}, skipping RL evaluation.")

    # ── Plot ───────────────────────────────────────────────────────────────────
    names   = list(results.keys())
    metrics = ['reward', 'complete', 'jfi', 'steps']
    labels  = ['Episode Reward', 'Completion Rate', "Jain's Fairness Index", 'Steps to End']
    colors  = ['#4CAF50' if 'RL' in n else '#2196F3' for n in names]

    fig, axes = plt.subplots(2, 2, figsize=(14, 8))
    fig.suptitle('Baselines vs RL (mid-training checkpoint)', fontsize=13)
    for ax, metric, ylabel in zip(axes.flat, metrics, labels):
        vals = [results[n][metric] for n in names]
        bars = ax.bar(names, vals, color=colors)
        ax.set_title(ylabel)
        ax.set_xticks(range(len(names)))
        ax.set_xticklabels(names, rotation=20, ha='right', fontsize=8)
        ax.bar_label(bars, fmt='%.2f', fontsize=7, padding=2)
        ax.grid(True, alpha=0.3, axis='y')
    plt.tight_layout()

    os.makedirs('results', exist_ok=True)
    out = 'results/checkpoint_eval.png'
    plt.savefig(out, dpi=150)
    print(f"\nSaved: {out}")

    # ── Summary table ──────────────────────────────────────────────────────────
    print("\n" + "="*68)
    print(f"{'Algorithm':20s} {'Reward':>8} {'Complete':>9} {'JFI':>7} {'Steps':>6}")
    print("-"*68)
    for name, d in results.items():
        rl_tag = " *" if 'RL' in name else "  "
        print(f"{name:20s} {d['reward']:8.3f} {d['complete']:9.2f} {d['jfi']:7.3f} {d['steps']:6.0f}{rl_tag}")
    print("="*68)
    rl_keys = [k for k in results if 'RL' in k]
    if rl_keys:
        # Report whether proper VecNorm stats were used
        vecnorm_path = ckpt.replace('_steps.zip', '_steps_vecnorm.pkl')
        vn_status = "saved VecNorm" if os.path.exists(vecnorm_path) else "fresh VecNorm (approximate)"
        print(f"  * RL checkpoint: {os.path.basename(ckpt)} | normalisation: {vn_status}")

if __name__ == '__main__':
    main()
