
import os
import sys
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from sb3_contrib import MaskablePPO
from sb3_contrib.common.wrappers import ActionMasker
from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize

sys.path.append(os.getcwd())

from src.uav_comm.envs.core import UAVEnv
from src.uav_comm.agents.baselines import MultiUserBaselines

N_EPISODES = 30

# ──────────────────────────────────────────────────────────────────────────────
# Episode runner helpers
# ──────────────────────────────────────────────────────────────────────────────

def _jfi(values):
    """Jain's Fairness Index on an array of positive numbers."""
    if len(values) == 0:
        return 1.0
    s = np.sum(values)
    return s ** 2 / (len(values) * np.sum(values ** 2) + 1e-9)


def run_baseline_episode(env, action_fn, seed):
    """Run one episode with a deterministic baseline action function. Returns metrics dict."""
    obs, _ = env.reset(seed=seed)
    done = truncated = False
    total_reward = 0.0
    step = 0

    while not done and not truncated:
        action = action_fn()
        obs, reward, done, truncated, info = env.step(action)
        total_reward += reward
        step += 1

    n_users = env.num_users
    served = np.sum(env.progress >= env.needs)
    completion_rate = served / n_users

    # Fairness: JFI on final progress ratios of ALL users
    prog_ratios = env.progress / np.maximum(env.needs, 1e-6)
    fairness = _jfi(prog_ratios)

    # Time to completion (max_time if not fully served)
    time_used = env.current_time
    max_time = env.config['max_episode_time']
    completion_time = time_used if served == n_users else max_time

    return {
        'reward': total_reward,
        'completion_rate': completion_rate,
        'fairness_jfi': fairness,
        'completion_time': completion_time,
        'steps': step,
    }


def run_rl_episode(model, vec_env, seed, max_steps=120):
    """Run one RL episode with action masking. Returns metrics dict."""
    obs = vec_env.reset()
    inner = vec_env.envs[0].env   # ActionMasker → UAVEnv
    inner.reset(seed=seed)
    obs = vec_env.reset()

    total_reward = 0.0
    last_progress = np.zeros(inner.num_users)
    last_needs = np.ones(inner.num_users) * 10.0
    last_time = 0
    all_done = False
    for _ in range(max_steps):
        # Snapshot BEFORE step — DummyVecEnv auto-resets on done=True, wiping inner.progress.
        last_progress = inner.progress.copy()
        last_needs = inner.needs.copy()
        last_time = getattr(inner, 'current_time', 0)
        action_masks = np.array([e.env.get_action_mask() for e in vec_env.envs])
        action, _ = model.predict(obs, action_masks=action_masks, deterministic=True)
        obs, reward, done_arr, _ = vec_env.step(action)
        total_reward += float(reward[0])
        if done_arr[0]:
            all_done = True
            break

    n_users = inner.num_users
    served = np.sum(last_progress >= last_needs)
    completion_rate = served / n_users
    prog_ratios = last_progress / np.maximum(last_needs, 1e-6)
    fairness = _jfi(prog_ratios)
    completion_time = last_time if all_done else inner.config['max_episode_time']

    return {
        'reward': total_reward,
        'completion_rate': completion_rate,
        'fairness_jfi': fairness,
        'completion_time': completion_time,
    }


# ──────────────────────────────────────────────────────────────────────────────
# Main
# ──────────────────────────────────────────────────────────────────────────────

def evaluate_final():
    from src.uav_comm.utils.config_loader import load_config as load_env_config
    print("=" * 60)
    print("UAV Multi-User Scheduling — Benchmark")
    print("=" * 60)

    env_config = load_env_config()

    raw_env = UAVEnv(config=env_config)
    bl = MultiUserBaselines(raw_env)

    # ── RL setup ──────────────────────────────────────────────────────────────
    # Prefer the latest model if it has a companion VecNormalize, else fall back to
    # the best available checkpoint (largest step count with companion vecnorm pkl).
    def mask_fn(env):
        return env.get_action_mask()

    def _make_rl_env():
        return ActionMasker(UAVEnv(config=env_config), mask_fn)

    def _find_best_checkpoint():
        """Return sorted list of (model_path, vecnorm_path) candidates.

        Priority: has companion vecnorm pkl first (accurate eval), then largest step count.
        Caller should try each in order and break on first successful load.
        """
        ckpt_dir = "models/checkpoints"
        if not os.path.exists(ckpt_dir):
            return []
        pairs = []
        for f in os.listdir(ckpt_dir):
            if not f.endswith('.zip') or 'vecnorm' in f:
                continue
            vn = f.replace('_steps.zip', '_steps_vecnorm.pkl')
            vn_path = os.path.join(ckpt_dir, vn) if os.path.exists(os.path.join(ckpt_dir, vn)) else None
            try:
                steps = int(f.replace('ppo_uav_', '').replace('_steps.zip', ''))
            except ValueError:
                continue
            pairs.append((steps, os.path.join(ckpt_dir, f), vn_path))
        # Sort: has vecnorm first, then step count descending
        pairs.sort(key=lambda x: (x[2] is not None, x[0]), reverse=True)
        return [(p, vn) for _, p, vn in pairs]

    # Try loading best model
    final_model = "models/ppo_multi_user_latest"
    final_vn    = "models/vec_normalize_latest.pkl"
    rl_available = False
    vec_env = None
    rl_model = None

    if os.path.exists(final_model + ".zip") and os.path.exists(final_vn):
        try:
            vec_env = DummyVecEnv([_make_rl_env])
            vec_env = VecNormalize.load(final_vn, vec_env)
            vec_env.training = False; vec_env.norm_reward = False
            rl_model = MaskablePPO.load(final_model, env=vec_env)
            rl_available = True
            print(f"Loaded final RL model: {final_model}\n")
        except Exception as e:
            print(f"Could not load final model: {e}")

    if not rl_available:
        candidates = _find_best_checkpoint()
        if not candidates:
            print("WARNING: No RL checkpoints found. Run train_multi_user.py first.\n")
        for ck_model, ck_vn in candidates:
            try:
                vec_env = DummyVecEnv([_make_rl_env])
                if ck_vn:
                    vec_env = VecNormalize.load(ck_vn, vec_env)
                    vec_env.training = False; vec_env.norm_reward = False
                else:
                    vec_env = VecNormalize(vec_env, norm_obs=True, norm_reward=False, clip_obs=10.0,
                        norm_obs_keys=['needs','directions','distance','remaining_time','sinr_obs'])
                rl_model = MaskablePPO.load(ck_model, env=vec_env)
                rl_available = True
                vn_note = "(with saved VecNorm)" if ck_vn else "(fresh VecNorm — approximate)"
                print(f"Loaded RL checkpoint: {ck_model} {vn_note}\n")
                break
            except Exception as e:
                print(f"  Skipping {ck_model}: {e}")

    if not rl_available:
        print("WARNING: No RL model found — RL column will be skipped. Run train_multi_user.py first.\n")

    # ── Algorithm catalogue ────────────────────────────────────────────────────
    # Each entry: (label, category, action_fn)
    algorithms = [
        # Single-user baselines (all arrays → one user)
        ("S-Random",   "Single",  lambda: bl.single_random()),
        ("S-FCFS",     "Single",  lambda: bl.single_fcfs()),
        ("S-Greedy",   "Single",  lambda: bl.single_greedy()),
        ("S-RR",       "Single",  lambda: bl.single_round_robin()),
        ("S-PF",       "Single",  lambda: bl.single_proportional_fair()),
        # Multi-user baselines (each array independent)
        ("M-Random",   "Multi",   lambda: bl.multi_random()),
        ("M-FCFS",     "Multi",   lambda: bl.multi_fcfs()),
        ("M-Greedy",   "Multi",   lambda: bl.multi_greedy()),
        ("M-RR",       "Multi",   lambda: bl.multi_round_robin()),
        ("M-PF",       "Multi",   lambda: bl.multi_proportional_fair()),
        # Angular-separation-aware greedy (strongest non-RL multi baseline)
        ("M-Angular",  "Multi",   lambda: bl.multi_angular_greedy()),
    ]

    results = {}

    # ── Run baselines ─────────────────────────────────────────────────────────
    for label, category, action_fn in algorithms:
        print(f"  Evaluating {label} ...")
        ep_results = []
        for i in range(N_EPISODES):
            bl.reset()  # clear RR counter and PF EMA
            ep = run_baseline_episode(raw_env, action_fn, seed=i)
            ep_results.append(ep)
        results[label] = ep_results

    # ── Run RL ────────────────────────────────────────────────────────────────
    if rl_available:
        print("  Evaluating RL (MaskablePPO) ...")
        rl_eps = []
        for i in range(N_EPISODES):
            ep = run_rl_episode(rl_model, vec_env, seed=i)
            rl_eps.append(ep)
        results["RL-PPO"] = rl_eps

    # ── Print table ──────────────────────────────────────────────────────────
    metrics = ['reward', 'completion_rate', 'fairness_jfi', 'completion_time']
    headers = ['Algorithm', 'Reward', 'Completion%', 'JFI', 'Time(s)']

    print("\n" + "=" * 70)
    print(f"{'Algorithm':<14} {'Reward':>10} {'Completion%':>12} {'JFI':>8} {'Time(s)':>10}")
    print("-" * 70)
    for label, eps in results.items():
        r   = np.mean([e['reward'] for e in eps])
        c   = np.mean([e['completion_rate'] for e in eps]) * 100
        j   = np.mean([e['fairness_jfi'] for e in eps])
        t   = np.mean([e['completion_time'] for e in eps])
        print(f"{label:<14} {r:>10.2f} {c:>11.1f}% {j:>8.3f} {t:>10.2f}")
    print("=" * 70)

    # ── Plot ─────────────────────────────────────────────────────────────────
    labels = list(results.keys())
    n = len(labels)

    fig = plt.figure(figsize=(16, 10))
    gs = gridspec.GridSpec(2, 2, hspace=0.45, wspace=0.35)

    metric_info = [
        ('reward',          'Episode Reward',      gs[0, 0]),
        ('completion_rate', 'Completion Rate',     gs[0, 1]),
        ('fairness_jfi',    'Fairness (JFI)',       gs[1, 0]),
        ('completion_time', 'Completion Time (s)', gs[1, 1]),
    ]

    colors = ['#4C72B0'] * 5 + ['#DD8452'] * 6 + ['#55A868'] * 1  # single(5) / multi(6) / RL(1)

    for metric, ylabel, pos in metric_info:
        ax = fig.add_subplot(pos)
        means = [np.mean([e[metric] for e in results[l]]) for l in labels]
        stds  = [np.std( [e[metric] for e in results[l]]) for l in labels]
        xs = np.arange(n)
        bars = ax.bar(xs, means, yerr=stds, capsize=4, color=colors[:n], alpha=0.85, edgecolor='white')
        ax.set_xticks(xs)
        ax.set_xticklabels(labels, rotation=35, ha='right', fontsize=8)
        ax.set_ylabel(ylabel, fontsize=9)
        ax.grid(axis='y', alpha=0.3)
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)

    from matplotlib.patches import Patch
    legend_elements = [
        Patch(facecolor='#4C72B0', alpha=0.85, label='Single-user baselines'),
        Patch(facecolor='#DD8452', alpha=0.85, label='Multi-user baselines'),
        Patch(facecolor='#55A868', alpha=0.85, label='RL (MaskablePPO)'),
    ]
    fig.legend(handles=legend_elements, loc='upper center', ncol=3,
               bbox_to_anchor=(0.5, 1.01), frameon=False, fontsize=9)

    fig.suptitle('UAV Scheduling — Algorithm Comparison', y=1.04, fontsize=13)
    plt.savefig("evaluation_results.png", bbox_inches='tight', dpi=150)
    print("\nPlot saved to evaluation_results.png")


if __name__ == "__main__":
    evaluate_final()
