
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


def run_rl_episode(model, vec_env, raw_env, seed):
    """Run one RL episode. Uses normalised VecEnv for inference, raw env for metrics."""
    obs = vec_env.reset()
    done = False
    total_reward = 0.0

    while not done:
        action, _ = model.predict(obs, deterministic=True)
        obs, reward, done, _ = vec_env.step(action)
        total_reward += float(reward[0])

    # Extract metrics from the underlying raw env
    inner = vec_env.envs[0].env  # ActionMasker → UAVEnv
    n_users = inner.num_users
    served = np.sum(inner.progress >= inner.needs)
    completion_rate = served / n_users
    prog_ratios = inner.progress / np.maximum(inner.needs, 1e-6)
    fairness = _jfi(prog_ratios)
    completion_time = inner.current_time if served == n_users else inner.config['max_episode_time']

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
    model_path = "models/ppo_multi_user_latest"
    stats_path = "models/vec_normalize_latest.pkl"
    rl_available = os.path.exists(model_path + ".zip") and os.path.exists(stats_path)

    def mask_fn(env):
        return env.get_action_mask()

    if rl_available:
        vec_env = DummyVecEnv([lambda: ActionMasker(UAVEnv(config=env_config), mask_fn)])
        vec_env = VecNormalize.load(stats_path, vec_env)
        vec_env.training = False
        vec_env.norm_reward = False
        rl_model = MaskablePPO.load(model_path, env=vec_env)
        print(f"Loaded RL model from {model_path}\n")
    else:
        print("WARNING: No trained model found — RL column will be skipped.\n"
              "         Run train_multi_user.py first.\n")

    # ── Algorithm catalogue ────────────────────────────────────────────────────
    # Each entry: (label, category, action_fn)
    # action_fn must be called AFTER env.reset() so baselines see fresh state.
    algorithms = [
        # Single-user baselines (all arrays → one user)
        ("S-Random",   "Single", lambda: bl.single_random()),
        ("S-FCFS",     "Single", lambda: bl.single_fcfs()),
        ("S-Greedy",   "Single", lambda: bl.single_greedy()),
        ("S-RR",       "Single", lambda: bl.single_round_robin()),
        ("S-PF",       "Single", lambda: bl.single_proportional_fair()),
        # Multi-user baselines (each array independent)
        ("M-Random",   "Multi",  lambda: bl.multi_random()),
        ("M-FCFS",     "Multi",  lambda: bl.multi_fcfs()),
        ("M-Greedy",   "Multi",  lambda: bl.multi_greedy()),
        ("M-RR",       "Multi",  lambda: bl.multi_round_robin()),
        ("M-PF",       "Multi",  lambda: bl.multi_proportional_fair()),
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
            ep = run_rl_episode(rl_model, vec_env, raw_env, seed=i)
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

    colors = ['#4C72B0'] * 5 + ['#DD8452'] * 5 + ['#55A868'] * 1  # single / multi / RL

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
