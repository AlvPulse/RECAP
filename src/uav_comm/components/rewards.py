
import numpy as np

MAX_NEED = 10
# Maximum achievable throughput per user per step (used for normalisation).
# bandwidth=0.35 GHz, SINR≈500 linear (27 dB), dt=0.25 s  →  ~0.78 Gbits
_MAX_THR_PER_STEP = 0.35 * np.log2(1 + 500) * 0.25


def calculate_reward_function(throughputs_per_user, sinr_per_user, sinr_threshold_linear,
                               delay, progress, needs, active_users):
    """
    Reward design — everything is normalised to [0, ~1] so components are comparable.

    Component 1 – SINR quality  (primary: the RL's core contribution)
        Measures how well null-forming worked for users served THIS step.
        log2(1+SINR)/10 maps SINR=[0,500] → [0, 0.9].
        Only computed for users that received non-zero throughput this step.
        → Gives immediate feedback: "choosing those users led to this SINR."
        The RL can learn: large angular separation → better nulls → higher SINR → higher reward.

    Component 2 – Urgency-weighted normalised throughput  (fairness-aware service)
        Weight_i = (delay_i + 1) × remaining_need_i.
        Users who have waited longest AND have the most left to do get the highest weight.
        Throughput is normalised by the theoretical maximum per step.
        → Per-step signal: "did you serve the right user this step?"

    Component 3 – Min-progress ratio  (max-min fairness, state-level)
        min(progress_i / need_i) over active users.
        Catches persistent neglect that per-step urgency weighting may miss.

    Component 4 – Completion progress  (episode termination incentive)
    """
    n_active = int(np.sum(active_users))
    n_half_life = int(np.sum((needs / 2) > progress))

    # --- 1. SINR quality ---
    served = throughputs_per_user > 0
    served_active = served & active_users
    if np.sum(served_active) > 0:
        sinr_quality = float(np.mean(np.log2(1.0 + sinr_per_user[served_active]) / 10.0))
    else:
        sinr_quality = 0.0

    # --- 2. Urgency-weighted normalised throughput ---
    remaining = np.maximum(needs - progress, 0.0)
    raw_urgency = (delay + 1.0) * remaining
    raw_urgency[~active_users] = 0.0
    urgency_sum = raw_urgency.sum()
    urgency = raw_urgency / urgency_sum if urgency_sum > 1e-6 else np.zeros_like(raw_urgency)
    urgency_thr = float(np.dot(throughputs_per_user, urgency))
    norm_urgency_thr = min(urgency_thr / (_MAX_THR_PER_STEP + 1e-9), 1.0)

    # --- 3. Min-progress (max-min fairness) ---
    if n_active > 0:
        prog_ratios = progress[active_users] / np.maximum(needs[active_users], 1e-6)
        min_progress = float(np.min(prog_ratios))
    else:
        min_progress = 1.0

    # --- 4. Completion ---
    completion_reward = 1.0 / (n_active + 0.5) + 0.5 / (n_half_life + 1)

    # All components comparable; total raw ≈ 2–5 per step → /10 in env → VecNormalize handles rest
    raw_reward = (
        2.0 * sinr_quality        # null-forming quality  [0, ~1.8]
        + 2.0 * norm_urgency_thr  # fair service          [0, 2.0]
        + 1.0 * min_progress      # max-min fairness      [0, 1.0]
        + 0.3 * completion_reward # episode completion    [0, ~0.35]
    )

    return raw_reward, sinr_quality, norm_urgency_thr, min_progress, completion_reward


class RewardTracker:
    """Running normaliser — unused while VecNormalize handles reward normalisation externally."""
    def __init__(self):
        self.meanR = 0
        self.varR = 1
        self.countR = 1e-4

    def normalize(self, reward):
        self.countR += 1
        delta = reward - self.meanR
        self.meanR += delta / self.countR
        delta2 = reward - self.meanR
        self.varR = ((self.countR - 1) * self.varR + delta * delta2) / self.countR
        return np.clip((reward - self.meanR) / (np.sqrt(self.varR) + 1e-8), -10, 10)
