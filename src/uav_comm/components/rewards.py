
import numpy as np

MIN_PROGRESS = 1e-4
MAX_NEED = 10


def calculate_reward_function(throughputs_per_user, delay, progress, needs, active_users, current_time, max_episode_time):
    """
    Reward design rationale
    -----------------------
    The previous JFI-on-state reward had a credit-assignment problem: it told the agent
    how fair the CURRENT STATE was, but not which action this step was responsible for.
    The agent could not learn "I chose user 3 because they were falling behind."

    New design uses three components, each with a distinct role:

    1. Urgency-weighted throughput  (primary, immediate, action-correlated)
       Weight_i = (delay_i + 1) × remaining_deficit_i, normalised to sum=1.
       This directly rewards serving the most-neglected user THIS step.
       A user who has waited 8 steps with 90% remaining gets 8× the reward credit
       compared to a freshly-served user with 10% remaining.
       → Implements an LWDF-inspired (Largest Weighted Delay First) per-step objective.

    2. Min-progress ratio  (secondary, state-level max-min fairness)
       = min(progress_i / needs_i) over active users.
       Ranges [0, 1]. Directly measures how well the WORST-OFF user is doing.
       Complements urgency weighting: even if urgency rewards were given, this
       catches episodes where the agent still neglected one user.

    3. Delay-JFI  (tertiary, state-level delay equity)
       Kept as a softer secondary signal. Lower weight than before.

    4. Progress reward  (completion incentive)
       Increases as users finish. Ensures the agent still aims to complete the episode.
    """
    n_active = int(np.sum(active_users))
    n_half_life = int(np.sum((needs / 2) > progress))

    # --- 1. Urgency-weighted throughput ---
    remaining = np.maximum(needs - progress, 0.0)
    raw_urgency = (delay + 1.0) * remaining
    raw_urgency[~active_users] = 0.0
    urgency_sum = raw_urgency.sum()
    if urgency_sum > 1e-6:
        urgency = raw_urgency / urgency_sum
    else:
        urgency = np.zeros_like(raw_urgency)

    urgency_weighted_thr = float(np.dot(throughputs_per_user, urgency))
    total_thr = float(np.sum(throughputs_per_user))

    # --- 2. Min-progress ratio (max-min fairness) ---
    if n_active > 0:
        prog_ratios = progress[active_users] / np.maximum(needs[active_users], 1e-6)
        min_progress_ratio = float(np.min(prog_ratios))
    else:
        min_progress_ratio = 1.0

    # --- 3. Delay JFI (state-level equity signal, lower weight than before) ---
    if n_active > 1:
        d = delay[active_users]
        jfi_delay = float((np.sum(d) ** 2) / (n_active * np.sum(d ** 2) + 1e-6))
    else:
        jfi_delay = 1.0

    # --- 4. Progress reward ---
    progress_reward = 1.0 / (n_active + 0.5) + 0.5 / (n_half_life + 1)

    # New Reward Engineering:
    # Heavily weight total_thr (total bit rate) to reflect Shannon capacity / creating 4 perfect channels.
    # Deprioritize urgency and fairness to allow the agent to optimize for overall throughput.
    # Introduce a constant step penalty to force the UAV to finish the episode quickly.

    # Weights chosen:
    # total_thr ~ typically in Gbps, e.g. 0.5 - 5.0 Gbps. Weight of 5.0 makes it primary.
    # urgency_weighted_thr: reduced to 0.5 to keep a slight hint of fairness.
    # min_progress_ratio: reduced to 0.5.
    # jfi_delay: reduced to 0.1.
    # progress_reward: 0.2.
    # Constant step penalty: -2.0. This ensures the step reward is mostly negative unless throughput is huge,
    # driving the agent to complete the episode (all users satisfied) as soon as possible.

    step_penalty = -2.0

    raw_reward = (
        5.0 * total_thr
        + 0.5 * urgency_weighted_thr
        + 0.5 * min_progress_ratio
        + 0.1 * jfi_delay
        + 0.2 * progress_reward
        + step_penalty
    )

    return raw_reward, jfi_delay, urgency_weighted_thr, min_progress_ratio, total_thr


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
