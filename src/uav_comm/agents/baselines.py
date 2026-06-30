
import numpy as np
from src.uav_comm.components.channel import total_path_loss, calculate_noise_level_db


def _top_k(scores, valid_indices, k):
    """Return k indices from valid_indices with highest score, cycling if k > len."""
    if len(valid_indices) == 0:
        return np.zeros(k, dtype=int)
    top = valid_indices[np.argsort(scores[valid_indices])[::-1]]
    return top[:k] if len(top) >= k else np.resize(top, k)


def _single(user_idx, num_arrays):
    return np.full(num_arrays, user_idx, dtype=int)


class MultiUserBaselines:
    """
    Baseline scheduling policies for the UAV environment.

    Single-user methods (prefix 'single_'): all num_arrays arrays point to ONE user.
    Multi-user  methods (prefix 'multi_'):  each array independently selects a user.

    Call reset() at the start of each evaluation episode to clear stateful baselines
    (round-robin counter and proportional-fair EMA).
    """

    def __init__(self, env, pf_time_constant=20):
        self.env = env
        self._pf_tc = pf_time_constant
        self._rr_ptr = 0
        self._pf_avg_rate = np.ones(env.num_users) * 1e-6

    def reset(self):
        self._rr_ptr = 0
        self._pf_avg_rate = np.ones(self.env.num_users) * 1e-6

    # ------------------------------------------------------------------
    # Shared helpers
    # ------------------------------------------------------------------

    def _active(self):
        return np.where(self.env.needs > self.env.progress)[0]

    def _channel_rate_estimate(self):
        """
        Estimate achievable rate for each user based on path loss only (no beamforming
        overhead). Used as the instantaneous-rate proxy in Proportional Fair scheduling.
        For single-user mode (all arrays → one user), array gain scales as num_arrays².
        """
        bw = self.env.config['bandwidth']
        noise_db = calculate_noise_level_db(bw)
        noise_lin = 10 ** (noise_db / 10)

        # Heterogeneous hardware: N is the sum of all elements across all arrays
        N = sum(len(locs[0]) for locs in self.env.array_configs)

        array_gain_db = 20 * np.log10(N)  # coherent combining gain

        rates = np.zeros(self.env.num_users)
        for i in range(self.env.num_users):
            d = np.linalg.norm(self.env.uav_position - self.env.locations[i])
            pl = total_path_loss(d, fading=False)
            sig_lin = 10 ** ((array_gain_db - pl) / 10)
            sinr = sig_lin / noise_lin
            rates[i] = bw * np.log2(1 + max(sinr, 0))
        return rates

    def _pf_update(self, served_idx, actual_rates):
        """Update EMA of average rates after a scheduling decision."""
        alpha = 1.0 / self._pf_tc
        for i in range(self.env.num_users):
            r = actual_rates[i] if i in served_idx else 0.0
            self._pf_avg_rate[i] = (1 - alpha) * self._pf_avg_rate[i] + alpha * r

    # ------------------------------------------------------------------
    # Single-user baselines (mode a: all arrays → one user)
    # ------------------------------------------------------------------

    def single_random(self):
        active = self._active()
        if len(active) == 0:
            return _single(0, self.env.num_arrays)
        return _single(np.random.choice(active), self.env.num_arrays)

    def single_greedy(self):
        """Serve user with most remaining need."""
        active = self._active()
        if len(active) == 0:
            return _single(0, self.env.num_arrays)
        remaining = self.env.needs - self.env.progress
        best = active[np.argmax(remaining[active])]
        return _single(best, self.env.num_arrays)

    def single_fcfs(self):
        """Serve closest active user (highest channel quality proxy)."""
        active = self._active()
        if len(active) == 0:
            return _single(0, self.env.num_arrays)
        dist = np.linalg.norm(self.env.locations - self.env.uav_position, axis=1)
        best = active[np.argmin(dist[active])]
        return _single(best, self.env.num_arrays)

    def single_round_robin(self):
        """Cycle through active users in a fixed order."""
        active = self._active()
        if len(active) == 0:
            return _single(0, self.env.num_arrays)
        uid = active[self._rr_ptr % len(active)]
        self._rr_ptr += 1
        return _single(uid, self.env.num_arrays)

    def single_proportional_fair(self):
        """
        Classic PF scheduler: serve argmax r_i(t) / R̄_i(t).
        r_i(t) = instantaneous estimated rate, R̄_i(t) = EMA average rate.
        Balances throughput and fairness — the standard telecom baseline.
        """
        active = self._active()
        if len(active) == 0:
            return _single(0, self.env.num_arrays)

        inst_rate = self._channel_rate_estimate()
        pf_metric = inst_rate / self._pf_avg_rate

        best = active[np.argmax(pf_metric[active])]
        action = _single(best, self.env.num_arrays)

        served_rates = np.zeros(self.env.num_users)
        served_rates[best] = inst_rate[best]
        self._pf_update({best}, served_rates)
        return action

    def single_lwdf(self):
        """
        Largest Weighted Delay First (LWDF): serve argmax W_i * r_i(t)
        Where W_i is the delay metric (e.g., waiting time or deficit).
        A SOTA queue management scheduler.
        """
        active = self._active()
        if len(active) == 0:
            return _single(0, self.env.num_arrays)

        inst_rate = self._channel_rate_estimate()
        # Using a combination of delay and remaining need for the weight
        delay_weight = self.env.delay + 1.0
        remaining_need = np.maximum(self.env.needs - self.env.progress, 0.0)
        lwdf_metric = delay_weight * remaining_need * inst_rate

        best = active[np.argmax(lwdf_metric[active])]
        return _single(best, self.env.num_arrays)

    def single_max_min(self):
        """
        Max-Min Fairness: serve the user with the lowest normalized progress ratio
        to ensure strict fairness across active users.
        """
        active = self._active()
        if len(active) == 0:
            return _single(0, self.env.num_arrays)

        prog_ratios = self.env.progress / np.maximum(self.env.needs, 1e-6)
        # We want to pick the user with the minimum progress ratio
        # To reuse argmax style logic, we invert it (or just use argmin)
        best = active[np.argmin(prog_ratios[active])]
        return _single(best, self.env.num_arrays)

    # ------------------------------------------------------------------
    # Multi-user baselines (mode b: each array independent)
    # ------------------------------------------------------------------

    def multi_random(self):
        active = self._active()
        if len(active) == 0:
            return np.zeros(self.env.num_arrays, dtype=int)
        return np.random.choice(active, size=self.env.num_arrays, replace=True)

    def multi_greedy(self):
        """Assign each array to a top-K unique user by remaining need."""
        active = self._active()
        if len(active) == 0:
            return np.zeros(self.env.num_arrays, dtype=int)
        remaining = self.env.needs - self.env.progress
        return _top_k(remaining, active, self.env.num_arrays)

    def multi_fcfs(self):
        """Assign each array to a top-K unique user by proximity."""
        active = self._active()
        if len(active) == 0:
            return np.zeros(self.env.num_arrays, dtype=int)
        dist = np.linalg.norm(self.env.locations - self.env.uav_position, axis=1)
        neg_dist = -dist
        return _top_k(neg_dist, active, self.env.num_arrays)

    def multi_round_robin(self):
        """Distribute arrays across active users in round-robin order."""
        active = self._active()
        if len(active) == 0:
            return np.zeros(self.env.num_arrays, dtype=int)
        selected = np.array([active[(self._rr_ptr + k) % len(active)]
                             for k in range(self.env.num_arrays)], dtype=int)
        self._rr_ptr = (self._rr_ptr + self.env.num_arrays) % max(len(active), 1)
        return selected

    def multi_proportional_fair(self):
        """
        PF for multi-array: assign each array to the user with highest r_i / R̄_i
        among those not yet selected this step (greedy per-array PF).
        """
        active = self._active()
        if len(active) == 0:
            return np.zeros(self.env.num_arrays, dtype=int)

        inst_rate = self._channel_rate_estimate()
        pf_metric = inst_rate / self._pf_avg_rate

        K = self.env.num_arrays
        selected = _top_k(pf_metric, active, K)

        served_rates = np.zeros(self.env.num_users)
        for uid in np.unique(selected):
            served_rates[uid] = inst_rate[uid]
        self._pf_update(set(selected.tolist()), served_rates)
        return selected

    def multi_lwdf(self):
        """
        Multi-array Largest Weighted Delay First (LWDF).
        """
        active = self._active()
        if len(active) == 0:
            return np.zeros(self.env.num_arrays, dtype=int)

        inst_rate = self._channel_rate_estimate()
        delay_weight = self.env.delay + 1.0
        remaining_need = np.maximum(self.env.needs - self.env.progress, 0.0)
        lwdf_metric = delay_weight * remaining_need * inst_rate

        return _top_k(lwdf_metric, active, self.env.num_arrays)

    def multi_max_min(self):
        """
        Multi-array Max-Min Fairness.
        """
        active = self._active()
        if len(active) == 0:
            return np.zeros(self.env.num_arrays, dtype=int)

        prog_ratios = self.env.progress / np.maximum(self.env.needs, 1e-6)
        # Sort ascending to get the users with smallest progress first
        neg_prog = -prog_ratios
        return _top_k(neg_prog, active, self.env.num_arrays)
