
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
    # H-MARL baselines (Strategist outputs sectors)
    # ------------------------------------------------------------------

    def hmarl_static(self):
        """Assigns each array to a fixed sector (e.g., Array 0 -> Sector 0)"""
        num_sectors = self.env.config.get('num_sectors', 4)
        return np.array([i % num_sectors for i in range(self.env.num_arrays)], dtype=int)

    def hmarl_random(self):
        """Randomly assigns arrays to any valid sector that contains users."""
        num_sectors = self.env.config.get('num_sectors', 4)
        # Replicate logic from env.get_action_mask() for H-MARL
        sector_mask = np.zeros(num_sectors, dtype=bool)
        sector_width = 360.0 / num_sectors
        active = self._active()

        for u in active:
            loc = self.env.locations[u]
            # Use environment's internal trigonometry parser to avoid unpacking errors from 3D to 2D
            _, phi = self.env._calculate_direction(loc)
            phi_mapped = (phi + 180.0) % 360.0
            s_idx = int(phi_mapped // sector_width)
            s_idx = min(s_idx, num_sectors - 1)
            sector_mask[s_idx] = True

        valid_sectors = np.where(sector_mask)[0]
        if len(valid_sectors) == 0:
            return np.zeros(self.env.num_arrays, dtype=int)

        return np.random.choice(valid_sectors, size=self.env.num_arrays, replace=True)

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
    def multi_angular_greedy(self):
        """
        Angular-separation-aware greedy: selects num_arrays users maximising
        pairwise angular separation from the UAV's perspective.

        This is the strongest non-RL baseline for the null-forming system —
        it directly addresses the same problem the RL is expected to learn.
        RL should outperform it by also accounting for urgency and SINR history.
        """
        active = self._active()
        K = self.env.num_arrays
        if len(active) == 0:
            return np.zeros(K, dtype=int)
        if len(active) <= K:
            return np.resize(active, K)

        # Compute azimuth angles (phi) for each user from the UAV
        azimuths = np.arctan2(
            self.env.locations[active, 1] - self.env.uav_position[1],
            self.env.locations[active, 0] - self.env.uav_position[0]
        )

        # Greedy selection: first pick the highest-urgency user, then iteratively
        # add the user whose minimum angular distance to already-selected users is maximal.
        remaining = self.env.needs - self.env.progress
        first = active[np.argmax(remaining[active])]
        selected_local = [np.where(active == first)[0][0]]

        while len(selected_local) < K:
            best_idx, best_sep = -1, -1.0
            for i in range(len(active)):
                if i in selected_local:
                    continue
                diffs = [abs(azimuths[i] - azimuths[j]) for j in selected_local]
                min_sep = min(min(d, 2 * np.pi - d) for d in diffs)
                if min_sep > best_sep:
                    best_sep, best_idx = min_sep, i
            selected_local.append(best_idx)

        return active[np.array(selected_local, dtype=int)]

    def multi_pinn_greedy(self):
        """
        PINN-Aware Greedy: Selects the most urgent users sequentially, explicitly
        rejecting any user that falls below an isolation score threshold on the
        environment's PINN surrogate matrix when paired with already-selected users.
        """
        active = self._active()
        K = self.env.num_arrays
        if len(active) == 0:
            return np.zeros(K, dtype=int)

        # Retrieve the PINN matrix from the environment's observation state
        obs = self.env._get_observation()
        N = self.env.num_users
        pinn_matrix = obs['pinn_interference_matrix'].reshape((N, N))

        remaining = self.env.needs - self.env.progress

        # Sort active users by urgency (highest remaining need first)
        sorted_active = active[np.argsort(-remaining[active])]

        selected = []

        # First array takes the absolute highest urgency user
        selected.append(sorted_active[0])

        # Subsequent arrays pick the highest urgency user that survives the PINN mask
        for _ in range(1, K):
            best_candidate = None

            for candidate in sorted_active:
                if candidate in selected:
                    continue

                # Check if the candidate is physically compatible with ALL already-selected users
                is_safe = True
                for already_picked in selected:
                    # Look for high isolation score
                    if pinn_matrix[candidate, already_picked] < 0.05: # very aggressive threshold to find any isolation
                        is_safe = False
                        break

                if is_safe:
                    best_candidate = candidate
                    break

            if best_candidate is not None:
                selected.append(best_candidate)
            else:
                # If no user is safe from interference, fallback to picking the same user
                # to trigger coherent combining (DoF sacrifice rather than interference suicide)
                selected.append(selected[-1])

        return np.array(selected, dtype=int)
