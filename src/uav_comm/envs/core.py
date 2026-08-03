
import gymnasium as gym
from gymnasium import spaces
import numpy as np

from src.uav_comm.components.channel import total_path_loss, calculate_noise_level_db
from src.uav_comm.components.antenna import array_locs, pert2d_null_multi, phase_code_finder, find_gain_of_tphi
from src.uav_comm.components.rewards import calculate_reward_function, MAX_NEED

DEFAULT_CONFIG = {
    'num_users': 8,
    'num_arrays': 4,
    'num_elements_regular': 8,
    'num_elements_irregular': 16,
    'conflict_threshold_deg': 20,
    'bandwidth': 0.35,
    'time_interval': 0.25,
    'sinr_threshold_db': 3,
    'uav_height': 50,
    'max_range': 500,
    'max_episode_time': 30,
    'uav_speed': 30,
    'switch_cost': 0.25,
    # 'single': all arrays beam to the same user (max combining gain on one user)
    # 'multi':  each array independently selects a user (serve multiple users per step)
    'operation_mode': 'multi',
}


class UAVEnv(gym.Env):
    def __init__(self, config=None):
        super().__init__()

        self.config = DEFAULT_CONFIG.copy()
        if config:
            self.config.update(config)

        self.num_users = self.config['num_users']
        self.num_arrays = self.config['num_arrays']
        self.num_elements_regular = self.config.get('num_elements_regular', 8)
        self.num_elements_irregular = self.config.get('num_elements_irregular', 16)

        self.bts_gain = 10 ** (50 / 10) * 10 ** (10 / 10)
        tx_dbm  = self.config.get('tx_power_dbm', 43)   # UAV transmit power (dBm)
        rx_dbi  = self.config.get('rx_gain_dbi', 0)     # User receive antenna gain (dBi)
        # uav_user_gain in linear mW; divided by 1000 when computing effective noise (→ W)
        self.uav_user_gain = 10 ** ((tx_dbm + rx_dbi) / 10)
        self.sinr_threshold_linear = 10 ** (self.config['sinr_threshold_db'] / 10)

        # Heterogeneous Hardware: alternate between regular and irregular panels
        self.array_configs = []
        for i in range(self.num_arrays):
            num_el = self.num_elements_regular if i % 2 == 0 else self.num_elements_irregular
            self.array_configs.append(array_locs(num_el))

        if self.config['operation_mode'] == 'single':
            self.action_space = spaces.Discrete(self.num_users)
        else:
            self.action_space = spaces.MultiDiscrete([self.num_users] * self.num_arrays)

        self.observation_space = spaces.Dict({
            'needs':          spaces.Box(low=0, high=1, shape=(self.num_users,), dtype=np.float64),
            'directions':     spaces.Box(low=-0.5, high=0.5, shape=(self.num_users,), dtype=np.float64),
            'distance':       spaces.Box(low=0, high=1, shape=(self.num_users,), dtype=np.float64),
            'user_satisfied': spaces.MultiBinary(self.num_users),
            'remaining_time': spaces.Box(low=0, high=1, shape=(1,), dtype=np.float64),
            # Normalised SINR from the previous step: direct feedback on null-forming quality.
            # log2(1+SINR)/10 maps SINR=[0,500] → [0,0.9]; zero for unsatisfied/unserved users.
            'sinr_obs':       spaces.Box(low=0, high=1, shape=(self.num_users,), dtype=np.float64),
        })

        self.reset()

    def reset(self, seed=None, options=None):
        if seed is not None:
            np.random.seed(seed)

        range_lim = self.config['max_range']
        self.locations = np.random.uniform(-range_lim, range_lim, (self.num_users, 2))
        self.uav_position = np.random.uniform(-range_lim / 2, range_lim / 2, (2,))
        self.needs = np.zeros(self.num_users)
        self.progress = np.zeros(self.num_users)
        self.sinr = np.zeros(self.num_users)
        self.delay = np.zeros(self.num_users)
        self.current_time = 0.0
        self.last_action = None

        # M-B Violation-Driven Growth Mask (Persists across episodes to learn spatial constraints)
        if not hasattr(self, 'learned_mask'):
            self.learned_mask = np.zeros((self.num_users, self.num_users), dtype=bool)

        for i in range(self.num_users):
            self._declare_need(i)

        return self._get_observation(), {}

    def step(self, action):
        self.current_time += self.config['time_interval']
        conflict_repaired = False

        # Resolve per-array user assignments
        if self.config['operation_mode'] == 'single':
            uid = int(action.item()) if isinstance(action, np.ndarray) else int(action)
            selected_users = np.full(self.num_arrays, uid, dtype=int)
        elif isinstance(action, (int, np.integer)):
            selected_users = np.full(self.num_arrays, int(action), dtype=int)
        elif isinstance(action, np.ndarray) and action.ndim == 0:
            selected_users = np.full(self.num_arrays, int(action.item()), dtype=int)
        else:
            selected_users = action.astype(int)

        # ---------------------------------------------------------------------
        # Option A: Soft Probabilistic Masking / Gambling
        # ---------------------------------------------------------------------
        # We no longer force a deterministic repair. Agents (including baselines)
        # must gamble on their chosen assignments. If they pick conflicting users,
        # the underlying physics engine (_calculate_sinr) will naturally yield a
        # poor SINR, resulting in 0 throughput. This allows the RL to learn
        # the optimal trade-off natively.

        self._calculate_sinr(selected_users)

        active_users = (self.needs > 0) & (self.needs > self.progress)
        for i in range(self.num_users):
            if active_users[i]:
                self.delay[i] += 1

        throughputs_per_user = np.zeros(self.num_users)
        for uid in np.unique(selected_users):
            if active_users[uid]:
                throughputs_per_user[uid] = self._serve_user(uid)

        self._update_uav_location(selected_users)

        raw_reward, sinr_quality, urgency_thr, min_progress, completion_rew = calculate_reward_function(
            throughputs_per_user, self.sinr, self.sinr_threshold_linear,
            self.delay, self.progress, self.needs, active_users
        )

        # Safety net: penalise targeting already-satisfied users (masking should prevent this).
        inactive = np.where(~active_users)[0]
        wrong_count = int(np.sum(np.isin(selected_users, inactive)))
        if wrong_count > 0:
            raw_reward -= wrong_count * 2

        # Switch cost applied at raw scale (consistent with /10 below)
        if self.last_action is not None and not np.array_equal(action, self.last_action):
            raw_reward -= self.config['switch_cost']

        self.last_action = np.copy(action) if isinstance(action, np.ndarray) else action

        reward = raw_reward / 10

        # # New-needs logic — disabled: self.needs is never 0 after reset, so this never fires.
        # if np.random.rand() < 0.2:
        #     if len(inactive) > 0:
        #         idx = np.random.choice(inactive)
        #         if self.needs[idx] == 0:
        #             self._declare_need(idx)

        done = bool(np.all(self.progress >= self.needs))
        if done:
            reward += 0.0002 * (self.config['max_episode_time'] - self.current_time) ** 2

        truncated = self.current_time >= self.config['max_episode_time']

        info = {
            "sinr_quality": float(sinr_quality),
            "urgency_thr":  float(urgency_thr),
            "min_progress": float(min_progress),
            "total_thr": float(np.sum(throughputs_per_user)),
            "conflict_repaired": int(conflict_repaired),
        }

        return self._get_observation(), reward, done, truncated, info

    def _declare_need(self, user_idx):
        self.needs[user_idx] = MAX_NEED

    def _serve_user(self, user_idx):
        if self.sinr[user_idx] > self.sinr_threshold_linear:
            cap = self.config['bandwidth'] * np.log2(1 + self.sinr[user_idx]) * self.config['time_interval']
            cap = min(cap, self.needs[user_idx] - self.progress[user_idx])
            self.progress[user_idx] += cap
            self.delay[user_idx] = 0
            return cap
        return 0.0

    def _calculate_sinr(self, selected_users):
        noise_db = calculate_noise_level_db(self.config['bandwidth'])
        signals = np.zeros(self.num_users)
        interferences = np.zeros(self.num_users)

        for array_idx, uid in enumerate(selected_users):
            loc = self.locations[uid]
            theta, phi = self._calculate_direction(loc)
            # 2D horizontal distance for signal path loss (consistent with channel model)
            R = np.linalg.norm(self.uav_position - loc)
            D, PhaseTable = self.array_configs[array_idx]

            # Deduplicate: each interfering user counted once per array.
            # Bug fix: when two arrays serve the same interference user (e.g. selected=[0,1,1,2]),
            # that user would appear twice in `others` without deduplication, causing double-counted
            # interference power and double null attempts.
            seen = set()
            others = []
            for i, u in enumerate(selected_users):
                if i != array_idx and u != uid and u not in seen:
                    others.append(u)
                    seen.add(u)

            if others:
                int_dirs, int_dists = self._calculate_interference_directions(others)
                q_bits = 1 if len(D) <= 8 else 2
                sig_db, int_db = pert2d_null_multi(
                    D, PhaseTable, theta, phi, R,
                    int_dirs[:, 0], int_dirs[:, 1], int_dists, noise_db,
                    quantization_bits=q_bits
                )
                signals[uid] += 10 ** (sig_db / 10)
                for k, int_uid in enumerate(others):
                    interferences[int_uid] += 10 ** (int_db[k] / 10)
            else:
                ph = phase_code_finder(D, PhaseTable, theta, phi)
                sig_db = find_gain_of_tphi(theta, phi, ph, D) - total_path_loss(R)
                signals[uid] += 10 ** (sig_db / 10)

        # uav_user_gain encodes TX power (dBm) × RX gain (dBi) in linear mW.
        # signals/interferences are normalised to 1 W TX, so we must divide the
        # thermal noise by the actual TX power (converted W) to make units consistent.
        # Without this, SINR ≈ 0 everywhere → zero throughput → no learning signal.
        p_tx_watts = self.uav_user_gain / 1000.0        # linear mW → W
        noise_eff  = 10 ** (noise_db / 10) / p_tx_watts # effective noise floor (dimensionless)
        self.sinr  = signals / (interferences + noise_eff)

        # M-B Violation-Driven Mask Growth
        sinr_db_vals = 10 * np.log10(self.sinr + 1e-9)
        violation_threshold = self.config['sinr_threshold_db']

        if hasattr(self, 'learned_mask'):
            for i, uid_i in enumerate(selected_users):
                for j, uid_j in enumerate(selected_users):
                    if i != j and uid_i != uid_j:
                        if sinr_db_vals[uid_i] < violation_threshold or sinr_db_vals[uid_j] < violation_threshold:
                            self.learned_mask[uid_i, uid_j] = True
                            self.learned_mask[uid_j, uid_i] = True

    def _calculate_direction(self, user_location):
        dx, dy = user_location - self.uav_position
        dist_3d = np.sqrt(dx ** 2 + dy ** 2 + self.config['uav_height'] ** 2)
        theta = np.degrees(np.arccos(self.config['uav_height'] / dist_3d))
        phi = np.degrees(np.arctan2(dy, dx))
        return theta, phi

    def _calculate_interference_directions(self, user_indices):
        dirs, dists = [], []
        for uid in user_indices:
            loc = self.locations[uid]
            dirs.append(self._calculate_direction(loc))
            # Fix: use 2D horizontal distance, consistent with signal path loss and
            # los_probability_uav() which expects horizontal distance for elevation angle.
            dists.append(np.linalg.norm(loc - self.uav_position))
        return np.array(dirs), np.array(dists)

    def _update_uav_location(self, selected_users):
        targets = [self._calculate_optimal_location(uid) for uid in np.unique(selected_users)]
        if targets:
            avg = np.mean(targets, axis=0)
            d = avg - self.uav_position
            norm = np.linalg.norm(d)
            if norm > 1e-6:
                self.uav_position += (d / norm) * self.config['uav_speed'] * self.config['time_interval']

    def _calculate_optimal_location(self, user_idx):
        loc = self.locations[user_idx]
        d = loc - self.uav_position
        norm = np.linalg.norm(d)
        if norm < 1e-6:
            return loc
        optimal_dist = norm / np.sqrt(self.bts_gain / self.uav_user_gain)
        return loc - (d / norm) * optimal_dist

    def get_action_mask(self):
        user_mask = (self.needs > self.progress).astype(bool)

        if self.config.get('enable_spatial_masking', True):
            # Call to ensure the latest mask components are updated, but we don't strictly need to do this
            # if the mask logic doesn't depend on the side-effects of _get_observation.
            # However, _get_observation sets _last_conflict_matrix, so we call it.
            self._get_observation()
            conflict_matrix = getattr(self, '_last_conflict_matrix', np.zeros(self.num_users * self.num_users)).reshape((self.num_users, self.num_users))

            active_conflicts = (conflict_matrix > 0) & np.outer(user_mask, user_mask)
            unsafe_users = np.any(active_conflicts, axis=1)

            final_mask = user_mask & (~unsafe_users)

            if not np.any(final_mask) and np.any(user_mask):
                final_mask = user_mask
        else:
            final_mask = user_mask

        if self.config.get('operation_mode', 'multi') == 'single':
            return user_mask
        else:
            # sb3_contrib MaskablePPO with MultiDiscrete expects flat array of shape
            # (sum(nvec),) = (num_users * num_arrays,) — all dimension masks concatenated.
            return np.tile(user_mask, self.num_arrays)

    def _get_observation(self):
        needs_remaining = np.maximum(self.needs - self.progress, 0.0)
        user_satisfied = needs_remaining <= 0
        distance = np.linalg.norm(self.locations - self.uav_position, axis=1)

        directions = np.zeros(self.num_users)
        for i in range(self.num_users):
            _, phi = self._calculate_direction(self.locations[i])
            directions[i] = phi
        directions[user_satisfied] = 0.0

        # Anti-Ratchet Asymmetric Epsilon-Probing
        epsilon_probe = 0.10
        if hasattr(self, 'learned_mask') and np.random.rand() < epsilon_probe:
            mask_indices = np.argwhere(self.learned_mask)
            if len(mask_indices) > 0:
                choice_idx = np.random.choice(len(mask_indices))
                idx = mask_indices[choice_idx]
                self.learned_mask[idx[0], idx[1]] = False

        # M-A Deterministic Base Mask (Pairwise Angular Separation)
        base_mask = np.zeros((self.num_users, self.num_users), dtype=np.int8)
        threshold = self.config.get('conflict_threshold_deg', 20.0)

        # Determine DoF Budget globally for the worst-case panel (8 elements -> rank ~ 7)
        # If the number of active users clustered tightly exceeds this rank, we hard-mask the cluster
        max_dof_budget = 7

        for i in range(self.num_users):
            if user_satisfied[i]:
                continue

            cluster_size = 0
            for j in range(self.num_users):
                if i != j and not user_satisfied[j]:
                    diff = (directions[i] - directions[j] + 180) % 360 - 180
                    if abs(diff) <= threshold:
                        base_mask[i, j] = 1
                        cluster_size += 1

            # M-DoF Cardinality Limit
            if cluster_size > max_dof_budget:
                base_mask[i, :] = 1 # Completely invalidate this user to prevent DoF exhaustion

        if hasattr(self, 'learned_mask'):
            conflict_matrix = np.logical_or(base_mask, self.learned_mask).astype(np.int8)
        else:
            conflict_matrix = base_mask

        remaining_time = np.clip(
            [(self.config['max_episode_time'] - self.current_time) / self.config['max_episode_time']],
            0.0, 1.0
        ).astype(np.float64)

        # Normalised SINR: tells the agent how well null-forming worked last step.
        # Zero for satisfied users (no longer relevant) and for unserved users (SINR≈0 by model).
        sinr_obs = np.clip(np.log2(1.0 + self.sinr) / 10.0, 0.0, 1.0).astype(np.float64)
        sinr_obs[user_satisfied] = 0.0

        # Store conflict matrix internally for other methods to access without polluting the Observation Space
        self._last_conflict_matrix = conflict_matrix.flatten()

        return {
            'needs':          (needs_remaining / MAX_NEED).astype(np.float64),
            'directions':     (directions / 360.0).astype(np.float64),
            'distance':       (distance / (self.config['max_range'] * 2)).astype(np.float64),
            'user_satisfied': user_satisfied.astype(np.int8),
            'remaining_time': remaining_time,
            'sinr_obs':       sinr_obs,
        }
