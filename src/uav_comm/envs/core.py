
import gymnasium as gym
from gymnasium import spaces
import numpy as np

from src.uav_comm.components.channel import total_path_loss, calculate_noise_level_db
from src.uav_comm.components.antenna import array_locs, pert2d_null_multi, phase_code_finder, find_gain_of_tphi
from src.uav_comm.components.rewards import calculate_reward_function, MAX_NEED

DEFAULT_CONFIG = {
    'num_users': 8,
    'num_arrays': 4,
    'num_elements_per_array': 8,
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
        self.num_elements_per_array = self.config['num_elements_per_array']

        self.bts_gain = 10 ** (50 / 10) * 10 ** (10 / 10)
        self.uav_user_gain = 10 ** (24 / 10) * 10 ** (0 / 10)
        self.sinr_threshold_linear = 10 ** (self.config['sinr_threshold_db'] / 10)

        self.array_configs = [array_locs(self.num_elements_per_array) for _ in range(self.num_arrays)]

        if self.config['operation_mode'] == 'single':
            # Mode a: one discrete user choice, all arrays point there
            self.action_space = spaces.Discrete(self.num_users)
        else:
            # Mode b: each array independently picks a user
            self.action_space = spaces.MultiDiscrete([self.num_users] * self.num_arrays)

        self.observation_space = spaces.Dict({
            'needs':          spaces.Box(low=0, high=1, shape=(self.num_users,), dtype=np.float64),
            'directions':     spaces.Box(low=-0.5, high=0.5, shape=(self.num_users,), dtype=np.float64),
            'distance':       spaces.Box(low=0, high=1, shape=(self.num_users,), dtype=np.float64),
            'user_satisfied': spaces.MultiBinary(self.num_users),
            'remaining_time': spaces.Box(low=0, high=1, shape=(1,), dtype=np.float64),
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

        for i in range(self.num_users):
            self._declare_need(i)

        return self._get_observation(), {}

    def step(self, action):
        self.current_time += self.config['time_interval']

        # Resolve per-array user assignments from action
        if self.config['operation_mode'] == 'single':
            uid = int(action.item()) if isinstance(action, np.ndarray) else int(action)
            selected_users = np.full(self.num_arrays, uid, dtype=int)
        elif isinstance(action, (int, np.integer)):
            selected_users = np.full(self.num_arrays, int(action), dtype=int)
        elif isinstance(action, np.ndarray) and action.ndim == 0:
            selected_users = np.full(self.num_arrays, int(action.item()), dtype=int)
        else:
            selected_users = action.astype(int)

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

        raw_reward, jfi_delay, urgency_thr, min_progress, total_thr = calculate_reward_function(
            throughputs_per_user, self.delay, self.progress, self.needs,
            active_users, self.current_time, self.config['max_episode_time']
        )

        # Safety net: penalise targeting already-satisfied users.
        # Action masking should prevent this; penalty is at raw scale (consistent with /10 below).
        inactive = np.where(~active_users)[0]
        wrong_count = int(np.sum(np.isin(selected_users, inactive)))
        if wrong_count > 0:
            raw_reward -= wrong_count * 2

        # Switch cost applied at raw scale so it stays proportionate after /10
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
            "JFI_delay": float(jfi_delay),
            "urgency_thr": float(urgency_thr),
            "min_progress": float(min_progress),
            "total_thr": float(total_thr),
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
            R = np.linalg.norm(self.uav_position - loc)
            D, PhaseTable = self.array_configs[array_idx]

            others = [u for i, u in enumerate(selected_users) if i != array_idx and u != uid]

            if others:
                int_dirs, int_dists = self._calculate_interference_directions(others)
                sig_db, int_db = pert2d_null_multi(
                    D, PhaseTable, theta, phi, R,
                    int_dirs[:, 0], int_dirs[:, 1], int_dists, noise_db
                )
                signals[uid] += 10 ** (sig_db / 10)
                for k, int_uid in enumerate(others):
                    interferences[int_uid] += 10 ** (int_db[k] / 10)
            else:
                ph = phase_code_finder(D, PhaseTable, theta, phi)
                sig_db = find_gain_of_tphi(theta, phi, ph, D) - total_path_loss(R)
                signals[uid] += 10 ** (sig_db / 10)

        noise_lin = 10 ** (noise_db / 10)
        self.sinr = signals / (interferences + noise_lin)

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
            dists.append(np.linalg.norm(np.append(loc - self.uav_position, self.config['uav_height'])))
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
        if self.config.get('operation_mode', 'multi') == 'single':
            # Discrete action space: mask shape = (num_users,)
            return user_mask
        else:
            # MultiDiscrete action space: flat mask shape = (num_users * num_arrays,)
            # sb3_contrib expects masks for all dimensions concatenated into one 1-D array.
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

        remaining_time = np.clip(
            [(self.config['max_episode_time'] - self.current_time) / self.config['max_episode_time']],
            0.0, 1.0
        ).astype(np.float64)

        return {
            'needs':          (needs_remaining / MAX_NEED).astype(np.float64),
            'directions':     (directions / 360.0).astype(np.float64),
            'distance':       (distance / (self.config['max_range'] * 2)).astype(np.float64),
            'user_satisfied': user_satisfied.astype(np.int8),
            'remaining_time': remaining_time,
        }
