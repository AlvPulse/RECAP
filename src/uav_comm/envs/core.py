
import gymnasium as gym
from gymnasium import spaces
import numpy as np
import copy

from src.uav_comm.components.channel import total_path_loss, calculate_noise_level_db, landa
from src.uav_comm.components.antenna import array_locs, pert2d_null_multi, phase_code_finder, find_gain_of_tphi
from src.uav_comm.components.rewards import calculate_reward_function, RewardTracker, MAX_NEED

# Default Configs
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
    'switch_cost': 0.25
}

class UAVEnv(gym.Env):
    def __init__(self, config=None):
        super(UAVEnv, self).__init__()

        self.config = DEFAULT_CONFIG.copy()
        if config:
            self.config.update(config)

        self.num_users = self.config['num_users']
        self.num_arrays = self.config['num_arrays']
        self.num_elements_per_array = self.config['num_elements_per_array']

        self.current_time = 0.0

        self.bts_gain = 10 ** (50 / 10) * 10 ** (10 / 10)
        self.uav_user_gain = 10 ** (24 / 10) * 10 ** (0 / 10)
        self.sinr_threshold_linear = 10 ** (self.config['sinr_threshold_db'] / 10)

        self.array_configs = [array_locs(self.num_elements_per_array) for _ in range(self.num_arrays)]

        # --- ACTION SPACE UPGRADE ---
        # Original: Discrete(num_users) - Single user selection
        # New: MultiDiscrete([num_users] * num_arrays) - Independent selection per array
        self.action_space = spaces.MultiDiscrete([self.num_users] * self.num_arrays)

        self.observation_space = spaces.Dict({
            'needs': spaces.Box(low=0, high=1, shape=(self.num_users,), dtype=np.float64),
            'directions': spaces.Box(low=-0.5, high=0.5, shape=(self.num_users,), dtype=np.float64),
            'distance': spaces.Box(low=0, high=1, shape=(self.num_users,), dtype=np.float64),
            'user_satisfied': spaces.MultiBinary(self.num_users)
        })

        self.reward_tracker = RewardTracker()
        self.reset()
        self.last_action = None

    def reset(self, seed=None, options=None):
        if seed:
            np.random.seed(seed)

        range_lim = self.config['max_range']
        self.locations = np.random.uniform(-range_lim, range_lim, (self.num_users, 2))
        self.uav_position = np.random.uniform(-range_lim/2, range_lim/2, (2,))
        self.needs = np.zeros(self.num_users)
        self.progress = np.zeros(self.num_users)
        self.sinr = np.zeros(self.num_users)
        self.delay = np.zeros(self.num_users)
        self.current_time = 0.0

        for user_idx in range(self.num_users):
            self._declare_need(user_idx)

        return self._get_observation(), {}

    def step(self, action):
        switch_cost = self.config['switch_cost']
        self.current_time += self.config['time_interval']

        # --- ACTION HANDLING ---
        # Handle scalar (single-user legacy) or array (multi-user)
        if isinstance(action, (int, np.integer)):
            selected_users = action * np.ones(self.num_arrays, dtype=int)
        elif isinstance(action, np.ndarray) and action.ndim == 0:
            selected_users = action.item() * np.ones(self.num_arrays, dtype=int)
        else:
            selected_users = action.astype(int)

        self._calculate_sinr(selected_users)

        active_users = (self.needs > 0) & (self.needs > self.progress)

        for i in range(self.num_users):
            if active_users[i]:
                self.delay[i] += 1

        Throughput = 0

        # In multi-user, an array serves a user.
        # A user might be served by MULTIPLE arrays (constructive) or distinct ones.
        # _serve_user uses self.sinr which aggregates signal/interference.
        # But wait, self.sinr is per USER. So we just iterate over UNIQUE selected users to accrue throughput?
        # Or iterate over arrays?
        # Throughput depends on SINR. SINR is calculated once per step based on configuration.
        # So we just update progress for ALL selected users based on their SINR.

        unique_selected_users = np.unique(selected_users)
        for user_idx in unique_selected_users:
            if active_users[user_idx]:
                Throughput += self._serve_user(user_idx)

        mean_distance = self._update_uav_location(selected_users)

        raw_reward, jfi, sinr_reward, prop_reward, ep_reward = calculate_reward_function(
            Throughput, mean_distance, self.delay, self.progress, self.needs, active_users, self.current_time
        )

        # Normalize
        # But wait, Env used stateful normalization.
        norm_reward = self.reward_tracker.normalize(raw_reward)
        reward = raw_reward / 10 # Preserving original scaling logic

        # Penalty for inactive users
        inactive_user_indices = np.where(~active_users)[0]
        matches = np.isin(selected_users, inactive_user_indices)
        Wrong_count = np.sum(matches)
        reward = reward - Wrong_count * 10

        # New needs logic
        if np.random.rand() < 0.2:
            if len(inactive_user_indices) > 0:
                selected_index = np.random.choice(inactive_user_indices)
                if self.needs[selected_index] == 0:
                    self._declare_need(selected_index)

        done = np.all(self.progress >= self.needs)
        if done:
            time_bonus_coeff = 0.0002
            bonus = time_bonus_coeff * (self.config['max_episode_time'] - self.current_time)**2
            reward += bonus

        Truncated = self.current_time >= self.config['max_episode_time']

        observation = self._get_observation()
        info = {"JFI": jfi, "Bandwidth": 0, "Thr_fairness": prop_reward} # Bandwidth placeholder

        if self.last_action is not None and not np.array_equal(action, self.last_action):
            reward -= switch_cost

        self.last_action = action
        return observation, reward, bool(done), Truncated, info

    def _declare_need(self, user_idx):
        self.needs[user_idx] = 10

    def _serve_user(self, user_idx):
        throughput = 0
        if self.sinr[user_idx] > self.sinr_threshold_linear:
            throughput = self.config['bandwidth'] * np.log2(1 + self.sinr[user_idx]) * self.config['time_interval']
            self.delay[user_idx] = 0
            if (self.progress[user_idx] + throughput > self.needs[user_idx]):
                throughput = self.needs[user_idx] - self.progress[user_idx]
            self.progress[user_idx] += throughput
        return throughput

    def _calculate_sinr(self, selected_users):
        Noise_level_db = calculate_noise_level_db(self.config['bandwidth'])
        # Initialize arrays with 0.0 Linear Power (Watts)
        signals = np.zeros(self.num_users)
        interferences = np.zeros(self.num_users)

        for array_idx, user_idx in enumerate(selected_users):
            user_location = self.locations[user_idx]
            direction = self._calculate_direction(user_location)

            other_user_indices = [u for i, u in enumerate(selected_users) if i != array_idx and u != user_idx]

            RuserUAV = np.linalg.norm(self.uav_position - user_location)
            D, PhaseTable = self.array_configs[array_idx]

            if other_user_indices:
                interference_directions, interference_distances = self._calculate_interference_directions(other_user_indices)
                Signal_db, InterferencedB = pert2d_null_multi(D, PhaseTable, direction[0], direction[1], RuserUAV,
                                                      interference_directions[:, 0], interference_directions[:, 1], interference_distances, Noise_level_db)
                Interference_linear = 10**(InterferencedB/10)

                # Accumulate signal power (Linear)
                signals[user_idx] += 10**(Signal_db/10)

                # Accumulate interference power (Linear)
                for idx, int_user_in_list in enumerate(other_user_indices):
                    interferences[int_user_in_list] += Interference_linear[idx]
            else:
                phBest = phase_code_finder(D, PhaseTable, direction[0], direction[1])
                Signal_db = find_gain_of_tphi(direction[0], direction[1], phBest, D) - total_path_loss(RuserUAV)
                signals[user_idx] += 10**(Signal_db/10)

        noise_linear = 10**(Noise_level_db/10)
        self.sinr = signals / (interferences + noise_linear)

    def _calculate_direction(self, user_location):
        diff_x, diff_y = user_location - self.uav_position
        distance = np.sqrt(diff_x ** 2 + diff_y ** 2 + self.config['uav_height'] ** 2)
        theta = np.degrees(np.arccos(self.config['uav_height'] / distance))
        phi = np.degrees(np.arctan2(diff_y, diff_x))
        return theta, phi

    def _calculate_interference_directions(self, user_indices):
        directions = []
        distances = []
        for user_idx in user_indices:
            user_location = self.locations[user_idx]
            direction = self._calculate_direction(user_location)
            distance = np.linalg.norm(np.append(user_location - self.uav_position, self.config['uav_height']))
            directions.append(direction)
            distances.append(distance)
        return np.array(directions), np.array(distances)

    def _update_uav_location(self, selected_users):
        # Move towards centroid of optimal positions for selected users
        optimal_locations = []
        unique_users = np.unique(selected_users)
        for user_idx in unique_users:
            optimal_location = self._calculate_optimal_location(user_idx)
            optimal_locations.append(optimal_location)

        if optimal_locations:
            avg_optimal_location = np.mean(optimal_locations, axis=0)
            direction = avg_optimal_location - self.uav_position
            if np.linalg.norm(direction) > 1e-6:
                direction = direction / np.linalg.norm(direction)
            self.uav_position += direction * self.config['uav_speed'] * self.config['time_interval']

        user_distance = np.linalg.norm(self.uav_position - self.locations, axis=1)
        # Mean distance of ACTIVE users? Or selected?
        # Env used selected.
        return np.mean(user_distance[unique_users]) if len(unique_users)>0 else 0

    def _calculate_optimal_location(self, user_idx):
        gain_ratio = self.bts_gain / self.uav_user_gain
        user_location = self.locations[user_idx]
        optimal_distance_ratio = np.sqrt(gain_ratio)
        user_distance = np.linalg.norm(self.uav_position - user_location)
        optimal_user_distance = user_distance / optimal_distance_ratio

        direction = user_location - self.uav_position
        if np.linalg.norm(direction) > 1e-6:
            direction = direction / np.linalg.norm(direction)

        optimal_location = user_location - direction * optimal_user_distance
        return optimal_location

    def get_action_mask(self):
        # Mask for MultiDiscrete is tricky in SB3.
        # Usually ActionMasker expects Discrete.
        # For MultiDiscrete, we might need a custom wrapper or just rely on penalty.
        # But let's return the basic boolean mask for active users.
        return (self.needs > self.progress)

    def _get_observation(self):
        needs_progress = self.needs - self.progress
        user_satisfied = needs_progress <= 0
        needs_progress[user_satisfied] = 0
        distance = np.linalg.norm(self.locations - self.uav_position, axis=1)

        user_directions = np.zeros((self.num_users,))
        for user_idx in range(self.num_users):
            user_location = self.locations[user_idx]
            direction = self._calculate_direction(user_location)
            user_directions[user_idx] = direction[1]
        user_directions[user_satisfied] = 0

        return {
            'needs': needs_progress / MAX_NEED,
            'directions': user_directions / 360,
            'distance': distance / self.config['max_range'] / 2,
            'user_satisfied': user_satisfied
        }
