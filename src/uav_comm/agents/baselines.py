
import numpy as np

class MultiUserBaselines:
    def __init__(self, env):
        self.env = env

    def multi_greedy(self):
        # Select Top K users with highest remaining needs
        # K = num_arrays
        remaining = self.env.needs - self.env.progress
        mask = (self.env.needs > self.env.progress)

        # Valid indices
        valid_indices = np.where(mask)[0]

        if len(valid_indices) == 0:
            return np.zeros(self.env.num_arrays, dtype=int)

        # Sort valid users by remaining need (descending)
        sorted_valid = valid_indices[np.argsort(remaining[valid_indices])[::-1]]

        # Select Top K
        K = self.env.num_arrays
        if len(sorted_valid) >= K:
            selected = sorted_valid[:K]
        else:
            # Pad with best user or cycle?
            # Greedy usually means serve most urgent.
            # If arrays > users, double up on best users?
            # Or just idle? But action space requires selection.
            # Let's cycle/fill.
            selected = np.resize(sorted_valid, K)

        return selected

    def multi_fcfs(self):
        # Select Top K closest users (FCFS proxy)
        distances = np.linalg.norm(self.env.locations - self.env.uav_position, axis=1)
        mask = (self.env.needs > self.env.progress)
        valid_indices = np.where(mask)[0]

        if len(valid_indices) == 0:
            return np.zeros(self.env.num_arrays, dtype=int)

        # Sort valid by distance (ascending)
        sorted_valid = valid_indices[np.argsort(distances[valid_indices])]

        K = self.env.num_arrays
        if len(sorted_valid) >= K:
            selected = sorted_valid[:K]
        else:
            selected = np.resize(sorted_valid, K)

        return selected
