
import numpy as np

MIN_PROGRESS = 1e-4
MAX_NEED = 10

def calculate_reward_function(throughput, mean_distance, delay, progress, needs, active_users, episode_time):
    # Constants
    time_bonus_coeff = 0.0002
    MAX_EPISODE_TIME = 30 # TODO: Pass this in config

    closeness_reward = 10 / (mean_distance + 10)

    user_weight = 1
    half_life_weight = 0.5
    throughput_weight = 0.2

    half_life_users = (needs / 2 > progress)
    n_half_life = np.sum(half_life_users)
    n_active = np.sum(active_users)

    episode_reward = (
        user_weight / (n_active + 0.5) +
        half_life_weight / (n_half_life + 1)
    )

    sinr_reward = (throughput**3) * throughput_weight

    # JFI for Delay
    sum_delay = np.sum(delay[active_users])
    sum_delay_squared = np.sum(delay[active_users] ** 2)
    jfi_delay = (sum_delay ** 2) / (n_active * sum_delay_squared + 1e-6)

    # Proportional Fairness
    proportional_fairness_reward = np.sum(np.log(np.maximum(progress[active_users] / MAX_NEED, MIN_PROGRESS)))

    alpha = 2
    beta = 0.1

    raw_reward = sinr_reward - alpha * jfi_delay + beta * proportional_fairness_reward - episode_reward

    # Stateful normalization would need to be handled by a class or external tracker.
    # For now, return raw components.

    return raw_reward, jfi_delay, sinr_reward, proportional_fairness_reward, episode_reward

class RewardTracker:
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
        norm_rew = (reward - self.meanR) / (np.sqrt(self.varR) + 1e-8)
        norm_rew = np.clip(norm_rew, -10, +10)
        return norm_rew
