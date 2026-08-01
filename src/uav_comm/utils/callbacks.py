
import mlflow
import numpy as np
from stable_baselines3.common.callbacks import BaseCallback

class MLflowCallback(BaseCallback):
    def __init__(self, verbose=0):
        super(MLflowCallback, self).__init__(verbose)

    def _on_step(self) -> bool:
        # Check if an episode just finished
        if len(self.model.ep_info_buffer) > 0 and len(self.model.ep_info_buffer) > 0:
            # We can log the latest episode stats
            info = self.model.ep_info_buffer[-1]
            if "r" in info:
                mlflow.log_metric("rollout/ep_rew_mean", info["r"], step=self.num_timesteps)
            if "l" in info:
                mlflow.log_metric("rollout/ep_len_mean", info["l"], step=self.num_timesteps)

        # We can also log internal values like loss if available in locals()
        # For PPO, losses are available after update, usually logged in 'train/' via Logger
        # This callback runs every step, so we might want to log only periodically or hook into logger.
        # SB3 logger integration is more complex. Simple metric logging is often sufficient.

        return True
