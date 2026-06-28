
import os
import mlflow
import numpy as np
from stable_baselines3.common.callbacks import BaseCallback


class VecNormalizeCheckpointCallback(BaseCallback):
    """Save VecNormalize statistics alongside each model checkpoint.

    The standard CheckpointCallback saves only model weights (.zip).
    This callback saves the VecNormalize pkl at the same cadence so that
    checkpoints can be evaluated with proper observation normalisation.
    """

    def __init__(self, save_freq, save_path, name_prefix="ppo_uav", verbose=0):
        super().__init__(verbose)
        self.save_freq = save_freq
        self.save_path = save_path
        self.name_prefix = name_prefix

    def _on_step(self) -> bool:
        if self.n_calls % self.save_freq == 0:
            vec_env = self.model.get_vec_normalize_env()
            if vec_env is not None:
                path = os.path.join(
                    self.save_path,
                    f"{self.name_prefix}_{self.num_timesteps}_steps_vecnorm.pkl"
                )
                vec_env.save(path)
        return True


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
