
import os
import mlflow
import numpy as np
from stable_baselines3.common.callbacks import BaseCallback
from src.uav_comm.utils.visualizer import EpisodeVisualizer

class GifEvalCallback(BaseCallback):
    """
    Evaluates the current policy every `eval_freq` steps and generates a GIF
    of the episode to provide interpretability and debug feedback mid-training.
    """
    def __init__(self, eval_env, eval_freq=50000, save_path="diagnostics/gifs", verbose=0):
        super().__init__(verbose)
        self.eval_env = eval_env
        self.eval_freq = eval_freq
        self.save_path = save_path
        os.makedirs(self.save_path, exist_ok=True)

    def _on_step(self) -> bool:
        if self.n_calls % self.eval_freq == 0:
            print(f"\n[GifEvalCallback] Generating mid-training GIF at step {self.num_timesteps}...")

            # Access the raw env to initialize the visualizer
            raw_env = self.eval_env.envs[0].env
            viz = EpisodeVisualizer(raw_env.config)

            obs = self.eval_env.reset()
            done = False

            while not done:
                # MaskablePPO requires action masks during prediction
                action_masks = np.array([e.env.get_action_mask() for e in self.eval_env.envs])
                action, _ = self.model.predict(obs, action_masks=action_masks, deterministic=True)

                obs, reward, done_arr, info = self.eval_env.step(action)

                # Fetch true environment state for the visualizer
                real_env = self.eval_env.envs[0].env

                # Calculate what the PINN penalty was for the action they just took
                n_users = real_env.num_users
                pinn_matrix = real_env._get_observation()['pinn_interference_matrix'].reshape((n_users, n_users))

                # Use true reward if available
                step_rew = info[0].get('true_reward', reward[0])

                viz.record_step(real_env, action[0], step_rew, pinn_matrix)

                done = done_arr[0]

            filename = os.path.join(self.save_path, f"training_progress_{self.num_timesteps // 1000}k.gif")
            viz.generate_gif(filename=filename)

        return True


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
