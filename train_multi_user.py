
import os
import sys
import yaml
import numpy as np
import mlflow
from datetime import datetime
from sb3_contrib import MaskablePPO
from sb3_contrib.common.wrappers import ActionMasker
from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize
from stable_baselines3.common.callbacks import CheckpointCallback

sys.path.append(os.getcwd())

from src.uav_comm.envs.core import UAVEnv
from src.uav_comm.utils.callbacks import MLflowCallback


def load_train_config(config_path="configs/train_config.yaml"):
    with open(config_path, 'r') as f:
        return yaml.safe_load(f)


def mask_fn(env):
    return env.get_action_mask()


def linear_schedule(initial_lr):
    """Linearly decay learning rate from initial_lr to 0 over training."""
    def schedule(progress_remaining):
        return initial_lr * progress_remaining
    return schedule


def train():
    print("Loading configs...")
    train_config = load_train_config()

    from src.uav_comm.utils.config_loader import load_config as load_env_config
    env_config = load_env_config()

    mlflow.set_experiment("UAV_MultiUser_RL")
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    run_name = f"ppo_multi_user_{timestamp}"

    print(f"Starting MLflow run: {run_name}")
    with mlflow.start_run(run_name=run_name):
        mlflow.log_params(train_config)
        mlflow.log_params(env_config)
        mlflow.log_artifact("configs/train_config.yaml")
        mlflow.log_artifact("configs/env_config.yaml")

        n_envs = train_config.get('n_envs', 4)
        print(f"Building {n_envs} parallel environments...")

        def make_env():
            e = UAVEnv(config=env_config)
            return ActionMasker(e, mask_fn)

        # DummyVecEnv runs envs sequentially but multiplies batch diversity.
        # On Linux, replace with SubprocVecEnv([make_env]*n_envs) for wall-clock speedup.
        env = DummyVecEnv([make_env] * n_envs)
        env = VecNormalize(
            env,
            norm_obs=True,
            norm_reward=True,
            clip_obs=10.0,
            norm_obs_keys=['needs', 'directions', 'distance', 'remaining_time'],
            # Note: conflict_matrix is 0/1 boolean, typically better not to normalize it strictly,
            # but VecNormalize will ignore it if not in norm_obs_keys in SB3, or we just leave it out.
        )

        print("Initialising MaskablePPO...")
        model = MaskablePPO(
            "MultiInputPolicy",
            env,
            verbose=1,
            learning_rate=linear_schedule(train_config['learning_rate']),
            n_steps=train_config['n_steps'],
            batch_size=train_config['batch_size'],
            gamma=train_config['gamma'],
            gae_lambda=train_config['gae_lambda'],
            clip_range=train_config['clip_range'],
            ent_coef=train_config['ent_coef'],
            vf_coef=train_config['vf_coef'],
            n_epochs=train_config['n_epochs'],
            policy_kwargs=train_config['policy_kwargs'],
        )

        total_steps = train_config['total_timesteps']
        print(f"Training for {total_steps} timesteps across {n_envs} envs "
              f"({total_steps // n_envs} steps each)...")

        checkpoint_cb = CheckpointCallback(
            save_freq=max(50_000 // n_envs, 1),
            save_path="models/checkpoints/",
            name_prefix="ppo_uav",
        )
        mlflow_cb = MLflowCallback()

        model.learn(total_timesteps=total_steps, callback=[checkpoint_cb, mlflow_cb])

        print("Saving model...")
        os.makedirs("models", exist_ok=True)
        model_path = f"models/{run_name}"
        model.save(model_path)
        env.save(f"models/{run_name}_vec_normalize.pkl")
        model.save("models/ppo_multi_user_latest")
        env.save("models/vec_normalize_latest.pkl")

        mlflow.log_artifact(model_path + ".zip")
        print(f"Model saved to {model_path}.zip")


if __name__ == "__main__":
    train()
