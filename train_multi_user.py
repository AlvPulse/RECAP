
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
    # ActionMasker might pass a wrapped env (e.g., Monitor).
    # We need to access get_action_mask from the base environment.
    # Gymnasium wrappers typically support direct access if the method exists,
    # but some versions/wrappers might not.
    # Using getattr or accessing unwrapped is safer.
    return env.unwrapped.get_action_mask()

def train():
    print("Loading Configs...")
    train_config = load_train_config()

    from src.uav_comm.utils.config_loader import load_config as load_env_config
    env_config = load_env_config()

    # MLflow Setup
    mlflow.set_experiment("UAV_MultiUser_RL")

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    run_name = f"ppo_multi_user_{timestamp}"

    print(f"Starting MLflow Run: {run_name}")
    with mlflow.start_run(run_name=run_name):
        # Log Hyperparameters
        mlflow.log_params(train_config)
        mlflow.log_params(env_config)

        # Log Config Artifacts
        mlflow.log_artifact("configs/train_config.yaml")
        mlflow.log_artifact("configs/env_config.yaml")

        from stable_baselines3.common.monitor import Monitor
        print("Initializing Multi-User Environment with Action Masking...")

        def make_env():
            env = UAVEnv(config=env_config)
            # Monitor is required for logging episode stats (reward, length) to MLflow
            env = Monitor(env)
            env = ActionMasker(env, mask_fn)
            return env

        env = DummyVecEnv([make_env])

        env = VecNormalize(env, norm_obs=True, norm_reward=True, clip_obs=10., norm_obs_keys=['needs', 'directions', 'distance'])

        print("Initializing MaskablePPO Agent...")
        model = MaskablePPO(
            "MultiInputPolicy",
            env,
            verbose=1,
            learning_rate=train_config['learning_rate'],
            n_steps=train_config['n_steps'],
            batch_size=train_config['batch_size'],
            gamma=train_config['gamma'],
            gae_lambda=train_config['gae_lambda'],
            clip_range=train_config['clip_range'],
            ent_coef=train_config['ent_coef'],
            vf_coef=train_config['vf_coef'],
            policy_kwargs=train_config['policy_kwargs']
        )

        print(f"Starting Training for {train_config['total_timesteps']} timesteps...")

        # Callbacks
        mlflow_callback = MLflowCallback()

        model.learn(total_timesteps=train_config['total_timesteps'], callback=mlflow_callback)

        print("Training Complete. Saving Model...")
        os.makedirs("models", exist_ok=True)
        model_save_path = f"models/{run_name}"
        model.save(model_save_path)
        env.save(f"models/{run_name}_vec_normalize.pkl")

        # Create a "latest" alias for easy evaluation
        model.save("models/ppo_multi_user_latest")
        env.save("models/vec_normalize_latest.pkl")

        # Log Model Artifact
        mlflow.log_artifact(model_save_path + ".zip")
        print(f"Model saved to {model_save_path}.zip and logged to MLflow.")

if __name__ == "__main__":
    train()
