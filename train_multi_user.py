
import os
import sys
import yaml
import numpy as np
from sb3_contrib import MaskablePPO
from sb3_contrib.common.wrappers import ActionMasker
from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize
from stable_baselines3.common.callbacks import CheckpointCallback

sys.path.append(os.getcwd())

from src.uav_comm.envs.core import UAVEnv

def load_train_config(config_path="configs/train_config.yaml"):
    with open(config_path, 'r') as f:
        return yaml.safe_load(f)

def mask_fn(env):
    return env.get_action_mask()

def train():
    print("Loading Configs...")
    config = load_train_config()

    # Load Env Config (Optional: can merge with train config or load separately)
    # For now assuming env params are in train config or we load env_config.yaml manually
    # Let's load env_config explicitly to pass it
    from src.uav_comm.utils.config_loader import load_config as load_env_config
    env_config = load_env_config()

    print("Initializing Multi-User Environment with Action Masking...")

    # Create env function that includes ActionMasker
    def make_env():
        env = UAVEnv(config=env_config)
        env = ActionMasker(env, mask_fn)
        return env

    # Wrap in DummyVecEnv for SB3
    env = DummyVecEnv([make_env])

    # Normalize Env? Typically good for PPO.
    # Exclude 'user_satisfied' as it is MultiBinary
    env = VecNormalize(env, norm_obs=True, norm_reward=True, clip_obs=10., norm_obs_keys=['needs', 'directions', 'distance'])

    print("Initializing MaskablePPO Agent (MultiDiscrete Support)...")
    # PPO naturally supports MultiDiscrete
    model = MaskablePPO(
        "MultiInputPolicy",
        env,
        verbose=1,
        learning_rate=config['learning_rate'],
        n_steps=config['n_steps'],
        batch_size=config['batch_size'],
        gamma=config['gamma'],
        gae_lambda=config['gae_lambda'],
        clip_range=config['clip_range'],
        ent_coef=config['ent_coef'],
        vf_coef=config['vf_coef'],
        policy_kwargs=config['policy_kwargs']
    )

    print(f"Starting Training for {config['total_timesteps']} timesteps...")
    checkpoint_callback = CheckpointCallback(save_freq=10000, save_path='./logs/', name_prefix='ppo_multi_user')

    model.learn(total_timesteps=config['total_timesteps'], callback=checkpoint_callback)

    print("Training Complete. Saving Model...")
    model.save("ppo_multi_user_final")
    env.save("vec_normalize.pkl")

if __name__ == "__main__":
    train()
