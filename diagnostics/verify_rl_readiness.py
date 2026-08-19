import os
import sys
import numpy as np
import yaml
from sb3_contrib.common.wrappers import ActionMasker
from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize
from stable_baselines3.common.env_checker import check_env

sys.path.append(os.getcwd())
from src.uav_comm.envs.core import UAVEnv

def mask_fn(env):
    return env.get_action_mask()

def verify_rl_readiness():
    print("--- Running RL Readiness & Fuzzing Diagnostics ---\n")

    with open("configs/env_config.yaml", "r") as f:
        env_config = yaml.safe_load(f)

    # 1. Native SB3 Check
    print("1. Executing strict SB3 Native Environment Check...")
    raw_env = UAVEnv(config=env_config)
    check_env(raw_env, warn=True)
    print("   [PASS] Native SB3 Check\n")

    # 2. Pipeline Setup
    print("2. Constructing VecNormalize Pipeline...")
    def make_env():
        e = UAVEnv(config=env_config)
        return ActionMasker(e, mask_fn)

    vec_env = DummyVecEnv([make_env])
    vec_env = VecNormalize(
        vec_env,
        norm_obs=True,
        norm_reward=True,
        clip_obs=10.0,
        norm_obs_keys=['needs', 'directions', 'distance', 'remaining_time', 'sinr_obs', 'pinn_interference_matrix'],
    )
    print("   [PASS] Pipeline Constructed\n")

    # 3. Fuzzing Loop
    print("3. Executing Fuzzing Loop (10,000 steps)...")
    obs = vec_env.reset()

    max_reward = -np.inf
    min_reward = np.inf

    for step in range(10000):
        # Sample random actions to stress-test the gambling penalties and PINN matrix
        action = [vec_env.action_space.sample()]
        obs, reward, done, info = vec_env.step(action)

        raw_obs = vec_env.envs[0].env._get_observation()
        space = vec_env.envs[0].env.observation_space

        # Check NaNs
        if np.isnan(reward).any() or np.isinf(reward).any():
            raise ValueError(f"CRITICAL: Reward contains NaN or Inf at step {step}: {reward}")

        max_reward = max(max_reward, reward[0])
        min_reward = min(min_reward, reward[0])

        # Check strict bounds on raw observations
        for key, val in raw_obs.items():
            if np.isnan(val).any() or np.isinf(val).any():
                raise ValueError(f"CRITICAL: Observation '{key}' contains NaN or Inf.")

            # The Box limits (MultiBinary doesn't have .low or .high)
            if hasattr(space[key], 'low') and hasattr(space[key], 'high'):
                low = space[key].low
                high = space[key].high

                if np.any(val < low - 1e-5) or np.any(val > high + 1e-5):
                    raise ValueError(f"CRITICAL: Observation '{key}' violated Box bounds!\nVal: {val}\nAllowed: [{low[0]}, {high[0]}]")

    print(f"   [PASS] 10,000 Fuzzing Steps Completed without Math/Bound Errors.")
    print(f"   Train Reward Range (Normalized): [{min_reward:.3f}, {max_reward:.3f}]\n")

    print("--- ALL DIAGNOSTICS PASSED. SYSTEM IS RL-READY. ---")

if __name__ == "__main__":
    verify_rl_readiness()
