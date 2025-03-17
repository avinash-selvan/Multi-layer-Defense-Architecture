import numpy as np
from stable_baselines3 import DQN
from minimal import SimplifiedCAGE  # Import MiniCAGE environment

# Initialize MiniCAGE environment
from stable_baselines3.common.vec_env import DummyVecEnv

# 🔍 Debugging Wrapper for SB3’s DummyVecEnv
class DebugDummyVecEnv(DummyVecEnv):
    def _save_obs(self, env_idx, obs):
        import traceback
        if isinstance(obs, tuple):
            obs = obs[0]  # ✅ Extract only the observation


        print(f"🛑 SB3 Buffer Issue: Trying to store {obs.shape}, {obs.dtype} at index {env_idx}")
        print(f"🔍 Buffer Expected Shape: {self.buf_obs[0].shape}")
        print(f"🔍 Full Trace:\n{''.join(traceback.format_stack())}")
        super()._save_obs(env_idx, obs)

# ✅ Replace SB3’s `DummyVecEnv` with Debug Version
from stable_baselines3.common.vec_env import DummyVecEnv
import gym

# ✅ Ensure Gym-compatible wrapping
class GymCompatibleEnv(SimplifiedCAGE):
    def __init__(self, num_envs=1):
        super().__init__(num_envs)
        self._np_random = np.random.default_rng()  # ✅ Initialize RNG for seeding
    
    def seed(self, seed=None):
            """✅ Ensure SB3 can set seeds correctly."""
            self._np_random = np.random.default_rng(seed)  # ✅ Set random seed
            print(f"🔍 Environment seeded with: {seed}")

    def reset(self, seed=None, options=None):
        """Ensure SB3 can pass a seed during reset."""
        if seed is not None:
            self.seed(seed)

        reset_output = super().reset()  # Fix: Store output

        if isinstance(reset_output, tuple) and len(reset_output) == 2:
            obs, info = reset_output
        else:
            obs = reset_output  # Fallback if only one value is returned
            info = {}           # Provide an empty info dictionary

        obs = np.array(obs, dtype=np.float32).flatten()
        return obs, info  # Explicitly return both


    def step(self, action):
        step_output = super().step(action)
        if len(step_output) == 5:
            return step_output
    
        obs, reward, done, info = step_output  # 🔥 Fix unpacking
        truncated = False  # ✅ Add truncated flag

        return np.array(obs, dtype=np.float32), reward, done, truncated, info  # ✅ Ensure output format


# ✅ Manually wrap it in DummyVecEnv
# env = DummyVecEnv([lambda: GymCompatibleEnv(num_envs=1)])
from stable_baselines3.common.env_util import make_vec_env

# ✅ SB3-Compliant Environment Wrapper
env = make_vec_env(lambda: GymCompatibleEnv(num_envs=1), n_envs=1)

# ✅ Print buffer setup details
print(f"🔍 Observation Space: {env.observation_space.shape}, Dtype: {env.observation_space.dtype}")

# ✅ Print observation space details
print(f"🔍 Final Observation Space: {env.observation_space.shape}, Dtype: {env.observation_space.dtype}")


# Create the DQN model
model = DQN("MlpPolicy", env, verbose=1, learning_rate=0.0001, buffer_size=50000, policy_kwargs={"normalize_images": False})
print(f"🔍 Expected Observation Space: {env.observation_space.shape}, Dtype: {env.observation_space.dtype}")

# Train the model
model.learn(total_timesteps=100000, progress_bar=True, log_interval=10)

# Save the trained model
model.save("dqn_cyber_defense")

# Test the trained model
obs = env.reset()
for _ in range(10):
    action, _ = model.predict(obs)
    obs, reward, done, _ = env.step(action)
    print(f"Action: {action}, Reward: {reward}")
    if done:
        obs = env.reset()


