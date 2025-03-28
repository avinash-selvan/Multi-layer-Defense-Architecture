from test_agent import B_line_minimal, React_restore_minimal
import numpy as np
from stable_baselines3 import DQN
from minimal import SimplifiedCAGE
from stable_baselines3.common.vec_env import DummyVecEnv
from stable_baselines3.common.env_util import make_vec_env
import gym
import matplotlib.pyplot as plt

class GymCompatibleEnv(SimplifiedCAGE):
    def __init__(self, num_envs=1):
        super().__init__(num_envs=num_envs, remove_bugs=True)
        self.red_agent = B_line_minimal()      # Add Red Agent
        self.blue_agent = React_restore_minimal()  # Add Blue Agent
        self._np_random = np.random.default_rng()  # Initialize RNG for seeding

        self.detection_success_rate = 0.7
        self.red_agent_success_rate = 0.3
        self.detection_log = np.zeros(13)

        # ✅ Fix: Initialize total reward tracker
        self.total_blue_reward = 0
    
    def seed(self, seed=None):
        """Set random seed for reproducibility."""
        self._np_random = np.random.default_rng(seed)

    def reset(self, seed=None, options=None):
        """Reset the environment and return initial observation."""
        if seed is not None:
            self.seed(seed)

        reset_output = super().reset()
        
        if isinstance(reset_output, tuple) and len(reset_output) == 2:
            obs, info = reset_output
        else:
            obs = reset_output
            info = {}
        
        # Ensure observation is properly formatted
        obs = np.array(obs, dtype=np.float32).flatten()
        print(f"Reset Observation Shape: {obs.shape}")
        return obs, info
    
    def _calculate_rewards(self):
        """New reward calculation method"""
        blue_reward = 0
        red_reward = 0
        
        # Detection bonus
        if self.blue_action == 2:  # Detection action
            if np.random.random() < self.detection_success_rate:
                blue_reward += 20
                red_reward -= 15
                self.detection_log = np.ones(13)
            else:
                # If Blue detects but no real attack happened, apply a penalty
                if self.red_success.sum() == 0:  # No actual attack success
                    blue_reward -= 10  # Penalize false detections
        
        # Penalize for doing nothing
        if self.blue_action == 0:
            blue_reward -= 5
            
        # Scale rewards
        blue_reward = np.clip(blue_reward / 10, -5, 5)
        red_reward = np.clip(red_reward / 10, -5, 5)
        
        self.total_blue_reward += blue_reward
        return blue_reward, red_reward

    def step(self, action):
        # Store the action for reference in reward calculation
        self.blue_action = action  # ✅ Add this line
        
        # Handle action format
        if isinstance(action, np.ndarray) and action.shape == (1,):
            action = action[0]
            
        # Get red action with new success rate
        red_action = [1] if np.random.random() < self.red_agent_success_rate else [0]
        
        # Environment step
        step_output = super().step(blue_action=action, red_action=red_action)
        
        # Calculate new rewards
        blue_reward, red_reward = self._calculate_rewards()  # ✅ Now it works
        
        # Handle output format
        if len(step_output) == 5:
            obs, _, done, truncated, info = step_output
        else:
            obs, _, done, info = step_output
            truncated = done
            
        # Early termination
        if self.total_blue_reward < -100:
            done = True
            
        # Normalize observation
        obs = np.array(obs, dtype=np.float32)
        obs[obs == -1] = 0  # Replace -1 with 0
        
        return obs, blue_reward, done, truncated, info


# Create environment - using DummyVecEnv directly for better control
# Create environment
env = DummyVecEnv([lambda: GymCompatibleEnv(num_envs=1)])

# Create model with new parameters
model = DQN(
    "MlpPolicy", env,
    learning_rate=5e-4,
    buffer_size=100000,
    batch_size=128,
    gamma=0.9,
    exploration_fraction=0.1,
    exploration_final_eps=0.1,
    target_update_interval=500,
    train_freq=4,
    gradient_steps=1,
    policy_kwargs=dict(net_arch=[256, 256]),
    verbose=1
)

# Add callback
from stable_baselines3.common.callbacks import EvalCallback
eval_callback = EvalCallback(env, best_model_save_path='./logs/',
                           eval_freq=1000, deterministic=True)

# Train longer
#model.learn(total_timesteps=20, callback=eval_callback)

# Test the environment manually before training
print("=== Testing Environment ===")
obs = env.reset()
for i in range(5):
    action = [env.action_space.sample()]  # Note the list for vectorized env
    obs, reward, done, info = env.step(action)
    print(f"Step {i}: Action {action}, Reward {reward}")
    print(f"Observation sample: {obs[0][:10]}...")  # First 10 elements
    if done[0]:  # Check done flag for first environment
        obs = env.reset()

# Train the model
print("\n=== Training Model ===")
model.learn(total_timesteps=50, progress_bar=True, log_interval=10)
model.save("dqn_cyber_defense")

# Evaluation parameters
NUM_EPISODES = 50
MAX_STEPS = 50

# Test random agent
print("\n=== Testing Random Agent ===")
random_rewards = []
for _ in range(NUM_EPISODES):
    obs = env.reset()
    total_reward = 0
    for _ in range(MAX_STEPS):
        action = [env.action_space.sample()]  # Note the list for vectorized env
        obs, reward, done, _ = env.step(action)
        total_reward += reward[0]  # Get reward from first environment
        if done[0]:  # Check done flag for first environment
            break
    random_rewards.append(total_reward)
    print(f"Episode reward: {total_reward}")

# Test trained agent
print("\n=== Testing Trained Agent ===")
trained_rewards = []
for _ in range(NUM_EPISODES):
    obs = env.reset()
    total_reward = 0
    for _ in range(MAX_STEPS):
        action, _ = model.predict(obs, deterministic=True)
        obs, reward, done, _ = env.step(action)
        total_reward += reward[0]  # Get reward from first environment
        if done[0]:  # Check done flag for first environment
            break
    trained_rewards.append(total_reward)
    print(f"Episode reward: {total_reward}")

# Print performance comparison
print(f"\nRandom Agent Average Reward: {np.mean(random_rewards):.2f}")
print(f"Trained Agent Average Reward: {np.mean(trained_rewards):.2f}")

# Plot results
plt.figure(figsize=(10, 5))
plt.plot(trained_rewards, label="Trained Agent")
plt.plot(random_rewards, label="Random Agent", linestyle="dashed")
plt.xlabel("Episodes")
plt.ylabel("Total Reward")
plt.legend()
plt.title("Trained vs. Random Agent Performance")
plt.savefig("training_plot.png")
plt.show()
