import numpy as np
import torch
from stable_baselines3 import DQN

# Load trained model (no retraining)
model = DQN.load("dqn_cyber_defense")

# Ensure model is in strict evaluation mode
model.policy.eval()

# Freeze model parameters
for param in model.policy.parameters():
    param.requires_grad = False

# New observation pattern - Example (Shape: (52,))
obs = np.array([
    1., 0., 0., -1., -1., -1., -1., -1., 1., 1., -1., 1., -1., -1., -1., -1., -1., -1.,
    -1., -1., -1., -1., -1., -1., -1., -1., -1., 1., 0., 0., -1., -1., -1., -1., -1., -1.,
    -1., -1., -1., -1., -1., -1., -1., -1., -1., -1., -1., -1., -1., -1., 0., 1.
], dtype=np.float32)

print("Initial Observation:", obs)

done = False
with torch.no_grad():  # Disable gradients
    while not done:
        # Get the model's action
        action, _ = model.predict(obs, deterministic=True)

        # Simulate environment step (replace with actual interaction logic)
        print(f"Action Taken: {action}")

        # Example: Simulated reward and done flag (replace with real values)
        reward = 5 if action == 2 else -45
        done = True

        print(f"Reward: {reward}, Done: {done}")

        # Send alert if an attack is detected
        if reward > 0:
            print("ALERT: Possible attack detected!")

# Final result
if reward > 0:
    print("Defense SUCCESS")
else:
    print("Defense FAILED")
