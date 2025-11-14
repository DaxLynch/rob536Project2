"""
Train a reinforcement learning agent using panda-gym and stable-baselines3.

This script demonstrates training a DDPG agent on the PandaReach-v2 environment.
You can modify the environment and algorithm as needed.
"""
import gymnasium as gym
import panda_gym
from stable_baselines3 import DDPG

# Create the environment
env = gym.make("PandaReach-v3")

# Create the DDPG model
# MultiInputPolicy is used for environments with multiple observation spaces
model = DDPG(policy="MultiInputPolicy", env=env)

# Train the agent for 30,000 timesteps
print("Starting training...")
model.learn(total_timesteps=30_000)
print("Training completed!")

# Save the trained model
model.save("panda_reach_ddpg")
print("Model saved as 'panda_reach_ddpg'")

