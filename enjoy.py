"""
Visualize a trained agent using stable-baselines3.

This script loads a trained model and displays the agent's behavior in the environment.
"""
import gymnasium as gym
import panda_gym
from sbx import TQC
import time

# Load the trained model
model_path = "./logs/tqc_sbx/PandaReach-v3_20251124_173555/best_model/best_model"
print(f"Loading model from {model_path}...")
env = gym.make("PandaReach-v3", render_mode="human", renderer="OpenGL")
model = TQC.load(model_path, env=env)

# Create the environment with rendering enabled

print("Starting visualization...")
print("Close the window to stop.")

# Run the agent in the environment
obs, info = env.reset()
total_reward = 0
episode_count = 0

try:
    while True:
        # Get action from the trained agent
        action, _states = model.predict(obs, deterministic=True)
        
        # Step the environment
        obs, reward, terminated, truncated, info = env.step(action)
        total_reward += reward
        time.sleep(1.0/12.0)
        # Reset if episode is done
        if terminated or truncated:
            episode_count += 1
            print(f"Episode {episode_count} finished with total reward: {total_reward:.2f}")
            obs, info = env.reset()
            total_reward = 0

except KeyboardInterrupt:
    print("\nVisualization stopped by user.")

finally:
    env.close()



