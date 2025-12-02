"""Visualize a trained agent."""
import gymnasium as gym
import panda_gym
from sbx import TQC
import time
import friction_env
# ===== CHANGE THIS FLAG =====
USE_PICK_AND_PLACE = True  # False = PandaReach, True = PandaPickAndPlace
# ============================
FRICTION_MODE = True
if FRICTION_MODE:
    env_id = "FrictionPickAndPlace-v1"
    #model_path = "best_models/pick_and_place_end_effector_friction_mode"
    model_path = "best_models/pick_and_place_friction_friction_jamie.zip"
elif USE_PICK_AND_PLACE:
    env_id = "PandaPickAndPlace-v3"
    model_path = "best_models/pick_and_place_end_effector_mode_std_friction"
else:
    env_id = "PandaReach-v3"
    model_path = "best_models/panda_reach_end_effector_std_friction"

print(f"Loading {env_id}...")
env = gym.make(env_id, render_mode="human", renderer="OpenGL")
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
        time.sleep(1.0/24.0)
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



