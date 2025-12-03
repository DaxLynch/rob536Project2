"""Visualize a trained agent."""
import gymnasium as gym
import panda_gym
from sbx import TQC
import time
import friction_env

# ===== CHANGE THESE FLAGS =====
# Options: "reach", "pick_ee", "pick_ee_friction", "pick_joints_friction"
MODE = "pick_ee"
# ==============================

if MODE == "pick_joints_friction":
    env_id = "ConstantFrictionPickAndPlaceJoints-v1"
    model_path = "best_models/constant_friction_joints_mode"
elif MODE == "pick_ee_friction":
    env_id = "FrictionPickAndPlace-v1"
    model_path = "best_models/pick_and_place_end_effector_friction_mode"
elif MODE == "pick_ee":
    env_id = "PandaPickAndPlace-v3"
    model_path = "best_models/pick_and_place_end_effector_mode_std_friction"
else:  # reach
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



