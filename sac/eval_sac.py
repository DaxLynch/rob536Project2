import gymnasium as gym
import panda_gym
from datetime import datetime
from stable_baselines3 import SAC
from stable_baselines3.common.vec_env import (
        DummyVecEnv,
        VecVideoRecorder,
        VecNormalize,
)           

# ----------------------
# 1. Make env for live viewing
# ----------------------
def make_env():
    return gym.make("PandaPickAndPlace-v3",
                    render_mode="rgb_array")  # for video frames

# Vectorize for video recording
raw_env = DummyVecEnv([make_env])

base_env = VecNormalize.load("vecnormalize_stats.pkl", raw_env)

# Disable training mode for evaluation
base_env.training = False
base_env.norm_reward = False   # recommended for HER tasks

# Wrap in video recorder
video_folder = "./videos/"
video_length = 2000

timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

eval_env = VecVideoRecorder(
    base_env,
    video_folder,
    record_video_trigger=lambda step: step % 100000 == 0,
    name_prefix=f"sac_panda_{timestamp}",
    video_length=video_length,
)

# Load model
model = SAC.load("sac_panda_pickplace_final", env=eval_env)

# Rollout and record simultaneously
obs = eval_env.reset()

for t in range(video_length):
    print(f"Recording frame {t}", flush=True)

    action, _ = model.predict(obs, deterministic=True)
    obs, reward, done, info = eval_env.step(action)

    if done.any():
        obs = eval_env.reset()

eval_env.close()
