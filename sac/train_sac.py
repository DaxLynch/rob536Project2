import gymnasium as gym
import panda_gym
from stable_baselines3 import SAC
from stable_baselines3.her import HerReplayBuffer
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.callbacks import CheckpointCallback,\
   BaseCallback
from stable_baselines3.common.vec_env import VecNormalize

import time
from collections import deque
from tqdm import tqdm
import numpy as np

# =========================================================
#   Success Rate + Best Model + Progress Bar Callback
# =========================================================
class TrainingCallback(BaseCallback):
    """
    - Track success rate
    - Save best model
    - Progress bar
    """
    def __init__(self, total_timesteps, update_freq=1000, verbose=0):
        super().__init__(verbose)
        self.total_timesteps = total_timesteps
        self.update_freq = update_freq

        # Track ETA
        self.start_time = None
        #self.speeds = deque(maxlen=50)
        self.sps_window = deque(maxlen=50)
        self.pbar = None
        self.last_update = 0

        # Track success rate
        self.episode_successes = []
        self.episode_counts = 0
        self.best_success_rate = -np.inf

    def _on_training_start(self) -> None:
        self.start_time = time.time()
        self.pbar = tqdm(total=self.total_timesteps, desc="Training", ncols=100)

    def _on_step(self) -> bool:
        # Increment by 1 per environment step
        self.pbar.update(1)
        steps = self.model.num_timesteps

        # Success rate tracking
        infos = self.locals["infos"]
        for info in infos:
            if "is_success" in info:
                # success flag appears once per episode end
                self.episode_successes.append(info["is_success"])
                self.episode_counts += 1

        if steps - self.last_update >= self.update_freq:
            self.last_update = steps

            # compute ETA periodically
            elapsed = time.time() - self.start_time
            sps = steps / elapsed
            #self.speeds.append(sps)
            self.sps_window.append(sps)
            avg_sps = sum(self.sps_window) / len(self.sps_window)

            remaining = self.total_timesteps - steps
            eta_seconds = remaining / max(avg_sps, 1e-9)

            hrs = int(eta_seconds // 3600)
            mins = int((eta_seconds % 3600) // 60)
            secs = int(eta_seconds % 60)

            if self.episode_counts > 0:
                success_rate = np.mean(self.episode_successes)
            else:
                success_rate = 0.0

            # Save best model
            if success_rate > self.best_success_rate:
                self.best_success_rate = success_rate
                self.model.save("best_model_sac_her_panda")

            # Update progess bar
            self.pbar.set_postfix({
                "succ": f"{success_rate:.2f}",
                "best": f"{self.best_success_rate:.2f}",
                "sps": f"{avg_sps:.1f}",
                "ETA": f"{hrs}h {mins}m {secs}s"
            })

        return True

    def _on_training_end(self) -> None:
        self.pbar.close()

# ----------------------
# 1. Create Env
# ----------------------

max_epsiode_steps = gym.make("PandaPickAndPlace-v3").spec.max_episode_steps
env = make_vec_env(
    "PandaPickAndPlace-v3",
    n_envs=1,
    env_kwargs=dict(
        reward_type="dense", # TODO: Modify reward function
    )
)

env = VecNormalize(
    env,
    norm_obs=True,
    norm_reward=True,
    clip_obs=10.0,
)

# Optional checkpointing
checkpoint_callback = CheckpointCallback(
    save_freq=50_000,
    save_path="./checkpoints/",
    name_prefix="sac_panda_pickplace"
)

# ----------------------
# 2. Create SAC model
# ----------------------

model = SAC(
    "MultiInputPolicy",
    env,
    replay_buffer_class=HerReplayBuffer,
    replay_buffer_kwargs=dict(
        n_sampled_goal=4,
        goal_selection_strategy="future",
        copy_info_dict=False
    ),
    buffer_size=1_000_000,
    batch_size=256,
    learning_starts=10_000,
    gradient_steps=1,
    train_freq=1,
    target_update_interval=1,
    verbose=1,
)

# ----------------------
# 3. Train
# ----------------------

TOTAL_STEPS = 1_000_000
#TOTAL_STEPS = 10_000
model.learn(
    total_timesteps=TOTAL_STEPS,
    callback=[checkpoint_callback, TrainingCallback(TOTAL_STEPS, update_freq = 200)]
)

model.save("sac_panda_pickplace_final")
model.save("sac_panda_pickplace_final_ac_only", include=["actor","critic"])
env.save("vecnormalize_stats.pkl")
