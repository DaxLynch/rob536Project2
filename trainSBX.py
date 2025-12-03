"""
Train a reinforcement learning agent using panda-gym and stable-baselines3.

This script demonstrates training a TQC agent on the PandaReach-v3 environment
with full logging, evaluation, and checkpointing capabilities.
"""
import os
from datetime import datetime
import gymnasium as gym
import panda_gym
from sbx import TQC  # Using SBX (Stable-Baselines-Jax) for faster training
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3 import HerReplayBuffer

from stable_baselines3.common.callbacks import (
    CheckpointCallback,
    EvalCallback,
    CallbackList,
)
from stable_baselines3.common.monitor import Monitor

# Import to register native friction environments
import friction_env  # Registers FrictionPickAndPlace-v1, ConstantFrictionPickAndPlace-v1

# ============================================================================
# Configuration
# ============================================================================
# Environment options:
#   "PandaPickAndPlace-v3"           - Standard (no friction in obs)
#   "FrictionPickAndPlace-v1"        - Random friction each episode (native, fast)
#   "ConstantFrictionPickAndPlace-v1" - Fixed friction=0.5 (for pre-training)
#   "BlindFrictionPickAndPlace-v1"    - Blind friction pick and place (no friction in obs)
ENV_NAME = "FrictionPickAndPlaceJoints-v1"
ALGO_NAME = "tqc_sbx"  # Using SBX (Jax-based) implementation
N_ENVS = 24  # Number of parallel environments
TOTAL_TIMESTEPS = 3_000_000
EVAL_FREQ = 10_000  # Evaluate every N steps (per environment)
SAVE_FREQ = 50_000  # Save checkpoint every N steps
N_EVAL_EPISODES = 10  # Number of episodes for evaluation

# Create timestamped log directory
exp_name = "VaryingFrictionJoints"
log_dir = None
timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
if exp_name is not None:
    log_dir = f"./logs/{ALGO_NAME}/{ENV_NAME}_{exp_name}_{timestamp}"
else:
    log_dir = f"./logs/{ALGO_NAME}/{ENV_NAME}_{timestamp}"
print(log_dir)
tensorboard_log = f"./logs/{ALGO_NAME}_tensorboard"
os.makedirs(log_dir, exist_ok=True)

print("=" * 80)
print(f"Training Configuration:")
print(f"  Environment: {ENV_NAME}")
print(f"  Algorithm: {ALGO_NAME.upper()}")
print(f"  Parallel Environments: {N_ENVS}")
print(f"  Total Timesteps: {TOTAL_TIMESTEPS:,}")
print(f"  Log Directory: {log_dir}")
print(f"  Tensorboard Log: {tensorboard_log}")
print("=" * 80)

# ============================================================================
# Create Environments
# ============================================================================
# Training environments with Monitor wrapper for episode statistics
env = make_vec_env(
    ENV_NAME,
    n_envs=N_ENVS,
    monitor_dir=log_dir,  # Automatically wraps with Monitor
)

# Separate evaluation environment
eval_env = make_vec_env(
    ENV_NAME,
    n_envs=1,
    monitor_dir=f"{log_dir}/eval",
)

# ============================================================================
# Create Callbacks
# ============================================================================
# Checkpoint callback - saves model periodically
checkpoint_callback = CheckpointCallback(
    save_freq=SAVE_FREQ // N_ENVS,  # Divide by n_envs for actual step count
    save_path=f"{log_dir}/checkpoints",
    name_prefix=f"{ALGO_NAME}_model",
    save_replay_buffer=True,
    save_vecnormalize=True,
)

# Evaluation callback - evaluates and saves best model
eval_callback = EvalCallback(
    eval_env,
    best_model_save_path=f"{log_dir}/best_model",
    log_path=f"{log_dir}/eval",
    eval_freq=EVAL_FREQ // N_ENVS,  # Divide by n_envs
    n_eval_episodes=N_EVAL_EPISODES,
    deterministic=True,
    render=False,
    verbose=1,
)

# Combine callbacks
callbacks = CallbackList([checkpoint_callback, eval_callback])

# ============================================================================
# Create and Train Model
# ============================================================================
# Create the TQC model with tensorboard logging
# And update the model creation section (around lines 100-106):
model = TQC(
    policy="MultiInputPolicy",
    env=env,
    learning_rate=0.001,
    buffer_size=1_000_000,
    batch_size=2048*2,
    gamma=0.95,
    tau=0.05,
    train_freq=16,
    gradient_steps=16,
    replay_buffer_class=HerReplayBuffer,
    replay_buffer_kwargs=dict(goal_selection_strategy='future', n_sampled_goal=4),
    policy_kwargs=dict(net_arch=[512, 512, 512], n_critics=2),
    verbose=1,
    tensorboard_log=tensorboard_log,
    device="cuda",
)

# Train the agent
print("\n" + "=" * 80)
print("Starting training...")
print("=" * 80)
print(f"\nTo monitor training progress, run:")
print(f"  tensorboard --logdir {tensorboard_log}\n")

model.learn(
    total_timesteps=TOTAL_TIMESTEPS,
    callback=callbacks,
    tb_log_name=f"{exp_name}_{timestamp}",
    progress_bar=True,
)

print("\n" + "=" * 80)
print("Training completed!")
print("=" * 80)

# ============================================================================
# Save Final Model
# ============================================================================
final_model_path = f"{log_dir}/final_model"
model.save(final_model_path)
print(f"\nFinal model saved to: {final_model_path}")
print(f"Best model saved to: {log_dir}/best_model")
print(f"Checkpoints saved to: {log_dir}/checkpoints")
print(f"Training logs saved to: {log_dir}")
print(f"\nTo visualize training:")
print(f"  tensorboard --logdir {tensorboard_log}")
print("=" * 80)

