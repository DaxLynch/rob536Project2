# Training Guide - Enhanced RL Training with Full Logging

This guide explains how to use the enhanced training scripts with comprehensive logging, evaluation, and visualization capabilities.

## Which Training Script to Use?

We provide **two training scripts** with identical features:

### `train.py` (Recommended - Full Logging)
- Uses **sb3_contrib** (PyTorch-based)
- ✅ **Full tensorboard metrics** including actor_loss, critic_loss, ent_coef
- ✅ Better compatibility across systems
- ✅ Standard for RL research
- Use this for research, debugging, and when you need all metrics

### `trainSBX.py` (Experimental - Faster)
- Uses **sbx** (Jax-based)
- ✅ Potentially faster training
- ⚠️ Limited tensorboard logging (no loss metrics)
- ⚠️ May have compatibility issues (requires Jax)
- Use this only for production training when speed is critical

**For the rest of this guide, examples use `train.py`, but all features work the same with `trainSBX.py`.**

## Quick Start

### 1. Start Training

```bash
python train.py          # Recommended: Full logging
# OR
python trainSBX.py      # Experimental: Faster but limited logging
```

This will:
- Train a TQC agent on PandaReach-v3 with 16 parallel environments
- Save logs to `logs/tqc/PandaReach-v3_TIMESTAMP/` (or `logs/tqc_sbx/` for SBX version)
- Create checkpoints every 50,000 steps
- Evaluate and save best model every 10,000 steps
- Log all metrics to Tensorboard

### 2. Monitor Training in Real-Time (3 Options)

#### Option A: Tensorboard (Recommended - Best Visualization)
```bash
# In a separate terminal
# For train.py:
tensorboard --logdir ./logs/tqc_tensorboard

# For trainSBX.py:
tensorboard --logdir ./logs/tqc_sbx_tensorboard

# Then open http://localhost:6006 in your browser
```

**What you'll see in Tensorboard:**

**With `train.py` (sb3_contrib - Full Metrics):**
- `rollout/ep_rew_mean`: Average episode reward
- `rollout/ep_len_mean`: Average episode length  
- `train/actor_loss`: Actor network loss ✅
- `train/critic_loss`: Critic network loss ✅
- `train/ent_coef`: Entropy coefficient ✅
- `eval/mean_reward`: Evaluation performance (best model indicator)
- `eval/mean_ep_length`: Evaluation episode length
- `time/fps`: Frames per second

**With `trainSBX.py` (sbx - Limited Metrics):**
- `rollout/ep_rew_mean`: Average episode reward
- `rollout/ep_len_mean`: Average episode length  
- `eval/mean_reward`: Evaluation performance
- `eval/mean_ep_length`: Evaluation episode length
- `time/fps`: Frames per second
- ⚠️ **Note:** Loss metrics (actor_loss, critic_loss) are typically NOT logged with SBX

#### Option B: Live Progress Monitor
```bash
# In a separate terminal, monitor the latest training run
python monitor_training.py --log-dir logs/tqc/PandaReach-v3_TIMESTAMP
```

This displays:
- Episodes completed
- Total timesteps
- Recent performance (last 100 episodes)
- Latest episode rewards
- Training rate

#### Option C: Built-in Progress Bar
The training script shows a progress bar in the terminal by default.

### 3. Analyze Results After Training

```bash
# Plot training curves
python plot_results.py --log-dir logs/tqc/PandaReach-v3_TIMESTAMP

# Use a larger smoothing window for noisy data
python plot_results.py --log-dir logs/tqc/PandaReach-v3_TIMESTAMP --window 100
```

This generates:
- Episode rewards over time (raw and smoothed)
- Episode lengths over time
- Cumulative rewards
- Statistical summary
- Saves plot as `training_plots.png` in the log directory

### 4. Test Your Trained Model

```bash
# Test the best model (from evaluation)
python enjoy.py --model logs/tqc/PandaReach-v3_TIMESTAMP/best_model/best_model.zip

# Test a specific checkpoint
python enjoy.py --model logs/tqc/PandaReach-v3_TIMESTAMP/checkpoints/tqc_model_100000_steps.zip

# Test the final model
python enjoy.py --model logs/tqc/PandaReach-v3_TIMESTAMP/final_model.zip
```

## Output Directory Structure

**For `train.py` (sb3_contrib):**
```
logs/
├── tqc_tensorboard/              # Tensorboard logs for all runs
│   └── PandaReach-v3_TIMESTAMP/  # Run-specific tensorboard data
└── tqc/                          # Algorithm-specific logs
    └── PandaReach-v3_TIMESTAMP/  # Training run directory
        ├── 0.monitor.csv         # Training episode data
        ├── best_model/           # Best model based on evaluation
        │   ├── best_model.zip
        │   └── evaluations.npz
        ├── checkpoints/          # Periodic checkpoints
        │   ├── tqc_model_50000_steps.zip
        │   ├── tqc_model_100000_steps.zip
        │   └── ...
        ├── eval/                 # Evaluation logs
        │   ├── 0.monitor.csv
        │   └── evaluations.npz
        ├── final_model.zip       # Final trained model
        └── training_plots.png    # Generated plots (after running plot_results.py)
```

**For `trainSBX.py` (sbx):**
```
logs/
├── tqc_sbx_tensorboard/          # Tensorboard logs for all runs
│   └── PandaReach-v3_TIMESTAMP/  # Run-specific tensorboard data
└── tqc_sbx/                      # Algorithm-specific logs
    └── PandaReach-v3_TIMESTAMP/  # Training run directory
        └── (same structure as above)
```

## When to Use Which Script?

### Use `train.py` (sb3_contrib) if:
- 🔬 You're doing research and need all metrics
- 📊 You need loss metrics for debugging or analysis
- 🤝 You're sharing code with collaborators (better compatibility)
- 📝 You're writing a paper (standard implementation)
- ❓ You're unsure which to use (default choice)

### Use `trainSBX.py` (sbx) if:
- ⚡ Speed is critical and you don't need loss metrics
- 🖥️ You have Jax/GPU setup optimized for Jax
- 🎯 You only care about final performance, not training dynamics
- ⚠️ You're willing to debug potential compatibility issues

**Bottom line:** Start with `train.py`. Only switch to `trainSBX.py` if you specifically need the speed and don't need loss metrics.

## Configuration Options

Edit the configuration section in `train.py` (or `trainSBX.py`):

```python
# Environment
ENV_NAME = "PandaReach-v3"        # Gym environment name
ALGO_NAME = "tqc"                 # Algorithm name for logging

# Training
N_ENVS = 16                       # Parallel environments (higher = faster)
TOTAL_TIMESTEPS = 1_000_000       # Total training steps

# Evaluation & Checkpointing
EVAL_FREQ = 10_000                # Evaluate every N steps (per env)
SAVE_FREQ = 50_000                # Save checkpoint every N steps
N_EVAL_EPISODES = 10              # Episodes per evaluation
```

### Performance Tips

**Parallel Environments (`N_ENVS`):**
- More environments = faster training (more samples per update)
- Rule of thumb: Use as many as your CPU cores (typically 4-16)
- Too many may cause diminishing returns or memory issues

**Evaluation Frequency (`EVAL_FREQ`):**
- More frequent = better tracking of best model, but slower training
- Less frequent = faster training, but may miss peak performance
- Recommended: 5,000 - 20,000 steps

**Checkpoint Frequency (`SAVE_FREQ`):**
- More frequent = more disk space, but better recovery options
- Less frequent = less disk space, but fewer recovery points
- Recommended: 50,000 - 100,000 steps

## Monitor File Format

The `monitor.csv` files contain episode-level data:

| Column | Description |
|--------|-------------|
| `r` | Episode reward (return) |
| `l` | Episode length (timesteps) |
| `t` | Wall-clock time since training start (seconds) |

These files are automatically created by the Monitor wrapper and can be loaded with:

```python
from stable_baselines3.common.monitor import load_results
df = load_results("logs/tqc/PandaReach-v3_TIMESTAMP")
```

## Comparing Multiple Runs

### Using Tensorboard

```bash
# Load multiple runs at once
tensorboard --logdir ./logs/tqc_tensorboard

# Tensorboard will show all runs in the same plot
# Use the checkboxes to select which runs to display
```

### Using Custom Analysis

```python
import pandas as pd
from stable_baselines3.common.monitor import load_results

# Load multiple runs
run1 = load_results("logs/tqc/PandaReach-v3_TIMESTAMP1")
run2 = load_results("logs/tqc/PandaReach-v3_TIMESTAMP2")

# Compare mean rewards
print(f"Run 1 mean reward: {run1['r'].mean():.4f}")
print(f"Run 2 mean reward: {run2['r'].mean():.4f}")
```

## Troubleshooting

### Training is slow
- Increase `N_ENVS` (more parallel environments)
- Check CPU usage (should be near 100% during training)
- Ensure you're not running other heavy processes

### Out of memory errors
- Decrease `N_ENVS`
- Decrease `SAVE_FREQ` to save less frequent checkpoints
- Close other applications

### Tensorboard not showing data
- Make sure training has started and generated some data
- Check that the `--logdir` path is correct
- Try refreshing the browser
- Wait a few seconds for data to appear

### Monitor files not found
- Ensure training has completed at least one episode
- Check the log directory path is correct
- The monitor files are created per environment (look for `0.monitor.csv`, `1.monitor.csv`, etc.)

## Advanced: Custom Callbacks

You can add your own callbacks to the training script. Example:

```python
from stable_baselines3.common.callbacks import BaseCallback

class CustomLoggingCallback(BaseCallback):
    def __init__(self, verbose=0):
        super().__init__(verbose)
    
    def _on_step(self):
        # This is called after each training step
        # Log custom metrics to tensorboard
        self.logger.record("custom/my_metric", some_value)
        return True  # Continue training

# Add to callbacks list in train.py
callbacks = CallbackList([
    checkpoint_callback,
    eval_callback,
    CustomLoggingCallback(),
])
```

## References

- [Stable-Baselines3 Documentation](https://stable-baselines3.readthedocs.io/)
- [Stable-Baselines3 Callbacks](https://stable-baselines3.readthedocs.io/en/master/guide/callbacks.html)
- [Tensorboard Guide](https://www.tensorflow.org/tensorboard/get_started)
- [RL Baselines3 Zoo](https://github.com/DLR-RM/rl-baselines3-zoo)

