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


This displays:
- Episodes completed
- Total timesteps
- Recent performance (last 100 episodes)
- Latest episode rewards
- Training rate

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
