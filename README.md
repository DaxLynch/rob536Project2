# Panda-Gym with Stable-Baselines3 and RL Baselines3 Zoo

This project uses panda-gym environments with stable-baselines3 and RL Baselines3 Zoo for reinforcement learning.

## Overview

**stable-baselines3** provides the core reinforcement learning algorithms (DDPG, TD3, SAC, PPO, etc.).

**RL Baselines3 Zoo (rl-zoo3)** provides:
- Training framework with scripts (`python -m rl_zoo3.train`)
- Visualization tools (`python -m rl_zoo3.enjoy`)
- Pre-tuned hyperparameters for various environments
- Experiment management and logging

See the [RL Baselines3 Zoo GitHub repository](https://github.com/DLR-RM/rl-baselines3-zoo) for more details.

## Installation

1. Create and activate a virtual environment (if not already done):

```bash
python3 -m venv venv
source venv/bin/activate  # On Linux/Mac
# or
venv\Scripts\activate  # On Windows
```

2. Install the required packages:

```bash
pip install -r requirements.txt
```

**Note:** This installs:
- `stable-baselines3[extra]` and `sb3-contrib`: Core RL algorithms (required for `train.py`)
- `sbx-rl`: Jax-based implementations (optional, only needed for `trainSBX.py`)
- `rl-zoo3`: Training framework and hyperparameters
- `matplotlib`, `pandas`, `tensorboard`: Visualization and logging tools

If your system has issues with `sbx-rl`, you can skip it and just use `train.py`.

## Training

📘 **For detailed training instructions, monitoring, and visualization, see [TRAINING_GUIDE.md](TRAINING_GUIDE.md)**

There are two ways to train agents:

### Method 1: Using Training Scripts (Recommended for Custom Training)

We provide **two training scripts** with identical features but different implementations:

#### Option A: `train.py` (Recommended - Full Logging)
Uses **sb3_contrib** (PyTorch-based) with comprehensive tensorboard logging:

```bash
source venv/bin/activate
python train.py
```

**Pros:**
- ✅ Full tensorboard metrics (actor_loss, critic_loss, ent_coef)
- ✅ Better compatibility across systems
- ✅ Standard implementation used in RL research
- ✅ More mature and tested

**Cons:**
- Slightly slower than Jax-based version

#### Option B: `trainSBX.py` (Experimental - Faster)
Uses **sbx** (Jax-based) for potentially faster training:

```bash
source venv/bin/activate
python trainSBX.py
```

**Pros:**
- ✅ Faster training on some systems
- ✅ Hardware acceleration with Jax

**Cons:**
- ⚠️ Limited tensorboard logging (no loss metrics)
- ⚠️ May have compatibility issues on some systems
- ⚠️ Requires additional dependencies (sbx-rl)

**Recommendation:** Use `train.py` for research and full metrics. Use `trainSBX.py` only if you need maximum speed and don't require loss metrics.

**Features:**
- **Parallel Environments**: Uses 16 parallel environments for faster training
- **Tensorboard Logging**: Real-time monitoring of losses, rewards, and other metrics
- **Periodic Checkpoints**: Saves model every 50,000 steps
- **Continuous Evaluation**: Evaluates model every 10,000 steps and saves best model
- **Progress Bar**: Visual progress indicator during training
- **Monitor Logs**: Episode statistics saved to CSV files

**Output Structure:**
```
logs/
└── tqc/
    └── PandaReach-v3_YYYYMMDD_HHMMSS/
        ├── 0.monitor.csv           # Training episode data
        ├── best_model/             # Best model based on evaluation
        ├── checkpoints/            # Periodic model checkpoints
        ├── eval/                   # Evaluation logs
        └── final_model.zip         # Final trained model
```

**Monitor Training in Real-Time:**
```bash
# In a separate terminal, run tensorboard
tensorboard --logdir ./logs/tqc_tensorboard
# Then open http://localhost:6006 in your browser
```

**Visualize Training Results:**
```bash
# After training, plot the results
python plot_results.py --log-dir logs/tqc/PandaReach-v3_YYYYMMDD_HHMMSS
```

**Configuration:**
Edit the configuration section at the top of `train.py`:
```python
N_ENVS = 16              # Number of parallel environments
TOTAL_TIMESTEPS = 1_000_000  # Total training steps
EVAL_FREQ = 10_000       # Evaluate every N steps
SAVE_FREQ = 50_000       # Save checkpoint every N steps
```

You can modify `train.py` to use different algorithms or environments:

```python
from stable_baselines3 import DDPG, TD3, SAC
# or
from sb3_contrib import TQC

# Change environment
env = gym.make("PandaReach-v3")  # or PandaPickAndPlace-v3, etc.

# Change algorithm
model = DDPG(policy="MultiInputPolicy", env=env)
```

### Method 2: Using RL Baselines3 Zoo

You can also train using RL Baselines3 Zoo's training script, which uses pre-tuned hyperparameters:

```bash
source venv/bin/activate
python -m rl_zoo3.train --algo tqc --env PandaReach-v3
```

Example commands:
- `python -m rl_zoo3.train --algo ddpg --env PandaReach-v3`
- `python -m rl_zoo3.train --algo tqc --env PandaReach-v3`
- `python -m rl_zoo3.train --algo sac --env PandaPickAndPlace-v3`

This method automatically saves models in the `logs/` directory with proper folder structure.

### Performance Notes

- **DDPG**: Trains quickly and has shown good performance on PandaReach-v3
- **TQC**: A better version of SAC (Twin Quantile Critic), but may require more tuning for optimal performance

### Speeding Up Training with Multiple CPU Cores

You can significantly speed up training by using **vectorized environments** to train on multiple CPU cores in parallel. There are two main approaches:

#### Method 1: Using train.py with Vectorized Environments

The updated `train.py` script now supports parallel training. Simply adjust the `N_ENVS` variable:

```python
N_ENVS = 4  # Use 4 parallel environments (adjust based on your CPU cores)
```

**Options:**
- **SubprocVecEnv**: True multiprocessing - uses multiple CPU cores (recommended for CPU-bound environments)
- **DummyVecEnv**: Single-process vectorization - faster than single env but no multiprocessing

**Example:**
```bash
python train.py  # Will use 4 parallel environments by default
```

**Performance Tips:**
- Set `N_ENVS` to the number of CPU cores you have (e.g., 4, 8, 16)
- With `N_ENVS` parallel environments, each training step collects `N_ENVS` experiences simultaneously
- This can provide **near-linear speedup** (e.g., 4x faster with 4 environments)
- Note: Total timesteps are shared across all environments, so training completes faster

#### Method 2: Using RL Baselines3 Zoo with --n-jobs

RL Baselines3 Zoo automatically uses vectorized environments when you specify `--n-jobs`:

```bash
# Train with 4 parallel environments
python -m rl_zoo3.train --algo tqc --env PandaReach-v3 --n-jobs 4

# Train with 8 parallel environments
python -m rl_zoo3.train --algo ddpg --env PandaReach-v3 --n-jobs 8
```

**Benefits:**
- Automatic environment vectorization
- Pre-tuned hyperparameters (for supported environments)
- Better logging and experiment management

**References:**
- [Stable-Baselines3 Vectorized Environments](https://stable-baselines3.readthedocs.io/en/master/guide/vec_envs.html)
- [RL Baselines3 Zoo Training Guide](https://rl-baselines3-zoo.readthedocs.io/en/master/guide/train.html)

## Viewing the Trained Agent

### Method 1: Using enjoy.py Script

The provided `enjoy.py` script is set up to visualize a DDPG model:

```bash
source venv/bin/activate
python enjoy.py
```

This loads the trained model and displays the agent interacting with the environment in a graphical window. The script includes examples for both DDPG and TQC models.

**For DDPG:**
```python
from stable_baselines3 import DDPG
model = DDPG.load("panda_reach_ddpg")
```

**For TQC:**
```python
from sb3_contrib import TQC
model = TQC.load("path/to/tqc/model.zip", env=env)
```

### Method 2: Using RL Baselines3 Zoo's Enjoy Script

You can also use RL Baselines3 Zoo's enjoy script:

```bash
source venv/bin/activate
python -m rl_zoo3.enjoy --algo ddpg --env PandaReach-v3 --folder . 
```

Or for TQC:
```bash
python -m rl_zoo3.enjoy --algo tqc --env PandaReach-v3 --folder logs 
```

## Available Environments

You can use various panda-gym environments:
- `PandaReach-v3`
- `PandaPickAndPlace-v3`
- `PandaPush-v3`
- `PandaSlide-v3`
- `PandaFlip-v3`
- `PandaStack-v3`

## Available Algorithms

### Main Algorithms (from stable_baselines3)

```python
from stable_baselines3 import A2C, DDPG, DQN, PPO, SAC, TD3
```

- **DDPG** (Deep Deterministic Policy Gradient)
- **TD3** (Twin Delayed DDPG)
- **SAC** (Soft Actor-Critic)
- **PPO** (Proximal Policy Optimization)
- **A2C** (Advantage Actor-Critic)
- **DQN** (Deep Q-Network)

### Other Algorithms (from sb3_contrib)

```python
from sb3_contrib import TQC,
```

- **TQC** (Twin Quantile Critic) - Enhanced version of SAC


## Hyperparameters

### Pre-tuned Hyperparameters

**Important Note:** RL Baselines3 Zoo only provides tuned hyperparameters for TQC, and these are configured by default for Panda v1 environments (e.g., `PandaPush-v1`, `PandaSlide-v1`). If you attempt to use the RL Zoo training script with a v3 environment (like `PandaPush-v3`), it will **not work out of the box** because no v3 entries exist in the default YAML files. The script will fail with an error like "No hyperparameters for PandaPush-v3".

To use the tuned hyperparameters for a v3 Panda environment, **you must manually edit the `tqc.yml` file** and change every instance of `-v1` to `-v3` for the respective environments you want to use. For example, you need to edit lines like:

```yaml
PandaPush-v1: her-defaults
```
to
```yaml
PandaPush-v3: her-defaults
```

Do this for all Panda environments you wish to use with v3 (e.g., Push, Slide, PickAndPlace, Stack). Without this change, RL Zoo will refuse to use these environments—even though the algorithm itself is compatible. **You will not be able to use the pre-tuned settings unless you make these manual edits.**

Also keep in mind:
- There are **no pre-tuned hyperparameters for Panda-v3 for any algorithm except TQC.**
- The other YAML files (`ppo.yml`, `ddpg.yml`, etc.) do **not** contain tuned parameters for any of the panda environments. Only `tqc.yml` includes Panda hyperparameters.
- You will have to tune other algorithms yourself if you want to use them with Panda v3.

### Quick steps to use the pre-tuned TQC hyperparameters for v3
1. **Open and edit** the TQC hyperparameters file:
   ```bash
   nano venv/lib/python3.12/site-packages/rl_zoo3/hyperparams/tqc.yml
   ```
2. **Replace** all occurrences of `-v1` with `-v3` for every Panda environment you want to use.

3. Now you can run Zoo training as normal, e.g.
   ```bash
   python -m rl_zoo3.train --algo tqc --env PandaPush-v3 --env-kwargs render_mode:human
   ```

If you want to see the (TQC-only) available settings for Panda environments, run:
```bash
cat venv/lib/python3.12/site-packages/rl_zoo3/hyperparams/tqc.yml | grep -A 20 "Panda"
```

You can also use this script to list algorithms:
```bash
source venv/bin/activate
python list_algorithms.py
```

**Summary:**  
- You must edit the TQC YAML file to use v3 panda environments with RL Zoo.  
- Only TQC is supported with pre-tuned Panda hyperparameters.  
- For other algorithms, hyperparameters are not available and would need to be created manually or tuned from scratch.
```

## Troubleshooting

### Import Errors

- **TQC**: Must be imported from `sb3_contrib`, not `stable_baselines3`
  ```python
  from sb3_contrib import TQC  # Correct
  from stable_baselines3 import TQC  # Wrong - will raise ImportError
  ```

- **Other algorithms**: Check `list_algorithms.py` to see which package contains each algorithm

### Hyperparameter Issues

- If hyperparameters don't work for v3 environments, modify the YAML files as described above
- For custom environments, you may need to tune hyperparameters from scratch

## References

- [stable-baselines3 documentation](https://stable-baselines3.readthedocs.io/)
- [RL Baselines3 Zoo GitHub](https://github.com/DLR-RM/rl-baselines3-zoo)
- [panda-gym documentation](https://panda-gym.readthedocs.io/)

pip install --upgrade "jax[cuda12]" 
pip install --upgrade tfp-nightly

This let me run sbx, gets 15 percent performance, plus seems to actually learn lol