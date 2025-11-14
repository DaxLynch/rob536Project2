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

Note: This installs `stable-baselines3[extra]` which includes additional dependencies. The `rl-zoo3` package is installed separately and provides the training framework and hyperparameters.

## Training

There are two ways to train agents:

### Method 1: Using train.py Script

The provided `train.py` script is set up to train a DDPG agent:

```bash
source venv/bin/activate
python train.py
```

This trains a DDPG agent on the `PandaReach-v3` environment for 30,000 timesteps and saves the model as `panda_reach_ddpg`.

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
