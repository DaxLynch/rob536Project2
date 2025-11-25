# Varying Friction for Panda-Gym Pick and Place

## Overview

This implementation adds domain randomization via varying object friction to the PandaPickAndPlace environment. The friction value is randomized each episode (0.1-2.0 range) and exposed to the agent as an observation.

## Native Environments (Recommended - Fast)

Native environments avoid wrapper overhead for JAX/SBX training:

| Environment | Description |
|-------------|-------------|
| `FrictionPickAndPlace-v1` | Random friction each episode (0.1-2.0) |
| `ConstantFrictionPickAndPlace-v1` | Fixed friction=0.5 (for pre-training) |

### Usage

In `trainSBX.py`, set the environment:
```python
ENV_NAME = "FrictionPickAndPlace-v1"  # Random friction (native, fast)
```

### Observation Space

- Original: 19 floats (robot + object state)
- With friction: 20 floats (original + friction as last element)

## Files

| File | Description |
|------|-------------|
| `friction_env.py` | **Native environments** - FrictionPickAndPlace-v1, ConstantFrictionPickAndPlace-v1 |
| `evaluate_friction.py` | Evaluation script for testing models |

## Evaluating a Model

```bash
# Random friction each episode
python evaluate_friction.py --model path/to/model.zip --episodes 100

# Test at specific friction values
python evaluate_friction.py --model path/to/model.zip --mode fixed

# Both modes with visualization
python evaluate_friction.py --model path/to/model.zip --mode both --render
```

## Technical Details

- **Friction Range:** 0.1 (slippery) to 2.0 (grippy)
- **Default Friction:** 0.5 (panda-gym default)
- **PyBullet API:** Uses `sim.set_lateral_friction('object', -1, value)`
- **Speed:** Native env ~1000 steps/sec vs wrapper ~50 steps/sec

