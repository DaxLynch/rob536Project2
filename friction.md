# Varying Friction for Panda-Gym Pick and Place

## Overview

This implementation adds domain randomization via varying object friction to the PandaPickAndPlace-v3 environment. The friction value is randomized each episode (0.1-2.0 range) and exposed to the agent as an observation.

## Files Changed

| File | Change |
|------|--------|
| `friction_wrapper.py` | **NEW** - Gymnasium wrapper for friction randomization |
| `evaluate_friction.py` | **NEW** - Evaluation script for testing models |
| `trainSBX.py` | **MODIFIED** - Added `USE_FRICTION` toggle (2 lines) |

## Usage

### Training with Varying Friction

In `trainSBX.py`, change line 24:

```python
USE_FRICTION = True  # Change from False to True
```

Then run training as normal:
```bash
python trainSBX.py
```

### Evaluating a Model

Test a standard-trained model on varying friction:

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
- **Observation:** Friction added as `obs['friction']` (shape: [1])

## Wrapper Classes

- `FrictionWrapper`: Random friction each reset (for training)
- `FixedFrictionWrapper`: Fixed friction value (for controlled evaluation)

