# Reward Functions

This module provides pluggable reward functions for evaluating trajectory quality in the Pythonformer pipeline.

## Overview

Rewards are computed **AFTER** trajectory generation (not during) for:
- Filtering high-quality trajectories
- Post-analysis and dataset curation
- Future RL training preparation

## Files

- `functions.py` - Reward function definitions and base classes
- `compute_rewards.py` - Standalone script for recomputing rewards on existing trajectories
- `__init__.py` - Module exports

## Usage

### In Pipeline (Automatic)

Configure in your YAML config:

```yaml
dataset:
  reward_function: "simple"  # Options: "simple", "efficiency", "normalized"
  enable_rewards: true       # Set to false to disable reward computation
```

### Standalone Script

Recompute rewards on existing trajectory files:

```bash
# Analyze rewards
python -m datagenie.pythonformer.reward_functions.compute_rewards \
    --input outputs/pythonformer_oolong/traces_*.jsonl \
    --reward-function efficiency \
    --analyze-only

# Filter high-quality trajectories
python -m datagenie.pythonformer.reward_functions.compute_rewards \
    --input outputs/pythonformer_oolong/traces_*.jsonl \
    --reward-function simple \
    --min-reward 0.9 \
    --output outputs/high_quality.jsonl

# Filter correct answers only
python -m datagenie.pythonformer.reward_functions.compute_rewards \
    --input outputs/pythonformer_oolong/traces_*.jsonl \
    --reward-function efficiency \
    --correct-only \
    --output outputs/correct_only.jsonl
```

### Programmatic Usage

```python
from datagenie.pythonformer.reward_functions import (
    get_reward_function, TrajectoryMetrics
)

# Get a reward function
reward_fn = get_reward_function("efficiency")

# Create metrics from trajectory
metrics = TrajectoryMetrics(
    answer_correct=True,
    num_turns=8,
    num_code_blocks=3,
    num_errors=1,
    max_turns=30,
    intermediate_rewards=[0.0, -0.05, 0.0],
    success=True,
)

# Compute reward
result = reward_fn.compute(metrics)
print(f"Total reward: {result['total_reward']}")
print(f"Metadata: {result['metadata']}")
```

## Available Reward Functions

### 1. Simple Correctness (`simple`)

Binary reward based only on answer correctness.

**Rewards:**
- Correct: `+1.0`
- Incorrect: `-0.5`
- Unknown: `0.0`

**Use case:** Basic filtering of correct vs incorrect trajectories.

**Example:**
```python
reward_fn = get_reward_function("simple")
```

### 2. Efficiency Aware (`efficiency`)

Considers correctness, efficiency (fewer turns), and code quality (fewer errors).

**Rewards for correct answers:**
- Base: `+1.0`
- Efficiency bonus: up to `+0.5` (fewer code blocks = higher bonus)
- Error penalty: `-0.1` per error

**Rewards for incorrect answers:**
- Base: `-0.5`
- Effort penalty: up to `-0.3` (more turns wasted = bigger penalty)

**Use case:** Preferring concise, clean solutions.

**Example:**
```python
reward_fn = get_reward_function(
    "efficiency",
    base_correct_reward=2.0,  # Custom parameters
    max_efficiency_bonus=1.0,
)
```

### 3. Normalized Reward (`normalized`)

Simple reward normalized by trajectory length for fair comparison.

**Rewards:**
- Correct: `+1.0`
- Incorrect: `-0.5`
- Metadata includes `reward_per_turn` and `reward_per_code_block`

**Use case:** Comparing efficiency across different problem difficulties.

**Example:**
```python
reward_fn = get_reward_function("normalized")
```

## Creating Custom Reward Functions

Subclass `RewardFunction` and implement the `compute()` method:

```python
from datagenie.pythonformer.reward_functions import RewardFunction, TrajectoryMetrics

class MyCustomReward(RewardFunction):
    def compute(self, metrics: TrajectoryMetrics) -> dict:
        # Your custom logic here
        if metrics.answer_correct:
            reward = 10.0 - metrics.num_errors
        else:
            reward = -5.0
        
        return {
            "total_reward": reward,
            "final_step_reward": reward,
            "metadata": {
                "reward_function": self.name,
                "custom_metric": metrics.num_turns / metrics.num_code_blocks,
            }
        }

# Register it
from datagenie.pythonformer.reward_functions.functions import REWARD_FUNCTIONS
REWARD_FUNCTIONS["custom"] = MyCustomReward
```

Then use it in config:
```yaml
dataset:
  reward_function: "custom"
```

## Disabling Rewards

To disable reward computation during generation (useful for testing):

```yaml
dataset:
  enable_rewards: false
```

Or programmatically:
```python
config.dataset.enable_rewards = False
```

When disabled:
- `total_reward` will be `0.0`
- `step_rewards` will only contain intermediate execution rewards
- `reward_metadata` will be `None`

You can still compute rewards later using the standalone script.

## Output Format

Rewards are saved in trajectory files:

```json
{
  "task_id": "...",
  "answer_correct": true,
  "total_reward": 1.45,
  "step_rewards": [0.0, -0.05, 0.0, 1.5],
  "num_errors": 1,
  "reward_metadata": {
    "reward_function": "EfficiencyAware",
    "base_reward": 1.0,
    "efficiency_bonus": 0.5,
    "quality_penalty": 0.05
  }
}
```

## Development

When developing new reward functions:

1. Add your function to `functions.py`
2. Register it in `REWARD_FUNCTIONS` dict
3. Test it with the standalone script
4. Document it in this README

### Testing

```bash
# Test on a small sample
python -m datagenie.pythonformer.reward_functions.compute_rewards \
    --input outputs/sample.jsonl \
    --reward-function your_function \
    --analyze-only
```

## Future Work

- [ ] Add reward functions for specific task types (math, coding, reasoning)
- [ ] Support multi-objective rewards (correctness + efficiency + readability)
- [ ] Add reward shaping for RL training
- [ ] Support custom reward functions from external files
- [ ] Add reward visualization tools
