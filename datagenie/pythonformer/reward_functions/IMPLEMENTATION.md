# Reward System Implementation

## Overview

The reward system has been refactored into a separate, pluggable module for better organization and flexibility.

## Structure

```
datagenie/pythonformer/reward_functions/
├── __init__.py              # Module exports
├── functions.py             # Reward function definitions
├── compute_rewards.py       # Standalone script for recomputing rewards
├── README.md                # User documentation
└── IMPLEMENTATION.md        # This file
```

## Key Features

### 1. Pluggable Architecture

Reward functions are defined as classes inheriting from `RewardFunction`:

```python
class RewardFunction(ABC):
    @abstractmethod
    def compute(self, metrics: TrajectoryMetrics) -> Dict[str, Any]:
        pass
```

### 2. Toggle to Enable/Disable

Rewards can be disabled during generation:

```yaml
dataset:
  enable_rewards: false  # No reward computation
```

When disabled:
- `total_reward` = sum of intermediate rewards only (errors, etc.)
- `reward_metadata` = None
- Rewards can be computed later using the standalone script

### 3. Standalone Reward Computation

The `compute_rewards.py` script allows recomputing rewards on existing trajectories:

```bash
python -m datagenie.pythonformer.reward_functions.compute_rewards \
    --input outputs/traces.jsonl \
    --reward-function efficiency \
    --output outputs/filtered.jsonl
```

### 4. Multiple Reward Functions

Three built-in reward functions:

1. **SimpleCorrectness**: Binary reward (+1 correct, -0.5 incorrect)
2. **EfficiencyAware**: Considers correctness, efficiency, and code quality
3. **NormalizedReward**: Normalized by trajectory length

### 5. Custom Reward Functions

Users can define custom reward functions:

```python
from datagenie.pythonformer.reward_functions import RewardFunction

class MyReward(RewardFunction):
    def compute(self, metrics):
        # Custom logic
        return {
            "total_reward": ...,
            "final_step_reward": ...,
            "metadata": {...}
        }
```

## Integration with Pipeline

### Pipeline Changes

1. **Import**: Changed from `rewards.py` to `reward_functions/`
2. **Toggle**: Added `enable_rewards` check in `__init__`
3. **Conditional**: Reward computation only runs if enabled
4. **Metadata**: `reward_metadata` is None when disabled

### Config Changes

Added two new fields to `DatasetConfig`:

```python
enable_rewards: bool = True
reward_function: str = "simple"
```

## Usage Examples

### Enable Rewards (Default)

```yaml
dataset:
  enable_rewards: true
  reward_function: "efficiency"
```

### Disable Rewards

```yaml
dataset:
  enable_rewards: false
```

### Compute Later

```bash
# Generate without rewards
python -m datagenie.pythonformer.run \
    --config configs/oolong_no_rewards.yaml \
    --limit 10

# Compute rewards later
python -m datagenie.pythonformer.reward_functions.compute_rewards \
    --input outputs/traces.jsonl \
    --reward-function efficiency \
    --analyze-only
```

## Benefits

1. **Separation of Concerns**: Reward logic is separate from pipeline logic
2. **Flexibility**: Easy to swap reward functions or disable entirely
3. **Testability**: Reward functions can be tested independently
4. **Extensibility**: Users can add custom reward functions
5. **Post-Processing**: Rewards can be computed/recomputed after generation

## Development Workflow

When developing new reward functions:

1. Add function to `functions.py`
2. Register in `REWARD_FUNCTIONS` dict
3. Test with standalone script
4. Document in `README.md`
5. Use in pipeline via config

## Future Enhancements

- [ ] Support loading custom reward functions from external files
- [ ] Add reward visualization tools
- [ ] Support multi-objective rewards
- [ ] Add reward shaping for RL training
- [ ] Create reward function templates for common patterns
