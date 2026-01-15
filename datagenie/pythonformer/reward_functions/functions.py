"""
Reward functions for trajectory evaluation.

Rewards are computed AFTER trajectory generation for:
- Filtering high-quality trajectories
- Post-analysis and dataset curation
- Future RL training preparation

Users can define custom reward functions by subclassing RewardFunction.
"""

from typing import Optional, Dict, Any
from abc import ABC, abstractmethod
from dataclasses import dataclass


@dataclass
class TrajectoryMetrics:
    """Metrics extracted from a trajectory for reward calculation."""
    answer_correct: Optional[bool]  # True/False/None (unknown)
    num_turns: int
    num_code_blocks: int
    num_errors: int
    max_turns: int  # Config limit
    intermediate_rewards: list  # Rewards from execution (errors, etc.)
    success: bool  # Whether trajectory completed


class RewardFunction(ABC):
    """Base class for reward functions."""
    
    @abstractmethod
    def compute(self, metrics: TrajectoryMetrics) -> Dict[str, Any]:
        """
        Compute reward based on trajectory metrics.
        
        Args:
            metrics: Trajectory metrics
            
        Returns:
            Dict with:
                - total_reward: float
                - final_step_reward: float (reward for final outcome)
                - metadata: dict (optional, for analysis)
        """
        pass
    
    @property
    def name(self) -> str:
        """Name of the reward function."""
        return self.__class__.__name__


class SimpleCorrectness(RewardFunction):
    """
    Simple binary reward: +1 for correct, -0.5 for incorrect, 0 for unknown.
    
    This is the simplest reward function - just cares about correctness.
    Good for basic filtering of correct vs incorrect trajectories.
    """
    
    def compute(self, metrics: TrajectoryMetrics) -> Dict[str, Any]:
        if metrics.answer_correct is True:
            final_reward = 1.0
        elif metrics.answer_correct is False:
            final_reward = -0.5
        else:
            final_reward = 0.0
        
        total = sum(metrics.intermediate_rewards) + final_reward
        
        return {
            "total_reward": total,
            "final_step_reward": final_reward,
            "metadata": {
                "reward_function": self.name,
            }
        }


class EfficiencyAware(RewardFunction):
    """
    Reward that considers both correctness and efficiency.
    
    Rewards:
    - Correct answers get base reward + efficiency bonus
    - Fewer turns = higher bonus
    - Errors reduce the reward
    - Incorrect answers get penalty scaled by wasted effort
    
    Good for preferring concise, clean solutions.
    """
    
    def __init__(
        self,
        base_correct_reward: float = 1.0,
        max_efficiency_bonus: float = 0.5,
        error_penalty_per_error: float = 0.1,
        base_incorrect_penalty: float = -0.5,
        max_effort_penalty: float = -0.3,
    ):
        self.base_correct_reward = base_correct_reward
        self.max_efficiency_bonus = max_efficiency_bonus
        self.error_penalty_per_error = error_penalty_per_error
        self.base_incorrect_penalty = base_incorrect_penalty
        self.max_effort_penalty = max_effort_penalty
    
    def compute(self, metrics: TrajectoryMetrics) -> Dict[str, Any]:
        if metrics.answer_correct is True:
            # Base success reward
            reward = self.base_correct_reward
            
            # Efficiency bonus: fewer code blocks = higher bonus
            if metrics.max_turns > 0:
                efficiency_ratio = 1.0 - (metrics.num_code_blocks / metrics.max_turns)
                efficiency_bonus = self.max_efficiency_bonus * max(0.0, efficiency_ratio)
                reward += efficiency_bonus
            else:
                efficiency_bonus = 0.0
            
            # Quality penalty: errors reduce reward
            quality_penalty = self.error_penalty_per_error * metrics.num_errors
            reward -= quality_penalty
            
            final_reward = reward
            metadata = {
                "reward_function": self.name,
                "base_reward": self.base_correct_reward,
                "efficiency_bonus": efficiency_bonus if metrics.max_turns > 0 else 0.0,
                "quality_penalty": quality_penalty,
            }
            
        elif metrics.answer_correct is False:
            # Base penalty
            reward = self.base_incorrect_penalty
            
            # Effort penalty: more turns wasted = bigger penalty
            if metrics.max_turns > 0:
                effort_ratio = metrics.num_code_blocks / metrics.max_turns
                effort_penalty = self.max_effort_penalty * effort_ratio
                reward += effort_penalty  # Adding negative value
            else:
                effort_penalty = 0.0
            
            final_reward = reward
            metadata = {
                "reward_function": self.name,
                "base_penalty": self.base_incorrect_penalty,
                "effort_penalty": effort_penalty if metrics.max_turns > 0 else 0.0,
            }
        else:
            # Unknown correctness
            final_reward = 0.0
            metadata = {
                "reward_function": self.name,
                "note": "unknown_correctness",
            }
        
        total = sum(metrics.intermediate_rewards) + final_reward
        
        return {
            "total_reward": total,
            "final_step_reward": final_reward,
            "metadata": metadata,
        }


class NormalizedReward(RewardFunction):
    """
    Normalized reward that scales by trajectory length.
    
    Returns reward per turn and reward per code block for fair comparison
    across trajectories of different lengths.
    
    Good for comparing efficiency across different problem difficulties.
    """
    
    def __init__(self, base_correct: float = 1.0, base_incorrect: float = -0.5):
        self.base_correct = base_correct
        self.base_incorrect = base_incorrect
    
    def compute(self, metrics: TrajectoryMetrics) -> Dict[str, Any]:
        if metrics.answer_correct is True:
            final_reward = self.base_correct
        elif metrics.answer_correct is False:
            final_reward = self.base_incorrect
        else:
            final_reward = 0.0
        
        total = sum(metrics.intermediate_rewards) + final_reward
        
        # Normalized metrics
        reward_per_turn = total / max(1, metrics.num_turns)
        reward_per_code = total / max(1, metrics.num_code_blocks)
        
        return {
            "total_reward": total,
            "final_step_reward": final_reward,
            "metadata": {
                "reward_function": self.name,
                "reward_per_turn": reward_per_turn,
                "reward_per_code_block": reward_per_code,
            }
        }


# Registry of available reward functions
REWARD_FUNCTIONS = {
    "simple": SimpleCorrectness,
    "efficiency": EfficiencyAware,
    "normalized": NormalizedReward,
}


def get_reward_function(name: str, **kwargs) -> RewardFunction:
    """
    Get a reward function by name.
    
    Args:
        name: Name of reward function ("simple", "efficiency", "normalized")
        **kwargs: Arguments to pass to reward function constructor
        
    Returns:
        RewardFunction instance
        
    Example:
        reward_fn = get_reward_function("efficiency", base_correct_reward=2.0)
    """
    if name not in REWARD_FUNCTIONS:
        raise ValueError(f"Unknown reward function: {name}. Available: {list(REWARD_FUNCTIONS.keys())}")
    
    return REWARD_FUNCTIONS[name](**kwargs)
