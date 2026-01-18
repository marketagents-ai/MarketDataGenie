"""
Reward Functions Module

This module provides pluggable reward functions for evaluating trajectory quality.
Rewards are computed AFTER trajectory generation for filtering and analysis.

Usage:
    from datagenie.pythonformer.reward_functions import get_reward_function
    
    reward_fn = get_reward_function("simple")
    reward_result = reward_fn.compute(metrics)

Available Functions:
    - simple: Binary correctness (+1 correct, -0.5 incorrect)
    - efficiency: Considers correctness, efficiency, and code quality
    - normalized: Normalized rewards for fair comparison across lengths

To disable rewards during generation, set `enable_rewards: false` in config.
"""

from datagenie.pythonformer.reward_functions.functions import (
    RewardFunction,
    TrajectoryMetrics,
    SimpleCorrectness,
    EfficiencyAware,
    NormalizedReward,
    REWARD_FUNCTIONS,
    get_reward_function,
)

__all__ = [
    "RewardFunction",
    "TrajectoryMetrics",
    "SimpleCorrectness",
    "EfficiencyAware",
    "NormalizedReward",
    "REWARD_FUNCTIONS",
    "get_reward_function",
]
