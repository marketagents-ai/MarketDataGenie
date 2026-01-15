#!/usr/bin/env python3
"""
Compute or recompute rewards for existing trajectory files.

This script allows you to:
1. Apply different reward functions to existing trajectories
2. Filter trajectories by reward thresholds
3. Analyze reward distributions
4. Create filtered datasets

Usage:
    # Recompute rewards with a different function
    python -m datagenie.pythonformer.compute_rewards \
        --input outputs/pythonformer_oolong/traces_*.jsonl \
        --reward-function efficiency \
        --output outputs/filtered_traces.jsonl
    
    # Filter by reward threshold
    python -m datagenie.pythonformer.compute_rewards \
        --input outputs/pythonformer_oolong/traces_*.jsonl \
        --reward-function simple \
        --min-reward 0.9 \
        --output outputs/high_quality.jsonl
    
    # Just analyze without saving
    python -m datagenie.pythonformer.compute_rewards \
        --input outputs/pythonformer_oolong/traces_*.jsonl \
        --reward-function efficiency \
        --analyze-only
"""

import json
import argparse
from pathlib import Path
from typing import List, Dict, Any, Optional
from collections import defaultdict

from datagenie.pythonformer.reward_functions import (
    RewardFunction, TrajectoryMetrics, get_reward_function, REWARD_FUNCTIONS
)


def load_trajectories(input_path: str) -> List[Dict[str, Any]]:
    """Load trajectories from JSONL file."""
    trajectories = []
    with open(input_path, 'r') as f:
        for line in f:
            if line.strip():
                trajectories.append(json.loads(line))
    return trajectories


def compute_reward_for_trajectory(
    trajectory: Dict[str, Any],
    reward_fn: RewardFunction,
    max_turns: int = 30,
) -> Dict[str, Any]:
    """
    Compute reward for a single trajectory.
    
    Args:
        trajectory: Trajectory dict from traces file
        reward_fn: Reward function to use
        max_turns: Max turns config (for efficiency calculations)
        
    Returns:
        Updated trajectory dict with new rewards
    """
    # Extract metrics from trajectory
    metrics = TrajectoryMetrics(
        answer_correct=trajectory.get("answer_correct"),
        num_turns=trajectory.get("num_turns", 0),
        num_code_blocks=trajectory.get("num_code_blocks", 0),
        num_errors=trajectory.get("num_errors", 0),
        max_turns=max_turns,
        intermediate_rewards=trajectory.get("step_rewards", []),
        success=trajectory.get("success", False),
    )
    
    # Compute reward
    reward_result = reward_fn.compute(metrics)
    
    # Update trajectory
    updated = trajectory.copy()
    updated["total_reward"] = reward_result["total_reward"]
    updated["reward_metadata"] = reward_result.get("metadata", {})
    
    # Add final step reward to step_rewards if not already there
    step_rewards = list(trajectory.get("step_rewards", []))
    if len(step_rewards) == 0 or step_rewards[-1] != reward_result["final_step_reward"]:
        step_rewards.append(reward_result["final_step_reward"])
    updated["step_rewards"] = step_rewards
    
    return updated


def analyze_rewards(trajectories: List[Dict[str, Any]]) -> None:
    """Print reward distribution analysis."""
    if not trajectories:
        print("No trajectories to analyze")
        return
    
    rewards = [t["total_reward"] for t in trajectories]
    correct = [t for t in trajectories if t.get("answer_correct") is True]
    incorrect = [t for t in trajectories if t.get("answer_correct") is False]
    unknown = [t for t in trajectories if t.get("answer_correct") is None]
    
    print("\n" + "="*60)
    print("REWARD ANALYSIS")
    print("="*60)
    
    print(f"\nTotal trajectories: {len(trajectories)}")
    print(f"  Correct:   {len(correct)} ({len(correct)/len(trajectories)*100:.1f}%)")
    print(f"  Incorrect: {len(incorrect)} ({len(incorrect)/len(trajectories)*100:.1f}%)")
    print(f"  Unknown:   {len(unknown)} ({len(unknown)/len(trajectories)*100:.1f}%)")
    
    print(f"\nReward Statistics:")
    print(f"  Mean:   {sum(rewards)/len(rewards):.3f}")
    print(f"  Median: {sorted(rewards)[len(rewards)//2]:.3f}")
    print(f"  Min:    {min(rewards):.3f}")
    print(f"  Max:    {max(rewards):.3f}")
    
    if correct:
        correct_rewards = [t["total_reward"] for t in correct]
        print(f"\nCorrect Answer Rewards:")
        print(f"  Mean:   {sum(correct_rewards)/len(correct_rewards):.3f}")
        print(f"  Median: {sorted(correct_rewards)[len(correct_rewards)//2]:.3f}")
        print(f"  Min:    {min(correct_rewards):.3f}")
        print(f"  Max:    {max(correct_rewards):.3f}")
    
    if incorrect:
        incorrect_rewards = [t["total_reward"] for t in incorrect]
        print(f"\nIncorrect Answer Rewards:")
        print(f"  Mean:   {sum(incorrect_rewards)/len(incorrect_rewards):.3f}")
        print(f"  Median: {sorted(incorrect_rewards)[len(incorrect_rewards)//2]:.3f}")
        print(f"  Min:    {min(incorrect_rewards):.3f}")
        print(f"  Max:    {max(incorrect_rewards):.3f}")
    
    # Reward distribution
    print(f"\nReward Distribution:")
    bins = [
        ("< -0.5", lambda r: r < -0.5),
        ("[-0.5, 0)", lambda r: -0.5 <= r < 0),
        ("[0, 0.5)", lambda r: 0 <= r < 0.5),
        ("[0.5, 1.0)", lambda r: 0.5 <= r < 1.0),
        (">= 1.0", lambda r: r >= 1.0),
    ]
    for label, condition in bins:
        count = sum(1 for r in rewards if condition(r))
        pct = count / len(rewards) * 100
        print(f"  {label:12s}: {count:4d} ({pct:5.1f}%)")
    
    # Efficiency analysis (if metadata available)
    if trajectories[0].get("reward_metadata"):
        print(f"\nEfficiency Metrics:")
        avg_turns = sum(t["num_turns"] for t in trajectories) / len(trajectories)
        avg_code = sum(t["num_code_blocks"] for t in trajectories) / len(trajectories)
        avg_errors = sum(t["num_errors"] for t in trajectories) / len(trajectories)
        print(f"  Avg turns:       {avg_turns:.1f}")
        print(f"  Avg code blocks: {avg_code:.1f}")
        print(f"  Avg errors:      {avg_errors:.1f}")
        
        if correct:
            avg_turns_correct = sum(t["num_turns"] for t in correct) / len(correct)
            avg_code_correct = sum(t["num_code_blocks"] for t in correct) / len(correct)
            print(f"\n  Correct answers:")
            print(f"    Avg turns:       {avg_turns_correct:.1f}")
            print(f"    Avg code blocks: {avg_code_correct:.1f}")
    
    print("="*60 + "\n")


def main():
    parser = argparse.ArgumentParser(
        description="Compute or recompute rewards for trajectory files",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__
    )
    
    parser.add_argument(
        "--input", "-i",
        required=True,
        help="Input JSONL file with trajectories"
    )
    
    parser.add_argument(
        "--reward-function", "-r",
        default="simple",
        choices=list(REWARD_FUNCTIONS.keys()),
        help="Reward function to use (default: simple)"
    )
    
    parser.add_argument(
        "--output", "-o",
        help="Output JSONL file (if not specified, prints to stdout)"
    )
    
    parser.add_argument(
        "--min-reward",
        type=float,
        help="Minimum reward threshold for filtering"
    )
    
    parser.add_argument(
        "--max-reward",
        type=float,
        help="Maximum reward threshold for filtering"
    )
    
    parser.add_argument(
        "--correct-only",
        action="store_true",
        help="Only include trajectories with correct answers"
    )
    
    parser.add_argument(
        "--max-turns",
        type=int,
        default=30,
        help="Max turns config for efficiency calculations (default: 30)"
    )
    
    parser.add_argument(
        "--analyze-only",
        action="store_true",
        help="Only analyze rewards, don't save output"
    )
    
    args = parser.parse_args()
    
    # Load trajectories
    print(f"Loading trajectories from {args.input}...")
    trajectories = load_trajectories(args.input)
    print(f"Loaded {len(trajectories)} trajectories")
    
    # Get reward function
    reward_fn = get_reward_function(args.reward_function)
    print(f"Using reward function: {reward_fn.name}")
    
    # Recompute rewards
    print("Computing rewards...")
    updated_trajectories = []
    for traj in trajectories:
        updated = compute_reward_for_trajectory(traj, reward_fn, args.max_turns)
        updated_trajectories.append(updated)
    
    # Filter if requested
    filtered = updated_trajectories
    if args.min_reward is not None:
        filtered = [t for t in filtered if t["total_reward"] >= args.min_reward]
        print(f"Filtered to {len(filtered)} trajectories with reward >= {args.min_reward}")
    
    if args.max_reward is not None:
        filtered = [t for t in filtered if t["total_reward"] <= args.max_reward]
        print(f"Filtered to {len(filtered)} trajectories with reward <= {args.max_reward}")
    
    if args.correct_only:
        filtered = [t for t in filtered if t.get("answer_correct") is True]
        print(f"Filtered to {len(filtered)} trajectories with correct answers")
    
    # Analyze
    analyze_rewards(filtered)
    
    # Save output
    if not args.analyze_only:
        if args.output:
            output_path = Path(args.output)
            output_path.parent.mkdir(parents=True, exist_ok=True)
            
            with open(output_path, 'w') as f:
                for traj in filtered:
                    f.write(json.dumps(traj, ensure_ascii=False) + "\n")
            
            print(f"Saved {len(filtered)} trajectories to {output_path}")
        else:
            # Print to stdout
            for traj in filtered:
                print(json.dumps(traj, ensure_ascii=False))


if __name__ == "__main__":
    main()
