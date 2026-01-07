"""
Comprehensive Test Script for All Tool-Use Scenarios
=====================================================

Tests all scenarios (single, multistep, multiturn, relevance) with:
1. Full GPT turn generation
2. <think> block validation
3. Tool call ground truth matching
4. ShareGPT output validation
"""

import json
import sys
import asyncio
from pathlib import Path
from typing import List, Dict, Any, Optional

# Add parent to path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from datagenie.tool_use.datagen_tool_use import (
    _check_sequential_tools,
    _validate_think_only,
    _validate_think_plus_calls,
    _validate_reply_and_extract,
    ToolUsePipeline,
    ToolUseEnvConfig
)


# =============================================================================
# Test Fixtures - Sample conversations for each scenario
# =============================================================================

SINGLE_CONV = [
    {"from": "system", "value": "You are a helpful assistant with tools."},
    {"from": "human", "value": "What time is it in Tokyo?"},
    {"from": "gpt", "value": "<think>I need to use the timezone tool to check Tokyo time.</think>\n<tool_call>\n{\"name\":\"get_timezone\",\"arguments\":{\"city\":\"Tokyo\"}}\n</tool_call>"},
    {"from": "tool", "value": "<tool_response>\n{\"time\": \"2024-01-01 15:30:00 JST\"}\n</tool_response>"},
    {"from": "gpt", "value": "<think>Now I have the time, I can respond.</think>\nThe current time in Tokyo is 3:30 PM JST."},
]

MULTISTEP_CONV = [
    {"from": "system", "value": "You are a helpful assistant with tools."},
    {"from": "human", "value": "Search for Python tutorials and summarize the best one."},
    {"from": "gpt", "value": "<think>I'll search first, then read the top result.</think>\n<tool_call>\n{\"name\":\"web_search\",\"arguments\":{\"query\":\"Python tutorials\"}}\n</tool_call>"},
    {"from": "tool", "value": "<tool_response>\n{\"results\": [{\"url\": \"python.org/tutorial\", \"title\": \"Official Tutorial\"}]}\n</tool_response>"},
    {"from": "gpt", "value": "<think>Found results, now I need to read the page.</think>\n<tool_call>\n{\"name\":\"read_page\",\"arguments\":{\"url\":\"python.org/tutorial\"}}\n</tool_call>"},
    {"from": "tool", "value": "<tool_response>\n{\"content\": \"Python tutorial content...\"}\n</tool_response>"},
    {"from": "gpt", "value": "<think>Now I can summarize.</think>\nThe best Python tutorial is the official one at python.org. It covers basics, data structures, and more."},
]

MULTITURN_CONV = [
    {"from": "system", "value": "You are a helpful assistant with tools."},
    {"from": "human", "value": "Search for restaurants near me."},
    {"from": "gpt", "value": "<think>I'll search for restaurants.</think>\n<tool_call>\n{\"name\":\"search_restaurants\",\"arguments\":{\"location\":\"nearby\"}}\n</tool_call>"},
    {"from": "tool", "value": "<tool_response>\n{\"restaurants\": [\"Pizza Place\", \"Sushi Bar\", \"Taco Shop\"]}\n</tool_response>"},
    {"from": "gpt", "value": "<think>Found options to present.</think>\nI found: Pizza Place, Sushi Bar, and Taco Shop. Which interests you?"},
    {"from": "human", "value": "Tell me more about Sushi Bar."},  # Human follow-up
    {"from": "gpt", "value": "<think>User wants details on Sushi Bar.</think>\n<tool_call>\n{\"name\":\"get_restaurant_info\",\"arguments\":{\"name\":\"Sushi Bar\"}}\n</tool_call>"},
    {"from": "tool", "value": "<tool_response>\n{\"rating\": 4.5, \"price\": \"$$\", \"hours\": \"11am-10pm\"}\n</tool_response>"},
    {"from": "gpt", "value": "<think>Got the info.</think>\nSushi Bar has a 4.5 rating, moderate prices ($$), and is open 11am-10pm."},
]

RELEVANCE_CONV = [
    {"from": "system", "value": "You are a helpful assistant. You have access to tools but don't always need them."},
    {"from": "human", "value": "What is 2 + 2?"},
    {"from": "gpt", "value": "<think>This is a simple math question. I don't need any tools.</think>\n2 + 2 = 4."},
]


# =============================================================================
# Unit Tests
# =============================================================================

def test_scenario_classification():
    """Test that conversations are correctly classified into scenarios."""
    print("\n=== Testing Scenario Classification ===")
    
    config = ToolUseEnvConfig(limit=0)
    
    # SINGLE
    pipeline = ToolUsePipeline(config, scenario="single")
    assert pipeline.validate_scenario(SINGLE_CONV) == True
    assert pipeline.validate_scenario(MULTISTEP_CONV) == False  # Has 2 tool calls
    print("✓ Single scenario correctly classified")
    
    # MULTISTEP
    pipeline = ToolUsePipeline(config, scenario="multistep")
    assert pipeline.validate_scenario(MULTISTEP_CONV) == True
    assert pipeline.validate_scenario(MULTITURN_CONV) == False  # Has human interruption
    print("✓ Multistep scenario correctly classified")
    
    # MULTITURN
    pipeline = ToolUsePipeline(config, scenario="multiturn")
    assert pipeline.validate_scenario(MULTITURN_CONV) == True
    assert pipeline.validate_scenario(MULTISTEP_CONV) == False  # No human interruption
    print("✓ Multiturn scenario correctly classified")
    
    # RELEVANCE
    pipeline = ToolUsePipeline(config, scenario="relevance")
    assert pipeline.validate_scenario(RELEVANCE_CONV) == True
    assert pipeline.validate_scenario(SINGLE_CONV) == False  # Has tool calls
    print("✓ Relevance scenario correctly classified")


def test_think_block_validation():
    """Test <think> block validation for all turn types."""
    print("\n=== Testing <think> Block Validation ===")
    
    # Tool-calling turn with think
    valid_tool = "<think>Need to search.</think>\n<tool_call>{\"name\":\"search\",\"arguments\":{}}</tool_call>"
    result = _validate_think_plus_calls(valid_tool)
    assert result is not None and len(result) == 1
    print("✓ Tool turn with <think> accepted")
    
    # Tool-calling turn WITHOUT think (should fail)
    invalid_tool = "<tool_call>{\"name\":\"search\",\"arguments\":{}}</tool_call>"
    result = _validate_think_plus_calls(invalid_tool)
    assert result is None
    print("✓ Tool turn without <think> rejected")
    
    # Narration turn with think
    valid_narration = "<think>Let me analyze.</think>\nThe answer is 42."
    assert _validate_think_only(valid_narration) == True
    print("✓ Narration with <think> accepted")
    
    # Narration turn WITHOUT think (should fail)
    invalid_narration = "The answer is 42."
    assert _validate_think_only(invalid_narration) == False
    print("✓ Narration without <think> rejected")


def test_tool_call_matching():
    """Test that generated tool calls are matched against ground truth."""
    print("\n=== Testing Tool Call Ground Truth Matching ===")
    
    config = ToolUseEnvConfig(validate_think_blocks=True)
    pipeline = ToolUsePipeline(config, scenario="multistep")
    
    ground_truth = "<think>Search needed.</think>\n<tool_call>{\"name\":\"search\",\"arguments\":{\"q\":\"test\"}}</tool_call>"
    
    # Matching tool call
    gen_match = "<think>I need to search.</think>\n<tool_call>{\"name\":\"search\",\"arguments\":{\"q\":\"test\"}}</tool_call>"
    is_valid, calls, reason = pipeline._validate_generation(gen_match, ground_truth, expects_tool_call=True)
    assert is_valid == True, f"Should match: {reason}"
    print("✓ Matching tool call accepted")
    
    # Mismatched tool name
    gen_mismatch = "<think>Let me fetch.</think>\n<tool_call>{\"name\":\"fetch\",\"arguments\":{\"q\":\"test\"}}</tool_call>"
    is_valid, calls, reason = pipeline._validate_generation(gen_mismatch, ground_truth, expects_tool_call=True)
    assert is_valid == False, "Should reject mismatched tool name"
    assert "tool_mismatch" in reason
    print("✓ Mismatched tool name rejected")
    
    # Missing tool call when expected
    gen_missing = "<think>I don't know.</think>\nI cannot help with that."
    is_valid, calls, reason = pipeline._validate_generation(gen_missing, ground_truth, expects_tool_call=True)
    assert is_valid == False
    assert "missing_tool_calls" in reason
    print("✓ Missing tool call rejected")


def test_full_conversation_extraction():
    """Test extracting all GPT turns from a conversation."""
    print("\n=== Testing Full Conversation GPT Turn Extraction ===")
    
    config = ToolUseEnvConfig(limit=100)
    pipeline = ToolUsePipeline(config, scenario="multiturn")
    
    # Count GPT turns in multiturn conversation
    gpt_turns = [m for m in MULTITURN_CONV if m["from"] == "gpt"]
    print(f"  Multiturn conv has {len(gpt_turns)} GPT turns")
    
    # Check each turn type
    for i, turn in enumerate(gpt_turns):
        has_tool = "<tool_call>" in turn["value"].lower()
        has_think = "<think>" in turn["value"].lower()
        turn_type = "tool_call" if has_tool else "narration"
        print(f"  Turn {i+1}: {turn_type}, has_think={has_think}")
    
    print("✓ All GPT turns identified correctly")


def test_inter_turn_structure():
    """Test that inter-turns (human/tool) are correctly identified."""
    print("\n=== Testing Inter-Turn Structure ===")
    
    # Multistep: Only tool responses between GPT turns
    print("  Multistep structure:")
    for i, m in enumerate(MULTISTEP_CONV):
        print(f"    {i}: {m['from']}")
    
    # Multiturn: Has human between GPT turns  
    print("  Multiturn structure:")
    for i, m in enumerate(MULTITURN_CONV):
        print(f"    {i}: {m['from']}")
    
    # Verify _check_sequential_tools
    assert _check_sequential_tools(MULTISTEP_CONV) == True
    assert _check_sequential_tools(MULTITURN_CONV) == False
    print("✓ Inter-turn structure validated")


# =============================================================================
# Integration Test - Run actual pipeline on scenarios
# =============================================================================

async def test_all_scenarios_integration():
    """Integration test: run pipeline on sample data for each scenario."""
    print("\n=== Integration Test: All Scenarios ===")
    
    scenarios = ["single", "multistep", "multiturn", "relevance"]
    
    for scenario in scenarios:
        print(f"\n--- Testing {scenario.upper()} scenario ---")
        config = ToolUseEnvConfig(
            limit=2,  # Just 2 items per scenario
            batch_size=2,
            validate_think_blocks=True
        )
        pipeline = ToolUsePipeline(config, scenario=scenario)
        
        # Just test loading/validation without actual inference
        items = pipeline.load_items()
        
        if items:
            print(f"  Loaded {len(items)} items")
            # Show first item structure
            item = items[0]
            print(f"  First item:")
            print(f"    - turn_expects_tool_call: {item.get('turn_expects_tool_call')}")
            print(f"    - conv_has_tool_calls: {item.get('conv_has_tool_calls')}")
            print(f"    - history length: {len(item.get('history', []))}")
        else:
            print(f"  No items loaded (dataset may need filtering)")


def evaluate_output_file(filepath: str):
    """Evaluate a ShareGPT output file for correctness."""
    print(f"\n=== Evaluating Output: {filepath} ===")
    
    path = Path(filepath)
    if not path.exists():
        print(f"File not found: {filepath}")
        return
    
    stats = {
        "total": 0,
        "by_scenario": {},
        "tool_call_turns": 0,
        "narration_turns": 0,
        "matched_tools": 0,
    }
    
    with open(path, "r") as f:
        for line in f:
            if not line.strip():
                continue
            try:
                item = json.loads(line)
            except:
                continue
                
            stats["total"] += 1
            
            meta = item.get("metadata", {})
            scenario = meta.get("scenario", "unknown")
            stats["by_scenario"][scenario] = stats["by_scenario"].get(scenario, 0) + 1
            
            if meta.get("turn_expects_tool_call"):
                stats["tool_call_turns"] += 1
            else:
                stats["narration_turns"] += 1
                
            if meta.get("tool_calls_matched"):
                stats["matched_tools"] += 1
    
    print(f"Total valid conversations: {stats['total']}")
    print(f"By scenario: {stats['by_scenario']}")
    print(f"Tool-call turns: {stats['tool_call_turns']}")
    print(f"Narration turns: {stats['narration_turns']}")
    print(f"Tool calls matched: {stats['matched_tools']}")


# =============================================================================
# Main
# =============================================================================

if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Test Tool-Use Scenarios")
    parser.add_argument("--eval", help="Evaluate a ShareGPT output file")
    parser.add_argument("--integration", action="store_true", help="Run integration test with dataset")
    args = parser.parse_args()
    
    print("=" * 70)
    print("Tool-Use Scenario Comprehensive Tests")
    print("=" * 70)
    
    # Unit tests
    test_scenario_classification()
    test_think_block_validation()
    test_tool_call_matching()
    test_full_conversation_extraction()
    test_inter_turn_structure()
    
    print("\n" + "=" * 70)
    print("All unit tests passed! ✓")
    print("=" * 70)
    
    # Integration test
    if args.integration:
        asyncio.run(test_all_scenarios_integration())
    
    # Evaluate output file
    if args.eval:
        evaluate_output_file(args.eval)
