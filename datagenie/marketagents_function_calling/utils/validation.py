"""Validation utilities for tool calls and messages."""

import json
from typing import List, Dict, Any, Tuple


def validate_tool_calls(
    generated_calls: List[Dict[str, Any]],
    expected_answers: List[Dict[str, Any]],
    require_matching_args: bool = True
) -> Tuple[bool, str]:
    """
    Validate generated tool calls against expected answers.
    
    Args:
        generated_calls: Tool calls from model output
        expected_answers: Expected tool calls from dataset
        require_matching_args: Whether to require exact argument matching
        
    Returns:
        Tuple of (is_valid, reason)
    """
    if not generated_calls:
        return False, "No tool calls generated"
    
    for expected in expected_answers:
        found = False
        expected_name = expected.get('name', '')
        
        for call in generated_calls:
            call_name = call.get('function', {}).get('name', call.get('name', ''))
            
            if call_name == expected_name:
                if require_matching_args:
                    try:
                        gen_args = call.get('function', {}).get('arguments', '{}')
                        if isinstance(gen_args, str):
                            gen_args = json.loads(gen_args)
                        exp_args = expected.get('arguments', {})
                        if gen_args == exp_args:
                            found = True
                            break
                    except json.JSONDecodeError:
                        continue
                else:
                    found = True
                    break
        
        if not found:
            return False, f"Missing expected tool call: {expected_name}"
    
    return True, "Valid"


def validate_message(message: Dict[str, Any], msg_type: str = None) -> Tuple[bool, str]:
    """
    Validate message format.
    
    Args:
        message: Message dict to validate
        msg_type: Expected message type ('tool', 'user', 'assistant', etc.)
        
    Returns:
        Tuple of (is_valid, reason)
    """
    if not isinstance(message, dict):
        return False, "Message must be a dictionary"
    
    if 'role' not in message:
        return False, "Message missing 'role' field"
    
    if msg_type == 'tool':
        if message.get('role') != 'tool':
            return False, f"Expected tool message, got {message.get('role')}"
        if not message.get('content'):
            return False, "Tool message missing content"
        if not message.get('name'):
            return False, "Tool message missing name"
    
    if msg_type == 'user':
        if message.get('role') != 'user':
            return False, f"Expected user message, got {message.get('role')}"
        if not message.get('content'):
            return False, "User message missing content"
    
    if msg_type == 'assistant':
        if message.get('role') != 'assistant':
            return False, f"Expected assistant message, got {message.get('role')}"
    
    return True, "Valid"


def validate_tool_results(tool_results: Dict[str, Any]) -> Tuple[bool, str]:
    """
    Validate tool results structure.
    
    Args:
        tool_results: Dict with 'messages' key containing tool result messages
        
    Returns:
        Tuple of (is_valid, reason)
    """
    messages = tool_results.get('messages', [])
    
    if not messages:
        return False, "No tool result messages"
    
    # Group messages by function name
    messages_by_name = {}
    
    for message in messages:
        if message.get('role') != 'tool':
            return False, f"Invalid message role: {message.get('role')}"
        
        name = message.get('name')
        content = message.get('content')
        
        if not name:
            return False, "Tool message missing name"
        
        if not content:
            return False, f"Missing content for tool '{name}'"
        
        if name not in messages_by_name:
            messages_by_name[name] = []
        messages_by_name[name].append(content)
    
    # Check that messages with same name have consistent structure
    for name, contents in messages_by_name.items():
        if len(contents) > 1:
            if isinstance(contents[0], dict):
                reference_keys = set(contents[0].keys())
                for content in contents[1:]:
                    if isinstance(content, dict):
                        if set(content.keys()) != reference_keys:
                            return False, f"Inconsistent keys for tool '{name}'"
    
    return True, "Valid"
