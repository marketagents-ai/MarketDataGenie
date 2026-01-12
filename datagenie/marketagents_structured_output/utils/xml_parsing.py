"""XML tool call parsing utilities for Hermes-style responses."""

import re
import json
import ast
from typing import Dict, Any, Optional, List


def parse_xml_tool_call(text: str) -> Optional[Dict[str, Any]]:
    """
    Parse a single <tool_call> XML tag from text.
    
    Args:
        text: Text containing <tool_call>...</tool_call>
        
    Returns:
        Parsed tool call dict with 'name' and 'arguments', or None
    """
    if not text:
        return None
    
    # Find tool_call content
    pattern = r"<tool_call>\s*(.*?)\s*</tool_call>"
    match = re.search(pattern, text, flags=re.DOTALL | re.IGNORECASE)
    
    if not match:
        return None
    
    raw_json = match.group(1).strip()
    
    # Try to parse as JSON
    try:
        obj = json.loads(raw_json)
        return obj
    except json.JSONDecodeError:
        pass
    
    # Try Python literal eval
    try:
        obj = ast.literal_eval(raw_json)
        return obj
    except Exception:
        pass
    
    # Try to fix common issues (single quotes -> double quotes)
    try:
        fixed = re.sub(r"'([^']*)':", r'"\1":', raw_json)
        fixed = re.sub(r":\s*'([^']*)'", r':"\1"', fixed)
        obj = json.loads(fixed)
        return obj
    except Exception:
        pass
    
    return None


def extract_tool_call_arguments(result: Dict[str, Any]) -> Optional[Dict[str, Any]]:
    """
    Extract tool call arguments from agent result.
    
    Handles both:
    1. Native tool calls (result has direct fields)
    2. XML tool calls (result has 'raw' field with XML)
    
    Args:
        result: Agent execution result
        
    Returns:
        Extracted arguments dict, or None
    """
    if not result:
        return None
    
    # Check for raw XML response
    if 'raw' in result and isinstance(result['raw'], str):
        raw_text = result['raw']
        
        # Parse XML tool call
        parsed = parse_xml_tool_call(raw_text)
        if parsed:
            # Return the arguments if present
            if 'arguments' in parsed:
                return parsed['arguments']
            # Or return the whole object minus 'name'
            if 'name' in parsed:
                args = {k: v for k, v in parsed.items() if k != 'name'}
                return args
            return parsed
    
    # Native tool call - result already has the fields
    # Filter out internal fields
    internal_fields = {'raw', 'error', 'tool_call_id', 'name'}
    args = {k: v for k, v in result.items() if k not in internal_fields}
    
    if args:
        return args
    
    return result


def parse_all_xml_tool_calls(text: str) -> List[Dict[str, Any]]:
    """
    Parse all <tool_call> tags from text.
    
    Args:
        text: Text potentially containing multiple tool calls
        
    Returns:
        List of parsed tool call dicts
    """
    if not text:
        return []
    
    pattern = r"<tool_call>\s*(.*?)\s*</tool_call>"
    matches = re.findall(pattern, text, flags=re.DOTALL | re.IGNORECASE)
    
    results = []
    for raw_json in matches:
        raw_json = raw_json.strip()
        
        # Try JSON
        try:
            obj = json.loads(raw_json)
            results.append(obj)
            continue
        except json.JSONDecodeError:
            pass
        
        # Try literal eval
        try:
            obj = ast.literal_eval(raw_json)
            results.append(obj)
            continue
        except Exception:
            pass
    
    return results
