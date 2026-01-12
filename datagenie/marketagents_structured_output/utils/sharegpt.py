"""ShareGPT format conversion for structured output datasets."""

import json
import re
from typing import List, Dict, Any, Optional

from datagenie.marketagents_structured_output.utils.reasoning import (
    strip_reasoning_reminder,
    REASONING_REMINDER,
)


def to_sharegpt_format(
    messages: List[Dict[str, Any]],
    json_schema: Dict[str, Any],
    metadata: Optional[Dict[str, Any]] = None,
    include_reasoning_instructions: bool = False
) -> Dict[str, Any]:
    """
    Convert conversation messages to ShareGPT format for structured output.
    
    Args:
        messages: List of message dicts with role/content
        json_schema: The JSON schema used for structured output
        metadata: Optional metadata (task, category, etc.)
        include_reasoning_instructions: Whether reasoning instructions are in system prompt
        
    Returns:
        ShareGPT formatted dict with conversations, schema, and metadata
    """
    conversations = []
    
    for msg in messages:
        role = msg.get("role", "")
        content = msg.get("content", "")
        
        # Map roles to ShareGPT format
        if role == "system":
            sharegpt_role = "system"
        elif role == "user":
            sharegpt_role = "human"
            # Strip reasoning reminder from user messages
            content = strip_reasoning_reminder(content)
        elif role == "assistant":
            sharegpt_role = "gpt"
        else:
            continue
        
        # Format assistant messages with structured output
        if role == "assistant":
            # Check if content has <think> blocks - preserve as-is for reasoning
            if "<think>" in content and "</think>" in content:
                # Content has reasoning - extract think block and tool call
                # Format: <think>...</think><tool_call>{"name": "...", "arguments": {...}}</tool_call>
                think_match = re.search(r'<think>(.*?)</think>', content, re.DOTALL)
                tool_match = re.search(r'<tool_call>\s*(.*?)\s*</tool_call>', content, re.DOTALL)
                
                if think_match and tool_match:
                    think_content = think_match.group(1).strip()
                    tool_json_str = tool_match.group(1).strip()
                    
                    # Parse tool call to extract just the arguments (actual data)
                    try:
                        tool_obj = json.loads(tool_json_str)
                        if isinstance(tool_obj, dict) and 'arguments' in tool_obj:
                            # Format: <think>reasoning</think>\n{actual_data}
                            actual_data = json.dumps(tool_obj['arguments'], indent=2)
                            content = f"<think>{think_content}</think>\n{actual_data}"
                        elif isinstance(tool_obj, dict) and 'name' not in tool_obj:
                            # Tool call is just the data itself
                            actual_data = json.dumps(tool_obj, indent=2)
                            content = f"<think>{think_content}</think>\n{actual_data}"
                    except json.JSONDecodeError:
                        # Keep as-is if parsing fails
                        pass
                # If only think block found but no tool_call, keep as-is
            else:
                # No reasoning - check if content is JSON and format it nicely
                try:
                    json_obj = json.loads(content) if isinstance(content, str) else content
                    if isinstance(json_obj, dict):
                        # Just format the JSON nicely, no markdown code block
                        content = json.dumps(json_obj, indent=2)
                except (json.JSONDecodeError, TypeError):
                    # Not JSON, keep as-is
                    pass
        
        conversations.append({
            "from": sharegpt_role,
            "value": content
        })
    
    # Build result
    result = {
        "conversations": conversations,
        "json_schema": json.dumps(json_schema) if isinstance(json_schema, dict) else json_schema,
    }
    
    # Add metadata
    if metadata:
        result.update(metadata)
    
    return result


def format_system_prompt_with_schema(
    base_prompt: str,
    json_schema: Dict[str, Any],
    include_reasoning: bool = False
) -> str:
    """
    Format system prompt with embedded JSON schema.
    
    Args:
        base_prompt: Base system prompt text
        json_schema: JSON schema to embed
        include_reasoning: Whether to include reasoning instructions
        
    Returns:
        Formatted system prompt
    """
    schema_str = json.dumps(json_schema, indent=2)
    
    reasoning_prefix = ""
    if include_reasoning:
        reasoning_prefix = """You are a deep thinking AI, you may use extremely long chains of thought to deeply consider the problem and deliberate with yourself via systematic reasoning processes to help come to a correct solution prior to answering. You should enclose your thoughts and internal monologue inside <think> </think> tags, and then provide your solution or response to the problem.

"""
    
    prompt = f"""{reasoning_prefix}{base_prompt}

You must respond with a valid JSON object that conforms to the following JSON Schema:

<json_schema>
{schema_str}
</json_schema>

Ensure your response is valid JSON that can be parsed and validated against this schema."""
    
    return prompt
