"""
Reasoning validation utilities for <think></think> tag handling.

Ported from datagenie/tool_use/datagen_tool_use.py and add_synthetic_reasoning.py
"""

import re
import json
import ast
from typing import Optional, List, Dict, Any, Tuple


def normalize_tool_call_json(txt: str) -> str:
    """
    Normalise assistant replies so that:
      • the original <think> … </think> block is preserved
      • every <tool_call> … </tool_call> block is converted to
        canonical JSON (double‑quoted, valid JSON) even if the
        model used Python literal formatting.
    """
    m = re.match(r"^\s*(<think>[\s\S]*?</think>)\s*", txt, flags=re.IGNORECASE)
    if not m:
        return txt
    think_part = m.group(1)

    def _convert(match: re.Match) -> str:
        raw = match.group(1).strip()
        try:
            obj = ast.literal_eval(raw)
            return f"<tool_call>{json.dumps(obj, separators=(',', ':'))}</tool_call>"
        except Exception:
            pass
        try:
            json_like = re.sub(r"'([^']*)':", r'"\1":', raw)
            json_like = re.sub(r":\s*'([^']*)'", r':"\1"', json_like)
            json.loads(json_like)
            return f"<tool_call>{json_like}</tool_call>"
        except Exception:
            return match.group(0)

    tail = txt[len(m.group(0)):]
    tail = re.sub(
        r"<tool_call>\s*([\s\S]*?)\s*</tool_call>",
        _convert,
        tail,
        flags=re.DOTALL | re.IGNORECASE,
    )
    out = think_part + tail
    out = re.sub(r"\s*<tool_call>\s*", "\n<tool_call>\n", out)
    out = re.sub(r"\s*</tool_call>\s*", "\n</tool_call>\n", out)
    return out


def validate_think_only(txt: str) -> bool:
    """
    A narration / summary turn must:
    • start with exactly one <think> … </think> block
    • contain **no** <tool_call> tags anywhere
    """
    txt = normalize_tool_call_json(txt)
    if not isinstance(txt, str):
        return False

    think_blocks = re.findall(r"<think>[\s\S]*?</think>", txt, flags=re.IGNORECASE)
    if len(think_blocks) != 1:
        return False

    if not re.match(r"^\s*<think>", txt, flags=re.IGNORECASE):
        return False

    if re.search(r"<tool_call\s*>", txt, flags=re.IGNORECASE):
        return False

    return True


def validate_think_block(txt: str) -> Tuple[bool, str]:
    """
    Validate that text contains a properly formatted <think> block.
    
    Returns:
        Tuple of (is_valid, reason)
    """
    if not txt or not isinstance(txt, str):
        return False, "empty_or_invalid_input"
    
    # Check for presence of think tags
    think_match = re.search(r"<think>([\s\S]*?)</think>", txt, flags=re.IGNORECASE)
    if not think_match:
        return False, "no_think_block"
    
    # Check that think block is at the start (after optional whitespace)
    if not re.match(r"^\s*<think>", txt, flags=re.IGNORECASE):
        return False, "think_not_at_start"
    
    # Check think block has content
    think_content = think_match.group(1).strip()
    if not think_content:
        return False, "empty_think_block"
    
    # Check for multiple think blocks (should only have one)
    think_blocks = re.findall(r"<think>[\s\S]*?</think>", txt, flags=re.IGNORECASE)
    if len(think_blocks) > 1:
        return False, "multiple_think_blocks"
    
    return True, "valid"


def extract_think_content(txt: str) -> Optional[str]:
    """
    Extract the content from within <think></think> tags.
    
    Returns:
        The content inside the think block, or None if not found
    """
    match = re.search(r"<think>([\s\S]*?)</think>", txt, flags=re.IGNORECASE)
    if match:
        return match.group(1).strip()
    return None


def extract_content_after_think(txt: str) -> Optional[str]:
    """
    Extract content that comes after the </think> tag.
    
    Returns:
        The content after the think block, or None if not found
    """
    match = re.search(r"</think>\s*([\s\S]*)", txt, flags=re.IGNORECASE)
    if match:
        return match.group(1).strip()
    return None


def has_think_block(txt: str) -> bool:
    """Check if text contains a <think> block."""
    if not txt:
        return False
    return bool(re.search(r"<think>[\s\S]*?</think>", txt, flags=re.IGNORECASE))


# System prompt snippet for reasoning-enabled tool calling
REASONING_SYSTEM_PROMPT_ADDITION = """
You are a deep thinking AI, you may use extremely long chains of thought to deeply consider the problem and deliberate with yourself via systematic reasoning processes to help come to a correct solution prior to answering. You should enclose your thoughts and internal monologue inside <think> </think> tags, and then provide your solution or response to the problem.

IMPORTANT: You MUST always start your response with <think> tags to show your reasoning process, whether you are:
- Making tool calls
- Providing a direct answer
- Asking for clarification
- Summarizing results

After your </think> block, provide your tool calls or final response.
"""


def get_reasoning_system_prompt(base_prompt: str) -> str:
    """
    Augment a base system prompt with reasoning instructions.
    
    Args:
        base_prompt: The original system prompt
        
    Returns:
        System prompt with reasoning instructions prepended
    """
    return REASONING_SYSTEM_PROMPT_ADDITION.strip() + "\n\n" + base_prompt


def parse_xml_tool_calls(txt: str) -> List[Dict[str, Any]]:
    """
    Parse <tool_call> XML tags from model output (Hermes format).
    
    Args:
        txt: Model output text containing <tool_call> tags
        
    Returns:
        List of tool calls in OpenAI format (deduplicated)
    """
    import uuid
    
    tool_calls = []
    seen_signatures = set()  # For deduplication
    
    # Find all <tool_call> blocks
    pattern = r"<tool_call>\s*([\s\S]*?)\s*</tool_call>"
    matches = re.findall(pattern, txt, flags=re.IGNORECASE)
    
    for match in matches:
        try:
            # Try to parse as JSON
            tool_obj = json.loads(match.strip())
            
            name = tool_obj.get("name", "")
            arguments = tool_obj.get("arguments", {})
            
            # Convert arguments to JSON string if dict
            if isinstance(arguments, dict):
                arguments_str = json.dumps(arguments, sort_keys=True)
            else:
                arguments_str = str(arguments)
            
            # Deduplicate by signature (name + args)
            signature = f"{name}:{arguments_str}"
            if signature in seen_signatures:
                continue
            seen_signatures.add(signature)
            
            tool_calls.append({
                "id": f"chatcmpl-tool-{uuid.uuid4().hex[:16]}",
                "type": "function",
                "function": {
                    "name": name,
                    "arguments": arguments_str
                }
            })
        except json.JSONDecodeError:
            # Try to fix common JSON issues
            try:
                # Try ast.literal_eval for Python-style dicts
                tool_obj = ast.literal_eval(match.strip())
                name = tool_obj.get("name", "")
                arguments = tool_obj.get("arguments", {})
                
                if isinstance(arguments, dict):
                    arguments_str = json.dumps(arguments, sort_keys=True)
                else:
                    arguments_str = str(arguments)
                
                # Deduplicate
                signature = f"{name}:{arguments_str}"
                if signature in seen_signatures:
                    continue
                seen_signatures.add(signature)
                
                tool_calls.append({
                    "id": f"chatcmpl-tool-{uuid.uuid4().hex[:16]}",
                    "type": "function",
                    "function": {
                        "name": name,
                        "arguments": arguments_str
                    }
                })
            except:
                # Skip malformed tool calls
                continue
    
    return tool_calls


def has_xml_tool_call(txt: str) -> bool:
    """Check if text contains a <tool_call> XML tag."""
    if not txt:
        return False
    return bool(re.search(r"<tool_call>[\s\S]*?</tool_call>", txt, flags=re.IGNORECASE))


def has_incomplete_tool_call(txt: str) -> bool:
    """
    Check if text has an incomplete/truncated <tool_call> tag.
    This happens when max_tokens cuts off the response mid-JSON.
    """
    if not txt:
        return False
    
    # Has opening tag but no closing tag
    has_open = bool(re.search(r"<tool_call>", txt, flags=re.IGNORECASE))
    has_close = bool(re.search(r"</tool_call>", txt, flags=re.IGNORECASE))
    
    if has_open and not has_close:
        return True
    
    # Count opening vs closing tags
    open_count = len(re.findall(r"<tool_call>", txt, flags=re.IGNORECASE))
    close_count = len(re.findall(r"</tool_call>", txt, flags=re.IGNORECASE))
    
    return open_count > close_count


def has_malformed_tool_call(txt: str) -> Tuple[bool, str]:
    """
    Check if text has a malformed <tool_call> tag (e.g., reasoning inside tool_call).
    
    Returns:
        Tuple of (is_malformed, reason)
    """
    if not txt:
        return False, ""
    
    # Check if response starts with <tool_call> (should start with <think>)
    stripped = txt.strip()
    if stripped.lower().startswith("<tool_call>"):
        # Check if the content inside is reasoning, not JSON
        match = re.search(r"<tool_call>\s*([\s\S]*?)(?:</tool_call>|$)", txt, flags=re.IGNORECASE)
        if match:
            content = match.group(1).strip()
            # If content doesn't start with { or [, it's reasoning not JSON
            if content and not content.startswith('{') and not content.startswith('['):
                return True, "response_starts_with_tool_call_containing_reasoning"
            # Even if it looks like JSON, starting with <tool_call> without <think> is wrong
            # when generate_reasoning is enabled
        return True, "response_starts_with_tool_call_instead_of_think"
    
    # Check if <tool_call> appears before <think> (wrong order)
    tool_call_pos = txt.lower().find("<tool_call>")
    think_pos = txt.lower().find("<think>")
    
    if tool_call_pos != -1 and think_pos == -1:
        # Has tool_call but no think - check if content looks like reasoning
        match = re.search(r"<tool_call>\s*([\s\S]*?)(?:</tool_call>|$)", txt, flags=re.IGNORECASE)
        if match:
            content = match.group(1).strip()
            # If content doesn't start with { or [, it's probably reasoning not JSON
            if content and not content.startswith('{') and not content.startswith('['):
                return True, "tool_call_contains_reasoning"
    
    if tool_call_pos != -1 and think_pos != -1 and tool_call_pos < think_pos:
        return True, "tool_call_before_think"
    
    return False, ""
