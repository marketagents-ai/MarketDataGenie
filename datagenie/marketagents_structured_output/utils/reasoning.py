"""Reasoning validation utilities for <think> blocks."""

import re
from typing import Optional, Tuple


def has_think_block(text: str) -> bool:
    """Check if text contains a non-empty <think> block."""
    if not text:
        return False
    match = re.search(r"<think>(.*?)</think>", text, re.DOTALL | re.IGNORECASE)
    if not match:
        return False
    # Check that the think block has actual content
    content = match.group(1).strip()
    return bool(content)


def validate_think_block(text: str) -> Tuple[bool, Optional[str]]:
    """
    Validate that text has properly formatted <think> block.
    
    Args:
        text: Text to validate
        
    Returns:
        Tuple of (is_valid, error_message)
    """
    if not text:
        return False, "Empty text"
    
    # Check for opening tag
    if not re.search(r"<think>", text, re.IGNORECASE):
        return False, "Missing <think> opening tag"
    
    # Check for closing tag
    if not re.search(r"</think>", text, re.IGNORECASE):
        return False, "Missing </think> closing tag"
    
    # Check that <think> comes before </think>
    think_match = re.search(r"<think>", text, re.IGNORECASE)
    end_think_match = re.search(r"</think>", text, re.IGNORECASE)
    
    if think_match and end_think_match:
        if think_match.start() > end_think_match.start():
            return False, "</think> appears before <think>"
    
    # Check for content inside think block
    think_content = re.search(r"<think>(.*?)</think>", text, re.DOTALL | re.IGNORECASE)
    if think_content:
        content = think_content.group(1).strip()
        if not content:
            return False, "Empty <think> block"
    
    return True, None


def extract_think_content(text: str) -> Optional[str]:
    """Extract content from <think> block."""
    if not text:
        return None
    
    match = re.search(r"<think>(.*?)</think>", text, re.DOTALL | re.IGNORECASE)
    if match:
        return match.group(1).strip()
    return None


def extract_response_after_think(text: str) -> Optional[str]:
    """Extract response content after </think> tag."""
    if not text:
        return None
    
    match = re.search(r"</think>\s*(.*)", text, re.DOTALL | re.IGNORECASE)
    if match:
        return match.group(1).strip()
    return None


def get_reasoning_system_prompt() -> str:
    """Get the deep thinking system prompt prefix."""
    return """You are a deep thinking AI, you may use extremely long chains of thought to deeply consider the problem and deliberate with yourself via systematic reasoning processes to help come to a correct solution prior to answering. You should enclose your thoughts and internal monologue inside <think> </think> tags, and then provide your solution or response to the problem."""


# Reasoning reminder to append to follow-up user messages
# This helps ensure the model outputs <think> blocks on follow-up turns
# Will be stripped during ShareGPT conversion
REASONING_REMINDER = (
    "\n\n[IMPORTANT: Remember to start your response with <think></think> tags "
    "containing your reasoning before providing the output.]"
)


def add_reasoning_reminder(text: str) -> str:
    """Add reasoning reminder to user message."""
    if not text:
        return text
    return text + REASONING_REMINDER


def strip_reasoning_reminder(text: str) -> str:
    """Strip reasoning reminder from user message."""
    if not text:
        return text
    return text.replace(REASONING_REMINDER, "")
