"""Debug utilities for pretty printing messages."""

import json
from typing import List, Dict, Any, Optional


class Colors:
    """ANSI color codes."""
    RESET = "\033[0m"
    BOLD = "\033[1m"
    DIM = "\033[2m"
    
    RED = "\033[31m"
    GREEN = "\033[32m"
    YELLOW = "\033[33m"
    BLUE = "\033[34m"
    MAGENTA = "\033[35m"
    CYAN = "\033[36m"
    WHITE = "\033[37m"
    
    BRIGHT_RED = "\033[91m"
    BRIGHT_GREEN = "\033[92m"
    BRIGHT_YELLOW = "\033[93m"
    BRIGHT_BLUE = "\033[94m"
    BRIGHT_MAGENTA = "\033[95m"
    BRIGHT_CYAN = "\033[96m"
    BRIGHT_WHITE = "\033[97m"


def colorize(text: str, color: str) -> str:
    """Apply color to text."""
    return f"{color}{text}{Colors.RESET}"


def truncate_text(text: str, max_length: int = 500) -> str:
    """Truncate text with ellipsis."""
    if len(text) <= max_length:
        return text
    return text[:max_length] + f"\n... [{len(text) - max_length} chars truncated] ..."


def print_separator(char: str = "─", length: int = 60, color: str = Colors.DIM):
    """Print a separator line."""
    print(colorize(char * length, color))


def print_message(msg: Dict[str, Any], index: int, truncate: bool = True, max_length: int = 800):
    """Print a single message with formatting."""
    role = msg.get("role", "unknown")
    content = msg.get("content", "")
    
    # Color by role
    role_colors = {
        "system": Colors.BRIGHT_MAGENTA,
        "user": Colors.BRIGHT_CYAN,
        "assistant": Colors.BRIGHT_GREEN,
        "human": Colors.BRIGHT_CYAN,
        "gpt": Colors.BRIGHT_GREEN,
    }
    color = role_colors.get(role, Colors.WHITE)
    
    # Print role header
    print(colorize(f"[{index}] {role.upper()}", color + Colors.BOLD))
    
    # Print content
    if truncate and len(content) > max_length:
        content = truncate_text(content, max_length)
    
    # Highlight JSON in content
    if "```json" in content or content.strip().startswith("{"):
        print(colorize(content, Colors.YELLOW))
    elif "<think>" in content.lower():
        # Highlight think blocks
        import re
        think_pattern = r"(<think>.*?</think>)"
        parts = re.split(think_pattern, content, flags=re.DOTALL | re.IGNORECASE)
        for part in parts:
            if part.lower().startswith("<think>"):
                print(colorize(part, Colors.DIM + Colors.CYAN))
            else:
                print(part)
    else:
        print(content)


def print_messages(
    messages: List[Dict[str, Any]], 
    title: str = "Messages",
    truncate: bool = True,
    max_content_length: int = 800
):
    """Pretty print a list of messages with colors."""
    print()
    print_separator("═", 80, Colors.BRIGHT_WHITE)
    print(colorize(f" {title} ({len(messages)} messages)", Colors.BRIGHT_WHITE + Colors.BOLD))
    print_separator("═", 80, Colors.BRIGHT_WHITE)
    print()
    
    for i, msg in enumerate(messages):
        print_message(msg, i, truncate, max_content_length)
        if i < len(messages) - 1:
            print_separator("─", 60, Colors.DIM)


def print_response(response: str, title: str = "Response"):
    """Print a response with formatting."""
    print()
    print_separator("═", 60, Colors.BRIGHT_GREEN)
    print(colorize(f" {title}", Colors.BRIGHT_GREEN + Colors.BOLD))
    print_separator("═", 60, Colors.BRIGHT_GREEN)
    print()
    
    # Try to format as JSON
    try:
        obj = json.loads(response)
        print(colorize(json.dumps(obj, indent=2), Colors.YELLOW))
    except (json.JSONDecodeError, TypeError):
        print(response)
