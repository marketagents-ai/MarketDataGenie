"""
Debug utilities for pretty-printing messages with colors.
"""

import json
from typing import List, Dict, Any, Optional

# ANSI color codes
class Colors:
    RESET = "\033[0m"
    BOLD = "\033[1m"
    DIM = "\033[2m"
    
    # Foreground colors
    BLACK = "\033[30m"
    RED = "\033[31m"
    GREEN = "\033[32m"
    YELLOW = "\033[33m"
    BLUE = "\033[34m"
    MAGENTA = "\033[35m"
    CYAN = "\033[36m"
    WHITE = "\033[37m"
    
    # Bright foreground colors
    BRIGHT_BLACK = "\033[90m"
    BRIGHT_RED = "\033[91m"
    BRIGHT_GREEN = "\033[92m"
    BRIGHT_YELLOW = "\033[93m"
    BRIGHT_BLUE = "\033[94m"
    BRIGHT_MAGENTA = "\033[95m"
    BRIGHT_CYAN = "\033[96m"
    BRIGHT_WHITE = "\033[97m"
    
    # Background colors
    BG_BLACK = "\033[40m"
    BG_RED = "\033[41m"
    BG_GREEN = "\033[42m"
    BG_YELLOW = "\033[43m"
    BG_BLUE = "\033[44m"
    BG_MAGENTA = "\033[45m"
    BG_CYAN = "\033[46m"
    BG_WHITE = "\033[47m"


def colorize(text: str, color: str) -> str:
    """Wrap text with color codes."""
    return f"{color}{text}{Colors.RESET}"


def truncate_text(text: str, max_length: int = 500, show_end: int = 200) -> str:
    """
    Truncate text in the middle, preserving start and end.
    This ensures tool calls at the end of assistant messages are visible.
    
    Args:
        text: Text to truncate
        max_length: Maximum total length
        show_end: How many characters to show from the end
    """
    if len(text) <= max_length:
        return text
    
    # Calculate how much to show from start
    show_start = max_length - show_end - 30  # 30 chars for ellipsis message
    if show_start < 100:
        show_start = 100
        show_end = max_length - show_start - 30
    
    hidden = len(text) - show_start - show_end
    return f"{text[:show_start]}\n... [{hidden} chars hidden] ...\n{text[-show_end:]}"


def format_json_pretty(obj: Any, indent: int = 2) -> str:
    """Format object as pretty JSON."""
    try:
        return json.dumps(obj, indent=indent, ensure_ascii=False)
    except:
        return str(obj)


def print_separator(char: str = "─", length: int = 80, color: str = Colors.DIM):
    """Print a separator line."""
    print(colorize(char * length, color))


def print_message(msg: Dict[str, Any], index: int = 0, truncate: bool = True, max_content_length: int = 800):
    """
    Pretty print a single message with colors.
    
    Colors by role:
    - system: Magenta
    - user: Green
    - assistant: Blue
    - tool: Yellow
    """
    role = msg.get('role', 'unknown')
    content = msg.get('content', '')
    
    # Role colors
    role_colors = {
        'system': Colors.MAGENTA,
        'user': Colors.GREEN,
        'assistant': Colors.BLUE,
        'tool': Colors.YELLOW,
    }
    
    role_color = role_colors.get(role, Colors.WHITE)
    
    # Print role header
    role_header = f"[{index}] {role.upper()}"
    if role == 'tool':
        tool_name = msg.get('name', 'unknown')
        tool_call_id = msg.get('tool_call_id', '')[:16] if msg.get('tool_call_id') else ''
        role_header += f" ({tool_name})"
        if tool_call_id:
            role_header += f" id={tool_call_id}"
    
    # Add tool call indicator for assistant messages
    if role == 'assistant' and msg.get('tool_calls'):
        num_calls = len(msg['tool_calls'])
        role_header += colorize(f" [+{num_calls} tool call(s)]", Colors.CYAN)
    
    print(colorize(role_header, role_color + Colors.BOLD))
    
    # Print content
    if content:
        # For assistant messages with tool calls, use middle truncation to show end
        if role == 'assistant' and msg.get('tool_calls'):
            display_content = truncate_text(content, max_content_length, show_end=300) if truncate else content
        else:
            display_content = truncate_text(content, max_content_length) if truncate else content
        
        # Highlight special tags
        display_content = highlight_tags(display_content)
        
        print(display_content)
    
    # Print tool calls if present
    if msg.get('tool_calls'):
        print(colorize("  Tool Calls:", Colors.CYAN + Colors.BOLD))
        for tc in msg['tool_calls']:
            func = tc.get('function', {})
            tc_id = tc.get('id', '')[:16]
            name = func.get('name', 'unknown')
            args = func.get('arguments', '{}')
            
            # Parse and pretty print arguments
            try:
                args_obj = json.loads(args) if isinstance(args, str) else args
                args_pretty = json.dumps(args_obj, indent=2)
            except:
                args_pretty = args
            
            print(colorize(f"    → {name}", Colors.CYAN) + colorize(f" (id={tc_id})", Colors.DIM))
            # Indent arguments
            for line in args_pretty.split('\n'):
                print(colorize(f"      {line}", Colors.DIM))
    
    print()  # Empty line after message


def highlight_tags(text: str) -> str:
    """Highlight XML tags in text."""
    import re
    
    # Highlight <think> tags
    text = re.sub(
        r'(<think>)',
        colorize(r'\1', Colors.BRIGHT_MAGENTA + Colors.BOLD),
        text,
        flags=re.IGNORECASE
    )
    text = re.sub(
        r'(</think>)',
        colorize(r'\1', Colors.BRIGHT_MAGENTA + Colors.BOLD),
        text,
        flags=re.IGNORECASE
    )
    
    # Highlight <tool_call> tags
    text = re.sub(
        r'(<tool_call>)',
        colorize(r'\1', Colors.BRIGHT_CYAN + Colors.BOLD),
        text,
        flags=re.IGNORECASE
    )
    text = re.sub(
        r'(</tool_call>)',
        colorize(r'\1', Colors.BRIGHT_CYAN + Colors.BOLD),
        text,
        flags=re.IGNORECASE
    )
    
    # Highlight <tool_response> tags
    text = re.sub(
        r'(<tool_response>)',
        colorize(r'\1', Colors.BRIGHT_YELLOW + Colors.BOLD),
        text,
        flags=re.IGNORECASE
    )
    text = re.sub(
        r'(</tool_response>)',
        colorize(r'\1', Colors.BRIGHT_YELLOW + Colors.BOLD),
        text,
        flags=re.IGNORECASE
    )
    
    # Highlight <tools> tags
    text = re.sub(
        r'(<tools>)',
        colorize(r'\1', Colors.BRIGHT_GREEN + Colors.BOLD),
        text,
        flags=re.IGNORECASE
    )
    text = re.sub(
        r'(</tools>)',
        colorize(r'\1', Colors.BRIGHT_GREEN + Colors.BOLD),
        text,
        flags=re.IGNORECASE
    )
    
    return text


def print_messages(
    messages: List[Dict[str, Any]], 
    title: str = "Messages",
    truncate: bool = True,
    max_content_length: int = 800
):
    """
    Pretty print a list of messages with colors.
    
    Args:
        messages: List of message dicts
        title: Title to display
        truncate: Whether to truncate long content
        max_content_length: Max length before truncation
    """
    print()
    print_separator("═", 80, Colors.BRIGHT_WHITE)
    print(colorize(f" {title} ({len(messages)} messages)", Colors.BRIGHT_WHITE + Colors.BOLD))
    print_separator("═", 80, Colors.BRIGHT_WHITE)
    print()
    
    for i, msg in enumerate(messages):
        print_message(msg, i, truncate, max_content_length)
        if i < len(messages) - 1:
            print_separator("─", 60, Colors.DIM)


def print_chat_thread(
    thread_name: str,
    system_prompt: str,
    history: List[Any],
    new_message: Optional[str] = None,
    truncate: bool = True
):
    """
    Pretty print a ChatThread's state.
    
    Args:
        thread_name: Name of the thread
        system_prompt: System prompt content
        history: List of ChatMessage objects
        new_message: The new message being sent
        truncate: Whether to truncate long content
    """
    print()
    print_separator("═", 80, Colors.BRIGHT_CYAN)
    print(colorize(f" ChatThread: {thread_name}", Colors.BRIGHT_CYAN + Colors.BOLD))
    print_separator("═", 80, Colors.BRIGHT_CYAN)
    
    # Show system prompt only on turn 0 (when history is empty or very short)
    if len(history) <= 1:
        print(colorize("\n[SYSTEM]", Colors.MAGENTA + Colors.BOLD))
        display_prompt = truncate_text(system_prompt, 500, show_end=200) if truncate else system_prompt
        print(highlight_tags(display_prompt))
        print()
    
    # History - show last few messages for context
    if history:
        # Show last 3 messages max for context
        show_count = min(3, len(history))
        start_idx = len(history) - show_count
        
        print(colorize(f"[HISTORY] (showing last {show_count} of {len(history)} messages)", Colors.DIM + Colors.BOLD))
        print_separator("─", 40, Colors.DIM)
        
        for i, msg in enumerate(history[start_idx:], start=start_idx):
            role = msg.role.value if hasattr(msg.role, 'value') else str(msg.role)
            content = msg.content or ""
            
            role_colors = {
                'user': Colors.GREEN,
                'assistant': Colors.BLUE,
                'tool': Colors.YELLOW,
                'system': Colors.MAGENTA,
            }
            color = role_colors.get(role, Colors.WHITE)
            
            header = f"  [{i}] {role.upper()}"
            if hasattr(msg, 'tool_name') and msg.tool_name:
                header += f" ({msg.tool_name})"
            
            print(colorize(header, color + Colors.BOLD))
            
            # For tool messages, format as {"name": ..., "content": ...}
            if role == 'tool' and hasattr(msg, 'tool_name') and msg.tool_name:
                try:
                    # Try to parse content as JSON for pretty display
                    content_obj = json.loads(content) if content.startswith('{') else content
                    formatted = json.dumps({"name": msg.tool_name, "content": content_obj}, indent=2)
                    display_content = truncate_text(formatted, 400, show_end=150) if truncate else formatted
                except:
                    display_content = truncate_text(content, 300, show_end=100) if truncate else content
            else:
                display_content = truncate_text(content, 300, show_end=150) if truncate else content
            
            print(f"      {highlight_tags(display_content)}")
            print()
    
    # New message
    if new_message:
        print(colorize("[NEW MESSAGE]", Colors.GREEN + Colors.BOLD))
        display_msg = truncate_text(new_message, 400, show_end=150) if truncate else new_message
        print(f"  {display_msg}")
    
    print_separator("─", 80, Colors.BRIGHT_CYAN)
    print()


def print_response(
    content: str,
    tool_calls: Optional[List[Dict[str, Any]]] = None,
    title: str = "Model Response",
    truncate: bool = True,
    max_length: int = 1500
):
    """
    Pretty print a model response with colors.
    
    Args:
        content: Response content
        tool_calls: List of tool calls (if any)
        title: Title to display
        truncate: Whether to truncate
        max_length: Max length before truncation
    """
    print()
    print_separator("═", 80, Colors.BRIGHT_BLUE)
    print(colorize(f" {title}", Colors.BRIGHT_BLUE + Colors.BOLD))
    print_separator("═", 80, Colors.BRIGHT_BLUE)
    
    if content:
        display_content = truncate_text(content, max_length) if truncate else content
        print(highlight_tags(display_content))
    else:
        print(colorize("  (empty content)", Colors.DIM))
    
    if tool_calls:
        print()
        print(colorize("Tool Calls:", Colors.CYAN + Colors.BOLD))
        for tc in tool_calls:
            func = tc.get('function', {})
            tc_id = tc.get('id', '')[:16]
            name = func.get('name', 'unknown')
            args = func.get('arguments', '{}')
            
            print(colorize(f"  → {name}", Colors.CYAN) + colorize(f" (id={tc_id})", Colors.DIM))
            
            try:
                args_obj = json.loads(args) if isinstance(args, str) else args
                args_pretty = json.dumps(args_obj, indent=2)
                for line in args_pretty.split('\n'):
                    print(colorize(f"    {line}", Colors.DIM))
            except:
                print(colorize(f"    {args}", Colors.DIM))
    
    print_separator("─", 80, Colors.BRIGHT_BLUE)
    print()
