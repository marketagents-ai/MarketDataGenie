"""
Debug utilities for pretty-printing REPL output, state, and code blocks with colors.
"""

import re
import json
from typing import List, Dict, Any, Optional


# ============================================================
# ANSI Color Codes
# ============================================================

class Colors:
    """ANSI color codes for terminal output."""
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


# ============================================================
# Basic Utilities
# ============================================================

def colorize(text: str, color: str) -> str:
    """Wrap text with color codes."""
    return f"{color}{text}{Colors.RESET}"


def print_colored(text: str, color: str = Colors.RESET, bold: bool = False) -> None:
    """Print text with color."""
    prefix = Colors.BOLD if bold else ""
    print(f"{prefix}{color}{text}{Colors.RESET}")


def print_separator(char: str = "─", length: int = 80, color: str = Colors.DIM) -> None:
    """Print a separator line."""
    print(colorize(char * length, color))


def truncate_text(text: str, max_length: int = 500, show_end: int = 200) -> str:
    """
    Truncate text in the middle, preserving start and end.
    """
    if len(text) <= max_length:
        return text
    
    show_start = max_length - show_end - 30
    if show_start < 100:
        show_start = 100
        show_end = max_length - show_start - 30
    
    hidden = len(text) - show_start - show_end
    return f"{text[:show_start]}\n... [{hidden} chars hidden] ...\n{text[-show_end:]}"


# ============================================================
# Pythonformer-Specific Pretty Printers
# ============================================================

def print_header(title: str, char: str = "=", width: int = 60) -> None:
    """Print a colored header."""
    line = char * width
    print(f"{Colors.CYAN}{line}{Colors.RESET}")
    print(f"{Colors.BOLD}{Colors.CYAN}{title.center(width)}{Colors.RESET}")
    print(f"{Colors.CYAN}{line}{Colors.RESET}")


def print_subheader(title: str, color: str = Colors.YELLOW) -> None:
    """Print a colored subheader."""
    print(f"\n{Colors.BOLD}{color}▶ {title}{Colors.RESET}")


def print_code_block(code: str, language: str = "python", max_lines: int = 30) -> None:
    """Pretty print a code block with syntax highlighting hints."""
    lines = code.strip().split('\n')
    if len(lines) > max_lines:
        lines = lines[:max_lines] + [f"... ({len(lines) - max_lines} more lines)"]
    
    print(f"{Colors.DIM}┌{'─' * 58}┐{Colors.RESET}")
    print(f"{Colors.DIM}│{Colors.RESET} {Colors.BRIGHT_BLUE}{language}{Colors.RESET}")
    print(f"{Colors.DIM}├{'─' * 58}┤{Colors.RESET}")
    
    for i, line in enumerate(lines, 1):
        # Simple syntax highlighting
        highlighted = line
        # Comments
        if '#' in highlighted:
            idx = highlighted.index('#')
            highlighted = highlighted[:idx] + Colors.BRIGHT_BLACK + highlighted[idx:] + Colors.RESET
        # Strings (simple)
        highlighted = re.sub(r'(["\'])(.+?)\1', f'{Colors.GREEN}\\1\\2\\1{Colors.RESET}', highlighted)
        # Keywords
        for kw in ['import', 'from', 'def', 'class', 'return', 'if', 'else', 'for', 'while', 'try', 'except', 'with', 'as', 'print']:
            highlighted = re.sub(rf'\b({kw})\b', f'{Colors.MAGENTA}\\1{Colors.RESET}', highlighted)
        
        line_num = f"{Colors.DIM}{i:3}{Colors.RESET}"
        print(f"{Colors.DIM}│{Colors.RESET} {line_num} {highlighted}")
    
    print(f"{Colors.DIM}└{'─' * 58}┘{Colors.RESET}")


def print_repl_output(output: str, error: Optional[str] = None, 
                      execution_time_ms: int = 0, truncated: bool = False,
                      max_lines: int = 20) -> None:
    """Pretty print REPL output with colors."""
    print(f"\n{Colors.BOLD}{Colors.GREEN}📤 REPL Output{Colors.RESET}", end="")
    if execution_time_ms > 0:
        time_color = Colors.YELLOW if execution_time_ms > 1000 else Colors.DIM
        print(f" {time_color}({execution_time_ms}ms){Colors.RESET}", end="")
    print()
    
    print(f"{Colors.GREEN}┌{'─' * 58}┐{Colors.RESET}")
    
    if output:
        lines = output.strip().split('\n')
        if len(lines) > max_lines:
            lines = lines[:max_lines] + [f"... ({len(lines) - max_lines} more lines)"]
        
        for line in lines:
            # Truncate long lines
            if len(line) > 80:
                line = line[:77] + "..."
            print(f"{Colors.GREEN}│{Colors.RESET} {line}")
    else:
        print(f"{Colors.GREEN}│{Colors.RESET} {Colors.DIM}(no output){Colors.RESET}")
    
    if error:
        print(f"{Colors.GREEN}│{Colors.RESET}")
        print(f"{Colors.GREEN}│{Colors.RESET} {Colors.RED}Error: {error[:100]}{Colors.RESET}")
    
    if truncated:
        print(f"{Colors.GREEN}│{Colors.RESET} {Colors.YELLOW}[Output truncated]{Colors.RESET}")
    
    print(f"{Colors.GREEN}└{'─' * 58}┘{Colors.RESET}")


def print_state(state_formatted: str) -> None:
    """Pretty print the REPL state."""
    if not state_formatted or state_formatted == "(empty state)":
        return
    
    print(f"\n{Colors.BOLD}{Colors.BLUE}📊 State{Colors.RESET}")
    print(f"{Colors.BLUE}┌{'─' * 58}┐{Colors.RESET}")
    
    # Parse and colorize state components
    parts = state_formatted.split(' | ')
    for part in parts:
        if part.startswith('imports:'):
            modules = part.replace('imports:', '').strip()
            print(f"{Colors.BLUE}│{Colors.RESET} {Colors.CYAN}imports:{Colors.RESET} {modules}")
        elif part.startswith('functions:'):
            funcs = part.replace('functions:', '').strip()
            print(f"{Colors.BLUE}│{Colors.RESET} {Colors.MAGENTA}functions:{Colors.RESET} {funcs}")
        elif part.startswith('classes:'):
            classes = part.replace('classes:', '').strip()
            print(f"{Colors.BLUE}│{Colors.RESET} {Colors.YELLOW}classes:{Colors.RESET} {classes}")
        elif part.startswith('vars:'):
            vars_str = part.replace('vars:', '').strip()
            # Truncate if too long
            if len(vars_str) > 100:
                vars_str = vars_str[:97] + "..."
            print(f"{Colors.BLUE}│{Colors.RESET} {Colors.GREEN}vars:{Colors.RESET} {vars_str}")
        else:
            print(f"{Colors.BLUE}│{Colors.RESET} {part}")
    
    print(f"{Colors.BLUE}└{'─' * 58}┘{Colors.RESET}")


def print_final_answer(answer: str, boxed_answers: List[str] = None, 
                       answer_correct: Optional[bool] = None) -> None:
    """Pretty print the final answer."""
    status_icon = "✅" if answer_correct is True else "❌" if answer_correct is False else "📝"
    status_color = Colors.GREEN if answer_correct is True else Colors.RED if answer_correct is False else Colors.YELLOW
    
    print(f"\n{Colors.BOLD}{status_color}{status_icon} Final Answer{Colors.RESET}")
    print(f"{status_color}┌{'─' * 58}┐{Colors.RESET}")
    
    # Print answer (truncated if needed)
    lines = answer.strip().split('\n')
    for line in lines[:5]:
        if len(line) > 80:
            line = line[:77] + "..."
        print(f"{status_color}│{Colors.RESET} {line}")
    if len(lines) > 5:
        print(f"{status_color}│{Colors.RESET} {Colors.DIM}... ({len(lines) - 5} more lines){Colors.RESET}")
    
    if boxed_answers:
        print(f"{status_color}│{Colors.RESET}")
        print(f"{status_color}│{Colors.RESET} {Colors.BOLD}Boxed:{Colors.RESET} {', '.join(boxed_answers)}")
    
    print(f"{status_color}└{'─' * 58}┘{Colors.RESET}")


def print_task_start(task_id: str, prompt: str, expected: Optional[str] = None) -> None:
    """Print task start info."""
    print_header(f"Task: {task_id}")
    print(f"\n{Colors.BOLD}Prompt:{Colors.RESET}")
    # Truncate long prompts
    if len(prompt) > 300:
        print(f"  {prompt[:300]}...")
    else:
        print(f"  {prompt}")
    
    if expected:
        print(f"\n{Colors.BOLD}Expected:{Colors.RESET} {Colors.CYAN}{expected}{Colors.RESET}")


def print_task_result(success: bool, num_turns: int, num_code_blocks: int,
                      answer_correct: Optional[bool] = None) -> None:
    """Print task result summary."""
    status = "✅ SUCCESS" if success else "❌ FAILED"
    status_color = Colors.GREEN if success else Colors.RED
    
    print(f"\n{Colors.BOLD}{status_color}{status}{Colors.RESET}")
    print(f"  Turns: {num_turns}, Code blocks: {num_code_blocks}", end="")
    
    if answer_correct is True:
        print(f", Answer: {Colors.GREEN}✓ Correct{Colors.RESET}")
    elif answer_correct is False:
        print(f", Answer: {Colors.RED}✗ Incorrect{Colors.RESET}")
    else:
        print()


# ============================================================
# Message/Chat Pretty Printers (from marketagents)
# ============================================================

def highlight_tags(text: str) -> str:
    """Highlight XML tags in text."""
    # Highlight <python> tags
    text = re.sub(
        r'(<python>)',
        colorize(r'\1', Colors.BRIGHT_CYAN + Colors.BOLD),
        text,
        flags=re.IGNORECASE
    )
    text = re.sub(
        r'(</python>)',
        colorize(r'\1', Colors.BRIGHT_CYAN + Colors.BOLD),
        text,
        flags=re.IGNORECASE
    )
    
    # Highlight <repl> tags
    text = re.sub(
        r'(<repl>)',
        colorize(r'\1', Colors.BRIGHT_GREEN + Colors.BOLD),
        text,
        flags=re.IGNORECASE
    )
    text = re.sub(
        r'(</repl>)',
        colorize(r'\1', Colors.BRIGHT_GREEN + Colors.BOLD),
        text,
        flags=re.IGNORECASE
    )
    
    # Highlight <state> tags
    text = re.sub(
        r'(<state>)',
        colorize(r'\1', Colors.BRIGHT_BLUE + Colors.BOLD),
        text,
        flags=re.IGNORECASE
    )
    text = re.sub(
        r'(</state>)',
        colorize(r'\1', Colors.BRIGHT_BLUE + Colors.BOLD),
        text,
        flags=re.IGNORECASE
    )
    
    # Highlight <final_answer> tags
    text = re.sub(
        r'(<final_answer>)',
        colorize(r'\1', Colors.BRIGHT_YELLOW + Colors.BOLD),
        text,
        flags=re.IGNORECASE
    )
    text = re.sub(
        r'(</final_answer>)',
        colorize(r'\1', Colors.BRIGHT_YELLOW + Colors.BOLD),
        text,
        flags=re.IGNORECASE
    )
    
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
    
    return text


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
    
    role_colors = {
        'system': Colors.MAGENTA,
        'user': Colors.GREEN,
        'assistant': Colors.BLUE,
        'tool': Colors.YELLOW,
    }
    
    for i, msg in enumerate(messages):
        role = msg.get('role', 'unknown')
        content = msg.get('content', '')
        role_color = role_colors.get(role, Colors.WHITE)
        
        # Print role header
        role_header = f"[{i}] {role.upper()}"
        if role == 'tool':
            tool_name = msg.get('name', 'unknown')
            role_header += f" ({tool_name})"
        
        print(colorize(role_header, role_color + Colors.BOLD))
        
        if content:
            display_content = truncate_text(content, max_content_length) if truncate else content
            display_content = highlight_tags(display_content)
            print(display_content)
        
        print()
        if i < len(messages) - 1:
            print_separator("─", 60, Colors.DIM)
