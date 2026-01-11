"""Utility functions for the function calling pipeline."""

from datagenie.marketagents_function_calling.utils.validation import (
    validate_tool_calls,
    validate_message,
)
from datagenie.marketagents_function_calling.utils.sharegpt import to_sharegpt_format
from datagenie.marketagents_function_calling.utils.reasoning import (
    validate_think_block,
    extract_think_content,
    extract_content_after_think,
    has_think_block,
    get_reasoning_system_prompt,
    parse_xml_tool_calls,
    has_xml_tool_call,
    has_incomplete_tool_call,
    has_malformed_tool_call,
)
from datagenie.marketagents_function_calling.utils.debug import (
    print_messages,
    print_chat_thread,
    print_response,
    Colors,
)

__all__ = [
    "validate_tool_calls",
    "validate_message",
    "to_sharegpt_format",
    "validate_think_block",
    "extract_think_content",
    "extract_content_after_think",
    "has_think_block",
    "get_reasoning_system_prompt",
    "parse_xml_tool_calls",
    "has_xml_tool_call",
    "has_incomplete_tool_call",
    "has_malformed_tool_call",
    "print_messages",
    "print_chat_thread",
    "print_response",
    "Colors",
]
