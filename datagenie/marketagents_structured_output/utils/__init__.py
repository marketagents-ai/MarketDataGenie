"""Utility functions for structured output pipeline."""

from datagenie.marketagents_structured_output.utils.sharegpt import to_sharegpt_format
from datagenie.marketagents_structured_output.utils.reasoning import (
    validate_think_block,
    has_think_block,
    get_reasoning_system_prompt,
    REASONING_REMINDER,
    add_reasoning_reminder,
    strip_reasoning_reminder,
)
from datagenie.marketagents_structured_output.utils.debug import (
    print_messages,
    print_response,
)
from datagenie.marketagents_structured_output.utils.xml_parsing import (
    parse_xml_tool_call,
    extract_tool_call_arguments,
    parse_all_xml_tool_calls,
)

__all__ = [
    "to_sharegpt_format",
    "validate_think_block",
    "has_think_block",
    "get_reasoning_system_prompt",
    "REASONING_REMINDER",
    "add_reasoning_reminder",
    "strip_reasoning_reminder",
    "print_messages",
    "print_response",
    "parse_xml_tool_call",
    "extract_tool_call_arguments",
    "parse_all_xml_tool_calls",
]
