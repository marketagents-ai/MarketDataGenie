"""
Task-specific system prompts for Pythonformer.

Each task type (OOLONG, HotpotQA, Math, SWE, etc.) has its own specialized prompt
that guides the model on how to approach the problem.
"""

from datagenie.pythonformer.prompts.base import BASE_SYSTEM_PROMPT
from datagenie.pythonformer.prompts.oolong import OOLONG_SYSTEM_PROMPT
from datagenie.pythonformer.prompts.hotpotqa import HOTPOTQA_SYSTEM_PROMPT
from datagenie.pythonformer.prompts.swe import SWE_SYSTEM_PROMPT

__all__ = [
    "BASE_SYSTEM_PROMPT",
    "OOLONG_SYSTEM_PROMPT",
    "HOTPOTQA_SYSTEM_PROMPT",
    "SWE_SYSTEM_PROMPT",
]
