#!/usr/bin/env python3
"""
Add synthetic reasoning to Hermes tool-calling training data.

Post-processes Harmony-encoded JSONL to inject analysis channel content
that explains WHY each tool was selected.

Pipeline:
1. Parse existing Harmony tokens using openai_harmony
2. Detect parallel vs sequential tool calling patterns
3. Generate reasoning using LLM (Claude/GPT-4o)
4. Validate reasoning coherence with same LLM as judge
5. Insert analysis Message before tool calls
6. Re-render with openai_harmony

Usage:
    python scripts/add_synthetic_reasoning.py \
        --input hermes_function_calling_harmony.jsonl \
        --output hermes_with_reasoning.jsonl

    python scripts/add_synthetic_reasoning.py \
        --input hermes_function_calling_harmony.jsonl \
        --provider claude --dry-run --limit 10
"""

import argparse
import asyncio
import json
import logging
import os
import re
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

from openai_harmony import (
    Conversation,
    HarmonyEncoding,
    HarmonyEncodingName,
    Message,
    RenderConversationConfig,
    Role,
    load_harmony_encoding,
)

# Add parent directory to path
parent_dir = Path(__file__).parent.parent
if str(parent_dir) not in sys.path:
    sys.path.insert(0, str(parent_dir))

from common.api_retry import call_with_retry

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s | %(levelname)s | %(message)s'
)
logger = logging.getLogger(__name__)


# =============================================================================
# Constants
# =============================================================================

DEFAULT_MODELS = {
    "claude": "claude-sonnet-4-5-20250929",
    "openai": "gpt-4o",
}

# Stop/finish reasons that indicate truncation due to safety/refusal
CLAUDE_TRUNCATION_REASONS = {"content_filter", "end_turn_max_tokens", "max_tokens"}
OPENAI_TRUNCATION_REASONS = {"content_filter", "length"}


class TruncatedResponseError(Exception):
    """Raised when API response is truncated due to refusal, safety filter, or length limit."""

    def __init__(self, reason: str, provider: str):
        self.reason = reason
        self.provider = provider
        super().__init__(f"{provider} response truncated: {reason}")


# =============================================================================
# Data Classes
# =============================================================================

@dataclass
class ToolCallInfo:
    """Information about a tool call extracted from messages."""
    name: str
    arguments: str
    result: Optional[str] = None
    message_index: int = -1  # Index in the messages list


@dataclass
class ParsedSample:
    """Parsed components of a Harmony-encoded sample."""
    messages: List[Message]
    user_query: str
    tool_calls: List[ToolCallInfo]
    final_response: str
    available_tools: List[str]
    is_parallel: bool = False


@dataclass
class ProcessingStats:
    """Track processing statistics with task-safe increments (asyncio context)."""
    total: int = 0
    successful: int = 0
    failed_parse: int = 0
    failed_generate: int = 0
    failed_validate: int = 0
    no_tool_calls: int = 0
    parallel_samples: int = 0
    sequential_samples: int = 0
    synthetic_finals: int = 0  # Samples where final was generated
    post_tool_reasoning: int = 0  # Parallel samples with post-tool reasoning
    judge_failures_kept: int = 0  # Samples kept despite judge failure (fallback)
    extended_turns: int = 0  # Samples with additional simulated user turn
    extended_turn_failures: int = 0  # Failed to extend turn
    single_turn_samples: int = 0  # Single-turn samples (processed with analysis, not extended)
    api_truncations: int = 0  # API responses truncated due to refusal/safety/length
    # asyncio.Lock is callable, so each ProcessingStats instance gets its own lock
    _lock: asyncio.Lock = field(default_factory=asyncio.Lock, repr=False)

    # Valid stat field names for validation
    _VALID_FIELDS: frozenset = field(
        default_factory=lambda: frozenset({
            'total', 'successful', 'failed_parse', 'failed_generate',
            'failed_validate', 'no_tool_calls', 'parallel_samples',
            'sequential_samples', 'synthetic_finals', 'post_tool_reasoning',
            'judge_failures_kept', 'extended_turns', 'extended_turn_failures',
            'single_turn_samples', 'api_truncations'
        }),
        repr=False,
    )

    async def inc(self, **kwargs: int) -> None:
        """Task-safe increment of stat fields within async context. Usage: await stats.inc(total=1)"""
        async with self._lock:
            for field_name, amount in kwargs.items():
                if field_name not in self._VALID_FIELDS:
                    raise ValueError(f"Invalid stat field: {field_name}")
                current = getattr(self, field_name)
                setattr(self, field_name, current + amount)

    def report(self) -> str:
        """Generate statistics report."""
        success_rate = (self.successful / self.total * 100) if self.total > 0 else 0
        return f"""
=== Processing Statistics ===
Total processed:     {self.total:,}
Successful:          {self.successful:,}
  - Parallel:        {self.parallel_samples:,}
  - Sequential:      {self.sequential_samples:,}
Synthetic finals:    {self.synthetic_finals:,}
Post-tool reasoning: {self.post_tool_reasoning:,}
Extended turns:      {self.extended_turns:,}
Extended failures:   {self.extended_turn_failures:,}
Single-turn:         {self.single_turn_samples:,}
Judge fallbacks:     {self.judge_failures_kept:,}
API truncations:     {self.api_truncations:,}
No tool calls:       {self.no_tool_calls:,}
Failed parse:        {self.failed_parse:,}
Failed generate:     {self.failed_generate:,}
Failed validate:     {self.failed_validate:,}
Success rate:        {success_rate:.1f}%
"""


# =============================================================================
# Harmony Parser (using openai_harmony)
# =============================================================================

def extract_tool_names_from_namespace(text: str) -> List[str]:
    """
    Extract tool names from TypeScript-like namespace format.

    The format is:
        namespace functions {
            type tool_name = (_: {...}) => any;
        }

    Args:
        text: Developer message text containing namespace definitions

    Returns:
        List of tool names found
    """
    tools = []
    lines = text.split('\n')
    for line in lines:
        line = line.strip()
        # Look for "type NAME = " pattern
        if line.startswith('type ') and '=' in line:
            # Extract name between "type " and " ="
            parts = line.split('=', 1)
            if parts:
                name_part = parts[0].replace('type ', '').strip()
                if name_part:
                    tools.append(name_part)
    return tools


def parse_harmony_sample(text: str, enc: HarmonyEncoding) -> Optional[ParsedSample]:
    """
    Parse Harmony-encoded text using openai_harmony.

    Args:
        text: Harmony token-encoded string
        enc: Harmony encoding object

    Returns:
        ParsedSample object or None if parsing fails
    """
    try:
        # Parse to messages
        tokens = enc.encode(text, allowed_special='all')
        messages = enc.parse_messages_from_completion_tokens(tokens, strict=False)

        # Extract components
        user_query = ""
        tool_calls = []
        final_response = ""
        available_tools = []

        # Track indices for tool calls and responses
        tool_call_indices = []
        tool_response_indices = []

        for i, msg in enumerate(messages):
            role = msg.author.role if msg.author else None

            # Extract user query (first user message)
            if role == Role.USER and not user_query:
                if msg.content and hasattr(msg.content[0], 'text'):
                    user_query = msg.content[0].text

            # Extract tool definitions from developer message
            elif role == Role.DEVELOPER:
                if msg.content and hasattr(msg.content[0], 'text'):
                    dev_text = msg.content[0].text
                    # Tools are rendered as TypeScript namespace format
                    available_tools = extract_tool_names_from_namespace(dev_text)

            # Extract tool calls (assistant with recipient)
            elif role == Role.ASSISTANT and msg.recipient:
                tool_name = msg.recipient.replace('functions.', '')
                args = ""
                if msg.content and hasattr(msg.content[0], 'text'):
                    args = msg.content[0].text
                tool_calls.append(ToolCallInfo(
                    name=tool_name,
                    arguments=args,
                    message_index=i
                ))
                tool_call_indices.append(i)

            # Extract tool responses
            elif role == Role.TOOL:
                # Tool response author names have 'functions.' prefix, strip it to match call names
                raw_name = msg.author.name if msg.author else ""
                tool_name = raw_name.replace('functions.', '')
                result = ""
                if msg.content and hasattr(msg.content[0], 'text'):
                    result = msg.content[0].text
                # Match to corresponding tool call
                for tc in tool_calls:
                    if tc.name == tool_name and tc.result is None:
                        tc.result = result
                        break
                tool_response_indices.append(i)

            # Extract final response
            elif role == Role.ASSISTANT and msg.channel == 'final':
                if msg.content and hasattr(msg.content[0], 'text'):
                    final_response = msg.content[0].text

        # Detect parallel vs sequential tool calling
        # Parallel: multiple tool calls that all come before any tool responses
        # (or multiple tool calls with no responses at all)
        is_parallel = False
        if len(tool_calls) > 1 and tool_call_indices:
            if not tool_response_indices:
                # No responses yet - if multiple calls exist, they're parallel
                is_parallel = True
            else:
                # Check if all calls come before first response
                last_call_idx = max(tool_call_indices)
                first_response_idx = min(tool_response_indices)
                is_parallel = last_call_idx < first_response_idx

        return ParsedSample(
            messages=messages,
            user_query=user_query,
            tool_calls=tool_calls,
            final_response=final_response,
            available_tools=available_tools,
            is_parallel=is_parallel,
        )

    except Exception as e:
        logger.debug(f"Failed to parse Harmony text: {e}")
        return None


# =============================================================================
# Prompts
# =============================================================================

# Prompt for sequential tool calls (one at a time) - matches parallel structure
REASONING_PROMPT_SEQUENTIAL = '''You are analyzing a tool-calling conversation to explain the assistant's SEQUENTIAL decision-making process.

Given this interaction:
- User query: {user_query}
- Available tools: {available_tools}
{prior_context}
- Tool selected: {tool_name}
- Arguments used: {tool_args}
- Result received: {tool_result}
- Final response: {final_response}

Generate the assistant's internal reasoning that led to selecting this specific tool with these specific arguments.

The reasoning should explain:
1. What the user is asking for and what information is needed
2. Why this specific tool is the right choice from the available options
3. How the arguments were determined from the user's request
4. What outcome is expected from this tool call

Format as JSON array of steps:
[
  {{"step": 1, "explanation": "Brief description of this reasoning step", "output": "Concrete conclusion from this step"}},
  {{"step": 2, "explanation": "...", "output": "..."}},
  ...
]

Requirements:
- 2-4 steps maximum (concise reasoning)
- Each step should logically lead to the next
- Final step should conclude with the tool selection decision
- Match the actual tool and arguments used
- Focus on WHY this tool was chosen, not just WHAT it does
'''

# Prompt for parallel tool calls (multiple tools called together)
REASONING_PROMPT_PARALLEL = '''You are analyzing a tool-calling conversation where the assistant made MULTIPLE tool calls in parallel.

Given this interaction:
- User query: {user_query}
- Available tools: {available_tools}
- Tools called in parallel:
{tool_calls_summary}
- Final response: {final_response}

Generate the assistant's internal reasoning that led to calling these {num_tools} tools IN PARALLEL (simultaneously, not sequentially).

The reasoning should explain:
1. Why multiple tools are needed to fulfill this request
2. Why these specific tools were chosen
3. Why they can be called in parallel (independent operations)
4. The arguments chosen for each tool

Format as JSON array of steps:
[
  {{"step": 1, "explanation": "Brief description of this reasoning step", "output": "Concrete conclusion from this step"}},
  {{"step": 2, "explanation": "...", "output": "..."}},
  ...
]

Requirements:
- 3-5 steps maximum
- MUST mention that tools are being called in parallel/simultaneously
- Explain why parallel execution is appropriate (tasks are independent)
- Cover all {num_tools} tool selections in the reasoning
'''

# Prompt for POST-TOOL reasoning (after parallel tool responses, before final)
REASONING_PROMPT_POST_TOOL = '''You are analyzing a tool-calling conversation AFTER the tool results have been received.

The assistant called {num_tools} tools in parallel and received their results. Now generate the reasoning that synthesizes these results before composing the final response.

Given:
- User query: {user_query}
- Tool calls and results:
{tool_results_summary}
- Final response given: {final_response}

Generate the assistant's internal reasoning that synthesizes the tool results and leads to the final response.

The reasoning should explain:
1. What information was received from each tool
2. How the results relate to the user's original request
3. How to combine/synthesize the results
4. How to formulate the final response

Format as JSON array of steps:
[
  {{"step": 1, "explanation": "Brief description of this reasoning step", "output": "Concrete conclusion from this step"}},
  {{"step": 2, "explanation": "...", "output": "..."}},
  ...
]

Requirements:
- 2-4 steps maximum
- Reference the actual tool results received
- Show how results inform the final response
- Bridge from raw data to user-friendly answer
'''

# Prompt for SYNTHETIC FINAL response generation
SYNTHETIC_FINAL_PROMPT = '''You are completing a tool-calling conversation by generating the assistant's final response to the user.

Given:
- User query: {user_query}
- Tool calls and results:
{tool_results_summary}

Generate a natural, helpful final response that:
1. Directly answers the user's question using the tool results
2. Presents the information in a user-friendly format
3. Is concise but complete
4. Does NOT mention the tools or that tools were used (just give the answer naturally)

Respond with ONLY the final response text (no JSON, no explanation, just the response the assistant would give).
'''

JUDGE_PROMPT_SEQUENTIAL = '''Evaluate whether this reasoning naturally and logically leads to the tool selection.

REASONING STEPS:
{reasoning_text}

ACTUAL TOOL CALL:
{tool_name}({tool_args})

EVALUATION CRITERIA:
1. Does the reasoning correctly identify the user's need?
2. Does it logically conclude that this specific tool is appropriate?
3. Are the argument choices justified by the reasoning?

Respond in JSON:
{{"equivalent": true, "confidence": 0.95, "reasoning": "one sentence explanation"}}

Answer equivalent=true if the reasoning would naturally lead to this exact tool call.
Answer equivalent=false if the reasoning is contrived, illogical, or doesn't match the tool selection.
'''

JUDGE_PROMPT_PARALLEL = '''Evaluate whether this reasoning naturally and logically leads to the parallel tool calls.

REASONING STEPS:
{reasoning_text}

ACTUAL PARALLEL TOOL CALLS:
{tool_calls_summary}

EVALUATION CRITERIA:
1. Does the reasoning correctly identify all user needs?
2. Does it explain why MULTIPLE tools are being called?
3. Does it justify why these tools can be called in PARALLEL (independent operations)?
4. Are all argument choices justified by the reasoning?

Respond in JSON:
{{"equivalent": true, "confidence": 0.95, "reasoning": "one sentence explanation"}}

Answer equivalent=true if the reasoning naturally leads to these exact parallel tool calls.
Answer equivalent=false if the reasoning doesn't justify parallel execution or doesn't cover all tools.
'''

# Judge for post-tool reasoning
JUDGE_PROMPT_POST_TOOL = '''Evaluate whether this post-tool reasoning correctly synthesizes the tool results.

REASONING STEPS:
{reasoning_text}

TOOL RESULTS RECEIVED:
{tool_results_summary}

FINAL RESPONSE GIVEN:
{final_response}

EVALUATION CRITERIA:
1. Does the reasoning accurately reflect what the tools returned?
2. Does it correctly synthesize/combine the results?
3. Does it logically lead to the final response?

Respond in JSON:
{{"equivalent": true, "confidence": 0.95, "reasoning": "one sentence explanation"}}

Answer equivalent=true if the reasoning correctly bridges from tool results to final response.
Answer equivalent=false if it misrepresents results or doesn't support the final response.
'''

# =============================================================================
# Extended Turn Prompt (single-call structured generation)
# =============================================================================

# Single prompt that generates follow-up + reasoning + final together
EXTEND_TURN_PROMPT = '''You are extending a tool-calling conversation with one additional turn.

CONVERSATION SO FAR:
- User's original question: {original_question}
- Tools called: {tool_summary}
- Assistant's response: {previous_response}

YOUR TASK:
Generate a realistic follow-up from the user AND the assistant's response to it.

CRITICAL CONSTRAINT: The follow-up MUST be answerable using ONLY the information already available
from the previous tool results. Do NOT generate follow-ups that would require calling additional tools
or fetching new data. The assistant should reason over existing context, not make new tool calls.

Good follow-up types (answerable from existing context):
- "Why is X higher/lower than Y?" - comparing values from tool results
- "What does that mean for...?" - asking for interpretation of existing data
- "Can you explain why...?" - asking for clarification about the response
- "How does X relate to Y?" - connecting information already retrieved
- "Based on that, would you recommend...?" - asking for advice using existing info

Bad follow-up types (would require new tool calls - AVOID THESE):
- "What about [different stock/city/item]?" - needs new data lookup
- "Can you also check...?" - requests additional tool calls
- "What's the current price of...?" - needs fresh data retrieval

IMPORTANT: The follow-up should require the assistant to USE INFORMATION FROM THE PREVIOUS TURN.
This creates a multi-hop reasoning chain where context must be tracked across turns.

Respond with a JSON object:
{{
  "followup_question": "The user's follow-up question (1-2 sentences)",
  "reasoning_steps": [
    {{"step": 1, "explanation": "What this step does", "output": "The conclusion"}},
    {{"step": 2, "explanation": "...", "output": "..."}}
  ],
  "final_response": "The assistant's response to the follow-up (1-3 sentences)"
}}

Requirements:
- followup_question: Natural, specific to the conversation (not generic)
- reasoning_steps: 2-3 steps showing how previous context informs the answer
- final_response: References information from the tool results or previous response
- The reasoning MUST explain how previous turn data is used
'''

# Judge for validating the extended turn coherence
JUDGE_PROMPT_EXTEND = '''Evaluate whether this extended conversation turn is coherent and properly tracks context.

ORIGINAL CONVERSATION:
- User question: {original_question}
- Tool results: {tool_summary}
- Assistant response: {previous_response}

EXTENDED TURN:
- Follow-up: {followup_question}
- Reasoning: {reasoning_text}
- Final response: {final_response}

EVALUATION CRITERIA:
1. Is the follow-up question natural and specific to the conversation?
2. Can the follow-up be answered using ONLY the existing tool results (no new tool calls needed)?
3. Does the reasoning correctly use information from the previous turn?
4. Does the final response properly answer the follow-up using existing context?
5. Would this make sense as a real conversation?

Respond in JSON:
{{"equivalent": true, "confidence": 0.95, "reasoning": "one sentence explanation"}}

Answer equivalent=true if the extended turn is coherent, uses existing context, and doesn't require new tool calls.
Answer equivalent=false if:
- The follow-up would require calling additional tools (e.g., "What about [other item]?")
- It doesn't use previous context
- It seems generic or artificial
'''


# =============================================================================
# Synthetic Reasoning Pipeline
# =============================================================================

class SyntheticReasoningPipeline:
    """Generate and validate synthetic reasoning for tool selections."""

    MAX_REASONING_TOKENS = 16384
    LLM_TIMEOUT = 600.0  # 10 minutes for reasoning generation (can be slow with 16k tokens)

    def __init__(
        self,
        provider: str = "claude",
        model: Optional[str] = None,
        api_key: Optional[str] = None,
        min_confidence: float = 0.7,
        max_retries: int = 2,
    ):
        if not 0.0 <= min_confidence <= 1.0:
            raise ValueError(f"min_confidence must be between 0.0 and 1.0, got {min_confidence}")

        # Track truncations for stats reporting
        self.truncation_count = 0
        self._truncation_lock = asyncio.Lock()

        self.provider = provider
        self.model = model or DEFAULT_MODELS[provider]
        self.min_confidence = min_confidence
        self.max_retries = max_retries

        if provider == "claude":
            from anthropic import AsyncAnthropic
            self.client = AsyncAnthropic(
                api_key=api_key or os.getenv("ANTHROPIC_API_KEY")
            )
        else:
            from openai import AsyncOpenAI
            self.client = AsyncOpenAI(
                api_key=api_key or os.getenv("OPENAI_API_KEY")
            )

        logger.info(f"Pipeline initialized: provider={provider}, model={self.model}")

    async def _record_truncation(self, reason: str, sample_idx: int = -1) -> None:
        """Record a truncation event (thread-safe)."""
        async with self._truncation_lock:
            self.truncation_count += 1
            idx_str = f"[Sample {sample_idx}] " if sample_idx >= 0 else ""
            logger.warning(f"{idx_str}API truncation #{self.truncation_count}: {reason}")

    async def _call_llm(self, prompt: str, max_tokens: int = 4096) -> Optional[str]:
        """
        Make an LLM API call with retry.

        Raises:
            TruncatedResponseError: When response is truncated due to safety/refusal/length
        """
        try:
            if self.provider == "claude":
                response = await call_with_retry(
                    lambda: self.client.messages.create(
                        model=self.model,
                        max_tokens=max_tokens,
                        messages=[{"role": "user", "content": prompt}],
                    ),
                    operation_name=f"Claude API ({self.model})",
                    timeout=self.LLM_TIMEOUT,
                )

                # Check for truncation/refusal stop reasons
                stop_reason = response.stop_reason
                if stop_reason in CLAUDE_TRUNCATION_REASONS:
                    raise TruncatedResponseError(stop_reason, "Claude")

                # Handle empty content (refusal, content filter, etc.)
                if not response.content:
                    logger.warning(f"Claude returned empty content. stop_reason={stop_reason}")
                    # Empty content often indicates refusal even without explicit stop_reason
                    raise TruncatedResponseError(f"empty_content ({stop_reason})", "Claude")

                return response.content[0].text
            else:
                response = await call_with_retry(
                    lambda: self.client.chat.completions.create(
                        model=self.model,
                        max_tokens=max_tokens,
                        messages=[{"role": "user", "content": prompt}],
                    ),
                    operation_name=f"OpenAI API ({self.model})",
                    timeout=self.LLM_TIMEOUT,
                )

                # Check for truncation/refusal finish reasons
                if response.choices:
                    finish_reason = response.choices[0].finish_reason
                    if finish_reason in OPENAI_TRUNCATION_REASONS:
                        raise TruncatedResponseError(finish_reason, "OpenAI")

                # Handle empty choices
                if not response.choices or not response.choices[0].message.content:
                    finish_reason = response.choices[0].finish_reason if response.choices else 'no choices'
                    logger.warning(f"OpenAI returned empty content. finish_reason={finish_reason}")
                    raise TruncatedResponseError(f"empty_content ({finish_reason})", "OpenAI")

                return response.choices[0].message.content
        except TruncatedResponseError:
            raise  # Re-raise truncation errors for handling by callers
        except Exception as e:
            logger.warning(f"LLM call failed: {type(e).__name__}: {e}")
            raise  # Re-raise so callers can handle it

    def _parse_json_response(self, text: str) -> Optional[Any]:
        """Parse JSON from LLM response with fallback strategies."""
        if not text:
            return None

        # Strategy 1: Direct parse
        try:
            return json.loads(text)
        except json.JSONDecodeError:
            pass

        # Strategy 2: Extract from markdown code blocks
        code_block = re.search(r'```(?:json)?\s*([\s\S]*?)\s*```', text)
        if code_block:
            try:
                return json.loads(code_block.group(1))
            except json.JSONDecodeError:
                pass

        # Strategy 3: Find JSON array/object by matching brackets
        for start_char, end_char in [('[', ']'), ('{', '}')]:
            start_idx = text.find(start_char)
            if start_idx == -1:
                continue

            depth = 0
            for i, char in enumerate(text[start_idx:], start_idx):
                if char == start_char:
                    depth += 1
                elif char == end_char:
                    depth -= 1
                    if depth == 0:
                        try:
                            return json.loads(text[start_idx:i + 1])
                        except json.JSONDecodeError:
                            break

        return None

    def _validate_reasoning_steps(self, steps: Any) -> Optional[List[Dict[str, str]]]:
        """
        Validate and normalize reasoning steps from LLM response.

        Args:
            steps: Raw parsed response (expected to be a list of step dicts)

        Returns:
            List of validated step dicts with 'step', 'explanation', 'output' keys,
            or None if validation fails
        """
        if not isinstance(steps, list) or len(steps) < 1:
            return None

        valid_steps = []
        for step in steps:
            if isinstance(step, dict) and "explanation" in step and "output" in step:
                valid_steps.append({
                    "step": step.get("step", len(valid_steps) + 1),
                    "explanation": str(step["explanation"]),
                    "output": str(step["output"]),
                })

        return valid_steps if valid_steps else None

    async def generate_sequential_reasoning(
        self,
        user_query: str,
        tool_call: ToolCallInfo,
        final_response: str,
        available_tools: List[str],
        prior_tool_calls: Optional[List[ToolCallInfo]] = None,
        sample_idx: int = -1,
    ) -> Optional[List[Dict[str, str]]]:
        """Generate reasoning for a sequential tool call."""
        prior_context = ""
        if prior_tool_calls:
            prior_parts = []
            for i, tc in enumerate(prior_tool_calls, 1):
                prior_parts.append(f"- Prior tool call {i}: {tc.name}({tc.arguments})")
                if tc.result:
                    result_preview = tc.result[:500] + "..." if len(tc.result) > 500 else tc.result
                    prior_parts.append(f"  Result: {result_preview}")
            prior_context = "\n".join(prior_parts) + "\n"

        prompt = REASONING_PROMPT_SEQUENTIAL.format(
            user_query=user_query,
            available_tools=", ".join(available_tools) if available_tools else "unknown",
            prior_context=prior_context,
            tool_name=tool_call.name,
            tool_args=tool_call.arguments,
            tool_result=tool_call.result or "No result recorded",
            final_response=final_response,
        )

        try:
            response = await self._call_llm(prompt, max_tokens=self.MAX_REASONING_TOKENS)
            if not response:
                logger.warning(f"[Sample {sample_idx}] Sequential reasoning: LLM returned empty response")
                return None
            steps = self._parse_json_response(response)
            if steps is None:
                logger.warning(f"[Sample {sample_idx}] Sequential reasoning: Failed to parse JSON from response: {response[:500]}...")
                return None
            validated = self._validate_reasoning_steps(steps)
            if validated is None:
                logger.warning(f"[Sample {sample_idx}] Sequential reasoning: Steps failed validation: {steps}")
                return None
            return validated
        except TruncatedResponseError as e:
            await self._record_truncation(f"sequential reasoning: {e}", sample_idx)
        except (json.JSONDecodeError, KeyError, TypeError) as e:
            logger.debug(f"[Sample {sample_idx}] Failed to parse sequential reasoning response: {e}")
        except asyncio.TimeoutError as e:
            logger.warning(f"[Sample {sample_idx}] Sequential reasoning generation timed out: {e}")
        except Exception as e:
            logger.warning(f"[Sample {sample_idx}] Unexpected error in sequential reasoning: {e}", exc_info=True)

        return None

    async def validate_sequential_reasoning(
        self,
        reasoning_steps: List[Dict[str, str]],
        tool_call: ToolCallInfo,
        sample_idx: int = -1,
    ) -> Tuple[bool, float]:
        """Validate sequential reasoning."""
        reasoning_text = "\n".join([
            f"Step {s['step']}: {s['explanation']}\n{s['output']}"
            for s in reasoning_steps
        ])

        prompt = JUDGE_PROMPT_SEQUENTIAL.format(
            reasoning_text=reasoning_text,
            tool_name=tool_call.name,
            tool_args=tool_call.arguments,
        )

        try:
            response = await self._call_llm(prompt, max_tokens=500)
            result = self._parse_json_response(response)

            if isinstance(result, dict):
                is_valid = result.get("equivalent", False)
                confidence = float(result.get("confidence", 0.0))
                reasoning = result.get("reasoning", "")
                logger.info(f"[Sample {sample_idx}] Sequential judge: valid={is_valid}, conf={confidence:.2f}, reasoning={reasoning}")
                return (is_valid, confidence)
        except TruncatedResponseError as e:
            await self._record_truncation(f"sequential judge: {e}", sample_idx)
        except Exception as e:
            logger.debug(f"[Sample {sample_idx}] Sequential validation failed: {e}")

        return (False, 0.0)

    async def generate_and_validate_sequential(
        self,
        user_query: str,
        tool_call: ToolCallInfo,
        final_response: str,
        available_tools: List[str],
        prior_tool_calls: Optional[List[ToolCallInfo]] = None,
        fallback_on_failure: bool = False,
        sample_idx: int = -1,
    ) -> Tuple[Optional[List[Dict[str, str]]], bool]:
        """
        Generate and validate sequential reasoning with retries.

        Returns:
            Tuple of (reasoning_steps, is_fallback) where is_fallback indicates
            the reasoning was kept despite judge failure
        """
        last_reasoning = None

        for attempt in range(self.max_retries):
            reasoning = await self.generate_sequential_reasoning(
                user_query=user_query,
                tool_call=tool_call,
                final_response=final_response,
                available_tools=available_tools,
                prior_tool_calls=prior_tool_calls,
                sample_idx=sample_idx,
            )

            if reasoning is None:
                logger.debug(f"[Sample {sample_idx}] Attempt {attempt + 1}: Sequential generation failed")
                continue

            last_reasoning = reasoning

            is_valid, confidence = await self.validate_sequential_reasoning(
                reasoning_steps=reasoning,
                tool_call=tool_call,
                sample_idx=sample_idx,
            )

            if is_valid and confidence >= self.min_confidence:
                logger.debug(f"[Sample {sample_idx}] Attempt {attempt + 1}: Valid (confidence={confidence:.2f})")
                return (reasoning, False)
            else:
                logger.debug(f"[Sample {sample_idx}] Attempt {attempt + 1}: Invalid (valid={is_valid}, conf={confidence:.2f})")

        if fallback_on_failure and last_reasoning is not None:
            logger.info(f"[Sample {sample_idx}] Using fallback: keeping sequential reasoning despite judge failure")
            return (last_reasoning, True)

        return (None, False)

    async def generate_parallel_reasoning(
        self,
        user_query: str,
        tool_calls: List[ToolCallInfo],
        final_response: str,
        available_tools: List[str],
        sample_idx: int = -1,
    ) -> Optional[List[Dict[str, str]]]:
        """Generate reasoning for parallel tool calls."""
        tool_calls_summary = "\n".join([
            f"  {i+1}. {tc.name}({tc.arguments})"
            for i, tc in enumerate(tool_calls)
        ])

        prompt = REASONING_PROMPT_PARALLEL.format(
            user_query=user_query,
            available_tools=", ".join(available_tools) if available_tools else "unknown",
            tool_calls_summary=tool_calls_summary,
            final_response=final_response,
            num_tools=len(tool_calls),
        )

        try:
            response = await self._call_llm(prompt, max_tokens=self.MAX_REASONING_TOKENS)
            if not response:
                logger.warning(f"[Sample {sample_idx}] Parallel reasoning: LLM returned empty response")
                return None
            steps = self._parse_json_response(response)
            if steps is None:
                logger.warning(f"[Sample {sample_idx}] Parallel reasoning: Failed to parse JSON from response: {response[:500]}...")
                return None
            validated = self._validate_reasoning_steps(steps)
            if validated is None:
                logger.warning(f"[Sample {sample_idx}] Parallel reasoning: Steps failed validation: {steps}")
                return None
            return validated
        except TruncatedResponseError as e:
            await self._record_truncation(f"parallel reasoning: {e}", sample_idx)
        except (json.JSONDecodeError, KeyError, TypeError) as e:
            logger.debug(f"[Sample {sample_idx}] Failed to parse parallel reasoning response: {e}")
        except asyncio.TimeoutError as e:
            logger.warning(f"[Sample {sample_idx}] Parallel reasoning generation timed out: {e}")
        except Exception as e:
            logger.warning(f"[Sample {sample_idx}] Unexpected error in parallel reasoning: {e}", exc_info=True)

        return None

    async def validate_parallel_reasoning(
        self,
        reasoning_steps: List[Dict[str, str]],
        tool_calls: List[ToolCallInfo],
        sample_idx: int = -1,
    ) -> Tuple[bool, float]:
        """Validate parallel reasoning."""
        reasoning_text = "\n".join([
            f"Step {s['step']}: {s['explanation']}\n{s['output']}"
            for s in reasoning_steps
        ])

        tool_calls_summary = "\n".join([
            f"  {i+1}. {tc.name}({tc.arguments})"
            for i, tc in enumerate(tool_calls)
        ])

        prompt = JUDGE_PROMPT_PARALLEL.format(
            reasoning_text=reasoning_text,
            tool_calls_summary=tool_calls_summary,
        )

        try:
            response = await self._call_llm(prompt, max_tokens=500)
            result = self._parse_json_response(response)

            if isinstance(result, dict):
                is_valid = result.get("equivalent", False)
                confidence = float(result.get("confidence", 0.0))
                reasoning = result.get("reasoning", "")
                logger.info(f"[Sample {sample_idx}] Parallel judge: valid={is_valid}, conf={confidence:.2f}, reasoning={reasoning}")
                return (is_valid, confidence)
        except TruncatedResponseError as e:
            await self._record_truncation(f"parallel judge: {e}", sample_idx)
        except Exception as e:
            logger.debug(f"[Sample {sample_idx}] Parallel validation failed: {e}")

        return (False, 0.0)

    async def generate_and_validate_parallel(
        self,
        user_query: str,
        tool_calls: List[ToolCallInfo],
        final_response: str,
        available_tools: List[str],
        fallback_on_failure: bool = False,
        sample_idx: int = -1,
    ) -> Tuple[Optional[List[Dict[str, str]]], bool]:
        """
        Generate and validate parallel reasoning with retries.

        Args:
            fallback_on_failure: If True, return last generated reasoning even if
                                 validation failed (with is_fallback=True)

        Returns:
            Tuple of (reasoning_steps, is_fallback) where is_fallback indicates
            the reasoning was kept despite judge failure
        """
        last_reasoning = None

        for attempt in range(self.max_retries):
            reasoning = await self.generate_parallel_reasoning(
                user_query=user_query,
                tool_calls=tool_calls,
                final_response=final_response,
                available_tools=available_tools,
                sample_idx=sample_idx,
            )

            if reasoning is None:
                logger.debug(f"[Sample {sample_idx}] Attempt {attempt + 1}: Parallel generation failed")
                continue

            last_reasoning = reasoning  # Save for potential fallback

            is_valid, confidence = await self.validate_parallel_reasoning(
                reasoning_steps=reasoning,
                tool_calls=tool_calls,
                sample_idx=sample_idx,
            )

            if is_valid and confidence >= self.min_confidence:
                logger.debug(f"[Sample {sample_idx}] Attempt {attempt + 1}: Parallel valid (confidence={confidence:.2f})")
                return (reasoning, False)
            else:
                logger.debug(f"[Sample {sample_idx}] Attempt {attempt + 1}: Parallel invalid (valid={is_valid}, conf={confidence:.2f})")

        # All retries exhausted - use fallback if enabled
        if fallback_on_failure and last_reasoning is not None:
            logger.info(f"[Sample {sample_idx}] Using fallback: keeping parallel reasoning despite judge failure")
            return (last_reasoning, True)

        return (None, False)

    # =========================================================================
    # Post-Tool Reasoning (for parallel calls - synthesizes results before final)
    # =========================================================================

    async def generate_post_tool_reasoning(
        self,
        user_query: str,
        tool_calls: List[ToolCallInfo],
        final_response: str,
        sample_idx: int = -1,
    ) -> Optional[List[Dict[str, str]]]:
        """Generate reasoning that synthesizes tool results before final response."""
        tool_results_summary = "\n".join([
            f"  {i+1}. {tc.name}({tc.arguments}) -> {tc.result or 'No result'}"
            for i, tc in enumerate(tool_calls)
        ])

        prompt = REASONING_PROMPT_POST_TOOL.format(
            user_query=user_query,
            tool_results_summary=tool_results_summary,
            final_response=final_response,
            num_tools=len(tool_calls),
        )

        try:
            response = await self._call_llm(prompt, max_tokens=self.MAX_REASONING_TOKENS)
            if not response:
                logger.warning(f"[Sample {sample_idx}] Post-tool reasoning: LLM returned empty response")
                return None
            steps = self._parse_json_response(response)
            if steps is None:
                logger.warning(f"[Sample {sample_idx}] Post-tool reasoning: Failed to parse JSON from response: {response[:500]}...")
                return None
            validated = self._validate_reasoning_steps(steps)
            if validated is None:
                logger.warning(f"[Sample {sample_idx}] Post-tool reasoning: Steps failed validation: {steps}")
                return None
            return validated
        except TruncatedResponseError as e:
            await self._record_truncation(f"post-tool reasoning: {e}", sample_idx)
        except (json.JSONDecodeError, KeyError, TypeError) as e:
            logger.debug(f"[Sample {sample_idx}] Failed to parse post-tool reasoning response: {e}")
        except asyncio.TimeoutError as e:
            logger.warning(f"[Sample {sample_idx}] Post-tool reasoning generation timed out: {e}")
        except Exception as e:
            logger.warning(f"[Sample {sample_idx}] Unexpected error in post-tool reasoning: {e}", exc_info=True)

        return None

    async def validate_post_tool_reasoning(
        self,
        reasoning_steps: List[Dict[str, str]],
        tool_calls: List[ToolCallInfo],
        final_response: str,
        sample_idx: int = -1,
    ) -> Tuple[bool, float]:
        """Validate post-tool reasoning synthesizes results correctly."""
        reasoning_text = "\n".join([
            f"Step {s['step']}: {s['explanation']}\n{s['output']}"
            for s in reasoning_steps
        ])

        tool_results_summary = "\n".join([
            f"  {i+1}. {tc.name}({tc.arguments}) -> {tc.result or 'No result'}"
            for i, tc in enumerate(tool_calls)
        ])

        prompt = JUDGE_PROMPT_POST_TOOL.format(
            reasoning_text=reasoning_text,
            tool_results_summary=tool_results_summary,
            final_response=final_response,
        )

        try:
            response = await self._call_llm(prompt, max_tokens=500)
            result = self._parse_json_response(response)

            if isinstance(result, dict):
                is_valid = result.get("equivalent", False)
                confidence = float(result.get("confidence", 0.0))
                reasoning = result.get("reasoning", "")
                logger.info(f"[Sample {sample_idx}] Post-tool judge: valid={is_valid}, conf={confidence:.2f}, reasoning={reasoning}")
                return (is_valid, confidence)
        except TruncatedResponseError as e:
            await self._record_truncation(f"post-tool judge: {e}", sample_idx)
        except Exception as e:
            logger.debug(f"[Sample {sample_idx}] Post-tool validation failed: {e}")

        return (False, 0.0)

    async def generate_and_validate_post_tool(
        self,
        user_query: str,
        tool_calls: List[ToolCallInfo],
        final_response: str,
        fallback_on_failure: bool = False,
        sample_idx: int = -1,
    ) -> Tuple[Optional[List[Dict[str, str]]], bool]:
        """Generate and validate post-tool reasoning with retries."""
        last_reasoning = None

        for attempt in range(self.max_retries):
            reasoning = await self.generate_post_tool_reasoning(
                user_query=user_query,
                tool_calls=tool_calls,
                final_response=final_response,
                sample_idx=sample_idx,
            )

            if reasoning is None:
                logger.debug(f"[Sample {sample_idx}] Attempt {attempt + 1}: Post-tool generation failed")
                continue

            last_reasoning = reasoning

            is_valid, confidence = await self.validate_post_tool_reasoning(
                reasoning_steps=reasoning,
                tool_calls=tool_calls,
                final_response=final_response,
                sample_idx=sample_idx,
            )

            if is_valid and confidence >= self.min_confidence:
                logger.debug(f"[Sample {sample_idx}] Attempt {attempt + 1}: Post-tool valid (confidence={confidence:.2f})")
                return (reasoning, False)
            else:
                logger.debug(f"[Sample {sample_idx}] Attempt {attempt + 1}: Post-tool invalid (valid={is_valid}, conf={confidence:.2f})")

        if fallback_on_failure and last_reasoning is not None:
            logger.info(f"[Sample {sample_idx}] Using fallback: keeping post-tool reasoning despite judge failure")
            return (last_reasoning, True)

        return (None, False)

    # =========================================================================
    # Synthetic Final Response Generation
    # =========================================================================

    async def generate_synthetic_final(
        self,
        user_query: str,
        tool_calls: List[ToolCallInfo],
        sample_idx: int = -1,
    ) -> Optional[str]:
        """Generate a synthetic final response when dataset lacks one."""
        tool_results_summary = "\n".join([
            f"  {i+1}. {tc.name}({tc.arguments}) -> {tc.result or 'No result'}"
            for i, tc in enumerate(tool_calls)
        ])

        prompt = SYNTHETIC_FINAL_PROMPT.format(
            user_query=user_query,
            tool_results_summary=tool_results_summary,
        )

        try:
            response = await self._call_llm(prompt, max_tokens=2048)
            if not response:
                logger.warning(f"[Sample {sample_idx}] Synthetic final: LLM returned empty response")
                return None
            if response.strip():
                return response.strip()
            logger.warning(f"[Sample {sample_idx}] Synthetic final: LLM returned whitespace-only response")
            return None
        except TruncatedResponseError as e:
            await self._record_truncation(f"synthetic final: {e}", sample_idx)
        except asyncio.TimeoutError as e:
            logger.warning(f"[Sample {sample_idx}] Synthetic final generation timed out: {e}")
        except Exception as e:
            logger.warning(f"[Sample {sample_idx}] Unexpected error in synthetic final generation: {e}", exc_info=True)

        return None

    # =========================================================================
    # Extended Turn Generation (multi-hop conversation extension)
    # =========================================================================

    async def generate_extended_turn(
        self,
        original_question: str,
        tool_calls: List[ToolCallInfo],
        previous_response: str,
        sample_idx: int = -1,
    ) -> Optional[Dict[str, Any]]:
        """
        Generate an extended conversation turn (follow-up + reasoning + final).

        Uses a single LLM call to generate all three components together,
        ensuring coherence and proper context tracking.

        Args:
            original_question: The user's original question
            tool_calls: List of tool calls made in the first turn
            previous_response: The assistant's response to the original question
            sample_idx: Index of the sample for logging

        Returns:
            Dict with {followup_question, reasoning_steps, final_response} or None
        """
        tool_summary = "\n".join([
            f"  - {tc.name}({tc.arguments}) -> {tc.result or 'No result'}"
            for tc in tool_calls
        ])

        prompt = EXTEND_TURN_PROMPT.format(
            original_question=original_question,
            tool_summary=tool_summary,
            previous_response=previous_response,
        )

        try:
            response = await self._call_llm(prompt, max_tokens=self.MAX_REASONING_TOKENS)
            if not response:
                logger.warning(f"[Sample {sample_idx}] Extended turn: LLM returned empty response")
                return None
            result = self._parse_json_response(response)

            if result is None:
                logger.warning(f"[Sample {sample_idx}] Extended turn: Failed to parse JSON from response: {response[:500]}...")
                return None
            if not isinstance(result, dict):
                logger.warning(f"[Sample {sample_idx}] Extended turn: response is not a dict, got {type(result)}")
                return None

            # Validate required fields
            followup = result.get("followup_question", "")
            steps = result.get("reasoning_steps", [])
            final = result.get("final_response", "")

            if not followup or not steps or not final:
                logger.warning(f"[Sample {sample_idx}] Extended turn: missing fields - followup={bool(followup)}, steps={bool(steps)}, final={bool(final)}")
                return None

            # Validate steps structure using shared helper
            valid_steps = self._validate_reasoning_steps(steps)
            if not valid_steps:
                logger.warning(f"[Sample {sample_idx}] Extended turn: no valid reasoning steps")
                return None

            return {
                "followup_question": followup.strip(),
                "reasoning_steps": valid_steps,
                "final_response": final.strip(),
            }

        except TruncatedResponseError as e:
            await self._record_truncation(f"extended turn: {e}", sample_idx)
        except (json.JSONDecodeError, KeyError, TypeError) as e:
            logger.debug(f"[Sample {sample_idx}] Failed to parse extended turn response: {e}")
        except asyncio.TimeoutError as e:
            logger.warning(f"[Sample {sample_idx}] Extended turn generation timed out: {e}")
        except Exception as e:
            logger.warning(f"[Sample {sample_idx}] Unexpected error in extended turn generation: {e}", exc_info=True)
            return None

    async def validate_extended_turn(
        self,
        original_question: str,
        tool_calls: List[ToolCallInfo],
        previous_response: str,
        extended: Dict[str, Any],
        sample_idx: int = -1,
    ) -> Tuple[bool, float]:
        """Validate that the extended turn is coherent and tracks context."""
        tool_summary = "\n".join([
            f"  - {tc.name}({tc.arguments}) -> {tc.result or 'No result'}"
            for tc in tool_calls
        ])

        reasoning_text = "\n".join([
            f"Step {s['step']}: {s['explanation']}\n{s['output']}"
            for s in extended["reasoning_steps"]
        ])

        prompt = JUDGE_PROMPT_EXTEND.format(
            original_question=original_question,
            tool_summary=tool_summary,
            previous_response=previous_response,
            followup_question=extended["followup_question"],
            reasoning_text=reasoning_text,
            final_response=extended["final_response"],
        )

        try:
            response = await self._call_llm(prompt, max_tokens=500)
            result = self._parse_json_response(response)

            if isinstance(result, dict):
                is_valid = result.get("equivalent", False)
                confidence = float(result.get("confidence", 0.0))
                reasoning = result.get("reasoning", "")
                logger.info(f"[Sample {sample_idx}] Extended turn judge: valid={is_valid}, conf={confidence:.2f}, reasoning={reasoning}")
                return (is_valid, confidence)
        except TruncatedResponseError as e:
            await self._record_truncation(f"extended turn judge: {e}", sample_idx)
        except Exception as e:
            logger.debug(f"[Sample {sample_idx}] Extended turn validation failed: {e}")

        return (False, 0.0)

    async def generate_and_validate_extended_turn(
        self,
        original_question: str,
        tool_calls: List[ToolCallInfo],
        previous_response: str,
        fallback_on_failure: bool = False,
        sample_idx: int = -1,
    ) -> Tuple[Optional[Dict[str, Any]], bool]:
        """
        Generate and validate an extended turn with retries.

        Returns:
            Tuple of (extended_turn_dict, is_fallback)
        """
        last_extended = None

        for attempt in range(self.max_retries):
            extended = await self.generate_extended_turn(
                original_question=original_question,
                tool_calls=tool_calls,
                previous_response=previous_response,
                sample_idx=sample_idx,
            )

            if extended is None:
                logger.debug(f"[Sample {sample_idx}] Attempt {attempt + 1}: Extended turn generation failed")
                continue

            last_extended = extended

            is_valid, confidence = await self.validate_extended_turn(
                original_question=original_question,
                tool_calls=tool_calls,
                previous_response=previous_response,
                extended=extended,
                sample_idx=sample_idx,
            )

            if is_valid and confidence >= self.min_confidence:
                logger.debug(f"[Sample {sample_idx}] Attempt {attempt + 1}: Extended turn valid (confidence={confidence:.2f})")
                return (extended, False)
            else:
                logger.debug(f"[Sample {sample_idx}] Attempt {attempt + 1}: Extended turn invalid (valid={is_valid}, conf={confidence:.2f})")

        if fallback_on_failure and last_extended is not None:
            logger.info(f"[Sample {sample_idx}] Using fallback: keeping extended turn despite judge failure")
            return (last_extended, True)

        return (None, False)


# =============================================================================
# Analysis Insertion (using openai_harmony)
# =============================================================================

def flatten_reasoning_steps(steps: List[Dict[str, str]]) -> str:
    """Flatten structured JSON steps to natural text (no step numbers)."""
    flattened = []
    for step in steps:
        step_text = f"{step['explanation']}\n{step['output']}"
        flattened.append(step_text)
    return '\n\n'.join(flattened)


def insert_analysis_messages(
    messages: List[Message],
    parsed: ParsedSample,
    parallel_analysis: Optional[str] = None,
    sequential_analyses: Optional[List[str]] = None,
    post_tool_analysis: Optional[str] = None,
    synthetic_final: Optional[str] = None,
    extended_turn: Optional[Dict[str, Any]] = None,
) -> List[Message]:
    """
    Insert analysis messages before tool calls and optionally after tool responses.

    For parallel: single analysis before first tool call, optional post-tool analysis.
    For sequential: analysis before each tool call.

    Args:
        messages: Original list of Message objects
        parsed: Parsed sample with tool call info
        parallel_analysis: Single analysis text for parallel calls (before tools)
        sequential_analyses: List of analysis texts for sequential calls
        post_tool_analysis: Analysis after tool responses (for parallel calls)
        synthetic_final: Generated final response when dataset lacks one
        extended_turn: Dict with {followup_question, reasoning_steps, final_response} for multi-hop

    Returns:
        New list of messages with analysis inserted
    """
    result = []
    tool_call_positions = {tc.message_index for tc in parsed.tool_calls}

    # Find position of last tool response and final message
    last_tool_response_idx = -1
    final_msg_idx = -1

    for i, msg in enumerate(messages):
        role = msg.author.role if msg.author else None
        if role == Role.TOOL:
            last_tool_response_idx = i
        elif role == Role.ASSISTANT and msg.channel == 'final':
            final_msg_idx = i

    # For parallel: insert one analysis before first tool call
    first_tool_idx = min(tool_call_positions) if tool_call_positions else -1
    parallel_inserted = False
    post_tool_inserted = False

    for i, msg in enumerate(messages):
        # Insert parallel analysis before first tool call
        if parallel_analysis and i == first_tool_idx and not parallel_inserted:
            analysis_msg = Message.from_role_and_content(
                Role.ASSISTANT, parallel_analysis
            ).with_channel('analysis')
            result.append(analysis_msg)
            parallel_inserted = True

        # Insert sequential analysis before each tool call
        if sequential_analyses and i in tool_call_positions:
            # Find which tool call this is
            for j, tc in enumerate(parsed.tool_calls):
                if tc.message_index == i and j < len(sequential_analyses):
                    analysis_msg = Message.from_role_and_content(
                        Role.ASSISTANT, sequential_analyses[j]
                    ).with_channel('analysis')
                    result.append(analysis_msg)
                    break

        result.append(msg)

        # Insert post-tool analysis after last tool response
        if post_tool_analysis and i == last_tool_response_idx and not post_tool_inserted:
            analysis_msg = Message.from_role_and_content(
                Role.ASSISTANT, post_tool_analysis
            ).with_channel('analysis')
            result.append(analysis_msg)
            post_tool_inserted = True

    # Add synthetic final if needed (no existing final in messages)
    if synthetic_final and final_msg_idx == -1:
        final_msg = Message.from_role_and_content(
            Role.ASSISTANT, synthetic_final
        ).with_channel('final')
        result.append(final_msg)

    # Add extended turn (follow-up user message + analysis + final)
    if extended_turn:
        # Fix consecutive user issue: remove ALL trailing USER messages
        # (e.g., acknowledgments like "Thank you") before adding the follow-up
        # question to avoid U,U sequences. Using a while loop handles cases where
        # multiple user messages exist at the end.
        removed_count = 0
        while result:
            last_msg = result[-1]
            # Use getattr for safe attribute access in case of edge cases
            last_role = getattr(last_msg.author, 'role', None) if last_msg.author else None
            if last_role == Role.USER:
                result.pop()
                removed_count += 1
            else:
                break
        if removed_count > 0:
            logger.debug(f"Removed {removed_count} trailing USER message(s) before extended turn")

        # Guard: if all messages were USER messages, we can't extend (extremely rare edge case)
        if not result:
            logger.warning("All messages were USER messages after removal - cannot extend turn")
            return result

        # User's follow-up question
        followup_msg = Message.from_role_and_content(
            Role.USER, extended_turn["followup_question"]
        )
        result.append(followup_msg)

        # Analysis for the follow-up (reasoning about how to use context)
        followup_analysis = flatten_reasoning_steps(extended_turn["reasoning_steps"])
        analysis_msg = Message.from_role_and_content(
            Role.ASSISTANT, followup_analysis
        ).with_channel('analysis')
        result.append(analysis_msg)

        # Final response to the follow-up
        final_msg = Message.from_role_and_content(
            Role.ASSISTANT, extended_turn["final_response"]
        ).with_channel('final')
        result.append(final_msg)

    return result


# =============================================================================
# Main Processing
# =============================================================================

async def process_sample(
    text: str,
    enc: HarmonyEncoding,
    pipeline: SyntheticReasoningPipeline,
    stats: ProcessingStats,
    fallback_on_failure: bool = False,
    extend_turns: bool = False,
    sample_idx: int = -1,
) -> Optional[Tuple[str, bool]]:
    """
    Process a single sample to add synthetic reasoning.

    Features:
    - Pre-tool reasoning (before each tool call)
    - Post-tool reasoning (for parallel calls, after all responses)
    - Synthetic final generation (when dataset lacks final response)
    - Fallback mode (keep reasoning even if judge fails)
    - Extended turns (add a follow-up user question + reasoning + response)

    Returns:
        Tuple of (modified_text, is_fallback) or None if processing fails.
        is_fallback is True if any reasoning was kept despite judge failure.
    """
    await stats.inc(total=1)

    # Parse using openai_harmony
    parsed = parse_harmony_sample(text, enc)
    if parsed is None:
        logger.warning(f"[Sample {sample_idx}] Failed to parse Harmony sample")
        await stats.inc(failed_parse=1)
        return None

    # Skip if no tool calls
    if not parsed.tool_calls:
        await stats.inc(no_tool_calls=1)
        return (text, False)  # Return unchanged, no fallback

    # Count turns in this sample (number of user messages)
    num_turns = sum(1 for msg in parsed.messages if msg.author and msg.author.role == Role.USER)
    is_multi_turn = num_turns > 1

    # Per-sample config: drop previous analysis for multi-turn, preserve for single-turn
    # This ensures single-turn samples learn to generate analysis, while multi-turn
    # matches inference behavior (model doesn't see its own previous analysis)
    sample_config = RenderConversationConfig(auto_drop_analysis=is_multi_turn)

    # Track single-turn samples for stats
    if not is_multi_turn:
        await stats.inc(single_turn_samples=1)

    # Check if we need a synthetic final
    needs_synthetic_final = not parsed.final_response
    synthetic_final = None

    if needs_synthetic_final:
        # Try to generate synthetic final (works best with tool results, but try anyway)
        synthetic_final = await pipeline.generate_synthetic_final(
            user_query=parsed.user_query,
            tool_calls=parsed.tool_calls,
            sample_idx=sample_idx,
        )
        if synthetic_final:
            await stats.inc(synthetic_finals=1)
            # Use the synthetic final for reasoning generation
            parsed.final_response = synthetic_final
        else:
            logger.debug(f"[Sample {sample_idx}] Failed to generate synthetic final, proceeding anyway")

    # Generate reasoning based on parallel vs sequential
    parallel_analysis = None
    sequential_analyses = None
    post_tool_analysis = None
    any_fallback = False

    # OPTIMIZATION: When extend_turns is True AND sample is multi-turn, skip generating
    # tool call analysis because it will be dropped anyway (previous turns' analysis is dropped).
    # Single-turn samples still get their analysis generated normally.
    if extend_turns and is_multi_turn:
        logger.debug("Skipping tool call analysis (multi-turn extend mode, will be dropped)")
        # Skip to extended turn generation below
        pass
    elif parsed.is_parallel and len(parsed.tool_calls) > 1:
        # PARALLEL: Generate single unified reasoning before tools
        logger.debug(f"Processing parallel: {[tc.name for tc in parsed.tool_calls]}")

        reasoning, is_fallback = await pipeline.generate_and_validate_parallel(
            user_query=parsed.user_query,
            tool_calls=parsed.tool_calls,
            final_response=parsed.final_response,
            available_tools=parsed.available_tools,
            fallback_on_failure=fallback_on_failure,
            sample_idx=sample_idx,
        )

        if reasoning is None:
            logger.warning(f"[Sample {sample_idx}] Parallel reasoning generation failed")
            await stats.inc(failed_generate=1)
            return None

        if is_fallback:
            any_fallback = True

        parallel_analysis = flatten_reasoning_steps(reasoning)

        # PARALLEL: Also generate post-tool reasoning (after responses, before final)
        # Only if we have tool results to synthesize
        has_results = any(tc.result for tc in parsed.tool_calls)
        if has_results and parsed.final_response:
            post_reasoning, post_fallback = await pipeline.generate_and_validate_post_tool(
                user_query=parsed.user_query,
                tool_calls=parsed.tool_calls,
                final_response=parsed.final_response,
                fallback_on_failure=fallback_on_failure,
                sample_idx=sample_idx,
            )

            if post_reasoning:
                post_tool_analysis = flatten_reasoning_steps(post_reasoning)
                await stats.inc(post_tool_reasoning=1)
                if post_fallback:
                    any_fallback = True
            else:
                logger.debug(f"[Sample {sample_idx}] Post-tool reasoning failed, continuing without it")

        await stats.inc(parallel_samples=1)

    else:
        # SEQUENTIAL: Generate reasoning for each tool call
        sequential_analyses = []

        for i, tool_call in enumerate(parsed.tool_calls):
            prior_calls = parsed.tool_calls[:i] if i > 0 else None

            reasoning, is_fallback = await pipeline.generate_and_validate_sequential(
                user_query=parsed.user_query,
                tool_call=tool_call,
                final_response=parsed.final_response,
                available_tools=parsed.available_tools,
                prior_tool_calls=prior_calls,
                fallback_on_failure=fallback_on_failure,
                sample_idx=sample_idx,
            )

            if reasoning is None:
                logger.warning(f"[Sample {sample_idx}] Sequential reasoning generation failed for tool {i+1}/{len(parsed.tool_calls)}: {tool_call.name}")
                await stats.inc(failed_generate=1)
                return None

            if is_fallback:
                any_fallback = True

            sequential_analyses.append(flatten_reasoning_steps(reasoning))

        # SEQUENTIAL with synthetic final: Also need post-tool reasoning
        # This explains how tool results lead to the (synthetic) final response
        if synthetic_final:
            has_results = any(tc.result for tc in parsed.tool_calls)
            if has_results:
                post_reasoning, post_fallback = await pipeline.generate_and_validate_post_tool(
                    user_query=parsed.user_query,
                    tool_calls=parsed.tool_calls,
                    final_response=synthetic_final,  # Use the synthetic final we generated
                    fallback_on_failure=fallback_on_failure,
                    sample_idx=sample_idx,
                )

                if post_reasoning:
                    post_tool_analysis = flatten_reasoning_steps(post_reasoning)
                    await stats.inc(post_tool_reasoning=1)
                    if post_fallback:
                        any_fallback = True
                else:
                    logger.debug(f"[Sample {sample_idx}] Post-tool reasoning for synthetic final failed, continuing without it")

        await stats.inc(sequential_samples=1)

    # Track fallback usage
    if any_fallback:
        await stats.inc(judge_failures_kept=1)

    # Generate extended turn if requested (only for multi-turn samples)
    extended_turn = None
    if extend_turns and is_multi_turn:
        # For multi-turn + extend_turns, we MUST produce an extended turn or fail
        # (we skipped analysis generation as an optimization, so we need the extended turn)
        if not parsed.final_response:
            logger.warning(f"[Sample {sample_idx}] Multi-turn extend mode: no final_response, skipping")
            await stats.inc(extended_turn_failures=1)
            return None

        # Need a final response to extend from (either original or synthetic)
        final_to_extend = synthetic_final if synthetic_final else parsed.final_response

        extended, ext_fallback = await pipeline.generate_and_validate_extended_turn(
            original_question=parsed.user_query,
            tool_calls=parsed.tool_calls,
            previous_response=final_to_extend,
            fallback_on_failure=fallback_on_failure,
            sample_idx=sample_idx,
        )

        if extended:
            extended_turn = extended
            await stats.inc(extended_turns=1)
            if ext_fallback:
                any_fallback = True
        else:
            await stats.inc(extended_turn_failures=1)
            logger.warning(f"[Sample {sample_idx}] Extended turn generation failed, skipping")
            return None  # Skip samples where extended turn fails

    # Insert analysis messages
    try:
        new_messages = insert_analysis_messages(
            messages=parsed.messages,
            parsed=parsed,
            parallel_analysis=parallel_analysis,
            sequential_analyses=sequential_analyses,
            post_tool_analysis=post_tool_analysis,
            synthetic_final=synthetic_final,
            extended_turn=extended_turn,
        )

        # Re-render using openai_harmony with per-sample config
        conv = Conversation.from_messages(new_messages)
        rendered_tokens = enc.render_conversation_for_training(conv, sample_config)
        new_text = enc.decode(rendered_tokens)

        await stats.inc(successful=1)
        return (new_text, any_fallback)

    except Exception as e:
        logger.warning(f"[Sample {sample_idx}] Failed to render: {e}")
        await stats.inc(failed_validate=1)
        return None


async def process_file(
    input_path: Path,
    output_path: Optional[Path],
    pipeline: SyntheticReasoningPipeline,
    limit: Optional[int] = None,
    dry_run: bool = False,
    concurrency: int = 10,
    fallback_on_failure: bool = False,
    extend_turns: bool = False,
) -> ProcessingStats:
    """
    Process all samples in a JSONL file.

    Args:
        input_path: Path to input JSONL
        output_path: Path to output JSONL
        pipeline: SyntheticReasoningPipeline instance
        limit: Max samples to process
        dry_run: If True, don't write output
        concurrency: Number of concurrent API calls
        fallback_on_failure: If True, keep reasoning even when judge fails
        extend_turns: If True, add a follow-up user turn with reasoning
    """
    enc = load_harmony_encoding(HarmonyEncodingName.HARMONY_GPT_OSS)
    stats = ProcessingStats()
    output_records = []

    # Read input
    logger.info(f"Reading from {input_path}")
    with open(input_path, 'r', encoding='utf-8') as f:
        samples = [json.loads(line) for line in f if line.strip()]

    if limit:
        samples = samples[:limit]

    logger.info(f"Processing {len(samples)} samples with concurrency={concurrency}")
    if fallback_on_failure:
        logger.info("Fallback mode enabled: keeping reasoning on judge failure")
    if extend_turns:
        logger.info("Extended turns mode enabled: adding follow-up user turns (multi-turn only)")
    logger.info("auto_drop_analysis is set per-sample: True for multi-turn, False for single-turn")

    # Process with semaphore
    semaphore = asyncio.Semaphore(concurrency)

    total_samples = len(samples)
    progress_interval = max(1, total_samples // 20)  # Log ~20 times during processing
    completed_count = 0
    completed_lock = asyncio.Lock()

    async def process_with_sem(sample: Dict, idx: int) -> Optional[Tuple[str, bool]]:
        nonlocal completed_count
        async with semaphore:
            text = sample.get("text", "")
            result = await process_sample(
                text, enc, pipeline, stats,
                fallback_on_failure=fallback_on_failure,
                extend_turns=extend_turns,
                sample_idx=idx,
            )
            # Track completions and log progress
            async with completed_lock:
                completed_count += 1
                # Log first 3 completions, then every progress_interval
                if completed_count <= 3 or completed_count % progress_interval == 0 or completed_count == total_samples:
                    logger.info(f"Progress: {completed_count}/{total_samples} completed ({100*completed_count/total_samples:.1f}%)")
            return result

    # Process all
    logger.info(f"Starting {total_samples} samples with concurrency={concurrency} (first progress after ~{concurrency} API calls complete)...")
    tasks = [process_with_sem(s, i) for i, s in enumerate(samples)]
    results = await asyncio.gather(*tasks)

    # Collect results with fallback tracking
    for i, result in enumerate(results):
        if result is not None:
            text, is_fallback = result
            output_records.append({"text": text, "is_fallback": is_fallback})

    # Copy truncation count from pipeline to stats (pipeline tracks this internally)
    stats.api_truncations = pipeline.truncation_count

    # Output
    if dry_run:
        logger.info(f"[DRY RUN] Would write {len(output_records)} records")
        # Print first few samples using openai_harmony parsing
        for i, record in enumerate(output_records[:3]):
            logger.info(f"\n=== Sample {i + 1} ===")
            text = record["text"]

            try:
                # Parse using openai_harmony instead of regex
                tokens = enc.encode(text, allowed_special='all')
                messages = enc.parse_messages_from_completion_tokens(tokens, strict=False)

                analysis_count = 0
                for msg in messages:
                    role = msg.author.role if msg.author else None

                    # Show analysis blocks
                    if role == Role.ASSISTANT and msg.channel == 'analysis':
                        analysis_count += 1
                        if msg.content and hasattr(msg.content[0], 'text'):
                            content = msg.content[0].text
                            preview = content[:400] + "..." if len(content) > 400 else content
                            logger.info(f"Analysis block {analysis_count}:\n{preview}")

                    # Show final response
                    elif role == Role.ASSISTANT and msg.channel == 'final':
                        if msg.content and hasattr(msg.content[0], 'text'):
                            content = msg.content[0].text
                            preview = content[:300] + "..." if len(content) > 300 else content
                            logger.info(f"Final response: {preview}")

            except Exception as e:
                logger.warning(f"Failed to parse sample {i + 1} for display: {e}")
    else:
        logger.info(f"Writing {len(output_records)} records to {output_path}")
        with open(output_path, 'w', encoding='utf-8') as f:
            for record in output_records:
                f.write(json.dumps(record, ensure_ascii=False) + '\n')
        logger.info(f"Done! Output written to {output_path}")

    return stats


# =============================================================================
# CLI
# =============================================================================

def main():
    parser = argparse.ArgumentParser(
        description="Add synthetic reasoning to Hermes tool-calling training data",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Features:
  - Pre-tool reasoning: Explains WHY each tool was selected
  - Post-tool reasoning: Synthesizes parallel tool results before final response
  - Synthetic finals: Generates final response when dataset lacks one
  - Judge validation: Validates reasoning coherence with LLM judge
  - Fallback mode: Keeps reasoning even if judge fails (use with --fallback)
  - Extended turns: Adds a follow-up user question with analysis (use with --extend-turns)

Examples:
  # Basic usage
  python scripts/add_synthetic_reasoning.py -i hermes.jsonl -o hermes_with_reasoning.jsonl

  # Dry run with limit
  python scripts/add_synthetic_reasoning.py -i hermes.jsonl --dry-run --limit 5

  # Enable fallback mode (keep reasoning even if judge fails)
  python scripts/add_synthetic_reasoning.py -i hermes.jsonl -o output.jsonl --fallback

  # Add multi-hop follow-up turns
  python scripts/add_synthetic_reasoning.py -i hermes.jsonl -o output.jsonl --extend-turns

  # Use OpenAI instead of Claude
  python scripts/add_synthetic_reasoning.py -i hermes.jsonl -o output.jsonl --provider openai
"""
    )
    parser.add_argument("--input", "-i", type=str, required=True,
                        help="Input JSONL file with Harmony-encoded samples")
    parser.add_argument("--output", "-o", type=str, default="hermes_with_reasoning.jsonl",
                        help="Output JSONL file path")
    parser.add_argument("--provider", type=str, default="claude", choices=["claude", "openai"],
                        help="LLM provider for reasoning generation")
    parser.add_argument("--model", type=str, default=None,
                        help="Model name override (default: sonnet-4.5 for Claude, gpt-4o for OpenAI)")
    parser.add_argument("--min-confidence", type=float, default=0.7,
                        help="Minimum judge confidence threshold (default: 0.7)")
    parser.add_argument("--max-retries", type=int, default=2,
                        help="Max retry attempts per reasoning generation (default: 2)")
    parser.add_argument("--concurrency", type=int, default=10,
                        help="Number of concurrent API calls (default: 10)")
    parser.add_argument("--limit", type=int, default=None,
                        help="Max samples to process (for testing)")
    parser.add_argument("--dry-run", action="store_true",
                        help="Process samples but don't write output file")
    parser.add_argument("--fallback", action="store_true",
                        help="Keep reasoning even when judge validation fails")
    parser.add_argument("--extend-turns", action="store_true",
                        help="Add a follow-up user turn with analysis for multi-hop reasoning")
    parser.add_argument("--verbose", "-v", action="store_true",
                        help="Enable debug logging")
    parser.add_argument("--validate", type=str, default=None, metavar="FILE",
                        help="Validate an output JSONL file for consecutive USER messages")

    args = parser.parse_args()

    # Handle validation mode
    if args.validate:
        validate_output_file(Path(args.validate))
        return

    if args.verbose:
        logging.getLogger().setLevel(logging.DEBUG)

    pipeline = SyntheticReasoningPipeline(
        provider=args.provider,
        model=args.model,
        min_confidence=args.min_confidence,
        max_retries=args.max_retries,
    )

    input_path = Path(args.input)
    output_path = Path(args.output) if not args.dry_run else None

    if not input_path.exists():
        logger.error(f"Input file not found: {input_path}")
        sys.exit(1)

    stats = asyncio.run(process_file(
        input_path=input_path,
        output_path=output_path,
        pipeline=pipeline,
        limit=args.limit,
        dry_run=args.dry_run,
        concurrency=args.concurrency,
        fallback_on_failure=args.fallback,
        extend_turns=args.extend_turns,
    ))

    logger.info(stats.report())


def validate_conversation_flow(text: str, enc: HarmonyEncoding) -> Tuple[bool, str]:
    """
    Validate that a conversation has no consecutive USER messages.

    Args:
        text: Harmony-encoded conversation text
        enc: Harmony encoding object

    Returns:
        Tuple of (is_valid, flow_string) where flow_string shows the message pattern
    """
    try:
        tokens = enc.encode(text, allowed_special='all')
        messages = enc.parse_messages_from_completion_tokens(tokens, strict=False)

        # Build role flow string
        roles = []
        for msg in messages:
            if msg.author:
                role = msg.author.role
                if role == Role.USER:
                    roles.append('U')
                elif role == Role.ASSISTANT:
                    channel = msg.channel or 'default'
                    if channel == 'analysis':
                        roles.append('Aa')
                    elif channel == 'final':
                        roles.append('Af')
                    elif msg.recipient:  # Tool call
                        roles.append('At')
                    else:
                        roles.append('A')
                elif role == Role.TOOL:
                    roles.append('T')
                elif role == Role.DEVELOPER:
                    roles.append('D')
                else:
                    roles.append('?')

        flow = ','.join(roles)

        # Check for consecutive USER messages
        is_valid = ',U,U,' not in f',{flow},'
        return (is_valid, flow)

    except Exception as e:
        return (False, f"PARSE_ERROR: {e}")


def validate_output_file(output_path: Path) -> None:
    """
    Validate an output JSONL file for consecutive user messages.

    Usage:
        python scripts/add_synthetic_reasoning.py --validate hermes_extended_turns.jsonl
    """
    enc = load_harmony_encoding(HarmonyEncodingName.HARMONY_GPT_OSS)

    with open(output_path, 'r', encoding='utf-8') as f:
        samples = [json.loads(line) for line in f if line.strip()]

    consecutive_user_count = 0
    invalid_samples = []

    for idx, sample in enumerate(samples):
        text = sample.get("text", "")
        is_valid, flow = validate_conversation_flow(text, enc)

        if not is_valid:
            consecutive_user_count += 1
            invalid_samples.append((idx, flow))

    logger.info(f"=== Validation Results for {output_path} ===")
    logger.info(f"Total samples: {len(samples)}")
    logger.info(f"Samples with consecutive USER messages: {consecutive_user_count}")

    if consecutive_user_count > 0:
        logger.warning("Invalid samples (first 10):")
        for idx, flow in invalid_samples[:10]:
            logger.warning(f"  idx={idx}: {flow[:100]}...")
    else:
        logger.info("✓ All samples have valid conversation flow!")


if __name__ == "__main__":
    main()
