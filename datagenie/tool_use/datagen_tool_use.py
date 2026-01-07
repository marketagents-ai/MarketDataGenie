"""
Tool-Use Data Generation Script using Minference
================================================

This script loads the `interstellarninja/toolace_hermes_sequential_tool_use` dataset,
extracts conversation prefixes where the model is expected to generate a turn
(tool call or response), and runs parallel inference using `minference.lite`.

It uses "Teacher Forcing" for multi-turn conversations:
- Past history is taken from the Ground Truth dataset (including tool results).
- The model generates the next turn.
- We do NOT execute tools live; we rely on the dataset annotations.

Usage:
    python datagenie/tool_use/datagen_tool_use.py --limit 10 --batch_size 8
"""

import re
import sys
import json
import asyncio
import argparse
import time
import os
from pathlib import Path
from typing import List, Dict, Any, Optional
from dataclasses import dataclass, field

from datasets import load_dataset


def _check_sequential_tools(conv: List[Dict[str, str]]) -> bool:
    """
    Return True when every assistant tool‑calling turn is followed only by the
    corresponding <tool_response> messages from the system before the next
    assistant <tool_call>. Allow concluding narration after all tool calls are done.
    """
    tool_indices = [
        i
        for i, m in enumerate(conv)
        if m["from"] in ("gpt", "assistant") and "<tool_call>" in m["value"].lower()
    ]

    # No tool calls at all
    if not tool_indices:
        return False

    # Check sequences between tool calls
    for i in range(len(tool_indices) - 1):
        start, end = tool_indices[i], tool_indices[i + 1]
        # Messages strictly between two tool‑calling turns
        in_between = conv[start + 1 : end]
        # Only <tool_response> allowed between tool calls
        if any(m["from"] != "tool" for m in in_between):
            return False

    # For the last tool call, only check up to the next tool response
    # (allow narration after that)
    last_tool_idx = tool_indices[-1]
    next_responses = [
        i
        for i, m in enumerate(conv[last_tool_idx + 1 :], start=last_tool_idx + 1)
        if m["from"] == "tool"
    ]
    if not next_responses:  # No tool response after last tool call
        return False

    # Check sequence after last tool call up to its response
    last_response_idx = next_responses[0]
    in_between = conv[last_tool_idx + 1 : last_response_idx + 1]
    if any(m["from"] != "tool" for m in in_between):
        return False

    return True


# ---------------------------------------------------------------------------
# Validation helpers (ported from tool_use_multiturn_server.py)
# ---------------------------------------------------------------------------
def _normalize_tool_call_json(txt: str) -> str:
    """
    Normalise assistant replies so that:
      • the original <think> … </think> block is preserved
      • every <tool_call> … </tool_call> block is converted to
        canonical JSON (double‑quoted, valid JSON) even if the
        model used Python literal formatting.
    """
    import ast
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


def _canonicalise_tool_json(raw: str) -> Optional[str]:
    """
    Try to parse raw as JSON, then as Python literal, and return canonical json.dumps.
    """
    import ast
    try:
        obj = json.loads(raw)
        return json.dumps(obj, separators=(",", ":"))
    except Exception:
        pass
    try:
        obj = ast.literal_eval(raw)
        return json.dumps(obj, separators=(",", ":"))
    except Exception:
        return None


def _validate_think_only(txt: str) -> bool:
    """
    A narration / summary turn must:
    • start with exactly one <think> … </think> block
    • contain **no** <tool_call> tags anywhere
    """
    txt = _normalize_tool_call_json(txt)
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


def _validate_think_plus_calls(txt: str) -> Optional[List[dict]]:
    """
    Validate an assistant reply that *must* contain exactly one <think>…</think>
    block followed by one **or more** <tool_call>…</tool_call> blocks.
    Returns the parsed list of tool‑call JSON objects on success, otherwise None.
    """
    txt = _normalize_tool_call_json(txt)

    if re.search(r"<tool_response\s*>", txt, flags=re.IGNORECASE):
        return None

    m = re.match(r"\s*(<think>[\s\S]*?</think>)", txt, flags=re.IGNORECASE)
    if not m:
        return None
    think_block = m.group(1)
    rest = txt[len(think_block):]

    tc_pattern = r"\s*<tool_call>\s*([\s\S]*?)\s*</tool_call>\s*"

    tool_calls = []
    while True:
        m_tc = re.match(tc_pattern, rest, flags=re.IGNORECASE)
        if not m_tc:
            break
        raw_json = m_tc.group(1)
        canon = _canonicalise_tool_json(raw_json)
        if canon is None:
            return None
        tool_calls.append(json.loads(canon))
        rest = rest[m_tc.end():]

    if not tool_calls or rest.strip():
        return None
    return tool_calls


def _validate_reply_and_extract(txt: str, require_think: bool = True) -> Optional[List[dict]]:
    """
    Unified validator:
    - If the reply contains <tool_call>, validate & return tool calls.
    - If it has no <tool_call>, require a single narration-only <think> block and return [].
    - Return None on validation failure.
    """
    if re.search(r"<tool_call\s*>", txt, flags=re.IGNORECASE):
        return _validate_think_plus_calls(txt)
    # narration-only case
    if require_think:
        return [] if _validate_think_only(txt) else None
    return []  # No validation needed if require_think=False
from tqdm.asyncio import tqdm
from dotenv import load_dotenv

load_dotenv()


# Minference imports
from minference.lite.models import (
    LLMConfig,
    LLMClient,
    ResponseFormat,
    ChatThread,
    ChatMessage,
    MessageRole,
    SystemPrompt,
    ProcessedOutput
)
from minference.lite.inference import InferenceOrchestrator
from minference.enregistry import EntityRegistry

# Initialize EntityRegistry (Singleton) to setup logging
EntityRegistry()


from pydantic import BaseModel, Field


class ToolUseEnvConfig(BaseModel):
    model: str = Field(default="Hermes-4-70B", description="Model ID")
    dataset_name: str = Field(default="interstellarninja/hermes_reasoning_tool_use", description="Dataset path")
    split: str = Field(default="train", description="Dataset split")
    output_dir: str = Field(default="outputs/tool_use", description="Output directory")
    batch_size: int = Field(default=8, description="Batch size")
    max_tokens: int = Field(default=4096, description="Max tokens")
    temperature: float = Field(default=0.4, description="Temperature")
    limit: Optional[int] = Field(default=None, description="Limit items")
    start_offset: int = Field(default=0, description="Start offset")
    # Validation options
    validate_think_blocks: bool = Field(default=True, description="Validate <think> blocks in generations")
    require_think_for_augmentation: bool = Field(default=False, description="When augmenting existing data, require <think> generation")
    # Output options  
    output_sharegpt: bool = Field(default=True, description="Output in ShareGPT format")


class ToolUsePipeline:
    def __init__(self, config: ToolUseEnvConfig, scenario: str = "sequential"):
        self.config = config
        self.scenario = scenario
        self.orchestrator = InferenceOrchestrator()
        
        # Ensure output directory exists
        path = Path(self.config.output_dir)
        path.mkdir(parents=True, exist_ok=True)
        
        # Output files
        ts = int(time.time())
        self.output_file = path / f"tool_use_results_{self.scenario}_{ts}.jsonl"
        self.sharegpt_file = path / f"tool_use_sharegpt_{self.scenario}_{ts}.jsonl"
        
        # Stats tracking
        self.valid_count = 0
        self.invalid_count = 0

    def _to_sharegpt_conversation(
        self, 
        system_prompt: str,
        history: List[ChatMessage],
        model_output: str
    ) -> List[Dict[str, str]]:
        """
        Convert history + model output to ShareGPT format conversation.
        """
        conversation = []
        
        # Add system message
        if system_prompt:
            conversation.append({"from": "system", "value": system_prompt})
        
        # Add history messages
        for msg in history:
            role_map = {
                MessageRole.user: "human",
                MessageRole.assistant: "gpt", 
                MessageRole.tool: "tool",
                MessageRole.system: "system"
            }
            sharegpt_role = role_map.get(msg.role, "human")
            conversation.append({"from": sharegpt_role, "value": msg.content})
        
        # Add model output as final gpt turn
        conversation.append({"from": "gpt", "value": model_output})
        
        return conversation

    def _validate_generation(
        self, 
        output: str, 
        ground_truth: str,
        expects_tool_call: bool = True
    ) -> tuple[bool, Optional[List[dict]], str]:
        """
        Validate that the generation meets quality criteria.
        
        Returns:
            (is_valid, extracted_tool_calls, reason)
        """
        if not output or not output.strip():
            return False, None, "empty_output"
            
        # Validate <think> block if required
        if self.config.validate_think_blocks:
            result = _validate_reply_and_extract(output, require_think=True)
            if result is None:
                return False, None, "invalid_format"
        else:
            # Still extract tool calls for comparison
            result = _validate_reply_and_extract(output, require_think=False)
            if result is None:
                result = []
        
        generated_calls = result
        
        # Extract ground truth tool calls for comparison
        gt_calls = _validate_think_plus_calls(ground_truth) if "<tool_call>" in ground_truth.lower() else []
        
        # For tool-calling turns, validate tool calls match
        if expects_tool_call:
            if len(generated_calls) == 0:
                return False, generated_calls, "missing_tool_calls"
            
            # Compare tool call names (order-sensitive)
            gen_names = [c.get("name", "") for c in generated_calls]
            gt_names = [c.get("name", "") for c in (gt_calls or [])]
            
            if gen_names != gt_names:
                return False, generated_calls, f"tool_mismatch: expected {gt_names}, got {gen_names}"
        else:
            # Narration turn - should NOT have tool calls
            if len(generated_calls) > 0:
                return False, generated_calls, "unexpected_tool_calls"
        
        return True, generated_calls, "valid"

    def save_sharegpt_result(self, conversation: List[Dict[str, str]], metadata: Optional[Dict] = None):
        """Save a validated conversation in ShareGPT format."""
        item = {"conversations": conversation}
        if metadata:
            item["metadata"] = metadata
        with open(self.sharegpt_file, "a", encoding="utf-8") as f:
            f.write(json.dumps(item, ensure_ascii=False) + "\n")

    def validate_scenario(self, conv: List[Dict[str, Any]]) -> bool:
        """
        Validate if the conversation matches the requested scenario.
        Logic adapted from ToolUseMultiturnEnv.
        """
        # Basic validation
        if len(conv) < 2:
            return False
            
        tool_indices = [
            i
            for i, m in enumerate(conv)
            if m["from"] in ("gpt", "assistant")
            and "<tool_call>" in m["value"].lower()
        ]
        
        target = self.scenario

        # === RELEVANCE ===
        if target == "relevance":
            # Relevance training requires an existing *non‑tool* assistant message
            first_asst_idx = next(
                (i for i, m in enumerate(conv) if m["from"] in ("gpt", "assistant")),
                None
            )
            # Skip conversations lacking the first GPT reply or if first reply calls a tool
            if first_asst_idx is None:
                return False
                
            asst_msg = conv[first_asst_idx]["value"]
            if "<tool_call" in asst_msg.lower():
                return False 
            
            # No tool calls allowed in the entire conversation/prefix we are interested in
            if any("<tool_call" in m["value"].lower() for m in conv if m["from"] in ("gpt", "assistant")):
                return False
                
            return True

        # For other scenarios, we NEED tool calls
        if not tool_indices:
            return False

        human_after_first_tool = any(
            i > tool_indices[0] and m["from"] == "human" for i, m in enumerate(conv)
        )

        # === SINGLE ===
        if target == "single":
            # For single-turn, we accept only EXACTLY one tool calling turn in the prefix?
            # Or just filter datasets? Reference says `len(tool_indices) == 1`.
            return len(tool_indices) == 1

        # === MULTISTEP (aka Sequential) ===
        if target in ("multistep", "sequential"):
            # Must have >= 2 tool calls
            # First assistant must be tool call
            # No human interruption
            # Sequential pattern
            if len(tool_indices) >= 2:
                first_assistant_idx = next(
                    (i for i, m in enumerate(conv) if m["from"] in ("gpt", "assistant")),
                    None
                )
                
                if (
                    first_assistant_idx == tool_indices[0]
                    and not human_after_first_tool
                    and _check_sequential_tools(conv)
                ):
                    return True
            return False

        # === MULTITURN ===
        if target == "multiturn":
            # >= 2 tool calls
            # First assistant is tool call
            # Human interruption IS required
            if len(tool_indices) >= 2:
                first_assistant_idx = next(
                    (i for i, m in enumerate(conv) if m["from"] in ("gpt", "assistant")),
                    None
                )
                if first_assistant_idx != tool_indices[0]:
                    return False
                    
                if not human_after_first_tool:
                    return False
                
                return True

        return False

    def load_items(self) -> List[Dict[str, Any]]:
        """
        Load dataset and prepare items.
        Each item is a conversation prefix ending just before a GPT turn.
        """
        print(f"Loading dataset {self.config.dataset_name}...")
        try:
            ds = load_dataset(self.config.dataset_name, split=self.config.split)
        except Exception as e:
            print(f"Error loading dataset: {e}")
            return []

        items = []
        
        # Iterate through dataset rows
        iterable = ds
        # Note: Pagination logic removed for simplicity as we need to filter first. 
        # But we still respect config.limit after filtering.
        
        count = 0
        for row in iterable:
            if self.config.limit and count >= self.config.limit:
                break
                
            conv = row.get("conversations", [])
            if not conv:
                continue

            # Check scenario category using logic from ValidateScenario
            if not self.validate_scenario(conv):
                continue
            
            count += 1
            
            # Identify GPT turns to target
            # Strategy: For every "gpt" turn in the dataset, we create a test case 
            # where context = all previous messages.
            # We skip system prompts if they are redundant or handle them carefully.
            
            system_msg = None
            history: List[ChatMessage] = []
            
            for i, msg in enumerate(conv):
                role = msg.get("from")
                content = msg.get("value")
                
                if role == "system":
                    system_msg = content
                    continue
                
                if role == "gpt":
                    # This is a target generation point. 
                    # We want to generate THIS turn given history.
                    # Check if this specific turn contains tool_call
                    turn_has_tool_call = "<tool_call>" in content.lower()
                    # Check if full conversation has any tool calls
                    conv_has_tool_calls = any(
                        "<tool_call>" in m.get("value", "").lower() 
                        for m in conv if m.get("from") in ("gpt", "assistant")
                    )
                    items.append({
                        "id": f"{id(row)}_{i}",  # Unique-ish ID
                        "system_prompt": system_msg,
                        "history": list(history), # Deep copy of current history
                        "ground_truth": content,
                        "turn_expects_tool_call": turn_has_tool_call,  # This specific turn
                        "conv_has_tool_calls": conv_has_tool_calls,    # Full conversation
                        "scenario_detected": self.scenario,
                        "original_row": row # Keep metadata if needed
                    })
                
                # Append to history for NEXT turns
                minf_role = MessageRole.user if role == "human" else \
                           MessageRole.assistant if role == "gpt" else \
                           MessageRole.tool if role == "tool" else MessageRole.user
                
                msg_kwargs = {"role": minf_role, "content": content}
                
                if minf_role == MessageRole.tool:
                    # Tool messages MUST have a tool_call_id
                    # We'll generate a consistent one based on index if implicit
                    # or use a placeholder.
                    msg_kwargs["oai_tool_call_id"] = f"call_{i}"
                    msg_kwargs["tool_name"] = "unknown_tool" # fallback
                
                history.append(ChatMessage(**msg_kwargs))

        print(f"Prepared {len(items)} generation targets from dataset (filtered by {self.scenario}).")
        return items

    async def process_batch(self, batch: List[Dict[str, Any]]):
        """
        Run inference for a batch of items.
        """
        threads = []
        
        for item in batch:
            # Construct ChatThread
            sys_content = item.get("system_prompt") or (
                "You are a deep thinking AI, you may use extremely long chains of thought to deeply "
                "consider the problem and deliberate with yourself via systematic reasoning processes "
                "to help come to a correct solution prior to answering. You should enclose your thoughts "
                "and internal monologue inside <think> </think> tags, and then provide your solution "
                "or response to the problem."
            )
            sys_prompt = SystemPrompt(name="sys", content=sys_content)
            
            llm_config = LLMConfig(
                client=LLMClient.litellm,
                model=self.config.model,
                temperature=self.config.temperature,
                response_format=ResponseFormat.text,
                max_tokens=self.config.max_tokens
            )
            
            ct = ChatThread(
                name="tool-gen",
                system_prompt=sys_prompt,
                llm_config=llm_config,
                history=item["history"], # Pre-filled history
                new_message=None # No new user message, we are continuing from history
            )
            
            # If the last message in history was USER or TOOL, we are ready to generate ASSISTANT.
            # Minference Orchestrator usually expects `new_message` to be the user prompt.
            # HACK: If we just want to continue generation, we might need to handle it.
            # Looking at minference implementation, if new_message is None, it might fail or do nothing depending on logic.
            # However, `process_turn` usually appends new_message. 
            # We should verify if minference supports "continue generation" without new_message easily.
            # Assuming `run_parallel_ai_completion` handles it if we pass the thread state.
            # Actually, standard ChatThread logic appends `new_message` if present.
            # If we want to generate immediately based on history, we leave `new_message` None, 
            # ensuring the last message in `history` is indeed the trigger (Human or Tool).
            
            threads.append(ct)

        # Run parallel inference
        # Note: If minference requires a non-None new_message, we might need to pop the last message from history 
        # and use it as `new_message`. Let's do that to be safe and consistent with standard flows.
        
        valid_threads = []
        valid_items = []
        
        for i, ct in enumerate(threads):
            if ct.history:
                last_msg = ct.history[-1]
                # We can't easily "pop" from the typed list if it's strictly managed, but ChatThread.history is usually a list.
                # We simply set new_message = last_msg.content and remove it from history
                # But we need to preserve the role. Minference `new_message` is usually assumed User.
                # If the last message was TOOL, we need to be careful.
                # In `minference`, `add_user_message` handles appending.
                # Let's try running as-is (with None) first. If models.py implies it just sends messages to LLM, it should work.
                pass
            valid_threads.append(ct)
            valid_items.append(batch[i])

        try:
            outputs = await self.orchestrator.run_parallel_ai_completion(valid_threads)
            print(f"Orchestrator returned {len(outputs) if outputs else 0} outputs")
        except Exception as e:
            print(f"Batch execution failed: {e}")
            return

        if not outputs:
            print("No outputs received from orchestrator.")
            return

        # Process outputs
        for item, output in zip(valid_items, outputs):
            model_output = output.content if output else ""
            
            if not model_output:
                print(f"Output for item {item['id']} is None/Empty")
                self.invalid_count += 1
                continue
            
            # Use pre-computed flags from load_items
            expects_tool_call = item.get("turn_expects_tool_call", False)
            conv_has_tool_calls = item.get("conv_has_tool_calls", False)
            ground_truth = item.get("ground_truth", "")
            
            # Validate the generation with ground truth matching
            is_valid, extracted_calls, reason = self._validate_generation(
                model_output, 
                ground_truth=ground_truth,
                expects_tool_call=expects_tool_call
            )
            
            if is_valid:
                self.valid_count += 1
                
                # Build ShareGPT conversation
                sys_content = item.get("system_prompt") or (
                    "You are a deep thinking AI, you may use extremely long chains of thought to deeply "
                    "consider the problem and deliberate with yourself via systematic reasoning processes "
                    "to help come to a correct solution prior to answering. You should enclose your thoughts "
                    "and internal monologue inside <think> </think> tags, and then provide your solution "
                    "or response to the problem."
                )
                
                conversation = self._to_sharegpt_conversation(
                    system_prompt=sys_content,
                    history=item["history"],
                    model_output=model_output
                )
                
                # Save in ShareGPT format with accurate metadata
                metadata = {
                    "scenario": self.scenario,
                    "turn_expects_tool_call": expects_tool_call,
                    "conv_has_tool_calls": conv_has_tool_calls,
                    "tool_calls_matched": True,
                    "num_tool_calls": len(extracted_calls) if extracted_calls else 0
                }
                self.save_sharegpt_result(conversation, metadata)
            else:
                self.invalid_count += 1
                print(f"Validation failed for {item['id']}: {reason}")
            
            # Also save raw result for debugging
            result = {
                "id": item["id"],
                "model_output": model_output,
                "ground_truth": ground_truth,
                "valid": is_valid,
                "validation_reason": reason,
                "extracted_tool_calls": extracted_calls,
                "system_prompt": item["system_prompt"],
                "history_tail": [h.model_dump(mode='json') for h in item["history"][-2:]]
            }
            self.save_result(result)

    def save_result(self, result: Dict[str, Any]):
        with open(self.output_file, "a", encoding="utf-8") as f:
            f.write(json.dumps(result, ensure_ascii=False) + "\n")

    async def run(self):
        items = self.load_items()
        
        # Determine batching
        batch_size = self.config.batch_size
        total_batches = (len(items) + batch_size - 1) // batch_size
        
        print(f"Starting generation: {len(items)} items in {total_batches} batches.")
        
        for i in tqdm(range(0, len(items), batch_size)):
            batch = items[i : i + batch_size]
            await self.process_batch(batch)
        
        # Print summary
        total = self.valid_count + self.invalid_count
        print(f"\n=== Generation Complete ===")
        print(f"Total processed: {total}")
        print(f"Valid (saved to ShareGPT): {self.valid_count}")
        print(f"Invalid (failed validation): {self.invalid_count}")
        print(f"\nShareGPT output: {self.sharegpt_file}")
        print(f"Raw results: {self.output_file}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Tool-Use Data Generation")
    parser.add_argument("--scenario", default="sequential", choices=["sequential", "multistep", "multiturn", "relevance", "single"], help="Scenario to generate data for")
    parser.add_argument("--dataset", default=None, help="Override dataset path")
    parser.add_argument("--limit", type=int, default=None, help="Limit number of items")
    parser.add_argument("--batch_size", type=int, default=8, help="Parallel batch size")
    parser.add_argument("--model", default="Hermes-4-70B", help="Model ID")
    parser.add_argument("--validate_think", action="store_true", default=True, help="Validate <think> blocks in generations")
    parser.add_argument("--no_validate_think", action="store_false", dest="validate_think", help="Disable <think> block validation")
    
    args = parser.parse_args()
    
    dataset_name = args.dataset if args.dataset else "interstellarninja/hermes_reasoning_tool_use"
    
    config = ToolUseEnvConfig(
        dataset_name=dataset_name,
        limit=args.limit,
        batch_size=args.batch_size,
        model=args.model,
        validate_think_blocks=args.validate_think
    )
    
    pipeline = ToolUsePipeline(config, scenario=args.scenario)
    asyncio.run(pipeline.run())
