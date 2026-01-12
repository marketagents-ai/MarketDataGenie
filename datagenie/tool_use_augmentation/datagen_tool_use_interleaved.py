"""
Interleaved Tool-Use with Thinking Data Generation Script
==========================================================

This script generates datasets where tool calls are interleaved within <think> tags,
using the nvidia/AceReason-Math dataset as input. The model can use calculator and
python_interpreter tools to solve math problems.

Key features:
1. Tool calls are INSIDE <think> tags, not after them
2. Uses stop-execute-continue pattern: stop at </tool_call>, execute tool, inject response
3. Final answer must be in \boxed{...} format
4. Validates correct tool usage and final answer

Dataset format (nvidia/AceReason-Math):
- problem: The math problem to solve
- answer: The expected numeric answer

Output format (ShareGPT):
```
[
  {"from": "system", "value": "<system prompt with tool definitions>"},
  {"from": "human", "value": "<problem>"},
  {"from": "gpt", "value": "<think>...<tool_call>...</tool_call>...</think>"},
  {"from": "tool", "value": "<tool_response>...</tool_response>"},
  {"from": "gpt", "value": "<think>...</think>\nThe answer is \boxed{...}"}
]
```

Usage:
    python datagenie/tool_use/datagen_tool_use_interleaved.py --limit 10 --batch_size 4
"""

import re
import json
import asyncio
import argparse
import time
import os
from pathlib import Path
from typing import List, Dict, Any, Optional, Tuple

from datasets import load_dataset

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
from minference.caregistry import CallableRegistry
from minference.lite.models import CallableTool

# Initialize Registries (Singleton) to setup logging
EntityRegistry()
CallableRegistry()


from pydantic import BaseModel, Field
# ---------------------------------------------------------------------------
# Tool execution functions
# ---------------------------------------------------------------------------

async def _execute_calculator(expr: str) -> Dict[str, Any]:
    """
    Execute a calculator expression safely.
    Returns {"value": result}
    """
    import math
    try:
        val = eval(expr, {"__builtins__": {}}, {"math": math})
        return {"value": val}
    except Exception as e:
        return {"error": str(e)}


async def _execute_python_interpreter(code: str) -> Dict[str, Any]:
    """
    Execute Python code via local code execution server.
    Returns {"stdout": output, "result": last_line} or {"error": message}
    """
    import httpx
    try:
        async with httpx.AsyncClient(timeout=10.0) as client:
            payload = {"code": code, "input": ""}
            resp = await client.post("http://localhost:5002/execute", json=payload)
            data = resp.json()
            
            # Check for execution errors from server
            if data.get("error"):
                return {"error": data.get("error")}
            
            output = data.get("output", "")
            return {
                "stdout": output,
                "result": output.strip()
            }
    except httpx.ConnectError:
        raise RuntimeError(
            "Python interpreter server not available at localhost:5002. "
            "Please ensure the code_exec_server Docker container is running."
        )
    except Exception as e:
        return {"error": str(e)}


async def execute_tool(tool_name: str, arguments: Dict[str, Any]) -> Dict[str, Any]:
    """Execute a tool call and return the result."""
    if tool_name == "calculator":
        return await _execute_calculator(arguments.get("expr", ""))
    elif tool_name == "python_interpreter":
        return await _execute_python_interpreter(arguments.get("code", ""))
    else:
        return {"error": f"Unknown tool: {tool_name}"}


# ---------------------------------------------------------------------------
# Validation helpers for interleaved thinking format
# ---------------------------------------------------------------------------

def _extract_think_blocks(txt: str) -> List[str]:
    """
    Extract all <think>...</think> blocks from text.
    Returns list of think block contents (without the tags).
    """
    pattern = r"<think>(.*?)</think>"
    matches = re.findall(pattern, txt, flags=re.DOTALL | re.IGNORECASE)
    return matches


def _extract_last_tool_call(txt: str) -> Optional[Dict[str, Any]]:
    """
    Extract the last <tool_call> from text, even if incomplete.
    Returns parsed JSON dict or None.
    """
    # Find the last tool_call opening tag
    pattern = r"<tool_call>\s*(.*?)$"
    match = re.search(pattern, txt, flags=re.DOTALL | re.IGNORECASE)
    if not match:
        return None
    
    raw_json = match.group(1).strip()
    
    # Try to parse as JSON
    import ast
    try:
        return json.loads(raw_json)
    except json.JSONDecodeError:
        pass
    
    # Try Python literal
    try:
        return ast.literal_eval(raw_json)
    except Exception:
        pass
    
    # Try to find complete JSON object within the string
    if '{' in raw_json:
        start = raw_json.find('{')
        brace_count = 0
        json_end = 0
        for i, char in enumerate(raw_json[start:], start=start):
            if char == '{':
                brace_count += 1
            elif char == '}':
                brace_count -= 1
                if brace_count == 0:
                    json_end = i + 1
                    break
        if json_end > 0:
            try:
                return json.loads(raw_json[:json_end])
            except json.JSONDecodeError:
                pass
    
    return None


def _is_new_tool_call(txt: str) -> bool:
    """
    Return True if there's an unresponded <tool_call> in txt.
    """
    pos = txt.rfind("<tool_call>")
    if pos == -1:
        return False
    return "</tool_response>" not in txt[pos:]


def _validate_boxed_answer(txt: str) -> Optional[str]:
    """
    Extract the boxed answer from text.
    Returns the content inside \boxed{...} or None.
    """
    pattern = r"\\boxed\{([^}]+)\}"
    match = re.search(pattern, txt)
    return match.group(1).strip().replace("%", "").replace("\\", "") if match else None


def _normalize_number(txt: str) -> str:
    """
    Attempt to extract a float from the string, ignoring common symbols.
    Returns the string representation of the float if successful, else the original normalized string.
    """
    if not txt:
        return ""
    
    # Remove common non-numeric symbols
    clean_txt = txt.strip().lower()
    for char in ["%", "$", ",", " "]:
        clean_txt = clean_txt.replace(char, "")
    
    # Remove common units (heuristic)
    for unit in ["cm", "m", "km", "kg", "g", "s", "min", "h", "rubles", "usd"]:
        if clean_txt.endswith(unit):
             clean_txt = clean_txt[:-len(unit)]
             break

    try:
        return str(float(clean_txt))
    except ValueError:
        return clean_txt


def _validate_interleaved_thinking(
    txt: str, 
    expected_answer: Optional[str] = None,
    require_boxed: bool = True
) -> Tuple[bool, Optional[str], str]:
    """
    Validate the interleaved thinking format.
    
    Checks:
    1. At least one <think> block exists
    2. Think block is properly closed
    3. (Optional) Final answer is in \boxed{...} format
    4. (Optional) Answer matches expected value
    
    Returns:
        (is_valid, boxed_answer, reason)
    """
    if not txt or not txt.strip():
        return False, None, "empty_output"
    
    # Check for think blocks
    if "<think>" not in txt.lower():
        return False, None, "no_think_block"
    
    # Check that think block is closed
    if "</think>" not in txt.lower():
        return False, None, "unclosed_think_block"
    
    # Check for boxed answer if required
    boxed_answer = _validate_boxed_answer(txt)
    if require_boxed and not boxed_answer:
        return False, None, "missing_boxed_answer"
    
    # Validate answer matches expected if provided
    if expected_answer and boxed_answer:
        val_generated = _normalize_number(boxed_answer)
        val_expected = _normalize_number(str(expected_answer))
        
        match = False
        try:
            # Try float comparison if both look like numbers
            f_gen = float(val_generated)
            f_exp = float(val_expected)
            import math
            if math.isclose(f_gen, f_exp, rel_tol=1e-5):
                match = True
        except ValueError:
            # Fallback to string comparison
            pass
            
        if not match and val_generated != val_expected:
             return False, boxed_answer, f"wrong_answer (expected {expected_answer}, got {boxed_answer})"
    
    # Check that tool calls are inside think blocks
    txt_without_think = re.sub(r"<think>.*?</think>", "", txt, flags=re.DOTALL | re.IGNORECASE)
    if "<tool_call>" in txt_without_think.lower():
        return False, boxed_answer, "tool_call_outside_think"
    
    return True, boxed_answer, "valid"


class InterleavedToolUseConfig(BaseModel):
    model: str = Field(default="Hermes-4-70B", description="Model ID")
    dataset_name: str = Field(default="nvidia/AceReason-Math", description="Dataset path")
    split: str = Field(default="train", description="Dataset split")
    output_dir: str = Field(default="outputs/tool_use_interleaved", description="Output directory")
    batch_size: int = Field(default=4, description="Batch size")
    max_tokens_per_call: int = Field(default=512, description="Max tokens per generation call")
    max_total_tokens: int = Field(default=4096, description="Max total tokens for entire response")
    temperature: float = Field(default=0.7, description="Temperature")
    limit: Optional[int] = Field(default=None, description="Limit items")
    max_tool_calls: int = Field(default=10, description="Maximum tool calls per problem")
    # Validation options
    require_boxed_answer: bool = Field(default=True, description="Require final answer in \\boxed{} format")
    validate_answer: bool = Field(default=True, description="Validate answer matches expected")
    # Output options  
    output_sharegpt: bool = Field(default=True, description="Output in ShareGPT format")



# ---------------------------------------------------------------------------
# Native Tool Definitions (using minference CallableTool)
# ---------------------------------------------------------------------------

async def get_calculator_tool() -> CallableTool:
    return CallableTool.from_callable(
        func=_execute_calculator,
        name="calculator",
        docstring="Evaluate a numeric Python expression. Returns {'value': result}.",
        strict_schema=True
    )

async def get_python_tool() -> CallableTool:
    return CallableTool.from_callable(
        func=_execute_python_interpreter,
        name="python_interpreter",
        docstring="Run a short Python snippet. Returns {'stdout': output, 'result': last_line}. Use print() to see output.",
        strict_schema=True
    )

# System prompt for interleaved thinking with NATIVE tools
# We DON'T need to define tools in XML here - the provider handles that.
# We DO need to enforce <think> tags.
INTERLEAVED_SYSTEM_PROMPT = """You are a deep-thinking AI that solves complex math problems.
1. You MUST enclose your reasoning in <think>...</think> tags.
2. You have access to tools (python_interpreter).
3. When you need to calculate something, USE A TOOL. Do not calculate manually.
4. The model will automatically format tool calls for you - just decide to use them.

CRITICAL:
- Thoughts go in <think> tags.
- Final answer must be in \\boxed{...} format.
- ALWAYS use python_interpreter for mathematical computations.
"""

class InterleavedToolUsePipeline:
    def __init__(self, config: InterleavedToolUseConfig):
        self.config = config
        self.orchestrator = InferenceOrchestrator()
        
        # Ensure output directory exists
        path = Path(self.config.output_dir)
        path.mkdir(parents=True, exist_ok=True)
        
        # Output files
        ts = int(time.time())
        self.output_file = path / f"tool_use_interleaved_results_{ts}.jsonl"
        self.sharegpt_file = path / f"tool_use_interleaved_sharegpt_{ts}.jsonl"
        
        # Stats tracking
        self.valid_count = 0
        self.invalid_count = 0
        self.answer_correct_count = 0
        self.answer_wrong_count = 0

    async def _generate_with_tools(
        self,
        initial_messages: List[Dict[str, str]],
        item_id: str
    ) -> Tuple[List[ChatMessage], bool, str, str]:
        """
        Generate a response with NATIVE interleaved tool execution.
        Returns: (history, success, reason, flattened_response_for_validation)
        """
        print(f"\n  [{item_id}] === Starting Native Tool Generation ===")
        
        # Initialize Tools - Restrict to Python Tool only for "Strict Usage"
        calc_tool = await get_calculator_tool()
        python_tool = await get_python_tool()
        available_tools = [python_tool] # Only pass Python tool to ChatThread
        
        # Prepare History
        user_content = ""
        for msg in initial_messages:
            if msg["role"] == "user":
                user_content = msg["content"]
        
        # Start with fresh history
        current_history = [
            ChatMessage(role=MessageRole.user, content=user_content)
        ]
        
        # Track full response for validation only
        full_flattened_response = ""
        
        max_iterations = self.config.max_tool_calls * 2
        
        sys_prompt = SystemPrompt(name="sys", content=INTERLEAVED_SYSTEM_PROMPT)
        
        llm_config = LLMConfig(
            client=LLMClient.litellm,
            model=self.config.model,
            temperature=self.config.temperature,
            response_format=ResponseFormat.auto_tools,  # Use Auto to allow Thinking + Tools
            max_tokens=self.config.max_tokens_per_call
        )
        
        conversation_history = current_history.copy()

        for iteration in range(max_iterations):
            print(f"\n  [{item_id}] --- Iteration {iteration} ---")
            
            ct = ChatThread(
                name=f"native-{item_id}",
                system_prompt=sys_prompt,
                llm_config=llm_config,
                history=conversation_history,
                tools=available_tools, # Restrict availability here
                new_message=None # We are continuing or checking for tools
            )
            
            # Execute one turn
            outputs = await self.orchestrator.run_parallel_ai_completion([ct])
            
            if not outputs:
                return conversation_history, False, "no_output", full_flattened_response
            
            output: ProcessedOutput = outputs[0]
            
            # Handle Content (Thoughts/Text)
            content_chunk = output.content or ""
            if content_chunk:
                print(f"  [{item_id}] Received Content: {content_chunk[:100]}...")
                full_flattened_response += content_chunk
            
            # Handle Tool Calls (Native)
            if output.json_object:  # This is populated for function calls in minference
                tool_name = output.json_object.name
                tool_args = output.json_object.object
                tool_call_id = output.json_object.tool_call_id
                
                print(f"  [{item_id}] ★ Native Tool Call: {tool_name}")
                print(f"  [{item_id}]   Args: {tool_args}")
                
                # Append to validation string
                full_flattened_response += f"\n<tool_call>\n{json.dumps({'name': tool_name, 'arguments': tool_args})}\n</tool_call>\n"

                # Execute Tool
                try:
                    # Find tool definition
                    tool_def = next((t for t in available_tools if t.name == tool_name), None)
                    if not tool_def:
                        result = {"error": f"Unknown tool {tool_name}"}
                    else:
                        if tool_name == "calculator":
                             result = await _execute_calculator(tool_args.get("expr", ""))
                        elif tool_name == "python_interpreter":
                             result = await _execute_python_interpreter(tool_args.get("code", ""))
                        else:
                             result = {"error": "Tool not implemented"}
                             
                    print(f"  [{item_id}]   Result: {result}")
                    
                    # Append result to validation string
                    full_flattened_response += f"<tool_response>\n{json.dumps(result)}\n</tool_response>\n"
                    
                    # Update conversation history with Tool Result for next turn
                    # 1. Add Assistant Message (with tool call)
                    assistant_msg = ChatMessage(
                        role=MessageRole.assistant,
                        content=content_chunk, # Might include thoughts before the call
                        tool_call=tool_args,
                        tool_name=tool_name,
                        oai_tool_call_id=tool_call_id
                    )
                    conversation_history.append(assistant_msg)
                    
                    # 2. Add Tool Message
                    tool_msg = ChatMessage(
                        role=MessageRole.tool,
                        content=json.dumps(result),
                        tool_name=tool_name,
                        oai_tool_call_id=tool_call_id
                    )
                    conversation_history.append(tool_msg)
                    
                except Exception as e:
                    print(f"  [{item_id}] Tool Execution Error: {e}")
                    return conversation_history, False, "tool_error", full_flattened_response

            else:
                # No tool call - just text.
                # If we have content, we add it to history as simple assistant message
                if content_chunk:
                    conversation_history.append(ChatMessage(
                        role=MessageRole.assistant,
                        content=content_chunk
                    ))
                
                # Check for boxed answer - if present, we are likely done
                if "\\boxed{" in full_flattened_response:
                    # Append closing think if missing (for validation string)
                    if "<think>" in full_flattened_response and "</think>" not in full_flattened_response:
                        full_flattened_response += "\n</think>"
                    
                    print(f"  [{item_id}] ✓ Generation Complete (Boxed Answer Found)")
                    return conversation_history, True, "complete", full_flattened_response
                
                # Check for explicit completion without boxed answer (rare but possible failure case)
                if "</think>" in content_chunk:
                     if "\\boxed{" not in full_flattened_response:
                        print(f"  [{item_id}] ! Think closed but no boxed answer. Nudging...")
                        conversation_history.append(ChatMessage(
                            role=MessageRole.user,
                            content="Please provide your final answer in \\boxed{...} format."
                        ))
                        continue
                     else:
                        print(f"  [{item_id}] ✓ Generation Complete")
                        return conversation_history, True, "complete", full_flattened_response

                if not content_chunk and not output.json_object:
                    print(f"  [{item_id}] ! No content and no tool call. Stopping.")
                    return conversation_history, False, "stalled", full_flattened_response
                    
        return conversation_history, False, "max_iterations", full_flattened_response


    def _to_sharegpt_conversation(
        self, 
        problem: str,
        history: List[ChatMessage]
    ) -> List[Dict[str, str]]:
        """
        Convert history to ShareGPT format with explicit tool roles.
        Format: system -> human -> gpt(content + tool_call xml) -> tool(result) -> gpt ...
        Merges consecutive GPT messages (thought + tool call) into a single turn.
        """
        conversation = [
            {"from": "system", "value": INTERLEAVED_SYSTEM_PROMPT},
            {"from": "human", "value": problem}
        ]
        
        current_gpt_content = None
        
        for msg in history:
            if msg.role == MessageRole.user and msg.content == problem:
                continue # Skip initial user msg as we added it
                
            if msg.role == MessageRole.user:
                 # Flush any pending gpt content
                 if current_gpt_content is not None:
                     conversation.append({"from": "gpt", "value": current_gpt_content})
                     current_gpt_content = None
                 conversation.append({"from": "human", "value": msg.content})
            
            elif msg.role == MessageRole.assistant:
                content = msg.content or ""
                # If message has tool calls, format them as XML and append to content
                if msg.tool_call:
                    tool_call_json = json.dumps({"name": msg.tool_name, "arguments": msg.tool_call})
                    content += f"\n<tool_call>\n{tool_call_json}\n</tool_call>"
                
                if current_gpt_content is None:
                    current_gpt_content = content
                else:
                    current_gpt_content += content
            
            elif msg.role == MessageRole.tool:
                 # Flush any pending gpt content before tool response
                 if current_gpt_content is not None:
                     conversation.append({"from": "gpt", "value": current_gpt_content})
                     current_gpt_content = None
                 
                 # Format tool response
                 try:
                    # Ensure it's valid JSON for tool response
                    result_json = json.dumps(json.loads(msg.content))
                 except:
                    result_json = json.dumps({"result": msg.content}) # Fallback
                
                 conversation.append({"from": "tool", "value": result_json})
                
        # Flush final
        if current_gpt_content is not None:
            conversation.append({"from": "gpt", "value": current_gpt_content})
            
        return conversation

    def save_sharegpt_result(self, conversation: List[Dict[str, str]], metadata: Optional[Dict] = None):
        """Save a validated conversation in ShareGPT format."""
        item = {"conversations": conversation}
        if metadata:
            item["metadata"] = metadata
        with open(self.sharegpt_file, "a", encoding="utf-8") as f:
            f.write(json.dumps(item, ensure_ascii=False) + "\n")

    def load_items(self) -> List[Dict[str, Any]]:
        """
        Load dataset and prepare items.
        Returns list of problems to solve.
        """
        print(f"Loading dataset {self.config.dataset_name}...")
        try:
            ds = load_dataset(self.config.dataset_name, split=self.config.split)
        except Exception as e:
            print(f"Error loading dataset: {e}")
            return []

        items = []
        count = 0
        
        for idx, row in enumerate(ds):
            if self.config.limit and count >= self.config.limit:
                break
            
            problem = row.get("problem", "")
            answer = row.get("answer", "")
            
            if not problem or not answer:
                continue
            
            items.append({
                "id": f"problem_{idx}",
                "problem": problem,
                "answer": str(answer).strip()
            })
            count += 1

        print(f"Prepared {len(items)} problems from dataset.")
        return items

    async def process_item(self, item: Dict[str, Any]):
        """Process a single item with tool execution."""
        problem = item["problem"]
        expected_answer = item["answer"]
        item_id = item["id"]
        
        print(f"\n{'='*60}")
        print(f"Processing {item_id}")
        print(f"Problem: {problem[:100]}...")
        print(f"Expected answer: {expected_answer}")
        
        # Generate response with tool execution
        initial_messages = [
            {"role": "system", "content": INTERLEAVED_SYSTEM_PROMPT},
            {"role": "user", "content": problem}
        ]
        
        conversation_history, success, reason, flattened_response = await self._generate_with_tools(initial_messages, item_id)
        
        if not success:
            print(f"  [{item_id}] Generation failed: {reason}")
            self.invalid_count += 1
            self.save_result({
                "id": item_id,
                "problem": problem,
                "expected_answer": expected_answer,
                "response": flattened_response,
                "valid": False,
                "reason": reason
            })
            return
        
        # Validate the response (using flattened string for checking logic)
        is_valid, boxed_answer, validation_reason = _validate_interleaved_thinking(
            flattened_response,
            expected_answer=expected_answer if self.config.validate_answer else None,
            require_boxed=self.config.require_boxed_answer
        )
        
        if is_valid and validation_reason == "valid":
            self.valid_count += 1
            if self.config.validate_answer:
                self.answer_correct_count += 1
            print(f"  [{item_id}] ✓ Valid - Answer: {boxed_answer}")
            
            # Save in ShareGPT format using structured history
            conversation = self._to_sharegpt_conversation(problem, conversation_history)
            metadata = {
                "format": "interleaved_thinking",
                "expected_answer": expected_answer,
                "generated_answer": boxed_answer,
                "answer_correct": validation_reason == "valid"
            }
            self.save_sharegpt_result(conversation, metadata)
        else:
            self.invalid_count += 1
            if "wrong_answer" in validation_reason:
                self.answer_wrong_count += 1
            print(f"  [{item_id}] ✗ Invalid: {validation_reason}")
        
        # Save raw result
        self.save_result({
            "id": item_id,
            "problem": problem,
            "expected_answer": expected_answer,
            "response": flattened_response,
            "boxed_answer": boxed_answer,
            "valid": is_valid,
            "reason": validation_reason
        })

    def save_result(self, result: Dict[str, Any]):
        with open(self.output_file, "a", encoding="utf-8") as f:
            f.write(json.dumps(result, ensure_ascii=False) + "\n")

    async def run(self):
        items = self.load_items()
        
        if not items:
            print("No items to process")
            return
        
        print(f"\nStarting generation: {len(items)} items")
        print(f"Batch size: {self.config.batch_size}")
        
        # Process items  in batches (but sequentially for now due to tool execution complexity)
        for i in tqdm(range(len(items))):
            await self.process_item(items[i])
        
        # Print summary
        total = self.valid_count + self.invalid_count
        print(f"\n{'='*60}")
        print(f"=== Generation Complete ===")
        print(f"Total processed: {total}")
        print(f"Valid (saved to ShareGPT): {self.valid_count}")
        print(f"Invalid (failed validation): {self.invalid_count}")
        if self.config.validate_answer:
            print(f"Correct answers: {self.answer_correct_count}")
            print(f"Wrong answers: {self.answer_wrong_count}")
        print(f"\nShareGPT output: {self.sharegpt_file}")
        print(f"Raw results: {self.output_file}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Interleaved Tool-Use Data Generation")
    parser.add_argument("--dataset", default=None, help="Override dataset path")
    parser.add_argument("--limit", type=int, default=None, help="Limit number of items")
    parser.add_argument("--batch_size", type=int, default=4, help="Parallel batch size")
    parser.add_argument("--model", default="Hermes-4-70B", help="Model ID")
    parser.add_argument("--temperature", type=float, default=0.7, help="Temperature")
    parser.add_argument("--max_tool_calls", type=int, default=10, help="Max tool calls per problem")
    parser.add_argument("--validate_answer", action="store_true", default=True, help="Validate answer matches expected")
    parser.add_argument("--no_validate_answer", dest="validate_answer", action="store_false", help="Don't validate answer")
    
    args = parser.parse_args()
    
    dataset_name = args.dataset if args.dataset else "nvidia/AceReason-Math"
    
    config = InterleavedToolUseConfig(
        dataset_name=dataset_name,
        limit=args.limit,
        batch_size=args.batch_size,
        model=args.model,
        temperature=args.temperature,
        max_tool_calls=args.max_tool_calls,
        validate_answer=args.validate_answer
    )
    
    pipeline = InterleavedToolUsePipeline(config)
    asyncio.run(pipeline.run())

