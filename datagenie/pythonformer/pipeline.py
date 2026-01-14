"""
Pythonformer Dataset Generation Pipeline.

Generates interleaved reasoning + Python code action training data.
Uses the REPL server for code execution (same backend as inference).

XML Format:
- Assistant: <python>code with reasoning in comments</python>
- Final answer: <final_answer>answer</final_answer>
- Tool response: <repl>output</repl>

Usage:
    # Start the REPL server first
    python -m datagenie.pythonformer.server --port 5003
    
    # Then run the pipeline
    python -m datagenie.pythonformer.run --config configs/default_config.yaml --limit 100
"""

import json
import asyncio
import time
import random
import re
from pathlib import Path
from typing import List, Dict, Any, Optional, Tuple
from dataclasses import dataclass, field
from datetime import datetime

from datasets import load_dataset
from tqdm.asyncio import tqdm

from datagenie.pythonformer.config import (
    PythonformerConfig, EnvironmentType, ENV_TIPS, LLMClientType
)
from datagenie.pythonformer.repl_client import REPLClient, ExecutionResult, SubAgentCall
from datagenie.pythonformer.utils.debug import (
    Colors, print_colored, print_header, print_subheader,
    print_code_block, print_repl_output, print_state,
    print_final_answer, print_task_start, print_task_result
)


@dataclass
class Turn:
    """A single turn in the conversation."""
    role: str  # "user", "assistant", "tool"
    content: str
    code: Optional[str] = None


@dataclass
class TrajectoryResult:
    """Result of generating a trajectory."""
    success: bool
    final_answer: str
    turns: List[Turn]
    num_code_blocks: int = 0
    system_prompt: str = ""  # The system prompt used for this trajectory


@dataclass
class PipelineStats:
    """Track pipeline statistics."""
    total: int = 0
    successful: int = 0
    failed: int = 0
    answers_correct: int = 0
    answers_incorrect: int = 0
    answers_unknown: int = 0  # No expected answer to compare
    avg_turns: float = 0.0
    avg_code_blocks: float = 0.0
    total_turns: int = 0
    total_code_blocks: int = 0
    
    def record(self, result: TrajectoryResult, answer_correct: Optional[bool] = None) -> None:
        """Record a result."""
        self.total += 1
        if result.success:
            self.successful += 1
        else:
            self.failed += 1
        
        # Track answer correctness
        if answer_correct is True:
            self.answers_correct += 1
        elif answer_correct is False:
            self.answers_incorrect += 1
        else:
            self.answers_unknown += 1
        
        self.total_turns += len(result.turns)
        self.total_code_blocks += result.num_code_blocks
        self.avg_turns = self.total_turns / self.total
        self.avg_code_blocks = self.total_code_blocks / self.total
    
    def report(self) -> str:
        """Generate statistics report."""
        success_rate = (self.successful / self.total * 100) if self.total > 0 else 0
        validated = self.answers_correct + self.answers_incorrect
        accuracy = (self.answers_correct / validated * 100) if validated > 0 else 0
        return f"""
=== Pipeline Statistics ===
Total processed:    {self.total:,}
Successful:         {self.successful:,}
Failed:             {self.failed:,}
Success rate:       {success_rate:.1f}%
Avg turns:          {self.avg_turns:.1f}
Avg code blocks:    {self.avg_code_blocks:.1f}

=== Answer Validation ===
Correct:            {self.answers_correct:,}
Incorrect:          {self.answers_incorrect:,}
Unknown:            {self.answers_unknown:,}
Accuracy:           {accuracy:.1f}% (of {validated} validated)
"""


class PythonformerPipeline:
    """
    Pipeline for generating interleaved reasoning + code training data.
    
    Uses the REPL server for execution, ensuring consistency between
    training data generation and inference.
    """
    
    SYSTEM_PROMPT = """You are Pythonformer AI assistant that solves problems by reasoning and executing Python code.

## CRITICAL: Code Execution Required

You MUST write and execute Python code to solve problems. DO NOT attempt to solve problems mentally or provide answers without running code first. Your workflow is:

1. Write code in <python> tags
2. Wait for execution results in <repl> tags
3. Analyze results and iterate if needed
4. Only after successful code execution, provide <final_answer>

## Response Format

Your responses must include reasoning and python code within <python> </python> XML tags:

1. Python Code - Wrap all code in <python> tags. Include reasoning as comments:
<python>
# Okay, let's understand the problem
# We need to find...

import sympy as sp

# Define variables
x = sp.Symbol('x')

# Solve the equation
result = sp.solve(x**2 - 4, x)
print(f"Solutions: {{result}}")
</python>

2. Final Answer - ONLY after you have executed code and seen results, provide the final answer with the result in \\boxed{{}}:
<final_answer>
The solutions are $x = \\boxed{{2}}$ and $x = \\boxed{{-2}}$.
</final_answer>

For single answers:
<final_answer>
The answer is $\\boxed{{42}}$.
</final_answer>

## IMPORTANT RULES

1. NEVER give <final_answer> without executing at least one <python> block first
2. ALWAYS write code to solve the problem - do not solve mentally
3. Do NOT include <final_answer> in the same response as <python> blocks
4. Do NOT generate <repl>, <state>, or <sub_agent> tags - those are provided by the system
5. After writing <python> code, STOP and wait for execution results
6. Always put your final numerical/symbolic answer inside \\boxed{{}}
7. If your first approach fails, try alternative methods in code

## Execution Results

After you submit a <python> block, the system will execute it and return:

<repl>
Solutions: [-2, 2]
</repl>
<state>
imports: sympy | vars: x=Symbol('x'), result=[-2, 2]
</state>

The <state> tag shows the current REPL state including:
- Imported modules
- Defined functions and classes
- Variables with their types and values

Use this state information to track what's available for subsequent code blocks.

## Available in the Python Environment

- Standard library and common packages (numpy, pandas, sympy, scipy, json, re, etc.)
- `sub_agent(task, system_prompt=None)` - Invoke a sub-agent for semantic analysis

## Filesystem for Dynamic Context

Files are provided in <file> tags with name and type attributes:
<file name="data.csv" type="csv">
col1,col2
1,2
</file>

- `save_to_file(filename, content)` - Save content to workspace
- `read_file(filename, lines=N)` - Read file (optionally last N lines)  
- `list_files(pattern)` - List files matching pattern

## Guidelines

- Include reasoning as Python comments within <python> blocks
- Execute code to verify your approach before giving final answer
- REPL output is truncated to {max_output} characters
- Only use <final_answer> AFTER you have executed code and verified your solution
- Always include \\boxed{{}} around your final answer for validation
- If symbolic solving fails, try numerical methods
- Do NOT use `answer` as a variable name - it is reserved for the system

{env_tips}
"""

    OOLONG_SYSTEM_PROMPT = """You are Pythonformer AI assistant that solves long-context problems by programmatically analyzing documents with Python code.

## CRITICAL: Long Context Strategy

The context document is TOO LARGE to process in one pass. It has been saved to your workspace ({context_length:,} characters). You MUST use Python to:

1. Load and explore the context structure
2. Search, filter, and chunk as needed
3. Use `sub_agent()` for semantic analysis of chunks
4. Aggregate results programmatically
5. Verify your answer before submitting

## File Input Format

Long context files are provided in <file> tags:
<file name="context.txt" type="txt" chars="{context_length}">
[Content saved to workspace - use read_file('context.txt') to load]
</file>

## Response Format

Use <python> tags for code, <final_answer> for your answer:

<python>
# Step 1: Load and explore the context
context = read_file('context.txt')
lines = context.strip().split('\\n')
print(f"Total lines: {{len(lines)}}")
print(f"First 5 lines:\\n{{chr(10).join(lines[:5])}}")
</python>

## Available Tools

### File Operations
- `read_file('context.txt')` - Load the full context
- `read_file('context.txt', lines=100)` - Read last 100 lines
- `save_to_file(name, content)` - Save intermediate results
- `list_files(pattern)` - List workspace files

### Sub-Agent for Semantic Analysis
- `sub_agent(task, system_prompt=None)` - Invoke a sub-agent for semantic tasks

Use `sub_agent()` when you need semantic understanding. The response appears in <sub_agent> tags:
<python>
# Example: Classify or extract meaning from a chunk
chunk = '\\n'.join(lines[0:50])
result = sub_agent(
    task=f"How many dice rolls are mentioned in this text? Return just the number.\\n\\nText:\\n{{chunk}}",
    system_prompt="You are a precise counter. Return only the number."
)
print(f"Rolls in chunk: {{result}}")
</python>

The system returns sub-agent responses in <sub_agent> tags:
<sub_agent task="How many dice rolls...">
12
</sub_agent>

### Python Standard Library
- `re` for regex search/matching
- `collections.Counter` for counting
- String methods: split, find, count, etc.

## Strategy Examples

### Counting Pattern Occurrences
<python>
import re
context = read_file('context.txt')
# Count all dice rolls like "rolls a 15" or "rolled 20"
rolls = re.findall(r'rolls?\\s+(?:a\\s+)?(\\d+)', context, re.IGNORECASE)
print(f"Found {{len(rolls)}} rolls")
print(f"Sample rolls: {{rolls[:10]}}")
</python>

### Chunked Semantic Analysis with Sub-Agent
<python>
context = read_file('context.txt')
lines = context.split('\\n')
chunk_size = 200
results = []

for i in range(0, min(len(lines), 1000), chunk_size):
    chunk = '\\n'.join(lines[i:i+chunk_size])
    count = sub_agent(
        task=f"Count dice rolls in this D&D transcript chunk. Return just the number.\\n\\n{{chunk}}",
        system_prompt="You count dice rolls. Return only an integer."
    )
    results.append(int(count) if count.isdigit() else 0)
    print(f"Chunk {{i//chunk_size}}: {{results[-1]}} rolls")

total = sum(results)
print(f"Total rolls: {{total}}")
</python>

## IMPORTANT RULES

1. NEVER try to read the entire context in one LLM response - use Python
2. ALWAYS execute code before giving <final_answer>
3. Use `sub_agent()` for semantic tasks (understanding meaning, classification)
4. Use Python/regex for syntactic tasks (counting patterns, searching)
5. Put your final answer in \\boxed{{}}
6. Do NOT generate <repl>, <state>, or <sub_agent> tags - system provides those
7. Do NOT use `answer` as a variable name - it is reserved for the system

## Final Answer Format

<final_answer>
The answer is \\boxed{{84}}.
</final_answer>

{env_tips}
"""
    
    def __init__(self, config: PythonformerConfig):
        self.config = config
        self.stats = PipelineStats()
        
        # REPL client
        self.repl_client = REPLClient(server_url=config.repl.server_url)
        
        # LLM orchestrator for generating responses
        self.orchestrator = None
        self._init_orchestrator()
        
        # Setup output
        self.output_dir = Path(config.dataset.output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        ts = int(time.time())
        env_name = config.dataset.environment.value
        self.results_file = self.output_dir / f"traces_{env_name}_{ts}.jsonl"
        self.sharegpt_file = self.output_dir / f"sharegpt_{env_name}_{ts}.jsonl"
        self.chat_file = self.output_dir / f"chat_{env_name}_{ts}.jsonl"
        
        # Build and cache system prompt
        self.system_prompt = self._build_system_prompt(config.dataset.environment)
    
    def _init_orchestrator(self):
        """Initialize the LLM orchestrator."""
        try:
            from minference.lite.inference import InferenceOrchestrator
            from minference.lite.models import EntityRegistry
            
            # Initialize logger if needed
            if not hasattr(EntityRegistry, '_logger') or EntityRegistry._logger is None:
                import logging
                EntityRegistry._logger = logging.getLogger("minference")
            
            self.orchestrator = InferenceOrchestrator()
        except ImportError as e:
            print(f"Warning: minference not available ({e}), using OpenAI directly")
            self.orchestrator = None
        except Exception as e:
            print(f"Warning: Failed to init orchestrator ({e}), using OpenAI directly")
            self.orchestrator = None
    
    def _append_jsonl(self, path: Path, record: Dict[str, Any]) -> None:
        """Append a record to JSONL file."""
        with open(path, "a", encoding="utf-8") as f:
            f.write(json.dumps(record, ensure_ascii=False) + "\n")
    
    def _build_system_prompt(self, env_type: EnvironmentType, context_length: int = 0) -> str:
        """Build system prompt with environment-specific tips."""
        env_tips = ENV_TIPS.get(env_type, "")
        
        # Use OOLONG-specific prompt for long-context tasks
        if env_type == EnvironmentType.OOLONG:
            return self.OOLONG_SYSTEM_PROMPT.format(
                context_length=context_length,
                env_tips=env_tips
            )
        
        return self.SYSTEM_PROMPT.format(
            max_output=self.config.repl.max_output_chars,
            env_tips=env_tips
        )
    
    def _extract_python_blocks(self, text: str) -> List[str]:
        """Extract Python code from <python> tags."""
        pattern = r'<python>\s*(.*?)\s*</python>'
        return re.findall(pattern, text, re.DOTALL)
    
    def _extract_final_answer(self, text: str) -> Optional[str]:
        """Extract final answer from <final_answer> tags."""
        pattern = r'<final_answer>\s*(.*?)\s*</final_answer>'
        match = re.search(pattern, text, re.DOTALL)
        return match.group(1).strip() if match else None
    
    def _extract_boxed_answers(self, text: str) -> List[str]:
        """
        Extract answers from \\boxed{} format for validation.
        
        Returns list of boxed values (handles multiple boxes).
        """
        # Match \boxed{...} with nested braces support
        pattern = r'\\boxed\{([^{}]*(?:\{[^{}]*\}[^{}]*)*)\}'
        matches = re.findall(pattern, text)
        return [m.strip() for m in matches]
    
    def _has_hallucinated_repl(self, text: str) -> bool:
        """
        Check if the model hallucinated system output in its response.
        
        The model should NOT generate <repl>, <state>, or <sub_agent> tags - 
        those come from actual execution. If we see them in the assistant 
        response, the model is hallucinating.
        """
        # Check for <repl> tags in assistant response
        if re.search(r'<repl>', text):
            return True
        # Check for <state> tags in assistant response
        if re.search(r'<state>', text):
            return True
        # Check for <sub_agent> tags in assistant response
        if re.search(r'<sub_agent', text):
            return True
        return False
    
    async def _generate_response(
        self,
        messages: List[Dict[str, str]],
        system_prompt: str
    ) -> str:
        """Generate a response from the LLM."""
        if self.orchestrator:
            from minference.lite.models import (
                LLMConfig, ResponseFormat,
                ChatThread, ChatMessage, MessageRole, SystemPrompt
            )
            
            # Get LLM client from config
            llm_client = self.config.get_llm_client()
            
            # Build history with proper role mapping
            history = []
            for m in messages[:-1]:
                if m["role"] == "user":
                    history.append(ChatMessage(role=MessageRole.user, content=m["content"]))
                elif m["role"] == "assistant":
                    history.append(ChatMessage(role=MessageRole.assistant, content=m["content"]))
                elif m["role"] == "tool":
                    # Tool responses go as user messages with the content
                    history.append(ChatMessage(role=MessageRole.user, content=m["content"]))
            
            thread = ChatThread(
                name="gen",
                system_prompt=SystemPrompt(name="sys", content=system_prompt),
                llm_config=LLMConfig(
                    client=llm_client,
                    model=self.config.main_model,
                    temperature=self.config.main_temperature,
                    max_tokens=self.config.main_max_tokens,
                    response_format=ResponseFormat.text
                ),
                tools=[],
                history=history,
                new_message=messages[-1]["content"]
            )
            
            outputs = await self.orchestrator.run_parallel_ai_completion([thread])
            return outputs[0].content if outputs and outputs[0] else ""
        else:
            # Fallback to OpenAI directly
            from openai import OpenAI
            client = OpenAI()
            
            full_messages = [{"role": "system", "content": system_prompt}] + messages
            response = client.chat.completions.create(
                model=self.config.main_model,
                messages=full_messages,
                temperature=self.config.main_temperature,
                max_tokens=self.config.main_max_tokens,
            )
            return response.choices[0].message.content
    
    # ============================================================
    # Trajectory Generation
    # ============================================================
    
    async def generate_trajectory(
        self,
        prompt: str,
        context: Optional[str] = None,
        env_type: EnvironmentType = EnvironmentType.CODE,
    ) -> TrajectoryResult:
        """
        Generate a complete trajectory for a task.
        
        Args:
            prompt: The task/question
            context: Optional large context
            env_type: Environment type for tips
            
        Returns:
            TrajectoryResult with turns and final answer
        """
        # Create a NEW REPLClient instance for this task to ensure session isolation
        # This is critical for parallel execution - each task needs its own session
        repl_client = REPLClient(server_url=self.config.repl.server_url)
        
        # For OOLONG/long-context: don't pass context to session (too large)
        # Instead we'll save it to file after session creation
        session_context = None if env_type == EnvironmentType.OOLONG else context
        
        # Build sub-agent config from pipeline config
        sub_agent_config = {
            "model": self.config.sub_llm.model,
            "client": self.config.sub_llm.client.value,
            "temperature": self.config.sub_llm.temperature,
            "max_tokens": self.config.sub_llm.max_tokens,
        }
        
        # Create REPL session
        repl_client.create_session(
            context=session_context,
            max_output_chars=self.config.repl.max_output_chars,
            sub_agent_config=sub_agent_config
        )
        
        try:
            # For OOLONG: save context to file via code execution
            context_length = len(context) if context else 0
            if env_type == EnvironmentType.OOLONG and context:
                # Save context to file - escape for Python string
                # Use a chunked approach to avoid huge string literals
                chunk_size = 50000
                for i in range(0, len(context), chunk_size):
                    chunk = context[i:i+chunk_size]
                    # Escape the chunk for Python
                    escaped_chunk = chunk.replace('\\', '\\\\').replace("'", "\\'").replace('\n', '\\n').replace('\r', '\\r')
                    mode = 'w' if i == 0 else 'a'
                    init_code = f"with open('context.txt', '{mode}') as f: f.write('{escaped_chunk}')"
                    repl_client.execute(init_code)
                
                if self.config.debug:
                    print(f"Saved context to context.txt ({context_length:,} chars)")
            
            turns: List[Turn] = []
            num_code_blocks = 0
            
            # Get initial state from REPL (shows pre-imported modules)
            initial_state = repl_client.get_state()
            initial_state_str = initial_state.get("state_formatted", "")
            
            # Build system prompt (with context length for OOLONG)
            system_prompt = self._build_system_prompt(env_type, context_length)
            
            # Initial user message
            user_msg = prompt
            if context and env_type != EnvironmentType.OOLONG:
                # For non-OOLONG: mention context variable
                user_msg += f"\n\n[Context available as `context` variable ({len(context):,} chars) or in 'context.txt']"
            elif context and env_type == EnvironmentType.OOLONG:
                # For OOLONG: use <file> tag to indicate context location
                user_msg += f'\n\n<file name="context.txt" type="txt" chars="{context_length}">\n[Content saved to workspace - use read_file(\'context.txt\') to load]\n</file>'
            
            # Optionally prepend initial state info
            if initial_state_str and initial_state_str != "(empty state)":
                user_msg += f"\n\n[Initial REPL state: {initial_state_str}]"
            
            turns.append(Turn(role="user", content=user_msg))
            messages = [{"role": "user", "content": user_msg}]
            
            # Main generation loop
            for turn_idx in range(self.config.repl.max_turns):
                # Nudge model to conclude if it's been running too long
                if turn_idx >= 6 and num_code_blocks >= 3:
                    # Add a gentle nudge to wrap up
                    if turn_idx == 6:
                        nudge = "\n\n[Note: You've executed several code blocks. If you have enough information, please provide your <final_answer> with \\boxed{} now.]"
                        if messages[-1]["role"] == "tool":
                            messages[-1]["content"] += nudge
                    elif turn_idx >= 10:
                        # Stronger nudge
                        nudge = "\n\n[IMPORTANT: Please provide your <final_answer> now based on your analysis. Use \\boxed{} for the answer.]"
                        if messages[-1]["role"] == "tool":
                            messages[-1]["content"] += nudge
                
                # Generate response
                response = await self._generate_response(messages, system_prompt)
                
                if not response:
                    break
                
                # Check for hallucinated REPL output - model should NOT generate these
                if self._has_hallucinated_repl(response):
                    if self.config.debug:
                        print(f"Warning: Model hallucinated <repl>/<state>/<sub_agent> tags - stripping them")
                    # Strip hallucinated tags and re-extract
                    response = re.sub(r'<repl>.*?</repl>', '', response, flags=re.DOTALL)
                    response = re.sub(r'<state>.*?</state>', '', response, flags=re.DOTALL)
                    response = re.sub(r'<sub_agent[^>]*>.*?</sub_agent>', '', response, flags=re.DOTALL)
                    response = response.strip()
                
                # Extract Python blocks first
                python_blocks = self._extract_python_blocks(response)
                
                # Check for final answer
                final_answer = self._extract_final_answer(response)
                
                # Rule: Cannot give final_answer in same turn as python blocks
                # (must wait for execution results)
                if final_answer and python_blocks:
                    if self.config.debug:
                        print(f"Warning: Model gave final_answer with python blocks - executing code first")
                    # Remove the final_answer from response, execute code, continue
                    response = re.sub(r'<final_answer>.*?</final_answer>', '', response, flags=re.DOTALL)
                    response = response.strip()
                    final_answer = None
                
                # Rule: Must execute at least one code block before final answer
                if final_answer and num_code_blocks == 0:
                    if self.config.debug:
                        print(f"Warning: Model gave final_answer without any code execution - prompting for code")
                    # Instead of ignoring, prompt the model to write code
                    final_answer = None
                    # If there are no python blocks either, inject a prompt to write code
                    if not python_blocks:
                        # Add a system nudge to write code
                        nudge_msg = "You must execute Python code before providing a final answer. Please write code in <python> tags to solve this problem step by step."
                        turns.append(Turn(role="tool", content=nudge_msg))
                        messages.append({"role": "user", "content": nudge_msg})
                        continue  # Skip adding the assistant response, get new one
                
                turns.append(Turn(role="assistant", content=response))
                messages.append({"role": "assistant", "content": response})
                
                # If we have a valid final answer (after code execution), return
                if final_answer and num_code_blocks > 0:
                    return TrajectoryResult(
                        success=True,
                        final_answer=final_answer,
                        turns=turns,
                        num_code_blocks=num_code_blocks,
                        system_prompt=system_prompt
                    )
                
                if not python_blocks:
                    # No code blocks and no valid final answer - model might be done or confused
                    # Check if the response contains a boxed answer without proper tags
                    boxed_in_response = self._extract_boxed_answers(response)
                    if boxed_in_response and num_code_blocks > 0:
                        if self.config.debug:
                            print(f"Found \\boxed{{}} without <final_answer> tags - treating as final answer")
                        return TrajectoryResult(
                            success=True,
                            final_answer=response,  # Use the whole response as final answer
                            turns=turns,
                            num_code_blocks=num_code_blocks,
                            system_prompt=system_prompt
                        )
                    break
                
                for code in python_blocks:
                    # Pretty print code block in debug mode
                    if self.config.debug:
                        print_subheader(f"Executing Code Block #{num_code_blocks + 1}", Colors.MAGENTA)
                        print_code_block(code)
                    
                    result = repl_client.execute(code)
                    num_code_blocks += 1
                    
                    # Pretty print REPL output in debug mode
                    if self.config.debug:
                        print_repl_output(
                            output=result.output,
                            error=result.error,
                            execution_time_ms=result.execution_time_ms,
                            truncated=result.truncated
                        )
                        # Print sub-agent calls
                        for sub_call in result.sub_agent_calls:
                            print_colored(f"  <sub_agent> task: {sub_call.task[:100]}...", Colors.CYAN)
                            print_colored(f"  response: {sub_call.response}", Colors.GREEN)
                        if result.state_formatted and result.state_formatted != "(empty state)":
                            print_state(result.state_formatted)
                    
                    # Format observation with <repl> tags
                    obs_parts = [result.output] if result.output else []
                    if result.execution_time_ms > 100:
                        obs_parts.append(f"[Execution: {result.execution_time_ms}ms]")
                    if result.error:
                        obs_parts.append(f"[Error: {result.error}]")
                    if result.truncated:
                        obs_parts.append(f"[Output truncated]")
                    if result.files_created:
                        obs_parts.append(f"[Files created: {', '.join(result.files_created)}]")
                    
                    obs_content = "\n".join(obs_parts) if obs_parts else "(no output)"
                    
                    # Build tool response with <repl> and <state> tags
                    repl_response = f"<repl>\n{obs_content}\n</repl>"
                    
                    # Add sub-agent responses if any
                    for sub_call in result.sub_agent_calls:
                        task_preview = sub_call.task[:100] + "..." if len(sub_call.task) > 100 else sub_call.task
                        repl_response += f'\n<sub_agent task="{task_preview}">\n{sub_call.response}\n</sub_agent>'
                    
                    # Add state if there's meaningful state to show
                    if result.state_formatted and result.state_formatted != "(empty state)":
                        repl_response += f"\n<state>\n{result.state_formatted}\n</state>"
                    
                    turns.append(Turn(role="tool", content=repl_response, code=code))
                    messages.append({"role": "tool", "content": repl_response})
            
            # Max turns reached - no final answer
            return TrajectoryResult(
                success=False,
                final_answer="",
                turns=turns,
                num_code_blocks=num_code_blocks,
                system_prompt=system_prompt
            )
        
        finally:
            repl_client.delete_session()
    
    # ============================================================
    # Output Formatters
    # ============================================================
    
    def format_sharegpt(
        self,
        result: TrajectoryResult,
        task_id: str,
        mask_observations: bool = False,
    ) -> Dict[str, Any]:
        """Format trajectory as ShareGPT conversation with system prompt."""
        conversations = []
        
        # Use per-trajectory system prompt (important for OOLONG with varying context lengths)
        system_prompt = result.system_prompt if result.system_prompt else self.system_prompt
        conversations.append({"from": "system", "value": system_prompt})
        
        for turn in result.turns:
            if turn.role == "user":
                conversations.append({"from": "human", "value": turn.content})
            elif turn.role == "assistant":
                entry = {"from": "gpt", "value": turn.content}
                if mask_observations:
                    entry["loss_weight"] = 1.0
                conversations.append(entry)
            elif turn.role == "tool":
                entry = {"from": "tool", "value": turn.content}
                if mask_observations:
                    entry["loss_weight"] = 0.0  # Mask from loss
                conversations.append(entry)
        
        return {
            "id": task_id,
            "conversations": conversations,
            "metadata": {
                "success": result.success,
                "final_answer": result.final_answer,
                "num_turns": len(result.turns),
                "num_code_blocks": result.num_code_blocks,
            }
        }
    
    def format_chatml(
        self,
        result: TrajectoryResult,
        task_id: str,
    ) -> Dict[str, Any]:
        """Format trajectory with ChatML template."""
        parts = []
        
        # Use per-trajectory system prompt (important for OOLONG with varying context lengths)
        system_prompt = result.system_prompt if result.system_prompt else self.system_prompt
        parts.append(f"<|im_start|>system\n{system_prompt}<|im_end|>")
        
        for turn in result.turns:
            if turn.role == "user":
                parts.append(f"<|im_start|>user\n{turn.content}<|im_end|>")
            elif turn.role == "assistant":
                parts.append(f"<|im_start|>assistant\n{turn.content}<|im_end|>")
            elif turn.role == "tool":
                parts.append(f"<|im_start|>tool\n{turn.content}<|im_end|>")
        
        return {
            "id": task_id,
            "text": "\n".join(parts),
            "metadata": {
                "success": result.success,
                "final_answer": result.final_answer,
            }
        }

    # ============================================================
    # Dataset Field Helpers
    # ============================================================
    
    def _get_field(self, row: Dict[str, Any], field_name: str) -> Any:
        """Get a field from a dataset row using field mapping."""
        mapping = self.config.dataset.field_mapping
        mapped_name = mapping.get(field_name)
        
        if mapped_name is None:
            return None
        
        # Handle nested fields with dot notation
        if "." in mapped_name:
            parts = mapped_name.split(".")
            value = row
            for part in parts:
                if isinstance(value, dict) and part in value:
                    value = value[part]
                else:
                    return None
            return value
        
        return row.get(mapped_name)
    
    def _process_context(self, row: Dict[str, Any]) -> Optional[str]:
        """Process context field, handling complex nested structures."""
        processor = self.config.dataset.context_processor
        
        if processor == "hotpotqa":
            # HotpotQA has context as list of [title, sentences] pairs
            context_data = row.get("context", [])
            if not context_data:
                return None
            
            parts = []
            titles = context_data.get("title", []) if isinstance(context_data, dict) else []
            sentences_list = context_data.get("sentences", []) if isinstance(context_data, dict) else []
            
            for i, title in enumerate(titles):
                sentences = sentences_list[i] if i < len(sentences_list) else []
                if sentences:
                    text = " ".join(sentences)
                    parts.append(f"## {title}\n{text}")
            
            return "\n\n".join(parts) if parts else None
        
        elif processor == "oolong":
            # Oolong has context_window_text + question
            context_text = row.get("context_window_text", "")
            return context_text if context_text else None
        
        else:
            # Default: just get the mapped context field
            return self._get_field(row, "context")
    
    # ============================================================
    # Task Loaders
    # ============================================================
    
    def load_from_huggingface(
        self,
        dataset_name: str,
        config: Optional[str] = None,
        split: str = "train",
        limit: Optional[int] = None,
    ) -> List[Dict[str, Any]]:
        """
        Load tasks from a HuggingFace dataset using field mapping.
        
        Uses streaming to avoid downloading large datasets.
        
        Args:
            dataset_name: HuggingFace dataset name
            config: Dataset config (e.g., "distractor" for hotpotqa)
            split: Dataset split
            limit: Max number of tasks
            
        Returns:
            List of task dicts with id, prompt, expected_answer, context
        """
        print(f"Loading {dataset_name} (config={config}, split={split})...")
        print(f"Field mapping: {self.config.dataset.field_mapping}")
        
        # Use streaming to avoid downloading entire dataset
        if config:
            ds = load_dataset(dataset_name, config, split=split, streaming=True)
        else:
            ds = load_dataset(dataset_name, split=split, streaming=True)
        
        tasks = []
        for row in ds:
            if limit and len(tasks) >= limit:
                break
                
            task_id = self._get_field(row, "id")
            prompt = self._get_field(row, "prompt")
            expected = self._get_field(row, "expected_answer")
            context = self._process_context(row)
            
            if not prompt:
                continue
            
            # Generate ID if not present
            if not task_id:
                task_id = f"task_{len(tasks)}"
            
            tasks.append({
                "id": str(task_id),
                "prompt": prompt,
                "expected_answer": str(expected) if expected else None,
                "context": context,
            })
        
        print(f"Loaded {len(tasks)} tasks")
        return tasks
    
    def load_math_tasks(self, limit: Optional[int] = None) -> List[Dict[str, Any]]:
        """Load math tasks - fallback for math-python environment."""
        # Try configured dataset first
        if self.config.dataset.dataset_name:
            return self.load_from_huggingface(
                self.config.dataset.dataset_name,
                self.config.dataset.dataset_config,
                self.config.dataset.dataset_split,
                limit
            )
        
        # Fallback to Nemotron-Math-v2
        return self.load_from_huggingface(
            "nvidia/Nemotron-Math-v2",
            None,
            "medium",
            limit
        )
    
    def load_long_context_tasks(self, limit: Optional[int] = None) -> List[Dict[str, Any]]:
        """Load long-context tasks - fallback for oolong environment."""
        if self.config.dataset.dataset_name:
            return self.load_from_huggingface(
                self.config.dataset.dataset_name,
                self.config.dataset.dataset_config,
                self.config.dataset.dataset_split,
                limit
            )
        
        # Fallback to oolong
        return self.load_from_huggingface(
            "oolongbench/oolong-real",
            "dnd",
            "test",
            limit
        )
    
    def load_tasks(self, limit: Optional[int] = None) -> List[Dict[str, Any]]:
        """
        Load tasks based on environment type and config.
        
        Returns:
            List of task dicts
        """
        env = self.config.dataset.environment
        print(f"Environment: {env.value}")
        print(f"Dataset: {self.config.dataset.dataset_name}")
        print(f"Model: {self.config.main_model} ({self.config.main_client.value})")
        
        # If dataset is configured, use generic loader
        if self.config.dataset.dataset_name:
            return self.load_from_huggingface(
                self.config.dataset.dataset_name,
                self.config.dataset.dataset_config,
                self.config.dataset.dataset_split,
                limit
            )
        
        # Fallback based on environment
        if env == EnvironmentType.MATH_PYTHON:
            return self.load_math_tasks(limit)
        elif env == EnvironmentType.OOLONG:
            return self.load_long_context_tasks(limit)
        else:
            raise ValueError(f"No dataset configured for environment: {env}")
    
    # ============================================================
    # Task Processing
    # ============================================================
    
    async def process_task(self, task: Dict[str, Any]) -> Optional[Dict[str, Any]]:
        """
        Process a single task and generate trajectory.
        
        Args:
            task: Task dict with id, prompt, expected_answer, context
            
        Returns:
            Result dict or None if failed
        """
        task_id = task["id"]
        prompt = task["prompt"]
        context = task.get("context")
        expected = task.get("expected_answer")
        
        if self.config.debug:
            print_task_start(task_id, prompt, expected)
        
        try:
            result = await self.generate_trajectory(
                prompt=prompt,
                context=context,
                env_type=self.config.dataset.environment
            )
            
            # Extract boxed answers for validation
            boxed_answers = []
            if result.final_answer:
                boxed_answers = self._extract_boxed_answers(result.final_answer)
            
            # Simple validation against expected answer
            answer_correct = None
            if expected and boxed_answers:
                # Normalize for comparison (lowercase, strip whitespace)
                expected_norm = expected.lower().strip()
                for boxed in boxed_answers:
                    boxed_norm = boxed.lower().strip()
                    if boxed_norm == expected_norm or boxed_norm in expected_norm or expected_norm in boxed_norm:
                        answer_correct = True
                        break
                if answer_correct is None:
                    answer_correct = False
            
            # Record stats after computing answer_correct
            self.stats.record(result, answer_correct=answer_correct)
            
            # Build output record
            record = {
                "task_id": task_id,
                "prompt": prompt,
                "expected_answer": expected,
                "success": result.success,
                "final_answer": result.final_answer,
                "boxed_answers": boxed_answers,
                "answer_correct": answer_correct,
                "num_turns": len(result.turns),
                "num_code_blocks": result.num_code_blocks,
                "turns": [
                    {"role": t.role, "content": t.content, "code": t.code}
                    for t in result.turns
                ],
            }
            
            # Pretty print final answer and result in debug mode
            if self.config.debug:
                if result.final_answer:
                    print_final_answer(result.final_answer, boxed_answers, answer_correct)
                print_task_result(result.success, len(result.turns), result.num_code_blocks, answer_correct)
            
            # Save raw trace (thread-safe append)
            self._append_jsonl(self.results_file, record)
            
            # Save ShareGPT format
            if self.config.dataset.output_sharegpt:
                sharegpt = self.format_sharegpt(
                    result, task_id, 
                    mask_observations=self.config.dataset.mask_observations
                )
                self._append_jsonl(self.sharegpt_file, sharegpt)
            
            # Save ChatML format
            chatml = self.format_chatml(result, task_id)
            self._append_jsonl(self.chat_file, chatml)
            
            return record
            
        except Exception as e:
            print_colored(f"Error processing task {task_id}: {e}", Colors.RED)
            if self.config.debug:
                import traceback
                traceback.print_exc()
            # Record failed task in stats
            self.stats.total += 1
            self.stats.failed += 1
            self.stats.answers_unknown += 1
            return None
    
    # ============================================================
    # Batch Processing
    # ============================================================
    
    async def run_batch(self, tasks: List[Dict[str, Any]]) -> List[Optional[Dict[str, Any]]]:
        """
        Process a batch of tasks with concurrency control.
        
        Args:
            tasks: List of task dicts
            
        Returns:
            List of results (None for failed tasks)
        """
        semaphore = asyncio.Semaphore(self.config.dataset.batch_size)
        
        async def process_with_semaphore(task: Dict[str, Any]) -> Optional[Dict[str, Any]]:
            async with semaphore:
                try:
                    return await asyncio.wait_for(
                        self.process_task(task),
                        timeout=self.config.repl.timeout_seconds * self.config.repl.max_turns
                    )
                except asyncio.TimeoutError:
                    print(f"Task {task.get('id', 'unknown')} timed out")
                    self.stats.total += 1
                    self.stats.failed += 1
                    self.stats.answers_unknown += 1
                    return None
                except Exception as e:
                    print(f"Task {task.get('id', 'unknown')} failed: {e}")
                    self.stats.total += 1
                    self.stats.failed += 1
                    self.stats.answers_unknown += 1
                    return None
        
        results = await asyncio.gather(*[process_with_semaphore(t) for t in tasks])
        return list(results)
    
    # ============================================================
    # Main Runner
    # ============================================================
    
    async def run(self, limit: Optional[int] = None) -> None:
        """
        Run the pipeline with parallel batch processing.
        
        Args:
            limit: Override config limit
        """
        # Check server health
        if not self.repl_client.health_check():
            print("ERROR: REPL server not available!")
            print(f"Start it with: python -m datagenie.pythonformer.server --port 5003")
            return
        
        print("Loading tasks...")
        tasks = self.load_tasks(limit or self.config.dataset.limit)
        
        if not tasks:
            print("No tasks loaded!")
            return
        
        batch_size = self.config.dataset.batch_size
        print(f"\nProcessing {len(tasks)} tasks (batch_size={batch_size}, parallel={batch_size})...")
        print(f"Output: {self.output_dir}")
        print(f"ShareGPT: {self.sharegpt_file}")
        print()
        
        # Process tasks in batches with progress bar
        from tqdm.asyncio import tqdm as async_tqdm
        
        all_results = []
        for i in async_tqdm(range(0, len(tasks), batch_size), desc="Processing batches"):
            batch = tasks[i:i + batch_size]
            batch_results = await self.run_batch(batch)
            all_results.extend(batch_results)
        
        # Print stats
        print(self.stats.report())
        print(f"Tasks loaded: {len(tasks)}, Results collected: {len([r for r in all_results if r is not None])}")
        print(f"\nOutput files:")
        print(f"  Traces:   {self.results_file}")
        print(f"  ShareGPT: {self.sharegpt_file}")
        print(f"  ChatML:   {self.chat_file}")


async def main():
    """CLI entry point."""
    import argparse
    
    parser = argparse.ArgumentParser(description="Pythonformer Dataset Generation")
    parser.add_argument("--config", type=str, help="Path to config YAML")
    parser.add_argument("--limit", type=int, help="Limit number of tasks")
    parser.add_argument("--debug", action="store_true", help="Enable debug mode")
    args = parser.parse_args()
    
    # Load config
    if args.config:
        config = PythonformerConfig.from_yaml(args.config)
    else:
        config = PythonformerConfig()
    
    if args.debug:
        config.debug = True
    
    # Run pipeline
    pipeline = PythonformerPipeline(config)
    await pipeline.run(limit=args.limit)


if __name__ == "__main__":
    asyncio.run(main())
