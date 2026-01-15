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
from datagenie.pythonformer.reward_functions import (
    RewardFunction, TrajectoryMetrics, get_reward_function
)
from datagenie.pythonformer.prompts import (
    BASE_SYSTEM_PROMPT, OOLONG_SYSTEM_PROMPT, HOTPOTQA_SYSTEM_PROMPT
)
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
    # Reward tracking for RL/filtering
    total_reward: float = 0.0  # Cumulative reward across all steps
    step_rewards: List[float] = field(default_factory=list)  # Per-step rewards
    num_errors: int = 0  # Count of execution errors


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
    
    def __init__(self, config: PythonformerConfig):
        self.config = config
        self.stats = PipelineStats()
        
        # REPL client
        self.repl_client = REPLClient(server_url=config.repl.server_url)
        
        # LLM orchestrator for generating responses
        self.orchestrator = None
        self._init_orchestrator()
        
        # Reward function (pluggable, can be disabled)
        self.enable_rewards = getattr(config.dataset, 'enable_rewards', True)
        if self.enable_rewards:
            reward_fn_name = getattr(config.dataset, 'reward_function', 'simple')
            self.reward_function = get_reward_function(reward_fn_name)
            print(f"Reward function: {self.reward_function.name}")
        else:
            self.reward_function = None
            print("Reward computation: DISABLED")
        
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
        
        # Use task-specific prompts
        if env_type == EnvironmentType.OOLONG:
            return OOLONG_SYSTEM_PROMPT.format(
                context_length=context_length,
                env_tips=env_tips
            )
        elif env_type == EnvironmentType.HOTPOTQA:
            return HOTPOTQA_SYSTEM_PROMPT.format(
                context_length=context_length,
                env_tips=env_tips
            )
        
        # Default prompt for other tasks
        return BASE_SYSTEM_PROMPT.format(
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
            # Save context to file(s) based on environment type
            context_length = len(context) if context else 0
            
            if env_type == EnvironmentType.OOLONG and context:
                # OOLONG: Single large context file
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
            
            elif env_type == EnvironmentType.HOTPOTQA and context:
                # HotpotQA: Multiple document files for multi-hop reasoning
                # Parse context into separate documents and save each as a file
                if self.config.debug:
                    print(f"[DEBUG] HotpotQA context length: {len(context)} chars")
                    print(f"[DEBUG] Context starts with: {context[:200]}")
                
                documents = []
                current_doc = None
                
                for line in context.split('\n'):
                    if line.startswith('## '):
                        if current_doc:
                            documents.append(current_doc)
                        current_doc = {'title': line[3:].strip(), 'text': ''}
                    elif current_doc is not None:
                        current_doc['text'] += line + '\n'
                
                if current_doc:
                    documents.append(current_doc)
                
                if self.config.debug:
                    print(f"[DEBUG] Parsed {len(documents)} documents")
                    if documents:
                        print(f"[DEBUG] First doc title: {documents[0]['title']}")
                
                # Save each document as a separate file
                for i, doc in enumerate(documents):
                    # Create safe filename from title
                    safe_title = "".join(c if c.isalnum() or c in (' ', '-', '_') else '_' for c in doc['title'])
                    safe_title = safe_title.replace(' ', '_')[:50]  # Limit length
                    filename = f"doc_{i+1:02d}_{safe_title}.txt"
                    
                    if self.config.debug:
                        print(f"[DEBUG] Creating file: {filename} ({len(doc['text'])} chars)")
                    
                    # Escape content for Python string
                    escaped_text = doc['text'].replace('\\', '\\\\').replace("'", "\\'").replace('\n', '\\n').replace('\r', '\\r')
                    init_code = f"with open('{filename}', 'w') as f: f.write('{escaped_text}')"
                    result = repl_client.execute(init_code)
                    
                    if self.config.debug and result.error:
                        print(f"[DEBUG] Error creating {filename}: {result.error}")
                
                if self.config.debug:
                    print(f"Saved {len(documents)} documents to workspace ({context_length:,} chars total)")
            
            turns: List[Turn] = []
            num_code_blocks = 0
            
            # Reward tracking
            step_rewards: List[float] = []
            total_reward: float = 0.0
            num_errors: int = 0
            
            # Get initial state from REPL (shows pre-imported modules)
            initial_state = repl_client.get_state()
            initial_state_str = initial_state.get("state_formatted", "")
            
            # Build system prompt (with context length for OOLONG)
            system_prompt = self._build_system_prompt(env_type, context_length)
            
            # Initial user message
            user_msg = prompt
            if context and env_type == EnvironmentType.OOLONG:
                # For OOLONG: use <file> tag to indicate context location
                user_msg += f'\n\n<file name="context.txt" type="txt" chars="{context_length}">\n[Content saved to workspace - use read_file(\'context.txt\') to load]\n</file>'
            elif context and env_type == EnvironmentType.HOTPOTQA:
                # For HotpotQA: indicate multiple document files
                num_docs = context.count('\n## ') + (1 if context.startswith('## ') else 0)
                user_msg += f'\n\n<file name="documents" type="multiple" count="{num_docs}" chars="{context_length}">\n[{num_docs} documents saved to workspace as separate files]\n[Use list_files("doc_*.txt") to see all documents]\n[Use read_file("doc_XX_Title.txt") to read a specific document]\n</file>'
            elif context and env_type not in [EnvironmentType.OOLONG, EnvironmentType.HOTPOTQA]:
                # For other tasks: mention context variable
                user_msg += f"\n\n[Context available as `context` variable ({len(context):,} chars) or in 'context.txt']"
            
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
                        system_prompt=system_prompt,
                        total_reward=total_reward,
                        step_rewards=step_rewards,
                        num_errors=num_errors,
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
                            system_prompt=system_prompt,
                            total_reward=total_reward,
                            step_rewards=step_rewards,
                            num_errors=num_errors,
                        )
                    break
                
                for code in python_blocks:
                    # Pretty print code block in debug mode
                    if self.config.debug:
                        print_subheader(f"Executing Code Block #{num_code_blocks + 1}", Colors.MAGENTA)
                        print_code_block(code)
                    
                    result = repl_client.execute(code)
                    num_code_blocks += 1
                    
                    # Track rewards
                    step_rewards.append(result.reward)
                    total_reward += result.reward
                    if result.error:
                        num_errors += 1
                    
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
                system_prompt=system_prompt,
                total_reward=total_reward + self.config.repl.reward_on_failure if hasattr(self.config.repl, 'reward_on_failure') else total_reward - 0.1,
                step_rewards=step_rewards,
                num_errors=num_errors,
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
                # Reward tracking for filtering/analysis
                "total_reward": result.total_reward,
                "num_errors": result.num_errors,
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
            # HotpotQA has context as list of dicts with 'title' and 'sentences' keys
            # Example: [{"title": "Arthur's Magazine", "sentences": ["...", "..."]}, ...]
            context_data = row.get("context", [])
            
            if self.config.debug:
                print(f"[DEBUG] HotpotQA context_data type: {type(context_data)}")
                if isinstance(context_data, dict):
                    print(f"[DEBUG] Context is dict with keys: {context_data.keys()}")
                elif isinstance(context_data, list):
                    print(f"[DEBUG] Context is list with {len(context_data)} items")
            
            if not context_data:
                return None
            
            parts = []
            
            # Handle both formats: list of dicts OR dict with title/sentences lists
            if isinstance(context_data, dict):
                # Format: {"title": [...], "sentences": [[...], [...]]}
                titles = context_data.get("title", [])
                sentences_list = context_data.get("sentences", [])
                
                if self.config.debug:
                    print(f"[DEBUG] Dict format: {len(titles)} titles, {len(sentences_list)} sentence lists")
                
                for i, title in enumerate(titles):
                    if i < len(sentences_list):
                        sentences = sentences_list[i]
                        if sentences:
                            text = " ".join(sentences)
                            parts.append(f"## {title}\n{text}")
            
            elif isinstance(context_data, list):
                # Format: [{"title": "...", "sentences": [...]}, ...]
                if self.config.debug:
                    print(f"[DEBUG] List format: {len(context_data)} documents")
                
                for doc in context_data:
                    if isinstance(doc, dict):
                        title = doc.get("title", "")
                        sentences = doc.get("sentences", [])
                        if title and sentences:
                            text = " ".join(sentences)
                            parts.append(f"## {title}\n{text}")
            
            result = "\n\n".join(parts) if parts else None
            
            if self.config.debug:
                print(f"[DEBUG] Processed {len(parts)} documents, total length: {len(result) if result else 0} chars")
                if result:
                    print(f"[DEBUG] First 200 chars: {result[:200]}")
            
            return result
        
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
        elif env == EnvironmentType.HOTPOTQA:
            # Fallback to HotpotQA distractor
            return self.load_from_huggingface(
                "hotpotqa/hotpot_qa",
                "distractor",
                "validation",
                limit
            )
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
            
            # ============================================================
            # Compute reward using pluggable reward function (if enabled)
            # ============================================================
            reward_metadata = None
            
            if self.enable_rewards and self.reward_function:
                metrics = TrajectoryMetrics(
                    answer_correct=answer_correct,
                    num_turns=len(result.turns),
                    num_code_blocks=result.num_code_blocks,
                    num_errors=result.num_errors,
                    max_turns=self.config.repl.max_turns,
                    intermediate_rewards=result.step_rewards,
                    success=result.success,
                )
                
                reward_result = self.reward_function.compute(metrics)
                
                # Update trajectory result with computed rewards
                result.total_reward = reward_result["total_reward"]
                result.step_rewards.append(reward_result["final_step_reward"])
                reward_metadata = reward_result.get("metadata", {})
            else:
                # Rewards disabled - keep intermediate rewards only
                # total_reward will be sum of intermediate rewards (errors, etc.)
                pass
            
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
                # Reward tracking for filtering/analysis
                "total_reward": result.total_reward,
                "step_rewards": result.step_rewards,
                "num_errors": result.num_errors,
                "reward_metadata": reward_metadata,  # None if rewards disabled
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
