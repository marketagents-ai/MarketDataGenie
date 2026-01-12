"""
Main StructuredOutputPipeline class for dataset generation.

Supports two modes:
- Curriculum: Generate schemas/queries from task descriptions (CSV/JSONL)
- HuggingFace: Augment existing datasets to multi-turn
"""

import csv
import json
import asyncio
import time
from pathlib import Path
from typing import List, Dict, Any, Optional, Tuple
from dataclasses import dataclass, field
from datetime import datetime

from datasets import load_dataset, ReadInstruction
from tqdm.asyncio import tqdm
from tenacity import retry, stop_after_attempt, wait_random_exponential

from minference.lite.models import (
    LLMConfig, LLMClient, ResponseFormat,
    ChatThread, ChatMessage, MessageRole,
    SystemPrompt, ProcessedOutput, StructuredTool
)
from minference.lite.inference import InferenceOrchestrator

from datagenie.marketagents_structured_output.config import PipelineConfig, GenerationMode
from datagenie.marketagents_structured_output.agents import (
    create_schema_generator_agent,
    create_query_generator_agent,
    create_followup_agent,
    create_analysis_followup_agent,
    create_clarification_agent,
)
from datagenie.marketagents_structured_output.utils import (
    to_sharegpt_format,
    validate_think_block,
    has_think_block,
    get_reasoning_system_prompt,
    add_reasoning_reminder,
    strip_reasoning_reminder,
    print_messages,
    print_response,
    extract_tool_call_arguments,
)
from datagenie.marketagents_structured_output.utils.sharegpt import format_system_prompt_with_schema


@dataclass
class ProcessingStats:
    """Track processing statistics."""
    total: int = 0
    successful: int = 0
    failed_schema_gen: int = 0
    failed_query_gen: int = 0
    failed_structured_output: int = 0
    failed_followup: int = 0
    failed_analysis_followup: int = 0
    multi_turn_samples: int = 0
    analysis_followups: int = 0
    clarification_flows: int = 0
    failed_clarification: int = 0
    reasoning_generated: int = 0
    missing_think_blocks: int = 0
    truncated_conversations: int = 0
    _lock: asyncio.Lock = field(default_factory=asyncio.Lock, repr=False)
    
    async def inc(self, **kwargs: int) -> None:
        """Thread-safe increment of stat fields."""
        async with self._lock:
            for field_name, amount in kwargs.items():
                current = getattr(self, field_name, 0)
                setattr(self, field_name, current + amount)
    
    def report(self) -> str:
        """Generate statistics report."""
        success_rate = (self.successful / self.total * 100) if self.total > 0 else 0
        return f"""
=== Processing Statistics ===
Total processed:     {self.total:,}
Successful:          {self.successful:,}
Multi-turn samples:  {self.multi_turn_samples:,}
Analysis follow-ups: {self.analysis_followups:,}
Clarification flows: {self.clarification_flows:,}
Reasoning generated: {self.reasoning_generated:,}
Missing think blocks:{self.missing_think_blocks:,}
Truncated convos:    {self.truncated_conversations:,}
Failed schema gen:   {self.failed_schema_gen:,}
Failed query gen:    {self.failed_query_gen:,}
Failed struct output:{self.failed_structured_output:,}
Failed follow-up:    {self.failed_followup:,}
Failed analysis f/u: {self.failed_analysis_followup:,}
Failed clarification:{self.failed_clarification:,}
Success rate:        {success_rate:.1f}%
"""


class StructuredOutputPipeline:
    """
    Pipeline for generating structured output / JSON mode datasets.
    
    Supports two modes:
    1. Curriculum mode: Load tasks from CSV/JSONL, generate schemas and queries
    2. HuggingFace mode: Augment existing datasets to multi-turn
    """
    
    def __init__(self, config: PipelineConfig):
        self.config = config
        self.orchestrator = InferenceOrchestrator()
        self.stats = ProcessingStats()
        
        # Agent output tracking - nested by task_id
        self.agent_outputs: Dict[str, List[Dict[str, Any]]] = {}
        
        # Setup output directory
        self.output_dir = Path(config.output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        # Output files
        ts = int(time.time())
        mode_suffix = config.mode.value
        self.results_file = self.output_dir / f"structured_output_results_{mode_suffix}_{ts}.jsonl"
        self.sharegpt_file = self.output_dir / f"structured_output_sharegpt_{mode_suffix}_{ts}.jsonl"
        self.agent_outputs_file = self.output_dir / f"agent_outputs_{mode_suffix}_{ts}.jsonl"
    
    def _append_jsonl(self, path: Path, record: Dict[str, Any]) -> None:
        """Append a record to JSONL file."""
        with open(path, "a", encoding="utf-8") as f:
            f.write(json.dumps(record, ensure_ascii=False) + "\n")
    
    def _save_agent_output(
        self, 
        agent_name: str, 
        task_id: str, 
        input_data: Dict[str, Any],
        output_data: Optional[Dict[str, Any]],
        success: bool,
        error: Optional[str] = None
    ) -> None:
        """
        Save individual agent output for tracking and reuse.
        Outputs are accumulated per task_id and written when task completes.
        """
        record = {
            "agent_name": agent_name,
            "input_data": input_data,
            "output_data": output_data,
            "success": success,
            "error": error,
            "timestamp": datetime.now().isoformat()
        }
        
        # Accumulate outputs by task_id
        if task_id not in self.agent_outputs:
            self.agent_outputs[task_id] = []
        self.agent_outputs[task_id].append(record)
    
    def _flush_agent_outputs(self, task_id: str) -> None:
        """Write all accumulated agent outputs for a task to file."""
        if task_id in self.agent_outputs:
            record = {
                "id": task_id,
                "agents": self.agent_outputs[task_id]
            }
            self._append_jsonl(self.agent_outputs_file, record)
            # Clear from memory after writing
            del self.agent_outputs[task_id]
    
    # ============================================================
    # Task Loading Methods
    # ============================================================
    
    def load_curriculum_tasks(self, limit: Optional[int] = None) -> List[Dict[str, Any]]:
        """Load tasks from curriculum CSV or JSONL file."""
        curriculum_path = Path(self.config.curriculum_file)
        if not curriculum_path.exists():
            script_dir = Path(__file__).parent
            curriculum_path = script_dir / self.config.curriculum_file
        
        # Also check in json_mode folder
        if not curriculum_path.exists():
            alt_path = Path("datagenie/json_mode") / self.config.curriculum_file
            if alt_path.exists():
                curriculum_path = alt_path
        
        if not curriculum_path.exists():
            raise FileNotFoundError(f"Curriculum file not found: {self.config.curriculum_file}")
        
        tasks = []
        
        if curriculum_path.suffix == '.csv':
            with open(curriculum_path, 'r', encoding='utf-8') as f:
                reader = csv.DictReader(f)
                for row in reader:
                    if self.config.curriculum_categories:
                        if row.get('Category') not in self.config.curriculum_categories:
                            continue
                    if self.config.curriculum_subcategories:
                        if row.get('SubCategory') not in self.config.curriculum_subcategories:
                            continue
                    
                    # Parse schema if provided
                    schema = None
                    schema_str = row.get('Schema', '')
                    if schema_str:
                        try:
                            schema = json.loads(schema_str)
                        except json.JSONDecodeError:
                            pass
                    
                    tasks.append({
                        "id": f"curriculum_{len(tasks)}",
                        "category": row.get('Category', ''),
                        "subcategory": row.get('SubCategory', ''),
                        "task_description": row.get('Task', ''),
                        "schema": schema,
                        "mode": "curriculum"
                    })
        
        elif curriculum_path.suffix in ['.jsonl', '.json']:
            with open(curriculum_path, 'r', encoding='utf-8') as f:
                for line in f:
                    if line.strip():
                        row = json.loads(line)
                        if self.config.curriculum_categories:
                            if row.get('category') not in self.config.curriculum_categories:
                                continue
                        if self.config.curriculum_subcategories:
                            if row.get('subcategory') not in self.config.curriculum_subcategories:
                                continue
                        
                        tasks.append({
                            "id": row.get('id', f"curriculum_{len(tasks)}"),
                            "category": row.get('category', row.get('Category', '')),
                            "subcategory": row.get('subcategory', row.get('SubCategory', '')),
                            "task_description": row.get('task', row.get('Task', '')),
                            "schema": row.get('schema'),
                            "mode": "curriculum"
                        })
        
        if limit:
            tasks = tasks[:limit]
        
        print(f"Loaded {len(tasks)} curriculum tasks")
        return tasks
    
    def load_huggingface_tasks(
        self, 
        start_index: int = 0, 
        limit: Optional[int] = None
    ) -> List[Dict[str, Any]]:
        """Load tasks from HuggingFace dataset."""
        if not self.config.dataset_name:
            raise ValueError("No HuggingFace dataset specified")
        
        print(f"Loading HuggingFace dataset: {self.config.dataset_name}")
        
        if limit:
            end_index = start_index + limit
            ri = ReadInstruction('train', from_=start_index, to=end_index, unit='abs')
            dataset = load_dataset(self.config.dataset_name, split=ri)
        else:
            dataset = load_dataset(self.config.dataset_name, split=self.config.dataset_split)
        
        tasks = []
        for row in dataset:
            # Parse schema
            schema = row.get('schema', row.get('json_schema', {}))
            if isinstance(schema, str):
                try:
                    schema = json.loads(schema)
                except json.JSONDecodeError:
                    schema = {}
            
            # Extract query and output from conversations (ShareGPT format)
            # or from direct query/output fields
            conversations = row.get('conversations', [])
            query = ""
            output = ""
            
            if conversations:
                # ShareGPT format: [{"from": "system/human/gpt", "value": "..."}]
                for conv in conversations:
                    role = conv.get('from', '')
                    value = conv.get('value', '')
                    if role == 'human':
                        query = value
                    elif role == 'gpt':
                        output = value
            else:
                # Direct fields
                query = row.get('query', row.get('instruction', ''))
                output = row.get('output', row.get('response', ''))
            
            # Parse output if it's JSON string
            if isinstance(output, str) and output.strip():
                try:
                    output = json.loads(output)
                except json.JSONDecodeError:
                    pass  # Keep as string
            
            # Handle nested "raw" wrapper (some datasets store output as {"raw": "<json_string>"})
            if isinstance(output, dict) and 'raw' in output and len(output) == 1:
                raw_value = output['raw']
                if isinstance(raw_value, str):
                    try:
                        output = json.loads(raw_value)
                    except json.JSONDecodeError:
                        output = raw_value  # Keep as string if not valid JSON
                else:
                    output = raw_value
            
            if not query or not schema:
                continue
            
            tasks.append({
                "id": row.get('id', str(len(tasks))),
                "query": query,
                "schema": schema,
                "output": output,
                "category": row.get('category', ''),
                "subcategory": row.get('subcategory', ''),
                "mode": "huggingface"
            })
        
        print(f"Loaded {len(tasks)} HuggingFace tasks")
        return tasks
    
    # ============================================================
    # Agent Runner Methods
    # ============================================================
    
    @retry(wait=wait_random_exponential(multiplier=2, max=60), stop=stop_after_attempt(3))
    async def _run_schema_generator_agent(
        self, task_description: str, category: str, subcategory: str, task_id: str
    ) -> Optional[Dict]:
        """Run schema generation agent with retry."""
        cfg = self.config.agents.schema_generator
        agent = create_schema_generator_agent(
            task_description, category, subcategory,
            self.orchestrator, cfg.model,
            llm_client=cfg.get_llm_client(),
            temperature=cfg.temperature,
            max_tokens=cfg.max_tokens
        )
        try:
            result = await agent.execute()
            if not isinstance(result, dict):
                self._save_agent_output(
                    "schema_generator", task_id,
                    {"task_description": task_description, "category": category, "subcategory": subcategory},
                    None, success=False, error="Result not a dict"
                )
                return None
            # Handle XML tool call response
            extracted = extract_tool_call_arguments(result)
            output = extracted if extracted else result
            
            self._save_agent_output(
                "schema_generator", task_id,
                {"task_description": task_description, "category": category, "subcategory": subcategory},
                output, success=bool(output)
            )
            return output
        except Exception as e:
            self._save_agent_output(
                "schema_generator", task_id,
                {"task_description": task_description, "category": category, "subcategory": subcategory},
                None, success=False, error=str(e)
            )
            print(f"Schema generator agent failed: {e}")
            return None
    
    @retry(wait=wait_random_exponential(multiplier=2, max=60), stop=stop_after_attempt(3))
    async def _run_query_generator_agent(
        self, json_schema: Dict, task_description: str, task_id: str
    ) -> Optional[Dict]:
        """Run query generation agent with retry."""
        cfg = self.config.agents.query_generator
        agent = create_query_generator_agent(
            json_schema, task_description,
            self.orchestrator, cfg.model,
            llm_client=cfg.get_llm_client(),
            temperature=cfg.temperature,
            max_tokens=cfg.max_tokens
        )
        try:
            result = await agent.execute()
            if not isinstance(result, dict):
                self._save_agent_output(
                    "query_generator", task_id,
                    {"json_schema": json_schema, "task_description": task_description},
                    None, success=False, error="Result not a dict"
                )
                return None
            # Handle XML tool call response
            extracted = extract_tool_call_arguments(result)
            output = extracted if extracted else result
            
            self._save_agent_output(
                "query_generator", task_id,
                {"json_schema": json_schema, "task_description": task_description},
                output, success=bool(output and output.get("query"))
            )
            return output
        except Exception as e:
            self._save_agent_output(
                "query_generator", task_id,
                {"json_schema": json_schema, "task_description": task_description},
                None, success=False, error=str(e)
            )
            print(f"Query generator agent failed: {e}")
            return None
    
    @retry(wait=wait_random_exponential(multiplier=2, max=60), stop=stop_after_attempt(2))
    async def _run_followup_agent(
        self, messages: List[Dict], json_schema: Dict, task_id: str, turn: int
    ) -> Optional[Dict]:
        """Run follow-up query agent with retry."""
        cfg = self.config.agents.followup_generator
        agent = create_followup_agent(
            messages, json_schema, self.orchestrator, cfg.model,
            llm_client=cfg.get_llm_client(),
            temperature=cfg.temperature,
            max_tokens=cfg.max_tokens
        )
        try:
            result = await agent.execute()
            if not isinstance(result, dict):
                self._save_agent_output(
                    f"followup_generator_turn_{turn}", task_id,
                    {"messages_count": len(messages)},
                    None, success=False, error="Result not a dict"
                )
                return None
            # Handle XML tool call response
            extracted = extract_tool_call_arguments(result)
            output = extracted if extracted else result
            
            self._save_agent_output(
                f"followup_generator_turn_{turn}", task_id,
                {"messages_count": len(messages)},
                output, success=bool(output and output.get("content"))
            )
            return output
        except Exception as e:
            self._save_agent_output(
                f"followup_generator_turn_{turn}", task_id,
                {"messages_count": len(messages)},
                None, success=False, error=str(e)
            )
            print(f"Follow-up agent failed: {e}")
            return None
    
    @retry(wait=wait_random_exponential(multiplier=2, max=60), stop=stop_after_attempt(2))
    async def _run_analysis_followup_agent(
        self, messages: List[Dict], task_id: str
    ) -> Optional[Dict]:
        """Run analysis follow-up agent with retry."""
        cfg = self.config.agents.analysis_followup
        agent = create_analysis_followup_agent(
            messages, self.orchestrator, cfg.model,
            llm_client=cfg.get_llm_client(),
            temperature=cfg.temperature,
            max_tokens=cfg.max_tokens,
            generate_reasoning=self.config.generate_reasoning
        )
        try:
            result = await agent.execute()
            if not isinstance(result, dict):
                self._save_agent_output(
                    "analysis_followup", task_id,
                    {"messages_count": len(messages)},
                    None, success=False, error="Result not a dict"
                )
                return None
            # Handle XML tool call response
            extracted = extract_tool_call_arguments(result)
            output = extracted if extracted else result
            
            self._save_agent_output(
                "analysis_followup", task_id,
                {"messages_count": len(messages)},
                output, success=bool(output and output.get("followup_question") and output.get("response"))
            )
            return output
        except Exception as e:
            self._save_agent_output(
                "analysis_followup", task_id,
                {"messages_count": len(messages)},
                None, success=False, error=str(e)
            )
            print(f"Analysis follow-up agent failed: {e}")
            return None

    @retry(wait=wait_random_exponential(multiplier=2, max=60), stop=stop_after_attempt(2))
    async def _run_clarification_agent(
        self, original_query: str, assistant_response: str, json_schema: Dict, task_id: str
    ) -> Optional[Dict]:
        """Run clarification agent to provide missing details."""
        cfg = self.config.agents.clarification_agent
        agent = create_clarification_agent(
            original_query, assistant_response, json_schema,
            self.orchestrator, cfg.model,
            llm_client=cfg.get_llm_client(),
            temperature=cfg.temperature,
            max_tokens=cfg.max_tokens
        )
        try:
            result = await agent.execute()
            if not isinstance(result, dict):
                self._save_agent_output(
                    "clarification_agent", task_id,
                    {"original_query": original_query, "assistant_response": assistant_response[:200]},
                    None, success=False, error="Result not a dict"
                )
                return None
            # Handle XML tool call response
            extracted = extract_tool_call_arguments(result)
            output = extracted if extracted else result
            
            self._save_agent_output(
                "clarification_agent", task_id,
                {"original_query": original_query, "assistant_response": assistant_response[:200]},
                output, success=bool(output and output.get("content"))
            )
            return output
        except Exception as e:
            self._save_agent_output(
                "clarification_agent", task_id,
                {"original_query": original_query, "assistant_response": assistant_response[:200]},
                None, success=False, error=str(e)
            )
            print(f"Clarification agent failed: {e}")
            return None

    # ============================================================
    # Structured Output Generation
    # ============================================================
    
    async def _generate_structured_output(
        self,
        user_query: str,
        json_schema: Dict[str, Any],
        task_id: str,
        conversation_history: Optional[List[Dict[str, Any]]] = None,
    ) -> Tuple[bool, Optional[str], Optional[Dict[str, Any]]]:
        """
        Generate structured output for a user query given a JSON schema.
        
        When generate_reasoning=True:
          - Uses ResponseFormat.text with tool schema embedded in system prompt
          - Model outputs <think>...</think> followed by <tool_call>...</tool_call>
          - We parse both from content
        
        When generate_reasoning=False:
          - Uses ResponseFormat.tool with forced_output
          - Model outputs directly to tool_calls (no reasoning)
        
        Args:
            user_query: The user's query
            json_schema: JSON schema defining the expected output structure
            task_id: Task ID for tracking
            conversation_history: Optional previous conversation for context (follow-ups)
            
        Returns:
            Tuple of (success, assistant_content, json_output)
        """
        schema_json = json.dumps(json_schema, indent=2)
        cfg = self.config.agents.structured_output
        
        # Create tool definition for schema
        tool_def = {
            "name": "generate_output",
            "description": json_schema.get('description', 'Generate structured output conforming to the schema'),
            "parameters": json_schema
        }
        
        if self.config.generate_reasoning:
            # Text mode with embedded tools for reasoning support
            # Hermes format: embed tool in system prompt, model outputs <think> + <tool_call>
            tool_json = json.dumps(tool_def, indent=2)
            system_prompt = (
                "You are a deep thinking AI, you may use extremely long chains of thought to deeply "
                "consider the problem and deliberate with yourself via systematic reasoning processes "
                "to help come to a correct solution prior to answering. You should enclose your thoughts "
                "and internal monologue inside <think> </think> tags, and then provide your solution "
                "or response to the problem.\n\n"
                "IMPORTANT: You MUST start EVERY response with <think></think> tags, including:\n"
                "- When generating structured output\n"
                "- When updating or modifying previous output\n"
                "- When answering follow-up questions\n\n"
                "You have access to the following function:\n\n"
                f"{tool_json}\n\n"
                "After your reasoning in <think> tags, call the function using:\n"
                "<tool_call>\n"
                '{"name": "generate_output", "arguments": {...}}\n'
                "</tool_call>\n\n"
                "CRITICAL: The 'arguments' field must contain the actual data values that conform to the schema, "
                "NOT the schema definition itself. Generate real data based on the user's request."
            )
            
            # Build history from conversation context (for follow-ups)
            history = []
            if conversation_history:
                for msg in conversation_history:
                    role = msg.get("role", "")
                    content = msg.get("content", "")
                    if role == "user":
                        history.append(ChatMessage(role=MessageRole.user, content=content))
                    elif role == "assistant":
                        # For assistant messages, extract just the JSON data for clean context
                        # The model should see clean JSON, not XML tags or wrappers
                        clean_content = content
                        
                        # Check if content has <tool_call> tags - extract arguments
                        if "<tool_call>" in content and "</tool_call>" in content:
                            extracted = extract_tool_call_arguments({"raw": content})
                            if extracted:
                                clean_content = json.dumps(extracted, indent=2)
                        else:
                            # Plain JSON - try to parse and re-serialize cleanly
                            try:
                                json_obj = json.loads(content)
                                # Avoid nested "raw" wrappers
                                if isinstance(json_obj, dict) and 'raw' in json_obj and len(json_obj) == 1:
                                    inner = json_obj['raw']
                                    if isinstance(inner, str):
                                        try:
                                            json_obj = json.loads(inner)
                                        except json.JSONDecodeError:
                                            pass
                                clean_content = json.dumps(json_obj, indent=2)
                            except json.JSONDecodeError:
                                # Keep as-is if not valid JSON
                                pass
                        
                        history.append(ChatMessage(role=MessageRole.assistant, content=clean_content))
            
            chat_thread = ChatThread(
                name=f"structured-output-{task_id}",
                system_prompt=SystemPrompt(name="sys", content=system_prompt),
                llm_config=LLMConfig(
                    client=cfg.get_llm_client(),
                    model=cfg.model,
                    temperature=cfg.temperature,
                    max_tokens=cfg.max_tokens,
                    response_format=ResponseFormat.text
                ),
                tools=[],  # No tools - using text mode
                history=history,
                new_message=user_query
            )
        else:
            # Tool mode - native tool calling (no reasoning)
            system_prompt = (
                "You are a helpful assistant that generates structured JSON output. "
                "Your response must be valid JSON that conforms to the provided schema.\n\n"
                f"JSON Schema:\n{schema_json}\n\n"
                "You MUST call the generate_output function with your response. "
                "The arguments should contain actual data values, NOT the schema definition."
            )
            
            output_tool = StructuredTool(
                name="generate_output",
                description=json_schema.get('description', 'Generate structured output conforming to the schema'),
                json_schema=json_schema
            )
            
            # Build history from conversation context (for follow-ups)
            history = []
            if conversation_history:
                for msg in conversation_history:
                    role = msg.get("role", "")
                    content = msg.get("content", "")
                    if role == "user":
                        history.append(ChatMessage(role=MessageRole.user, content=content))
                    elif role == "assistant":
                        history.append(ChatMessage(role=MessageRole.assistant, content=content))
            
            chat_thread = ChatThread(
                name=f"structured-output-{task_id}",
                system_prompt=SystemPrompt(name="sys", content=system_prompt),
                llm_config=LLMConfig(
                    client=cfg.get_llm_client(),
                    model=cfg.model,
                    temperature=cfg.temperature,
                    max_tokens=cfg.max_tokens,
                    response_format=ResponseFormat.tool
                ),
                tools=[output_tool],
                history=history,
                new_message=user_query
            )
            chat_thread.forced_output = output_tool
        
        # Debug: Print request
        if self.config.debug_print_messages:
            print_messages(
                [{"role": "system", "content": system_prompt}, {"role": "user", "content": user_query}],
                title=f"Structured Output Request ({task_id})",
                truncate=True
            )
        
        try:
            outputs = await self.orchestrator.run_parallel_ai_completion([chat_thread])
            output = outputs[0] if outputs else None
        except Exception as e:
            print(f"Task {task_id}: Structured output generation failed: {e}")
            return False, None, None
        
        if not output:
            return False, None, None
        
        # Extract JSON from response
        json_output = None
        
        # First try json_object from native tool call
        if output.json_object and output.json_object.object:
            json_output = output.json_object.object
        
        # Handle XML tool call response (text mode or Hermes via litellm)
        if not json_output and output.content:
            extracted = extract_tool_call_arguments({"raw": output.content})
            if extracted:
                json_output = extracted
        
        # Fallback: try to parse JSON directly from content
        if not json_output and output.content:
            try:
                json_output = json.loads(output.content)
            except json.JSONDecodeError:
                # Try to extract JSON from markdown code block
                import re
                json_match = re.search(r'```(?:json)?\s*([\s\S]*?)\s*```', output.content)
                if json_match:
                    try:
                        json_output = json.loads(json_match.group(1))
                    except json.JSONDecodeError:
                        pass
        
        # Debug: Print response
        if self.config.debug_print_messages:
            response_str = output.content or ""
            if not response_str and json_output:
                response_str = json.dumps(json_output, indent=2)
            print_response(response_str, title=f"Structured Output Response ({task_id})")
        
        # Validate reasoning if enabled - STRICT: require <think> blocks when reasoning is enabled
        if self.config.generate_reasoning:
            if not output.content or not has_think_block(output.content):
                print(f"Task {task_id}: Missing <think> block in structured output - failing task")
                await self.stats.inc(missing_think_blocks=1)
                # Return failure - we need <think> blocks for training data quality
                return False, output.content, None
            else:
                await self.stats.inc(reasoning_generated=1)
        
        return bool(json_output), output.content, json_output
    
    async def _generate_structured_output_with_clarification(
        self,
        user_query: str,
        json_schema: Dict[str, Any],
        task_id: str,
        conversation_history: Optional[List[Dict[str, Any]]] = None,
        is_first_turn: bool = True,
        original_query: str = "",
    ) -> Tuple[bool, Optional[str], Optional[Dict[str, Any]], List[Dict[str, Any]]]:
        """
        Generate structured output with clarification flow support.
        
        If the model returns plain text (asking for clarification) instead of JSON,
        this method will generate a user clarification and retry.
        
        Args:
            user_query: The user's query
            json_schema: JSON schema defining the expected output structure
            task_id: Task ID for tracking
            conversation_history: Optional previous conversation for context
            is_first_turn: Whether this is the first turn (clarification only on first turn)
            original_query: Original user query (for clarification context)
            
        Returns:
            Tuple of (success, assistant_content, json_output, clarification_messages)
            clarification_messages contains any clarification exchanges that occurred
        """
        clarification_messages = []
        current_query = user_query
        current_history = conversation_history.copy() if conversation_history else []
        
        for clarification_turn in range(self.config.max_clarification_turns + 1):
            # Try to generate structured output
            success, content, json_output = await self._generate_structured_output(
                current_query, json_schema, f"{task_id}_clar{clarification_turn}",
                conversation_history=current_history if current_history else None
            )
            
            if success and json_output:
                # Got valid JSON output
                return True, content, json_output, clarification_messages
            
            # No JSON output - check if this is a clarification request
            if content and not json_output:
                # Model returned text without JSON - likely asking for clarification
                # or explaining why it can't fulfill the request
                
                # Check if clarification is enabled
                if not self.config.allow_clarification_flow:
                    if self.config.require_json_on_first_turn and is_first_turn:
                        print(f"Task {task_id}: First turn did not produce JSON (clarification disabled)")
                    # Return the text content - caller can decide what to do with it
                    return False, content, None, clarification_messages
                
                # Check if we've exhausted clarification turns
                if clarification_turn >= self.config.max_clarification_turns:
                    print(f"Task {task_id}: Max clarification turns ({self.config.max_clarification_turns}) reached")
                    return False, content, None, clarification_messages
                
                # Generate clarification response
                if self.config.debug_print_messages:
                    print(f"Task {task_id}: Model asked for clarification, generating user response...")
                
                clarification_result = await self._run_clarification_agent(
                    original_query or user_query, content, json_schema, task_id
                )
                
                if not clarification_result or not clarification_result.get('content'):
                    print(f"Task {task_id}: Failed to generate clarification response")
                    await self.stats.inc(failed_clarification=1)
                    return False, content, None, clarification_messages
                
                clarification_content = clarification_result['content']
                
                # Add the clarification exchange to messages
                clarification_messages.append({"role": "assistant", "content": content})
                clarification_messages.append({"role": "user", "content": clarification_content})
                
                # Update history for next attempt
                current_history.append({"role": "user", "content": current_query})
                current_history.append({"role": "assistant", "content": content})
                
                # Update query for next iteration
                current_query = clarification_content
                is_first_turn = False  # No longer first turn after clarification
                
                await self.stats.inc(clarification_flows=1)
                
                if self.config.debug_print_messages:
                    print(f"Task {task_id}: Clarification turn {clarification_turn + 1}, retrying with: {clarification_content[:100]}...")
                
                continue
            
            # No content at all - failure
            return False, None, None, clarification_messages
        
        # Should not reach here, but just in case
        return False, None, None, clarification_messages

    # ============================================================
    # Task Processing
    # ============================================================
    
    async def process_task(self, task: Dict[str, Any]) -> Optional[Dict]:
        """Process a single task through the full pipeline."""
        await self.stats.inc(total=1)
        
        task_id = task.get('id', str(time.time()))
        task_mode = task.get('mode', 'curriculum')
        
        if task_mode == 'curriculum':
            return await self._process_curriculum_task(task, task_id)
        else:
            return await self._process_huggingface_task(task, task_id)
    
    async def _process_curriculum_task(
        self, task: Dict[str, Any], task_id: str
    ) -> Optional[Dict]:
        """
        Process a curriculum task.
        
        Steps:
        1. Generate JSON schema (if not provided)
        2. Generate user query
        3. Generate structured output
        4. Optionally generate follow-up
        """
        task_description = task.get('task_description', '')
        category = task.get('category', '')
        subcategory = task.get('subcategory', '')
        provided_schema = task.get('schema')
        
        if not task_description:
            self._flush_agent_outputs(task_id)
            return None
        
        # Step 1: Get or generate JSON schema
        if provided_schema:
            json_schema = provided_schema
        else:
            schema_result = await self._run_schema_generator_agent(
                task_description, category, subcategory, task_id
            )
            
            if not schema_result or not schema_result.get('json_schema'):
                await self.stats.inc(failed_schema_gen=1)
                self._flush_agent_outputs(task_id)
                return None
            
            json_schema = schema_result.get('json_schema', {})
        
        # Step 2: Generate user query
        query_result = await self._run_query_generator_agent(
            json_schema, task_description, task_id
        )
        
        if not query_result or not query_result.get('query'):
            await self.stats.inc(failed_query_gen=1)
            self._flush_agent_outputs(task_id)
            return None
        
        user_query = query_result.get('query', '')
        expected_fields = query_result.get('expected_fields', [])
        
        # Step 3: Generate structured output (with clarification support)
        success, assistant_content, json_output, clarification_msgs = await self._generate_structured_output_with_clarification(
            user_query, json_schema, task_id,
            is_first_turn=True,
            original_query=user_query
        )
        
        if not success or not json_output:
            await self.stats.inc(failed_structured_output=1)
            self._flush_agent_outputs(task_id)
            return None
        
        # Build messages list
        # Start with any clarification exchanges
        messages = []
        
        if clarification_msgs:
            # Add original user query first
            messages.append({"role": "user", "content": user_query})
            # Add clarification exchanges
            messages.extend(clarification_msgs)
        else:
            # No clarification - just add user query
            messages.append({"role": "user", "content": user_query})
        
        # Add final assistant response
        # When reasoning is enabled, use full content (includes <think> blocks)
        # Otherwise, use clean JSON output
        if self.config.generate_reasoning and assistant_content:
            assistant_msg_content = assistant_content
        else:
            assistant_msg_content = json.dumps(json_output, indent=2)
        
        messages.append({"role": "assistant", "content": assistant_msg_content})
        
        # Step 4: Generate modification follow-ups (if enabled)
        # max_turns controls how many modification follow-ups to generate
        # Turn 1 is the initial query, so we do (max_turns - 1) modification follow-ups
        if self.config.generate_followup:
            num_modification_turns = self.config.max_turns - 1  # Reserve turn 1 for initial query
            
            for turn in range(1, num_modification_turns + 1):
                followup_result = await self._run_followup_agent(
                    messages, json_schema, task_id, turn=turn
                )
                
                if not followup_result or not followup_result.get('content'):
                    # No more follow-ups generated - stop
                    if self.config.debug_print_messages:
                        print(f"Task {task_id}: No follow-up generated at turn {turn}, stopping")
                    break
                
                followup_query = followup_result.get('content', '')
                
                # Add reasoning reminder to follow-up query when reasoning is enabled
                # This will be stripped during ShareGPT conversion
                followup_query_with_reminder = followup_query
                if self.config.generate_reasoning:
                    followup_query_with_reminder = add_reasoning_reminder(followup_query)
                
                messages.append({"role": "user", "content": followup_query_with_reminder})
                
                # Generate follow-up response with clarification support
                success_fu, content_fu, json_output_fu, clarification_msgs_fu = await self._generate_structured_output_with_clarification(
                    followup_query_with_reminder, json_schema, f"{task_id}_followup_{turn}",
                    conversation_history=messages[:-1],  # Pass history without the new user message
                    is_first_turn=False,  # This is a follow-up turn
                    original_query=followup_query
                )
                
                if success_fu and json_output_fu:
                    # Validate <think> block when reasoning is enabled
                    if self.config.generate_reasoning:
                        # content_fu should have the full response including <think> blocks
                        if not content_fu or not has_think_block(content_fu):
                            print(f"Task {task_id}: Follow-up turn {turn} missing <think> block - stopping but keeping previous valid turns")
                            await self.stats.inc(missing_think_blocks=1)
                            # Remove the follow-up user message and stop (keep previous valid conversation)
                            messages.pop()
                            break
                    
                    # Add any clarification exchanges that occurred
                    if clarification_msgs_fu:
                        # Remove the original followup user message (we'll rebuild)
                        messages.pop()
                        messages.append({"role": "user", "content": followup_query_with_reminder})
                        messages.extend(clarification_msgs_fu)
                    
                    # Use full content with reasoning if enabled
                    if self.config.generate_reasoning and content_fu:
                        followup_msg_content = content_fu
                        await self.stats.inc(reasoning_generated=1)
                    else:
                        followup_msg_content = json.dumps(json_output_fu, indent=2)
                    messages.append({"role": "assistant", "content": followup_msg_content})
                    await self.stats.inc(multi_turn_samples=1)
                elif content_fu and self.config.generate_reasoning:
                    # Model returned text response (e.g., explaining why it can't do the modification)
                    # This is valid if it has <think> blocks - treat as an explanation response
                    if has_think_block(content_fu):
                        messages.append({"role": "assistant", "content": content_fu})
                        await self.stats.inc(multi_turn_samples=1)
                        await self.stats.inc(reasoning_generated=1)
                        if self.config.debug_print_messages:
                            print(f"Task {task_id}: Follow-up turn {turn} got text explanation instead of JSON (valid)")
                    else:
                        # No <think> block - remove the follow-up user message and stop (keep previous valid conversation)
                        print(f"Task {task_id}: Follow-up turn {turn} text response missing <think> block - stopping but keeping previous valid turns")
                        messages.pop()
                        await self.stats.inc(missing_think_blocks=1)
                        break
                else:
                    # Remove the follow-up user message if response failed and stop (keep previous valid conversation)
                    print(f"Task {task_id}: Follow-up turn {turn} failed - stopping but keeping previous valid turns")
                    messages.pop()
                    await self.stats.inc(failed_followup=1)
                    break
        
        # Step 5: Generate analysis follow-up (if enabled)
        if self.config.generate_analysis_followup and len(messages) >= 2:
            analysis_result = await self._run_analysis_followup_agent(messages, task_id)
            
            if analysis_result:
                followup_q = analysis_result.get('followup_question', '')
                followup_resp = analysis_result.get('response', '')
                
                if followup_q and followup_resp:
                    # Validate reasoning if enabled
                    if self.config.generate_reasoning:
                        if not has_think_block(followup_resp):
                            print(f"Task {task_id}: Analysis follow-up missing <think> tags")
                            await self.stats.inc(missing_think_blocks=1)
                            await self.stats.inc(failed_analysis_followup=1)
                        else:
                            messages.append({"role": "user", "content": followup_q})
                            messages.append({"role": "assistant", "content": followup_resp})
                            await self.stats.inc(reasoning_generated=1)
                            await self.stats.inc(analysis_followups=1)
                    else:
                        messages.append({"role": "user", "content": followup_q})
                        messages.append({"role": "assistant", "content": followup_resp})
                        await self.stats.inc(analysis_followups=1)
                else:
                    await self.stats.inc(failed_analysis_followup=1)
            else:
                await self.stats.inc(failed_analysis_followup=1)
        
        # Build result
        result = {
            "id": task_id,
            "mode": "curriculum",
            "category": category,
            "subcategory": subcategory,
            "task_description": task_description,
            "json_schema": json_schema,
            "messages": messages,
            "timestamp": datetime.now().isoformat()
        }
        
        # Save results
        self._append_jsonl(self.results_file, result)
        
        if self.config.output_sharegpt:
            sharegpt_metadata = {
                "id": task_id,
                "source": "curriculum",
                "task": task_description,
                "category": category,
                "subcategory": subcategory
            }
            sharegpt = to_sharegpt_format(
                messages, json_schema, sharegpt_metadata,
                include_reasoning_instructions=self.config.generate_reasoning
            )
            if sharegpt:
                self._append_jsonl(self.sharegpt_file, sharegpt)
        
        # Debug: Print final messages
        if self.config.debug_print_messages:
            print_messages(
                messages,
                title=f"Final Messages for Task {task_id}",
                truncate=True,
                max_content_length=1000
            )
        
        # Flush agent outputs
        self._flush_agent_outputs(task_id)
        
        await self.stats.inc(successful=1)
        return result
    
    async def _process_huggingface_task(
        self, task: Dict[str, Any], task_id: str
    ) -> Optional[Dict]:
        """
        Process a HuggingFace dataset task.
        
        Augments existing single-turn to multi-turn.
        
        When generate_reasoning=True, we regenerate the first turn to include
        <think> blocks, rather than using the existing output from the dataset.
        """
        user_query = task.get('query', '')
        json_schema = task.get('schema', {})
        existing_output = task.get('output', '')
        
        if not user_query or not json_schema:
            self._flush_agent_outputs(task_id)
            return None
        
        # Parse existing output if string
        if isinstance(existing_output, str):
            try:
                existing_output = json.loads(existing_output)
            except json.JSONDecodeError:
                existing_output = {}
        
        # Handle nested "raw" wrapper
        if isinstance(existing_output, dict) and 'raw' in existing_output and len(existing_output) == 1:
            raw_value = existing_output['raw']
            if isinstance(raw_value, str):
                try:
                    existing_output = json.loads(raw_value)
                except json.JSONDecodeError:
                    existing_output = {"data": raw_value}  # Wrap in data key if not valid JSON
            else:
                existing_output = raw_value
        
        # Build initial messages
        messages = []
        
        # When reasoning is enabled, regenerate the first turn to include <think> blocks
        # Otherwise, use the existing output from the dataset
        if self.config.generate_reasoning:
            # Regenerate first turn with reasoning (with clarification support)
            if self.config.debug_print_messages:
                print(f"Task {task_id}: Regenerating first turn with reasoning enabled")
            
            success, content, json_output, clarification_msgs = await self._generate_structured_output_with_clarification(
                user_query, json_schema, task_id,
                is_first_turn=True,
                original_query=user_query
            )
            
            if not success or not json_output:
                await self.stats.inc(failed_structured_output=1)
                self._flush_agent_outputs(task_id)
                return None
            
            # Debug: Check if content has <think> block
            if self.config.debug_print_messages:
                has_think = "<think>" in (content or "")
                print(f"Task {task_id}: First turn content has <think>: {has_think}")
                if content:
                    print(f"Task {task_id}: First turn content preview: {content[:200]}...")
            
            # Build messages with any clarification exchanges
            if clarification_msgs:
                messages.append({"role": "user", "content": user_query})
                messages.extend(clarification_msgs)
            else:
                messages.append({"role": "user", "content": user_query})
            
            # Use full content with reasoning
            if content:
                first_assistant_content = content
            else:
                first_assistant_content = json.dumps(json_output, indent=2)
            
            messages.append({"role": "assistant", "content": first_assistant_content})
        else:
            # Use existing output from dataset (no reasoning)
            messages.append({"role": "user", "content": user_query})
            messages.append({
                "role": "assistant", 
                "content": json.dumps(existing_output, indent=2) if existing_output else ""
            })
        
        # Generate modification follow-ups if enabled
        # max_turns controls how many modification follow-ups to generate
        # Turn 1 is the initial query, so we do (max_turns - 1) modification follow-ups
        if self.config.generate_followup:
            num_modification_turns = self.config.max_turns - 1  # Reserve turn 1 for initial query
            
            for turn in range(1, num_modification_turns + 1):
                followup_result = await self._run_followup_agent(
                    messages, json_schema, task_id, turn=turn
                )
                
                if not followup_result or not followup_result.get('content'):
                    # No more follow-ups generated - stop
                    if self.config.debug_print_messages:
                        print(f"Task {task_id}: No follow-up generated at turn {turn}, stopping")
                    break
                
                followup_query = followup_result.get('content', '')
                
                # Add reasoning reminder to follow-up query when reasoning is enabled
                # This will be stripped during ShareGPT conversion
                followup_query_with_reminder = followup_query
                if self.config.generate_reasoning:
                    followup_query_with_reminder = add_reasoning_reminder(followup_query)
                
                messages.append({"role": "user", "content": followup_query_with_reminder})
                
                # Generate follow-up response with clarification support
                success_fu, content_fu, json_output_fu, clarification_msgs_fu = await self._generate_structured_output_with_clarification(
                    followup_query_with_reminder, json_schema, f"{task_id}_followup_{turn}",
                    conversation_history=messages[:-1],  # Pass history without the new user message
                    is_first_turn=False,  # This is a follow-up turn
                    original_query=followup_query
                )
                
                if success_fu and json_output_fu:
                    # Validate <think> block when reasoning is enabled
                    if self.config.generate_reasoning:
                        # content_fu should have the full response including <think> blocks
                        if not content_fu or not has_think_block(content_fu):
                            print(f"Task {task_id}: Follow-up turn {turn} missing <think> block - stopping but keeping previous valid turns")
                            await self.stats.inc(missing_think_blocks=1)
                            # Remove the follow-up user message and stop
                            messages.pop()
                            break
                    
                    # Add any clarification exchanges that occurred
                    if clarification_msgs_fu:
                        messages.pop()  # Remove original followup user message
                        messages.append({"role": "user", "content": followup_query_with_reminder})
                        messages.extend(clarification_msgs_fu)
                    
                    # Use full content with reasoning if enabled
                    if self.config.generate_reasoning and content_fu:
                        followup_msg_content = content_fu
                        await self.stats.inc(reasoning_generated=1)
                    else:
                        followup_msg_content = json.dumps(json_output_fu, indent=2)
                    messages.append({"role": "assistant", "content": followup_msg_content})
                    await self.stats.inc(multi_turn_samples=1)
                elif content_fu and self.config.generate_reasoning:
                    # Model returned text response (e.g., explaining why it can't do the modification)
                    # This is valid if it has <think> blocks - treat as an explanation response
                    if has_think_block(content_fu):
                        messages.append({"role": "assistant", "content": content_fu})
                        await self.stats.inc(multi_turn_samples=1)
                        await self.stats.inc(reasoning_generated=1)
                        if self.config.debug_print_messages:
                            print(f"Task {task_id}: Follow-up turn {turn} got text explanation instead of JSON (valid)")
                    else:
                        # No <think> block - remove the follow-up user message and stop (keep previous valid conversation)
                        print(f"Task {task_id}: Follow-up turn {turn} text response missing <think> block - stopping but keeping previous valid turns")
                        messages.pop()
                        await self.stats.inc(missing_think_blocks=1)
                        break
                else:
                    # Remove the follow-up user message if response failed and stop (keep previous valid conversation)
                    print(f"Task {task_id}: Follow-up turn {turn} failed - stopping but keeping previous valid turns")
                    messages.pop()
                    await self.stats.inc(failed_followup=1)
                    break
        
        # Generate analysis follow-up if enabled
        if self.config.generate_analysis_followup and len(messages) >= 2:
            analysis_result = await self._run_analysis_followup_agent(messages, task_id)
            
            if analysis_result:
                followup_q = analysis_result.get('followup_question', '')
                followup_resp = analysis_result.get('response', '')
                
                if followup_q and followup_resp:
                    if self.config.generate_reasoning and not has_think_block(followup_resp):
                        await self.stats.inc(missing_think_blocks=1)
                        await self.stats.inc(failed_analysis_followup=1)
                    else:
                        messages.append({"role": "user", "content": followup_q})
                        messages.append({"role": "assistant", "content": followup_resp})
                        await self.stats.inc(analysis_followups=1)
                        if self.config.generate_reasoning:
                            await self.stats.inc(reasoning_generated=1)
                else:
                    await self.stats.inc(failed_analysis_followup=1)
            else:
                await self.stats.inc(failed_analysis_followup=1)
        
        # Build result
        result = {
            "id": task_id,
            "mode": "huggingface",
            "json_schema": json_schema,
            "messages": messages,
            "timestamp": datetime.now().isoformat()
        }
        
        # Save results
        self._append_jsonl(self.results_file, result)
        
        if self.config.output_sharegpt:
            sharegpt_metadata = {
                "id": task_id,
                "source": self.config.dataset_name or "huggingface",
                "task": user_query,
                "category": "",
                "subcategory": ""
            }
            sharegpt = to_sharegpt_format(
                messages, json_schema, sharegpt_metadata,
                include_reasoning_instructions=self.config.generate_reasoning
            )
            if sharegpt:
                self._append_jsonl(self.sharegpt_file, sharegpt)
        
        # Flush agent outputs
        self._flush_agent_outputs(task_id)
        
        await self.stats.inc(successful=1)
        return result
    
    # ============================================================
    # Batch Processing
    # ============================================================
    
    async def run_batch(self, tasks: List[Dict]) -> List[Optional[Dict]]:
        """Process a batch of tasks with concurrency control."""
        semaphore = asyncio.Semaphore(self.config.batch_size)
        
        async def process_with_sem(task: Dict) -> Optional[Dict]:
            async with semaphore:
                try:
                    return await asyncio.wait_for(
                        self.process_task(task),
                        timeout=self.config.per_task_timeout
                    )
                except asyncio.TimeoutError:
                    print(f"Task {task.get('id', 'unknown')} timed out")
                    return None
                except Exception as e:
                    print(f"Task {task.get('id', 'unknown')} failed: {e}")
                    import traceback
                    traceback.print_exc()
                    return None
        
        results = await asyncio.gather(*[process_with_sem(t) for t in tasks])
        return list(results)
    
    async def run(
        self,
        start_index: int = 0,
        limit: Optional[int] = None
    ) -> None:
        """Run the full pipeline."""
        if self.config.mode == GenerationMode.CURRICULUM:
            print(f"Running in CURRICULUM mode")
            print(f"Loading curriculum from: {self.config.curriculum_file}")
            tasks = self.load_curriculum_tasks(limit=limit)
        else:
            print(f"Running in HUGGINGFACE mode")
            print(f"Loading dataset: {self.config.dataset_name}")
            tasks = self.load_huggingface_tasks(start_index=start_index, limit=limit)
        
        if not tasks:
            print("No tasks loaded!")
            return
        
        print(f"Processing {len(tasks)} tasks in batches of {self.config.batch_size}")
        
        batch_size = self.config.batch_size
        
        for i in tqdm(range(0, len(tasks), batch_size), desc="Processing batches"):
            batch = tasks[i:i + batch_size]
            await self.run_batch(batch)
        
        print(self.stats.report())
        print(f"\nResults saved to: {self.results_file}")
        if self.config.output_sharegpt:
            print(f"ShareGPT saved to: {self.sharegpt_file}")
        print(f"Agent outputs saved to: {self.agent_outputs_file}")
