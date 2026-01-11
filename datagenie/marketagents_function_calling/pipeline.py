"""
Main FunctionCallingPipeline class for dataset generation.

Supports two modes:
- Curriculum: Generate tools/queries from task descriptions
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
    SystemPrompt, ProcessedOutput
)
from minference.lite.inference import InferenceOrchestrator

from datagenie.marketagents_function_calling.config import PipelineConfig, GenerationMode, AgentLLMConfig
from datagenie.marketagents_function_calling.agents import (
    create_tool_generator_agent,
    create_query_generator_agent,
    create_docstring_agent,
    create_schema_agent,
    create_results_agent,
    create_followup_agent,
    create_clarification_agent,
    create_analysis_followup_agent,
)
from datagenie.marketagents_function_calling.utils import (
    validate_tool_calls,
    validate_message,
    to_sharegpt_format,
    validate_think_block,
    has_think_block,
    get_reasoning_system_prompt,
    parse_xml_tool_calls,
    has_xml_tool_call,
    has_incomplete_tool_call,
    has_malformed_tool_call,
    print_messages,
    print_chat_thread,
    print_response,
)


@dataclass
class AgentOutput:
    """Track individual agent outputs for debugging and reuse."""
    agent_name: str
    task_id: str
    input_data: Dict[str, Any]
    output_data: Optional[Dict[str, Any]]
    success: bool
    error: Optional[str] = None
    timestamp: str = field(default_factory=lambda: datetime.now().isoformat())


@dataclass
class ProcessingStats:
    """Track processing statistics."""
    total: int = 0
    successful: int = 0
    failed_tool_gen: int = 0
    failed_query_gen: int = 0
    failed_docstring: int = 0
    failed_tool_call: int = 0
    failed_validation: int = 0
    failed_results: int = 0
    failed_followup: int = 0
    multi_turn_samples: int = 0
    clarification_flows: int = 0  # Samples that needed clarification
    failed_clarification: int = 0  # Failed to generate clarification
    max_recursion_reached: int = 0  # Hit max recursion depth
    analysis_followups: int = 0  # Samples with analysis follow-up generated
    failed_analysis_followup: int = 0  # Failed to generate analysis follow-up
    reasoning_generated: int = 0  # Turns with reasoning in <think> tags
    missing_think_blocks: int = 0  # Turns missing required <think> blocks
    failed_reasoning_validation: int = 0  # Failed reasoning format validation
    truncated_tool_calls: int = 0  # Tool calls truncated due to max_tokens
    malformed_tool_calls: int = 0  # Malformed tool call tags
    truncated_conversations: int = 0  # Conversations truncated to last valid turn
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
Clarification flows: {self.clarification_flows:,}
Analysis follow-ups: {self.analysis_followups:,}
Reasoning generated: {self.reasoning_generated:,}
Missing think blocks:{self.missing_think_blocks:,}
Truncated convos:    {self.truncated_conversations:,}
Max recursion hit:   {self.max_recursion_reached:,}
Failed tool gen:     {self.failed_tool_gen:,}
Failed query gen:    {self.failed_query_gen:,}
Failed docstring:    {self.failed_docstring:,}
Failed tool call:    {self.failed_tool_call:,}
Truncated tool calls:{self.truncated_tool_calls:,}
Malformed tool calls:{self.malformed_tool_calls:,}
Failed clarification:{self.failed_clarification:,}
Failed analysis f/u: {self.failed_analysis_followup:,}
Failed reasoning:    {self.failed_reasoning_validation:,}
Failed validation:   {self.failed_validation:,}
Failed results:      {self.failed_results:,}
Failed follow-up:    {self.failed_followup:,}
Success rate:        {success_rate:.1f}%
"""


class FunctionCallingPipeline:
    """
    Multi-agent pipeline for generating function calling datasets.
    
    Supports two modes:
    1. Curriculum mode (default): Load tasks from CSV/JSONL, generate tools and queries
    2. HuggingFace mode: Augment existing single-turn datasets to multi-turn
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
        self.results_file = self.output_dir / f"function_calling_results_{mode_suffix}_{ts}.jsonl"
        self.sharegpt_file = self.output_dir / f"function_calling_sharegpt_{mode_suffix}_{ts}.jsonl"
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
                    
                    tasks.append({
                        "id": f"curriculum_{len(tasks)}",
                        "category": row.get('Category', ''),
                        "subcategory": row.get('SubCategory', ''),
                        "task_description": row.get('Task', ''),
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
        print(f"Loading HuggingFace dataset: {self.config.dataset_name}")
        
        if limit:
            end_index = start_index + limit
            ri = ReadInstruction('train', from_=start_index, to=end_index, unit='abs')
            dataset = load_dataset(self.config.dataset_name, split=ri)
        else:
            dataset = load_dataset(self.config.dataset_name, split=self.config.dataset_split)
        
        tasks = []
        for row in dataset:
            tools = row.get('tools', '[]')
            if isinstance(tools, str):
                try:
                    tools = json.loads(tools)
                except json.JSONDecodeError:
                    tools = []
            
            formatted_tools = []
            for tool in tools:
                if 'function' not in tool:
                    formatted_tools.append({"type": "function", "function": tool})
                else:
                    formatted_tools.append(tool)
            
            # Normalize tool schemas (fix common issues like "str" -> "string")
            formatted_tools = self._normalize_tool_schemas(formatted_tools)
            
            tasks.append({
                "id": row.get('id', str(len(tasks))),
                "query": row.get('query', ''),
                "tools": formatted_tools,
                "answers": row.get('answers', '[]'),
                "mode": "huggingface"
            })
        
        print(f"Loaded {len(tasks)} HuggingFace tasks")
        return tasks
    
    def _normalize_tool_schemas(self, tools: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """
        Normalize tool schemas to fix common issues from HuggingFace datasets.
        
        Fixes:
        - "str" -> "string"
        - "int" -> "integer" 
        - "bool" -> "boolean"
        - "float" -> "number"
        - "list" -> "array"
        - "dict" -> "object"
        """
        type_mapping = {
            "str": "string",
            "int": "integer",
            "bool": "boolean",
            "float": "number",
            "list": "array",
            "dict": "object"
        }
        
        def fix_schema(obj: Any) -> Any:
            if isinstance(obj, dict):
                result = {}
                for k, v in obj.items():
                    if k == "type" and isinstance(v, str) and v in type_mapping:
                        result[k] = type_mapping[v]
                    else:
                        result[k] = fix_schema(v)
                return result
            elif isinstance(obj, list):
                return [fix_schema(item) for item in obj]
            else:
                return obj
        
        return fix_schema(tools)
    
    def _tools_need_docstrings(self, tools: List[Dict[str, Any]]) -> bool:
        """Check if any tools are missing descriptions."""
        for tool in tools:
            func = tool.get('function', tool)
            description = func.get('description', '')
            if not description or description.strip() == '':
                return True
        return False
    
    # ============================================================
    # Agent Runner Methods
    # ============================================================
    
    @retry(wait=wait_random_exponential(multiplier=2, max=60), stop=stop_after_attempt(3))
    async def _run_tool_generator_agent(
        self, task_description: str, category: str, subcategory: str
    ) -> Optional[Dict]:
        """Run tool generation agent with retry."""
        cfg = self.config.agents.tool_generator
        agent = create_tool_generator_agent(
            task_description, category, subcategory,
            self.orchestrator, cfg.model,
            llm_client=cfg.get_llm_client(),
            temperature=cfg.temperature,
            max_tokens=cfg.max_tokens
        )
        try:
            result = await agent.execute()
            if not isinstance(result, dict):
                return None
            
            # Handle case where JSON parsing failed and we got raw string
            if 'raw' in result and 'tools' not in result:
                import json
                try:
                    parsed = json.loads(result['raw'])
                    return parsed
                except json.JSONDecodeError:
                    print(f"Tool generator returned truncated/invalid JSON")
                    return None
            
            return result
        except Exception as e:
            print(f"Tool generator agent failed: {e}")
            return None
    
    @retry(wait=wait_random_exponential(multiplier=2, max=60), stop=stop_after_attempt(3))
    async def _run_query_generator_agent(
        self, tools: List[Dict], task_description: str
    ) -> Optional[Dict]:
        """Run query generation agent with retry."""
        cfg = self.config.agents.query_generator
        agent = create_query_generator_agent(
            tools, task_description,
            self.orchestrator, cfg.model,
            llm_client=cfg.get_llm_client(),
            temperature=cfg.temperature,
            max_tokens=cfg.max_tokens
        )
        try:
            result = await agent.execute()
            return result if isinstance(result, dict) else None
        except Exception as e:
            print(f"Query generator agent failed: {e}")
            return None
    
    @retry(wait=wait_random_exponential(multiplier=2, max=60), stop=stop_after_attempt(3))
    async def _run_docstring_agent(self, tools: List[Dict]) -> Optional[Dict]:
        """Run docstring generation agent with retry."""
        cfg = self.config.agents.docstring_generator
        agent = create_docstring_agent(
            tools, self.orchestrator, cfg.model,
            llm_client=cfg.get_llm_client(),
            temperature=cfg.temperature,
            max_tokens=cfg.max_tokens
        )
        try:
            result = await agent.execute()
            return result if isinstance(result, dict) else None
        except Exception as e:
            print(f"Docstring agent failed: {e}")
            return None
    
    @retry(wait=wait_random_exponential(multiplier=2, max=60), stop=stop_after_attempt(3))
    async def _run_schema_agent(self, tool_calls: List[Dict]) -> Optional[Dict]:
        """Run schema generation agent with retry."""
        cfg = self.config.agents.schema_generator
        agent = create_schema_agent(
            tool_calls, self.orchestrator, cfg.model,
            llm_client=cfg.get_llm_client(),
            temperature=cfg.temperature,
            max_tokens=cfg.max_tokens
        )
        try:
            result = await agent.execute()
            return result if isinstance(result, dict) else None
        except Exception as e:
            print(f"Schema agent failed: {e}")
            return None
    
    @retry(wait=wait_random_exponential(multiplier=2, max=60), stop=stop_after_attempt(3))
    async def _run_results_agent(
        self, tool_calls: List[Dict], schemas: List[Dict], user_query: str
    ) -> Optional[Dict]:
        """Run results generation agent with retry."""
        cfg = self.config.agents.results_generator
        agent = create_results_agent(
            tool_calls, schemas, user_query,
            self.orchestrator, cfg.model,
            llm_client=cfg.get_llm_client(),
            temperature=cfg.temperature,
            max_tokens=cfg.max_tokens
        )
        try:
            result = await agent.execute()
            return result if isinstance(result, dict) else None
        except Exception as e:
            print(f"Results agent failed: {e}")
            return None
    
    @retry(wait=wait_random_exponential(multiplier=2, max=60), stop=stop_after_attempt(2))
    async def _run_followup_agent(
        self, messages: List[Dict], tools: List[Dict]
    ) -> Optional[Dict]:
        """Run follow-up query agent with retry."""
        cfg = self.config.agents.followup_generator
        agent = create_followup_agent(
            messages, tools, self.orchestrator, cfg.model,
            llm_client=cfg.get_llm_client(),
            temperature=cfg.temperature,
            max_tokens=cfg.max_tokens
        )
        try:
            result = await agent.execute()
            return result if isinstance(result, dict) else None
        except Exception as e:
            print(f"Follow-up agent failed: {e}")
            return None

    @retry(wait=wait_random_exponential(multiplier=2, max=60), stop=stop_after_attempt(2))
    async def _run_clarification_agent(
        self, original_query: str, assistant_response: str, tools: List[Dict]
    ) -> Optional[Dict]:
        """Run clarification agent to provide missing details."""
        cfg = self.config.agents.clarification_agent
        agent = create_clarification_agent(
            original_query, assistant_response, tools,
            self.orchestrator, cfg.model,
            llm_client=cfg.get_llm_client(),
            temperature=cfg.temperature,
            max_tokens=cfg.max_tokens
        )
        try:
            result = await agent.execute()
            return result if isinstance(result, dict) else None
        except Exception as e:
            print(f"Clarification agent failed: {e}")
            return None

    @retry(wait=wait_random_exponential(multiplier=2, max=60), stop=stop_after_attempt(2))
    async def _run_analysis_followup_agent(
        self, messages: List[Dict]
    ) -> Optional[Dict]:
        """
        Run analysis follow-up agent to generate non-tool-calling follow-up.
        
        This generates a user follow-up question that requires ANALYSIS of
        existing tool results, along with the assistant's response.
        No new tool calls are made - just reasoning over existing context.
        """
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
            return result if isinstance(result, dict) else None
        except Exception as e:
            print(f"Analysis follow-up agent failed: {e}")
            return None

    # ============================================================
    # Tool Calling Agent - Simulates tool calling workflow
    # ============================================================
    
    def _create_tool_calling_agent_thread(
        self,
        system_prompt: str,
        tools: List[Dict[str, Any]],
    ) -> ChatThread:
        """
        Create a ChatThread configured as a tool-calling agent.
        
        This is the "master" chat thread that simulates the assistant
        with access to tools. It maintains conversation state across
        multiple tool call rounds.
        """
        from minference.lite.models import StructuredTool
        
        # Convert tool dicts to StructuredTool objects
        structured_tools = []
        for tool in tools:
            func = tool.get('function', tool)
            structured_tools.append(StructuredTool(
                name=func.get('name', ''),
                description=func.get('description', ''),
                json_schema=func.get('parameters', {"type": "object", "properties": {}})
            ))
        
        cfg = self.config.agents.tool_calling
        chat_thread = ChatThread(
            name="tool-calling-agent",
            system_prompt=SystemPrompt(name="tool-agent-sys", content=system_prompt),
            llm_config=LLMConfig(
                client=cfg.get_llm_client(),
                model=cfg.model,
                temperature=cfg.temperature,
                response_format=ResponseFormat.auto_tools
            ),
            tools=structured_tools,
            history=[]
        )
        
        return chat_thread
    
    def _extract_all_tool_calls(
        self,
        output: ProcessedOutput,
        chat_thread: ChatThread
    ) -> List[Dict[str, Any]]:
        """
        Extract ALL tool calls from the raw completion response.
        
        minference only parses the first tool call into json_object,
        but the raw API response may contain multiple parallel tool calls.
        This method extracts all of them.
        
        Args:
            output: ProcessedOutput from inference
            chat_thread: ChatThread for context (to check provider)
            
        Returns:
            List of tool calls in OpenAI format
        """
        tool_calls = []
        
        try:
            raw_result = output.raw_output.raw_result
            
            # Handle OpenAI/vLLM format
            if isinstance(raw_result, dict):
                choices = raw_result.get('choices', [])
                if choices:
                    message = choices[0].get('message', {})
                    raw_tool_calls = message.get('tool_calls', [])
                    
                    for tc in raw_tool_calls:
                        tool_calls.append({
                            "id": tc.get('id', f"chatcmpl-tool-{len(tool_calls)}"),
                            "type": "function",
                            "function": {
                                "name": tc.get('function', {}).get('name', ''),
                                "arguments": tc.get('function', {}).get('arguments', '{}')
                            }
                        })
            
            # If no tool calls found in raw, fall back to json_object
            if not tool_calls and output.json_object and output.json_object.tool_call_id:
                tool_calls.append({
                    "id": output.json_object.tool_call_id,
                    "type": "function",
                    "function": {
                        "name": output.json_object.name,
                        "arguments": json.dumps(output.json_object.object, sort_keys=True)
                    }
                })
                
        except Exception as e:
            print(f"Error extracting tool calls: {e}")
            # Fall back to json_object
            if output.json_object and output.json_object.tool_call_id:
                tool_calls.append({
                    "id": output.json_object.tool_call_id,
                    "type": "function",
                    "function": {
                        "name": output.json_object.name,
                        "arguments": json.dumps(output.json_object.object, sort_keys=True)
                    }
                })
        
        return tool_calls
    
    async def _run_tool_calling_turn(
        self,
        chat_thread: ChatThread,
        user_message: str,
    ) -> Optional[ProcessedOutput]:
        """
        Run a single turn of tool calling.
        
        Adds the user message to the thread and runs inference.
        The chat_thread maintains history automatically.
        """
        chat_thread.new_message = user_message
        
        try:
            outputs = await self.orchestrator.run_parallel_ai_completion([chat_thread])
            return outputs[0] if outputs else None
        except Exception as e:
            print(f"Tool calling turn failed: {e}")
            return None
    
    async def _append_tool_results_to_thread(
        self,
        chat_thread: ChatThread,
        tool_results: List[Dict[str, Any]],
    ) -> None:
        """
        Append tool results to the chat thread history.
        
        After the assistant makes tool calls, we need to add the
        tool results before the next inference turn.
        """
        for result in tool_results:
            tool_msg = ChatMessage(
                role=MessageRole.tool,
                content=str(result.get('content', '')),
                tool_name=result.get('name'),
                oai_tool_call_id=result.get('tool_call_id')
            )
            chat_thread.history.append(tool_msg)
    
    async def _execute_tool_calling(
        self,
        messages: List[Dict[str, Any]],
        tools: List[Dict[str, Any]],
    ) -> Optional[ProcessedOutput]:
        """Execute tool calling completion using ChatThread."""
        from minference.lite.models import StructuredTool
        
        system_content = ""
        history = []
        user_message = None
        
        for msg in messages:
            if msg['role'] == 'system':
                system_content = msg['content']
            elif msg['role'] == 'user':
                user_message = msg['content']
                history.append(ChatMessage(role=MessageRole.user, content=msg['content']))
            elif msg['role'] == 'assistant':
                # Handle assistant messages with tool calls
                assistant_msg = ChatMessage(role=MessageRole.assistant, content=msg.get('content', ''))
                if msg.get('tool_calls'):
                    # Store tool calls info for proper reconstruction
                    assistant_msg.tool_calls = msg.get('tool_calls')
                history.append(assistant_msg)
            elif msg['role'] == 'tool':
                history.append(ChatMessage(
                    role=MessageRole.tool,
                    content=str(msg['content']),
                    tool_name=msg.get('name'),
                    oai_tool_call_id=msg.get('tool_call_id')
                ))
        
        # Convert tool dicts to StructuredTool objects
        structured_tools = []
        for tool in tools:
            func = tool.get('function', tool)
            structured_tools.append(StructuredTool(
                name=func.get('name', ''),
                description=func.get('description', ''),
                json_schema=func.get('parameters', {"type": "object", "properties": {}})
            ))
        
        cfg = self.config.agents.tool_calling
        chat_thread = ChatThread(
            name="tool-calling",
            system_prompt=SystemPrompt(name="sys", content=system_content),
            llm_config=LLMConfig(
                client=cfg.get_llm_client(),
                model=cfg.model,
                temperature=cfg.temperature,
                response_format=ResponseFormat.auto_tools
            ),
            tools=structured_tools,
            history=history[:-1] if history else [],
            new_message=user_message
        )
        
        try:
            outputs = await self.orchestrator.run_parallel_ai_completion([chat_thread])
            return outputs[0] if outputs else None
        except Exception as e:
            print(f"Tool calling failed: {e}")
            return None
    
    async def _run_tool_calling_workflow(
        self,
        messages: List[Dict[str, Any]],
        tools: List[Dict[str, Any]],
        user_query: str,
        task_id: str,
        max_turns: int = None
    ) -> Tuple[bool, List[Dict[str, Any]]]:
        """
        Run the complete tool calling workflow with a persistent ChatThread.
        
        This follows the pattern from datagen_salesforce.py:
        1. Initialize tool calling agent with system prompt and tools
        2. Send user query and get response
        3. If tool call -> generate results via schema_agent + results_agent -> append tool results
        4. If plain text on first turn -> clarification flow (if enabled)
        5. If plain text after tool calls -> completion (summary)
        6. Continue until no more tool calls or max turns reached
        
        Supports parallel tool calls by checking raw completion for multiple tool_calls.
        
        Args:
            messages: Initial messages list [system, user] - will be mutated
            tools: Available tools in OpenAI format
            user_query: The user's query
            task_id: Task ID for tracking
            max_turns: Maximum tool call turns (defaults to config.max_recursion_depth)
            
        Returns:
            Tuple of (success, messages)
        """
        from minference.lite.models import StructuredTool
        
        if max_turns is None:
            max_turns = self.config.max_recursion_depth
        
        # Extract system prompt from messages
        system_content = ""
        for msg in messages:
            if msg.get('role') == 'system':
                system_content = msg.get('content', '')
                break
        
        # Convert tools to StructuredTool objects
        structured_tools = []
        for tool in tools:
            func = tool.get('function', tool)
            structured_tools.append(StructuredTool(
                name=func.get('name', ''),
                description=func.get('description', ''),
                json_schema=func.get('parameters', {"type": "object", "properties": {}})
            ))
        
        # Build history from existing messages (for follow-up workflows)
        # Skip system message, convert the rest to ChatMessage objects
        initial_history = []
        for msg in messages:
            role = msg.get('role')
            content = msg.get('content', '')
            
            if role == 'system':
                continue  # System prompt is set separately
            elif role == 'user':
                initial_history.append(ChatMessage(role=MessageRole.user, content=content))
            elif role == 'assistant':
                initial_history.append(ChatMessage(role=MessageRole.assistant, content=content))
            elif role == 'tool':
                initial_history.append(ChatMessage(
                    role=MessageRole.tool,
                    content=content,
                    tool_name=msg.get('name'),
                    oai_tool_call_id=msg.get('tool_call_id')
                ))
        
        # Remove the last user message from history - it will be set as new_message
        # (The user_query parameter is the new message to process)
        if initial_history and initial_history[-1].role == MessageRole.user:
            # Check if the last user message matches user_query
            if initial_history[-1].content == user_query:
                initial_history = initial_history[:-1]
        
        # Create persistent ChatThread for tool calling agent
        cfg = self.config.agents.tool_calling
        
        # Use text format when reasoning is enabled (for XML tool calls)
        # Otherwise use auto_tools for native tool calling
        if self.config.generate_reasoning:
            response_format = ResponseFormat.text
        else:
            response_format = ResponseFormat.auto_tools
        
        tool_calling_thread = ChatThread(
            name=f"tool-calling-{task_id}",
            system_prompt=SystemPrompt(name="tool-agent-sys", content=system_content),
            llm_config=LLMConfig(
                client=cfg.get_llm_client(),
                model=cfg.model,
                temperature=cfg.temperature,
                max_tokens=cfg.max_tokens,
                response_format=response_format
            ),
            # Pass tools for native tool calling (auto_tools mode)
            # For text mode with reasoning, tools are embedded in system prompt
            tools=structured_tools if not self.config.generate_reasoning else [],
            history=initial_history  # Load existing conversation history
        )
        
        turn = 0
        called_tool_signatures = set()  # Track tool calls to detect loops (use set for O(1) lookup)
        current_query = user_query
        
        while turn < max_turns:
            # Set the new message for this turn
            # For continuation after tool results, new_message may be None
            tool_calling_thread.new_message = current_query
            
            # Skip if new_message is None and history is empty (nothing to process)
            if current_query is None and not tool_calling_thread.history:
                print(f"Task {task_id}: Skipping turn {turn} - no message and no history")
                break
            
            # Debug: Print ChatThread state before inference
            if self.config.debug_print_messages:
                print_chat_thread(
                    thread_name=f"tool-calling-{task_id} (turn {turn})",
                    system_prompt=system_content,
                    history=tool_calling_thread.history,
                    new_message=current_query,
                    truncate=True
                )
            
            # Run inference
            try:
                outputs = await self.orchestrator.run_parallel_ai_completion([tool_calling_thread])
                output = outputs[0] if outputs else None
            except Exception as e:
                print(f"Task {task_id}: Tool calling failed at turn {turn}: {e}")
                return False, messages
            
            if not output:
                print(f"Task {task_id}: No output at turn {turn} (new_message was: {repr(current_query)[:50]})")
                return False, messages
            
            # Check for empty content (might happen when continuing from tool results)
            if not output.content and current_query is None:
                print(f"Task {task_id}: Empty output when continuing from tool results at turn {turn}")
                # This might be a minference issue - try to continue anyway
            
            # Check for truncated tool calls (max_tokens issue)
            if self.config.generate_reasoning and output.content:
                if has_incomplete_tool_call(output.content):
                    print(f"Task {task_id}: Truncated tool call at turn {turn} - increase max_tokens")
                    await self.stats.inc(truncated_tool_calls=1)
                    return False, messages
                
                # Check for malformed tool calls (reasoning inside tool_call tag)
                is_malformed, reason = has_malformed_tool_call(output.content)
                if is_malformed:
                    print(f"Task {task_id}: Malformed tool call at turn {turn}: {reason}")
                    await self.stats.inc(malformed_tool_calls=1)
                    # Strict: fail the task - model confused <tool_call> with <think>
                    return False, messages
            
            # Extract tool calls - use XML parsing when reasoning is enabled
            if self.config.generate_reasoning and output.content:
                # Parse XML <tool_call> tags from text response
                all_tool_calls = parse_xml_tool_calls(output.content)
            else:
                # Use native tool call extraction
                all_tool_calls = self._extract_all_tool_calls(output, tool_calling_thread)
            
            # Check if there's at least one tool call
            has_tool_call = len(all_tool_calls) > 0
            
            # Debug: Print model response
            if self.config.debug_print_messages:
                print_response(
                    content=output.content or "",
                    tool_calls=all_tool_calls if has_tool_call else None,
                    title=f"Model Response (turn {turn})",
                    truncate=True,
                    max_length=2000
                )
            
            # Validate reasoning if enabled - <think> blocks REQUIRED on ALL assistant turns
            if self.config.generate_reasoning:
                if output.content:
                    has_reasoning = has_think_block(output.content)
                    if not has_reasoning:
                        print(f"Task {task_id}: Missing <think> block at turn {turn} (required for all assistant turns)")
                        await self.stats.inc(missing_think_blocks=1)
                        # This is a hard requirement - fail the task
                        return False, messages
                    else:
                        await self.stats.inc(reasoning_generated=1)
                        if self.config.validate_reasoning:
                            is_valid, reason = validate_think_block(output.content)
                            if not is_valid:
                                print(f"Task {task_id}: Reasoning validation failed at turn {turn}: {reason}")
                                await self.stats.inc(failed_reasoning_validation=1)
                                # Continue anyway - format issues are less critical than missing think block
            
            # Save tool calling agent output
            self._save_agent_output(
                f"tool_calling_turn_{turn}", task_id,
                {"user_query": user_query if turn == 0 else "(continuation from tool results)"},
                {"tool_calls": all_tool_calls, "content": output.content},
                success=has_tool_call or bool(output.content)
            )
            
            if has_tool_call:
                # Check for loops - if ALL tool calls have been seen before, it's a loop
                new_signatures = []
                for tc in all_tool_calls:
                    sig = f"{tc['function']['name']}:{tc['function']['arguments']}"
                    if sig not in called_tool_signatures:
                        new_signatures.append(sig)
                        called_tool_signatures.add(sig)
                
                if not new_signatures:
                    print(f"Task {task_id}: Loop detected - all tool calls already made")
                    messages.append({
                        "role": "assistant",
                        "content": output.content or "I've completed the requested actions based on the tool results."
                    })
                    return True, messages
                
                # Build assistant message with ALL tool calls (parallel)
                assistant_msg = {
                    "role": "assistant",
                    "content": output.content or "",
                    "tool_calls": all_tool_calls
                }
                messages.append(assistant_msg)
                
                # NOTE: Do NOT add assistant message to chat_thread.history here!
                # minference automatically adds it when processing the output.
                # Adding it manually causes duplicate messages.
                
                # Generate schema for ALL tool results
                schemas_result = await self._run_schema_agent(all_tool_calls)
                schemas = schemas_result.get('content_schemas', []) if schemas_result else []
                
                # Save schema agent output
                self._save_agent_output(
                    f"schema_generator_turn_{turn}", task_id,
                    {"tool_calls": all_tool_calls},
                    schemas_result,
                    success=bool(schemas_result)
                )
                
                # Generate tool results for ALL tool calls
                results_result = await self._run_results_agent(all_tool_calls, schemas, user_query)
                
                # Save results agent output
                self._save_agent_output(
                    f"results_generator_turn_{turn}", task_id,
                    {"tool_calls": all_tool_calls, "schemas": schemas, "user_query": user_query},
                    results_result,
                    success=bool(results_result)
                )
                
                if not results_result:
                    print(f"Task {task_id}: Results agent failed at turn {turn}")
                    await self.stats.inc(failed_results=1)
                    return False, messages
                
                # Process and append tool results to both messages list and chat thread
                tool_results_for_thread = []
                result_messages = results_result.get('messages', [])
                
                # For parallel tool calls, we need results for each call
                # Match results to tool calls by name, handling duplicates
                used_results = set()  # Track which result indices we've used
                
                for i, tool_call in enumerate(all_tool_calls):
                    tc_name = tool_call['function']['name']
                    tc_id = tool_call['id']
                    
                    # Find matching result by name (that hasn't been used yet)
                    matching_result = None
                    matching_idx = None
                    for idx, rm in enumerate(result_messages):
                        if rm.get('name') == tc_name and idx not in used_results:
                            matching_result = rm
                            matching_idx = idx
                            break
                    
                    # Fallback to index if no name match
                    if not matching_result and i < len(result_messages) and i not in used_results:
                        matching_result = result_messages[i]
                        matching_idx = i
                    
                    if matching_result:
                        if matching_idx is not None:
                            used_results.add(matching_idx)
                        
                        tool_msg = {
                            'role': 'tool',
                            'name': tc_name,
                            'tool_call_id': tc_id,
                            'content': matching_result.get('content', '{}')
                        }
                        
                        # Ensure content is string
                        if isinstance(tool_msg['content'], dict):
                            tool_msg['content'] = json.dumps(tool_msg['content'])
                        
                        # Validate and append
                        is_valid, reason = validate_message(tool_msg, 'tool')
                        if is_valid:
                            messages.append(tool_msg)
                            tool_results_for_thread.append(tool_msg)
                        else:
                            print(f"Task {task_id}: Invalid tool message for {tc_name} - {reason}")
                    else:
                        print(f"Task {task_id}: No result found for tool call {tc_name}")
                
                if not tool_results_for_thread:
                    print(f"Task {task_id}: No valid tool results generated")
                    return False, messages
                
                # Append tool results to chat thread history for next turn
                # The chat template will wrap these in <tool_response> tags automatically
                for tr in tool_results_for_thread:
                    tool_calling_thread.history.append(ChatMessage(
                        role=MessageRole.tool,
                        content=str(tr.get('content', '')),
                        tool_name=tr.get('name'),
                        oai_tool_call_id=tr.get('tool_call_id')
                    ))
                
                # IMPORTANT: minference doesn't auto-continue after tool results
                # We need to manually run inference with new_message=None to get the summary
                # The model should see tool results in history and generate a response
                tool_calling_thread.new_message = None
                
                # Debug: Print ChatThread state before summary inference
                if self.config.debug_print_messages:
                    print_chat_thread(
                        thread_name=f"tool-calling-{task_id} (summary after turn {turn})",
                        system_prompt=system_content,
                        history=tool_calling_thread.history,
                        new_message=None,
                        truncate=True
                    )
                
                # Run inference to generate summary/continuation after tool results
                try:
                    summary_outputs = await self.orchestrator.run_parallel_ai_completion([tool_calling_thread])
                    summary_output = summary_outputs[0] if summary_outputs else None
                except Exception as e:
                    print(f"Task {task_id}: Summary inference failed after turn {turn}: {e}")
                    # Return partial success - we have tool call and results
                    return True, messages
                
                if not summary_output or not summary_output.content:
                    print(f"Task {task_id}: No summary generated after tool results at turn {turn}")
                    # Return partial success - we have tool call and results
                    return True, messages
                
                # Debug: Print summary response
                if self.config.debug_print_messages:
                    print_response(
                        content=summary_output.content or "",
                        tool_calls=None,
                        title=f"Summary Response (after turn {turn})",
                        truncate=True,
                        max_length=2000
                    )
                
                # Check for malformed tool calls (reasoning inside <tool_call> tags)
                if self.config.generate_reasoning:
                    is_malformed, malform_reason = has_malformed_tool_call(summary_output.content)
                    if is_malformed:
                        print(f"Task {task_id}: Summary has malformed tool_call at turn {turn}: {malform_reason}")
                        await self.stats.inc(malformed_tool_calls=1)
                        # Strict: fail the task - model confused <tool_call> with <think>
                        return False, messages
                
                # Check if summary has <think> block when reasoning is enabled
                # This is REQUIRED for all assistant turns when generate_reasoning=True
                if self.config.generate_reasoning:
                    if not has_think_block(summary_output.content):
                        print(f"Task {task_id}: Summary missing <think> tags at turn {turn} - REQUIRED")
                        await self.stats.inc(missing_think_blocks=1)
                        # Strict: fail the task if summary is missing <think> block
                        return False, messages
                    else:
                        await self.stats.inc(reasoning_generated=1)
                        if self.config.validate_reasoning:
                            is_valid, reason = validate_think_block(summary_output.content)
                            if not is_valid:
                                print(f"Task {task_id}: Summary reasoning validation failed: {reason}")
                                await self.stats.inc(failed_reasoning_validation=1)
                
                # Check if summary contains more tool calls (model wants to continue)
                if self.config.generate_reasoning:
                    summary_tool_calls = parse_xml_tool_calls(summary_output.content)
                else:
                    summary_tool_calls = self._extract_all_tool_calls(summary_output, tool_calling_thread)
                
                if summary_tool_calls:
                    # Model made more tool calls - continue the loop
                    # Add the summary as assistant message and process tool calls in next iteration
                    assistant_msg = {
                        "role": "assistant",
                        "content": summary_output.content or "",
                        "tool_calls": summary_tool_calls
                    }
                    messages.append(assistant_msg)
                    
                    # Update tracking
                    for tc in summary_tool_calls:
                        sig = f"{tc['function']['name']}:{tc['function']['arguments']}"
                        called_tool_signatures.add(sig)
                    
                    # Continue to process these tool calls
                    turn += 1
                    current_query = None  # Will be handled in next iteration
                    
                    # Generate results for these tool calls
                    schemas_result = await self._run_schema_agent(summary_tool_calls)
                    schemas = schemas_result.get('content_schemas', []) if schemas_result else []
                    
                    results_result = await self._run_results_agent(summary_tool_calls, schemas, user_query)
                    if not results_result:
                        print(f"Task {task_id}: Results agent failed for summary tool calls")
                        return True, messages  # Partial success
                    
                    # Process tool results
                    result_messages = results_result.get('messages', [])
                    for i, tc in enumerate(summary_tool_calls):
                        tc_name = tc['function']['name']
                        tc_id = tc['id']
                        
                        matching_result = None
                        for rm in result_messages:
                            if rm.get('name') == tc_name:
                                matching_result = rm
                                break
                        
                        if not matching_result and i < len(result_messages):
                            matching_result = result_messages[i]
                        
                        if matching_result:
                            tool_msg = {
                                'role': 'tool',
                                'name': tc_name,
                                'tool_call_id': tc_id,
                                'content': matching_result.get('content', '{}')
                            }
                            if isinstance(tool_msg['content'], dict):
                                tool_msg['content'] = json.dumps(tool_msg['content'])
                            
                            messages.append(tool_msg)
                            tool_calling_thread.history.append(ChatMessage(
                                role=MessageRole.tool,
                                content=str(tool_msg['content']),
                                tool_name=tc_name,
                                oai_tool_call_id=tc_id
                            ))
                    
                    # Continue loop to generate next summary
                    continue
                else:
                    # No more tool calls - this is the final summary
                    messages.append({
                        "role": "assistant",
                        "content": summary_output.content
                    })
                    return True, messages
                
            else:
                # No tool call - assistant responded with text
                # This could be a clarification request (first turn) or completion (after tool calls)
                
                # Check if this is truly the first turn of a fresh conversation
                # (not a follow-up that already has history)
                has_prior_history = len([m for m in messages if m.get('role') in ('assistant', 'tool')]) > 0
                is_fresh_first_turn = turn == 0 and not has_prior_history
                
                if is_fresh_first_turn:
                    # First turn of fresh conversation without tool call
                    if self.config.allow_clarification_flow:
                        # Try clarification flow
                        assistant_text = output.content or ""
                        clarification_result = await self._run_clarification_agent(
                            user_query, assistant_text, tools
                        )
                        
                        # Save clarification agent output
                        self._save_agent_output(
                            "clarification_agent", task_id,
                            {"original_query": user_query, "assistant_response": assistant_text},
                            clarification_result,
                            success=bool(clarification_result and clarification_result.get('content'))
                        )
                        
                        if clarification_result and clarification_result.get('content'):
                            # Add assistant's clarification request to messages
                            messages.append({
                                "role": "assistant",
                                "content": assistant_text
                            })
                            
                            # Add user's clarification
                            clarification_content = clarification_result['content']
                            messages.append({
                                "role": "user",
                                "content": clarification_content
                            })
                            
                            # Also add to chat thread history
                            tool_calling_thread.history.append(ChatMessage(
                                role=MessageRole.assistant,
                                content=assistant_text
                            ))
                            
                            await self.stats.inc(clarification_flows=1)
                            
                            # Update query for next iteration
                            current_query = clarification_content
                            turn += 1
                            continue
                        else:
                            await self.stats.inc(failed_clarification=1)
                            print(f"Task {task_id}: Clarification failed")
                            # Ensure we don't return with a user message as the last message
                            if messages and messages[-1].get('role') == 'user':
                                messages.pop()
                            return False, messages
                    
                    elif self.config.require_tool_call_on_first_turn:
                        print(f"Task {task_id}: Fresh first turn did not produce tool call")
                        # Ensure we don't return with a user message as the last message
                        if messages and messages[-1].get('role') == 'user':
                            messages.pop()
                        return False, messages
                
                # Add final assistant message (summary after tool calls or direct response)
                assistant_content = output.content or ""
                
                # Note: <think> block validation already done at top of loop
                
                messages.append({
                    "role": "assistant",
                    "content": assistant_content
                })
                
                # Success - we have completed the workflow
                return True, messages
        
        # Max turns reached
        print(f"Task {task_id}: Max turns ({max_turns}) reached")
        await self.stats.inc(max_recursion_reached=1)
        
        # Ensure we don't return with a user message as the last message
        # This can happen if clarification flow added a user message but then we hit max_turns
        if messages and messages[-1].get('role') == 'user':
            # Remove the dangling user message
            removed_msg = messages.pop()
            print(f"Task {task_id}: Removed dangling user message to maintain valid conversation structure")
        
        # Return partial success if we have tool calls
        has_tool_calls = any(msg.get('tool_calls') for msg in messages)
        return has_tool_calls, messages
    
    async def _handle_tool_calls_recursive(
        self,
        messages: List[Dict],
        tools: List[Dict],
        user_query: str,
        depth: int = 0,
        is_first_turn: bool = True,
        original_query: str = "",
        called_tools: Optional[List[str]] = None
    ) -> Tuple[bool, List[Dict]]:
        """
        Recursively handle tool calls until completion or max depth.
        
        Supports clarification flows when allow_clarification_flow=True:
        - If first turn gets text response (clarification request), generate user clarification
        - Then retry tool calling with the clarification
        
        Completion conditions:
        - Model responds with text (no tool call) after at least one tool call
        - Max recursion depth reached (returns partial success if we have tool calls)
        - Same tool called with same args (loop detection)
        """
        if called_tools is None:
            called_tools = []
            
        if depth >= self.config.max_recursion_depth:
            print(f"Max recursion depth ({self.config.max_recursion_depth}) reached")
            # If we have at least one tool call, consider it a partial success
            has_tool_calls = any(msg.get('tool_calls') for msg in messages)
            await self.stats.inc(max_recursion_reached=1)
            return has_tool_calls, messages
        
        output = await self._execute_tool_calling(messages, tools)
        
        if not output:
            return False, messages
        
        assistant_msg = {
            "role": "assistant",
            "content": output.content or ""
        }
        
        # Check if there's a tool call in the output
        # Tool calls are stored in json_object when using auto_tools
        has_tool_call = output.json_object is not None and output.json_object.tool_call_id is not None
        
        if has_tool_call:
            tool_name = output.json_object.name
            tool_args = json.dumps(output.json_object.object, sort_keys=True)
            tool_signature = f"{tool_name}:{tool_args}"
            
            # Loop detection: check if we've called this exact tool with same args
            if tool_signature in called_tools:
                print(f"Loop detected: {tool_name} called with same arguments, stopping recursion")
                # Add the assistant message without tool call to end the conversation
                assistant_msg["content"] = output.content or "I've completed the requested actions."
                messages.append(assistant_msg)
                return True, messages
            
            called_tools.append(tool_signature)
            
            # Convert json_object to tool_calls format
            tool_call = {
                "id": output.json_object.tool_call_id,
                "type": "function",
                "function": {
                    "name": tool_name,
                    "arguments": tool_args
                }
            }
            tool_calls = [tool_call]
            assistant_msg["tool_calls"] = tool_calls
            messages.append(assistant_msg)
            
            schemas_result = await self._run_schema_agent(tool_calls)
            schemas = schemas_result.get('content_schemas', []) if schemas_result else []
            
            results_result = await self._run_results_agent(tool_calls, schemas, user_query)
            if not results_result:
                return False, messages
            
            for tool_msg in results_result.get('messages', []):
                # Force role to 'tool' since LLM might not set it correctly
                tool_msg['role'] = 'tool'
                
                # Ensure content is a string (JSON serialize if dict)
                if isinstance(tool_msg.get('content'), dict):
                    tool_msg['content'] = json.dumps(tool_msg['content'])
                
                # Copy tool_call_id from the original tool call if missing
                if not tool_msg.get('tool_call_id') and tool_calls:
                    # Match by name or use first tool call
                    for tc in tool_calls:
                        if tc['function']['name'] == tool_msg.get('name'):
                            tool_msg['tool_call_id'] = tc['id']
                            break
                    if not tool_msg.get('tool_call_id'):
                        tool_msg['tool_call_id'] = tool_calls[0]['id']
                
                is_valid, reason = validate_message(tool_msg, 'tool')
                if is_valid:
                    messages.append(tool_msg)
                else:
                    print(f"Invalid tool message: {reason}")
                    return False, messages
            
            return await self._handle_tool_calls_recursive(
                messages, tools, user_query, depth + 1, 
                is_first_turn=False, original_query=original_query,
                called_tools=called_tools
            )
        else:
            # No tool call - assistant responded with text (likely clarification request)
            if is_first_turn:
                if self.config.allow_clarification_flow:
                    # Try to generate clarification and retry
                    assistant_text = output.content or ""
                    
                    # Generate user clarification
                    clarification_result = await self._run_clarification_agent(
                        original_query or user_query,
                        assistant_text,
                        tools
                    )
                    
                    if clarification_result and clarification_result.get('content'):
                        # Add assistant's clarification request
                        messages.append(assistant_msg)
                        
                        # Add user's clarification response
                        clarification_msg = {
                            "role": "user",
                            "content": clarification_result['content']
                        }
                        messages.append(clarification_msg)
                        
                        await self.stats.inc(clarification_flows=1)
                        
                        # Retry tool calling with clarification
                        return await self._handle_tool_calls_recursive(
                            messages, tools, clarification_result['content'], 
                            depth + 1, is_first_turn=False, original_query=original_query,
                            called_tools=called_tools
                        )
                    else:
                        await self.stats.inc(failed_clarification=1)
                        print(f"Failed to generate clarification response")
                        return False, messages
                
                elif self.config.require_tool_call_on_first_turn:
                    # First turn must have tool call - this is a failure
                    print(f"First turn did not produce tool call (got clarification request)")
                    return False, messages
            
            messages.append(assistant_msg)
            return True, messages
    
    # ============================================================
    # Task Processing
    # ============================================================
    
    async def process_task(self, task: Dict[str, Any]) -> Optional[Dict]:
        """Process a single task through the full pipeline."""
        await self.stats.inc(total=1)
        
        task_id = task.get('id', str(time.time()))
        task_mode = task.get('mode', 'huggingface')
        
        # Mode-specific: Get tools and user query
        if task_mode == 'curriculum':
            task_description = task.get('task_description', '')
            category = task.get('category', '')
            subcategory = task.get('subcategory', '')
            
            if not task_description:
                self._flush_agent_outputs(task_id)
                return None
            
            # Generate tools
            tools_result = await self._run_tool_generator_agent(
                task_description, category, subcategory
            )
            
            # Save agent output
            self._save_agent_output(
                "tool_generator", task_id,
                {"task_description": task_description, "category": category, "subcategory": subcategory},
                tools_result,
                success=bool(tools_result and tools_result.get('tools'))
            )
            
            if not tools_result or not tools_result.get('tools'):
                await self.stats.inc(failed_tool_gen=1)
                self._flush_agent_outputs(task_id)
                return None
            
            tools = []
            for gen_tool in tools_result.get('tools', []):
                tools.append({
                    "type": "function",
                    "function": {
                        "name": gen_tool.get('name', ''),
                        "description": gen_tool.get('description', ''),
                        "parameters": gen_tool.get('parameters', {"type": "object", "properties": {}})
                    }
                })
            
            # Generate user query
            query_result = await self._run_query_generator_agent(tools, task_description)
            
            # Save agent output
            self._save_agent_output(
                "query_generator", task_id,
                {"tools": tools, "task_description": task_description},
                query_result,
                success=bool(query_result and query_result.get('query'))
            )
            
            if not query_result or not query_result.get('query'):
                await self.stats.inc(failed_query_gen=1)
                self._flush_agent_outputs(task_id)
                return None
            
            user_query = query_result.get('query', '')
            expected_answers = []
            
        else:
            user_query = task.get('query', '')
            tools = task.get('tools', [])
            expected_answers = task.get('answers', [])
            
            if not user_query or not tools:
                self._flush_agent_outputs(task_id)
                return None
            
            if isinstance(expected_answers, str):
                try:
                    expected_answers = json.loads(expected_answers)
                except json.JSONDecodeError:
                    expected_answers = []
        
        # Conditional docstring generation
        if self.config.generate_docstrings and self._tools_need_docstrings(tools):
            docstrings_result = await self._run_docstring_agent(tools)
            if docstrings_result:
                for tool in tools:
                    func = tool.get('function', tool)
                    for ds in docstrings_result.get('doc_strings', []):
                        if func.get('name') == ds.get('name'):
                            func['description'] = ds.get('doc_string', '')
                            break
        
        # ============================================================
        # Tool Calling Workflow (similar to original datagen_salesforce.py)
        # ============================================================
        
        # Build system prompt for tool calling agent
        if self.config.generate_reasoning:
            # Format tools as JSON for the prompt (Hermes format)
            # We embed tools in the system prompt because with ResponseFormat.text,
            # litellm may not apply the chat template's tool injection
            tools_json = json.dumps(tools, indent=2)
            
            # Deep thinking system prompt with embedded tools (matches Hermes format)
            # IMPORTANT: Emphasize <think> tags on ALL responses including summaries
            system_prompt = (
                "You are a deep thinking AI, you may use extremely long chains of thought to deeply "
                "consider the problem and deliberate with yourself via systematic reasoning processes "
                "to help come to a correct solution prior to answering. You should enclose your thoughts "
                "and internal monologue inside <think> </think> tags, and then provide your solution "
                "or response to the problem.\n\n"
                "IMPORTANT: You MUST start EVERY response with <think></think> tags, including:\n"
                "- When making tool calls\n"
                "- When summarizing tool results\n"
                "- When answering follow-up questions\n"
                "- When providing any response to the user\n\n"
                "You are a function calling AI model. You may call one or more functions to assist "
                "with the user query. Don't make assumptions about what values to plug into functions.\n\n"
                "You are provided with function signatures within <tools></tools> XML tags:\n"
                f"<tools>\n{tools_json}\n</tools>\n\n"
                "For each function call, return a json object with function name and arguments within "
                "<tool_call></tool_call> XML tags:\n"
                "<tool_call>\n"
                '{"name": "<function-name>", "arguments": <args-json-object>}\n'
                "</tool_call>"
            )
        else:
            # Standard function calling without reasoning (uses native tool calling API)
            system_prompt = (
                "You are a helpful assistant with access to tools. "
                "Use the available tools to help answer the user's questions. "
                "You may call one or more functions to assist with the user query. " 
                "Don't make assumptions about what values to plug into functions. "
                "Ask clarifying questions when you need additional information for function parameters. "         
                "When you have completed the user's request, provide a summary response."
            )
        
        # Initialize messages list (this accumulates the full conversation)
        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_query}
        ]
        
        # Run the tool calling workflow
        success, messages = await self._run_tool_calling_workflow(
            messages=messages,
            tools=tools,
            user_query=user_query,
            task_id=task_id
        )
        
        # Check if we have any tool calls (partial success)
        has_tool_calls = any(msg.get('tool_calls') for msg in messages)
        
        if not success and not has_tool_calls:
            await self.stats.inc(failed_tool_call=1)
            self._flush_agent_outputs(task_id)
            return None
        
        # If we hit max recursion but have tool calls, still save the result
        if not success and has_tool_calls:
            print(f"Task {task_id}: Partial success - saving with {sum(1 for m in messages if m.get('tool_calls'))} tool calls")
        
        # Validate (HuggingFace mode only)
        if task_mode == 'huggingface' and self.config.validate_tool_calls and expected_answers:
            all_tool_calls = []
            for msg in messages:
                if msg.get('tool_calls'):
                    all_tool_calls.extend(msg['tool_calls'])
            
            is_valid, reason = validate_tool_calls(
                all_tool_calls, expected_answers,
                self.config.require_matching_arguments
            )
            if not is_valid:
                await self.stats.inc(failed_validation=1)
                print(f"Validation failed: {reason}")
        
        # Generate follow-up query and run another tool calling round
        followup_result = await self._run_followup_agent(messages, tools)
        if followup_result and followup_result.get('content'):
            followup_content = followup_result.get('content', '')
            followup_msg = {"role": "user", "content": followup_content}
            messages.append(followup_msg)
            
            # Save follow-up agent output
            self._save_agent_output(
                "followup_generator", task_id,
                {"messages": messages[:-1], "tools": tools},
                followup_result,
                success=True
            )
            
            # Run another tool calling workflow for the follow-up
            success2, messages = await self._run_tool_calling_workflow(
                messages=messages,
                tools=tools,
                user_query=followup_content,
                task_id=f"{task_id}_followup"
            )
            
            if success2:
                await self.stats.inc(multi_turn_samples=1)
            else:
                # Follow-up workflow failed - remove the trailing user message
                # to maintain valid conversation structure
                if messages and messages[-1].get('role') == 'user':
                    messages.pop()
                    print(f"Task {task_id}: Removed follow-up user message after workflow failure")
        
        # Generate analysis follow-up (non-tool-calling) if enabled
        if self.config.generate_analysis_followup:
            # Only generate if we have tool results to analyze
            has_tool_results = any(msg.get('role') == 'tool' for msg in messages)
            # Only generate if last message is from assistant (not user)
            last_msg_role = messages[-1].get('role') if messages else None
            
            if has_tool_results and last_msg_role == 'assistant':
                analysis_result = await self._run_analysis_followup_agent(messages)
                
                if analysis_result:
                    followup_q = analysis_result.get('followup_question', '')
                    followup_resp = analysis_result.get('response', '')
                    
                    if followup_q and followup_resp:
                        # Validate <think> block in response when reasoning is enabled
                        if self.config.generate_reasoning:
                            # Check for malformed tool calls first
                            is_malformed, malform_reason = has_malformed_tool_call(followup_resp)
                            if is_malformed:
                                print(f"Task {task_id}: Analysis follow-up has malformed tool_call: {malform_reason} - keeping previous valid conversation")
                                await self.stats.inc(malformed_tool_calls=1)
                                await self.stats.inc(failed_analysis_followup=1)
                                # Don't add - keep conversation as-is (truncated to last valid turn)
                            elif not has_think_block(followup_resp):
                                print(f"Task {task_id}: Analysis follow-up response missing <think> tags - keeping previous valid conversation")
                                await self.stats.inc(missing_think_blocks=1)
                                await self.stats.inc(failed_analysis_followup=1)
                                # Don't add - keep conversation as-is (truncated to last valid turn)
                            else:
                                # Response has <think> block - add it
                                messages.append({"role": "user", "content": followup_q})
                                messages.append({"role": "assistant", "content": followup_resp})
                                await self.stats.inc(reasoning_generated=1)
                                await self.stats.inc(analysis_followups=1)
                                
                                # Save agent output
                                self._save_agent_output(
                                    "analysis_followup", task_id,
                                    {"messages_before": len(messages) - 2},
                                    analysis_result,
                                    success=True
                                )
                        else:
                            # No reasoning required - add directly
                            messages.append({"role": "user", "content": followup_q})
                            messages.append({"role": "assistant", "content": followup_resp})
                            await self.stats.inc(analysis_followups=1)
                            
                            # Save agent output
                            self._save_agent_output(
                                "analysis_followup", task_id,
                                {"messages_before": len(messages) - 2},
                                analysis_result,
                                success=True
                            )
                    else:
                        self._save_agent_output(
                            "analysis_followup", task_id,
                            {"messages_before": len(messages)},
                            analysis_result,
                            success=False,
                            error="Missing followup_question or response"
                        )
                        await self.stats.inc(failed_analysis_followup=1)
                else:
                    await self.stats.inc(failed_analysis_followup=1)
        
        # Final validation: Truncate to last valid assistant turn if needed
        # This ensures we don't lose valuable data when only the final turn is invalid
        if self.config.generate_reasoning and messages:
            truncated = False
            # Find the last valid assistant message (has <think> block and no malformed tool_call)
            while messages:
                last_msg = messages[-1]
                if last_msg.get('role') == 'assistant':
                    content = last_msg.get('content', '')
                    is_malformed, _ = has_malformed_tool_call(content)
                    if is_malformed or not has_think_block(content):
                        # Invalid assistant turn - remove it and any preceding user message
                        print(f"Task {task_id}: Truncating invalid final assistant turn (missing <think> or malformed)")
                        messages.pop()
                        truncated = True
                        # Also remove the preceding user message if it exists
                        if messages and messages[-1].get('role') == 'user':
                            messages.pop()
                        await self.stats.inc(missing_think_blocks=1)
                    else:
                        # Valid assistant turn - stop truncating
                        break
                elif last_msg.get('role') == 'user':
                    # Trailing user message - remove it
                    messages.pop()
                    truncated = True
                else:
                    # Tool or system message - stop
                    break
            
            if truncated:
                await self.stats.inc(truncated_conversations=1)
        
        # Build result
        result = {
            "id": task_id,
            "mode": task_mode,
            "tools": tools,
            "messages": messages,
            "timestamp": datetime.now().isoformat(),
            "complete": success  # Track if this was a complete or partial result
        }
        
        if task_mode == 'curriculum':
            result["category"] = task.get('category', '')
            result["subcategory"] = task.get('subcategory', '')
            result["task_description"] = task.get('task_description', '')
        else:
            result["expected_answers"] = expected_answers
        
        # Final safety check: ensure conversation doesn't end with user message
        # (should already be handled by truncation above, but double-check)
        if messages and messages[-1].get('role') == 'user':
            messages.pop()
            print(f"Task {task_id}: Removed trailing user message")
        
        # Validate no consecutive user messages
        prev_role = None
        consecutive_user_indices = []
        for i, msg in enumerate(messages):
            curr_role = msg.get('role')
            if curr_role == 'user' and prev_role == 'user':
                consecutive_user_indices.append(i)
            prev_role = curr_role
        
        # Remove consecutive user messages (keep the first one)
        for idx in reversed(consecutive_user_indices):
            messages.pop(idx)
            print(f"Task {task_id}: Removed consecutive user message at index {idx}")
        
        # Ensure we have at least one complete exchange (user + assistant)
        has_user = any(m.get('role') == 'user' for m in messages)
        has_assistant = any(m.get('role') == 'assistant' for m in messages)
        if not (has_user and has_assistant):
            print(f"Task {task_id}: Incomplete conversation - missing user or assistant turn")
            self._flush_agent_outputs(task_id)
            return None
        
        # Update result with cleaned messages
        result["messages"] = messages
        
        self._append_jsonl(self.results_file, result)
        
        if self.config.output_sharegpt:
            # Determine source based on mode
            if task_mode == 'huggingface':
                source = self.config.dataset_name
                # For HF mode, task is the user query, category/subcategory may not exist
                sharegpt_metadata = {
                    "id": result["id"],
                    "source": source,
                    "task": user_query,  # First user query as task
                    "category": "",
                    "subcategory": ""
                }
            else:
                source = "curriculum"
                # For curriculum mode, we have category/subcategory/task_description
                sharegpt_metadata = {
                    "id": result["id"],
                    "source": source,
                    "task": task.get('task_description', user_query),
                    "category": task.get('category', ''),
                    "subcategory": task.get('subcategory', '')
                }
            
            sharegpt = to_sharegpt_format(
                messages, tools, sharegpt_metadata,
                include_reasoning_instructions=self.config.generate_reasoning
            )
            if sharegpt:
                self._append_jsonl(self.sharegpt_file, sharegpt)
            else:
                print(f"Task {task_id}: ShareGPT conversion returned None - skipping")
        
        # Debug: Print final messages
        if self.config.debug_print_messages:
            print_messages(
                messages,
                title=f"Final Messages for Task {task_id}",
                truncate=True,
                max_content_length=1000
            )
        
        # Flush agent outputs for this task
        self._flush_agent_outputs(task_id)
        
        await self.stats.inc(successful=1)
        return result
    
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
