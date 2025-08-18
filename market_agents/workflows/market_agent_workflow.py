from typing import Dict, List, Any, Self, Type, Optional, Union, Literal
from pydantic import BaseModel, Field, model_validator
from datetime import datetime
import json
import os
from uuid import UUID
from market_agents.agents.market_agent import MarketAgent
from market_agents.agents.cognitive_steps import (
    ActionStep,
    CognitiveEpisode,
    PerceptionStep,
    ReflectionStep
)
from market_agents.environments.mechanisms.mcp_server import MCPServerActionSpace, MCPServerEnvironment
from market_agents.environments.mechanisms.chat import ToolEnabledChatActionSpace
from market_agents.environments.environment import MultiAgentEnvironment
from minference.lite.models import (
    CallableTool,
    StructuredTool,
    CallableMCPTool,
    ResponseFormat,
    Entity
)

from typing import Dict, List, Any, Optional, Union
from pydantic import BaseModel, Field

import logging
logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)

# Add a handler if none exists
if not logger.handlers:
    handler = logging.StreamHandler()
    handler.setLevel(logging.DEBUG)
    formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    handler.setFormatter(formatter)
    logger.addHandler(handler)

def _json_safe(obj):
    """Recursively convert common non-JSON-serializable objects to JSON-safe forms."""
    # Pydantic BaseModel
    if isinstance(obj, BaseModel):
        return _json_safe(obj.model_dump())
    # UUID -> str
    if isinstance(obj, UUID):
        return str(obj)
    # Entities from minference that may wrap pydantic/uuids
    if hasattr(obj, "model_dump") and callable(getattr(obj, "model_dump")):
        try:
            return _json_safe(obj.model_dump())
        except Exception:
            pass
    # dict
    if isinstance(obj, dict):
        return {str(_json_safe(k)): _json_safe(v) for k, v in obj.items()}
    # list/tuple/set
    if isinstance(obj, (list, tuple, set)):
        t = type(obj)
        return t(_json_safe(v) for v in obj)
    # fallback for simple types
    try:
        json.dumps(obj)
        return obj
    except TypeError:
        return str(obj)

# ADD BELOW THIS LINE
from typing import Any as _Any  # local alias to avoid confusion

def _extract_text(action_result: _Any) -> str:
    """Best-effort extraction of assistant text from action_result structures."""
    try:
        if isinstance(action_result, str):
            return action_result
        if isinstance(action_result, dict):
            info = action_result.get("info") or action_result
            if isinstance(info, dict) and "assistant_text" in info:
                return str(info.get("assistant_text") or "")
            msg = info.get("message") if isinstance(info, dict) else None
            if isinstance(msg, dict) and "content" in msg:
                return str(msg.get("content") or "")
        if hasattr(action_result, "model_dump"):
            return _extract_text(action_result.model_dump())
        if hasattr(action_result, "content"):
            return str(getattr(action_result, "content"))
    except Exception:
        pass
    return str(action_result)

# --- Helper functions for safe string formatting ---
class _SafeDefaultDict(dict):
    def __missing__(self, key):
        return "{" + key + "}"

def _escape_braces(val: Any) -> Any:
    if isinstance(val, str):
        return val.replace("{", "{{").replace("}", "}}")
    return val

class WorkflowStepIO(BaseModel):
    """Input/output values for workflow steps."""
    name: str = Field(..., description="Name of the input/output field")
    data: Union[Type[BaseModel], Dict[str, Any]] = Field(
        default_factory=dict,
        description="Schema (as BaseModel class) or actual data (as dict)"
    )

    def format_for_prompt(self) -> str:
        """Format the input data for use in prompts (robust to UUIDs/BaseModel instances)."""
        # If `data` is a BaseModel *class*, show its JSON schema
        if isinstance(self.data, type) and issubclass(self.data, BaseModel):
            return f"{self.name}:\n{self.data.model_json_schema()}"

        # If `data` is a BaseModel *instance*, dump to a plain dict first
        if isinstance(self.data, BaseModel):
            payload = self.data.model_dump()
        else:
            payload = self.data

        try:
            # Use default=str so UUIDs and other exotic types serialize cleanly
            text = json.dumps(payload, indent=2, ensure_ascii=False, default=str)
        except TypeError:
            # Last‑ditch fallback: string representation
            text = str(payload)
        return f"{self.name}:\n{text}"

    def get_formatted_data(self) -> Dict[str, Any]:
        """Get the data in a format suitable for workflow processing"""
        if isinstance(self.data, type) and issubclass(self.data, BaseModel):
            return {}
        if isinstance(self.data, BaseModel):
            return self.data.model_dump()
        return self.data

    model_config = {
        "arbitrary_types_allowed": True
    }

class CognitiveStepResult(BaseModel):
    """Base class for any cognitive step result"""
    step_type: str = Field(..., description="Type of cognitive step (perception, action, reflection, etc)")
    content: Any = Field(..., description="Result content from the step")
    metadata: Dict[str, Any] = Field(default_factory=dict, description="Step-specific metadata")

class EpisodeResult(BaseModel):
    """Result from a full cognitive episode with arbitrary steps"""
    steps: List[CognitiveStepResult] = Field(..., description="Results from each cognitive step")
    episode_metadata: Dict[str, Any] = Field(
        default_factory=dict, 
        description="Metadata about the episode execution"
    )

    @property
    def step_types(self) -> List[str]:
        """Get list of step types in this episode"""
        return [step.step_type for step in self.steps]
    
    def get_step_result(self, step_type: str) -> Optional[CognitiveStepResult]:
        """Get result for a specific step type"""
        for step in self.steps:
            if step.step_type == step_type:
                return step
        return None

class WorkflowStepResult(BaseModel):
    """Complete result from a workflow step execution"""
    step_id: str = Field(..., description="Identifier for the workflow step")
    status: Literal["completed", "failed"] = Field(..., description="Execution status")
    result: Union[EpisodeResult, CognitiveStepResult] = Field(
        ..., 
        description="The actual result, either from an episode or single step"
    )
    tool_results: List[Any] = Field(
        default_factory=list,
        description="Results from tool executions"
    )
    metadata: Dict[str, Any] = Field(
        default_factory=dict,
        description="General metadata about the step execution"
    )
    error: Optional[str] = Field(
        None,
        description="Error message if status is failed"
    )

    @property
    def is_episode(self) -> bool:
        """Check if this is an episode result"""
        return isinstance(self.result, EpisodeResult)

    @property
    def is_single_step(self) -> bool:
        """Check if this is a single step result"""
        return isinstance(self.result, CognitiveStepResult)
    
class WorkflowStep(Entity):
    """
    A step in a workflow that executes through MarketAgent's cognitive architecture.
    Can run as a single ActionStep or full cognitive episode.
    """
    name: str = Field(
        ...,
        description="Name identifier for this step"
    )
    environment_name: str = Field(
        ...,
        description="Name of the MCP server environment for this step"
    )
    tools: List[Union[CallableTool, StructuredTool, CallableMCPTool]] = Field(
        ..., 
        description="Tools to be executed in this step"
    )
    subtask: str = Field(
        ..., 
        description="Instruction to follow for this workflow step"
    )
    inputs: List[WorkflowStepIO] = Field(
        default_factory=list,
        description="Input schema definitions"
    )
    output: Optional[WorkflowStepIO] = Field(
        default=None,
        description="Output schema definitions"
    )
    run_full_episode: bool = Field(
        default=False,
        description="Whether to run full cognitive episode (perception->action->reflection) or just action step"
    )
    sequential_tools: bool = Field(
        default=True,
        description="Whether tools should be executed in sequence through ActionStep's workflow mode"
    )

    async def execute(
        self,
        agent: "MarketAgent",
        inputs: Dict[str, Any],
        mcp_servers: Dict[str, Union[MCPServerEnvironment, MultiAgentEnvironment]],
        workflow_task: Optional[str] = None
    ) -> Dict[str, Any]:
        """Execute this workflow step using MarketAgent's cognitive architecture."""
        try:
            # Check if MCP server environment is required (only if tools are needed)
            if self.tools and self.environment_name not in mcp_servers:
                raise ValueError(f"MCP server environment '{self.environment_name}' not found - required for tools: {[t.name for t in self.tools]}")

            # If no tools are needed, we can proceed without MCP server
            if not self.tools:
                logger.debug(f"No tools required for step '{self.name}', proceeding without MCP server")

            # Debug input state
            logger.debug(f"Executing step '{self.name}' with inputs: {inputs}")

            # Sanitize inputs for template formatting (escape any literal braces in values)
            _format_inputs = _SafeDefaultDict({k: _escape_braces(v) for k, v in (inputs or {}).items()})

            # Format the task with inputs
            formatted_task = workflow_task.format_map(_format_inputs) if workflow_task else ""
            logger.debug(f"Formatted workflow task: {formatted_task}")

            input_context = []
            for io in self.inputs:
                if io.name in inputs:
                    if isinstance(io.data, type) and issubclass(io.data, BaseModel):
                        validated_data = io.data.model_validate(inputs[io.name])
                        io.data = validated_data.model_dump()
                    else:
                        io.data = inputs[io.name]
                    input_context.append(io.format_for_prompt())

            formatted_context = "\n".join(input_context)
            logger.debug(f"Formatted context: {formatted_context}")

            # Format the subtask with inputs
            try:
                formatted_subtask = self.subtask.format_map(_format_inputs)
            except Exception as e:
                logger.error(f"[WorkflowStep:{self.name}] subtask formatting failed: {e}. Falling back to raw subtask.")
                formatted_subtask = self.subtask
            logger.debug(f"Formatted subtask: {formatted_subtask}")

            combined_task = f"""
            {formatted_task}

            ## Current Step: {self.name}

            ## Context:
            {formatted_context}

            ## Sub-task 
            {formatted_subtask}
            """.strip()

            agent.task = combined_task
            agent._refresh_prompts()

            env_info = dict(inputs) if inputs else {}
            env_info["prompt"] = combined_task
            # also provide chat-style messages as a fallback for any adapters
            if "messages" not in env_info or not env_info.get("messages"):
                env_info["messages"] = [{"role": "user", "content": combined_task}]

            # Handle environment setup based on type
            if self.environment_name in mcp_servers:
                environment = mcp_servers[self.environment_name]

                if isinstance(environment, MCPServerEnvironment) and self.tools:
                    # MCP server with tools - create restricted action space
                    selected_action_space = MCPServerActionSpace(
                        mechanism=environment.mechanism,
                        selected_tools=self.tools,
                        workflow=self.sequential_tools and len(self.tools) > 1
                    )

                    # Create a temporary environment with restricted tools
                    environment.action_space = selected_action_space
                    agent.chat_thread.tools = self.tools

                    # Add temporary environment to agent
                    agent.environments[self.environment_name] = environment
                    initial_history_len = len(environment.mechanism.tool_history.get("default", []))

                elif isinstance(environment, MultiAgentEnvironment):
                    # Regular environment (like ChatMechanism). If tools are provided, attach a tool-enabled action space.
                    if self.tools:
                        environment.action_space = ToolEnabledChatActionSpace(tools=self.tools)
                        agent.chat_thread.tools = self.tools
                        # Prefer workflow when multiple tools and sequential flag set
                        if self.sequential_tools and len(self.tools) > 1:
                            agent.chat_thread.llm_config.response_format = ResponseFormat.workflow
                            agent.chat_thread.workflow_step = 0
                        elif len(self.tools) > 1:
                            agent.chat_thread.llm_config.response_format = ResponseFormat.auto_tools
                            agent.chat_thread.workflow_step = None
                        else:
                            # Exactly one tool: force tool output
                            agent.chat_thread.llm_config.response_format = ResponseFormat.tool
                            agent.chat_thread.forced_output = self.tools[0]
                            agent.chat_thread.workflow_step = None
                        initial_history_len = 0
                    else:
                        # No tools: plain chat, force text response format
                        agent.chat_thread.tools = []
                        agent.chat_thread.llm_config.response_format = ResponseFormat.text
                        agent.chat_thread.forced_output = None
                        agent.chat_thread.workflow_step = None
                        initial_history_len = 0
                    # Bind environment
                    agent.environments[self.environment_name] = environment
                else:
                    # MCP server without tools - set up for LLM-only, force text output
                    agent.environments[self.environment_name] = environment
                    agent.chat_thread.tools = []
                    agent.chat_thread.llm_config.response_format = ResponseFormat.text
                    agent.chat_thread.forced_output = None
                    agent.chat_thread.workflow_step = None
                    initial_history_len = 0
            else:
                # No environment found
                raise ValueError(f"Environment '{self.environment_name}' not found in available environments")

            # Safety: if no tools are attached at this point, prefer text format
            if hasattr(agent, "chat_thread") and agent.chat_thread is not None and not agent.chat_thread.tools:
                agent.chat_thread.llm_config.response_format = ResponseFormat.text
                agent.chat_thread.forced_output = None
                agent.chat_thread.workflow_step = None

            # Create action step
            action_step = ActionStep(
                step_name=self.name,
                agent_id=agent.id,
                environment_name=self.environment_name,
                environment_info=env_info,
                action_space=self.tools[0] if not self.sequential_tools and self.tools else None
            )

            # Only switch to auto_tools when there are multiple tools and sequential execution is disabled
            if not self.sequential_tools and self.tools and len(self.tools) > 1:
                agent.chat_thread.llm_config.response_format = ResponseFormat.auto_tools
                agent.chat_thread.workflow_step = None

            if self.run_full_episode:
                # Run full cognitive episode
                episode = CognitiveEpisode(
                    steps=[PerceptionStep, ActionStep, ReflectionStep],
                    environment_name=self.environment_name,
                    metadata={
                        "workflow_step": self.name,
                        "tools": [t.name for t in self.tools]
                    }
                )
                results = await agent.run_episode(episode=episode)

                # Log a preview of the action step's text output for inspection
                try:
                    _action_text_preview = None
                    if isinstance(results, (list, tuple)) and len(results) > 1:
                        _action_text_preview = _extract_text(results[1])
                    if _action_text_preview:
                        logger.info(f"[WorkflowStep:{self.name}] episode action text -> {_action_text_preview[:200]}{'…' if len(_action_text_preview) > 200 else ''}")
                except Exception as _e:
                    logger.debug(f"[WorkflowStep:{self.name}] episode preview logging failed: {_e}")

                # Get tool results if MCP server was used
                if self.tools and self.environment_name in mcp_servers:
                    environment = mcp_servers[self.environment_name]
                    if isinstance(environment, MCPServerEnvironment):
                        print(f"Tool Execution History:\n {environment.mechanism.tool_history.get('default', [])}")
                        tool_results = environment.mechanism.tool_history.get("default", [])[initial_history_len:]
                    else:
                        tool_results = []
                else:
                    tool_results = []

                episode_result = EpisodeResult(
                    steps=[
                        CognitiveStepResult(
                            step_type="perception",
                            content=_json_safe(results[0])
                        ),
                        CognitiveStepResult(
                            step_type="action",
                            content=_json_safe(results[1])
                        ),
                        CognitiveStepResult(
                            step_type="reflection",
                            content=_json_safe(results[2])
                        )
                    ],
                    episode_metadata={
                        "workflow_step": self.name,
                        "tools": [t.name for t in self.tools]
                    }
                )

                return WorkflowStepResult(
                    step_id=self.name,
                    status="completed",
                    result=episode_result,
                    tool_results=tool_results,
                    metadata={
                        "agent_id": str(agent.id),
                        "environment": self.environment_name,
                        "tools_used": [t.name for t in self.tools],
                        "timestamp": datetime.utcnow().isoformat(),
                        "execution_type": "episode"
                    }
                )
            else:
                # Run just the action step
                action_result = await agent.run_step(step=action_step)
                logger.debug(f"[WorkflowStep:{self.name}] action step returned type={type(action_result)}")

                # Prefer plain text for StrAction; also log a short preview
                assistant_text = _extract_text(action_result)
                logger.info(f"[WorkflowStep:{self.name}] action text -> {assistant_text[:200]}{'…' if len(assistant_text) > 200 else ''}")

                # Get tool results if MCP server was used
                if self.tools and self.environment_name in mcp_servers:
                    environment = mcp_servers[self.environment_name]
                    if isinstance(environment, MCPServerEnvironment):
                        tool_results = environment.mechanism.tool_history.get("default", [])[initial_history_len:]
                    else:
                        tool_results = []
                else:
                    tool_results = []

                return WorkflowStepResult(
                    step_id=self.name,
                    status="completed",
                    result=CognitiveStepResult(
                        step_type="action",
                        content=assistant_text,
                        metadata={
                            "workflow_step": self.name,
                            "tools": [t.name for t in self.tools]
                        }
                    ),
                    tool_results=tool_results,
                    metadata={
                        "agent_id": str(agent.id),
                        "environment": self.environment_name,
                        "tools_used": [t.name for t in self.tools],
                        "timestamp": datetime.utcnow().isoformat(),
                        "execution_type": "action_step"
                    }
                )

        except Exception as e:
            return WorkflowStepResult(
                step_id=self.name,
                status="failed",
                result=CognitiveStepResult(
                    step_type="action",
                    content=None,
                    metadata={"error": str(e)}
                ),
                error=str(e),
                metadata={
                    "agent_id": str(agent.id),
                    "environment": self.environment_name,
                    "timestamp": datetime.utcnow().isoformat()
                }
            )

class WorkflowExecutionResult(BaseModel):
    """Result from a complete workflow execution"""
    workflow_id: str = Field(..., description="Identifier for the workflow")
    final_state: Dict[str, Any] = Field(
        default_factory=dict,
        description="Final state after all steps executed"
    )
    step_results: List[WorkflowStepResult] = Field(
        default_factory=list,
        description="Results from each workflow step"
    )
    tool_history: Dict[str, Dict[str, List[Dict[str, Any]]]] = Field(
        default_factory=dict,
        description="Tool execution history per environment and cohort"
    )

    @property
    def successful_steps(self) -> List[WorkflowStepResult]:
        """Get list of successfully completed steps"""
        return [step for step in self.step_results if step.status == "completed"]
    
    @property
    def failed_steps(self) -> List[WorkflowStepResult]:
        """Get list of failed steps"""
        return [step for step in self.step_results if step.status == "failed"]
    
    def get_step_result(self, step_id: str) -> Optional[WorkflowStepResult]:
        """Get result for a specific step"""
        for step in self.step_results:
            if step.step_id == step_id:
                return step
        return None

    def get_cohort_tool_history(self, environment: str, cohort: str = "default") -> List[Dict[str, Any]]:
        """Get tool history for a specific environment and cohort"""
        return self.tool_history.get(environment, {}).get(cohort, [])
        
class Workflow(Entity):
    """A workflow that orchestrates execution of steps across multiple environments."""
    name: str = Field(..., description="Name identifier for this workflow")
    task: str = Field(
        ..., 
        description="High-level task prompt for the entire workflow"
    )
    steps: List[WorkflowStep] = Field(..., description="Ordered sequence of workflow steps")
    mcp_servers: Dict[str, Union[MCPServerEnvironment, MultiAgentEnvironment]] = Field(
        ...,
        description="Environments for tool execution (MCP servers) or chat-based inference (MultiAgentEnvironment)"
    )

    model_config = {
        "arbitrary_types_allowed": True,
        "extra": "allow"
    }
    @classmethod
    def create(
        cls,
        name: str,
        task: str,
        steps: List[WorkflowStep],
        mcp_servers: Dict[str, MCPServerEnvironment],
    ) -> 'Workflow':
        """Create a new workflow instance while preserving the MCP servers."""
        import logging
        logger = logging.getLogger(__name__)
        logger.setLevel(logging.DEBUG)

        # Add a handler if none exists
        if not logger.handlers:
            handler = logging.StreamHandler()
            handler.setLevel(logging.DEBUG)
            formatter = logging.Formatter('%(levelname)s - %(message)s')
            handler.setFormatter(formatter)
            logger.addHandler(handler)

        # Create instance without validation
        instance = cls.model_construct(
            name=name,
            task=task,
            steps=steps,
            mcp_servers=mcp_servers
        )
        
        # Log the state
        logger.debug(f"Created workflow with environments: {list(instance.mcp_servers.keys())}")
        return instance
        
    async def execute(
        self,
        agent: "MarketAgent",
        initial_inputs: Dict[str, Any]
    ) -> WorkflowExecutionResult:
        """Execute the workflow across multiple environments."""
        import logging
        logger = logging.getLogger(__name__)
        logger.setLevel(logging.DEBUG)

        state = initial_inputs.copy()
        results: List[WorkflowStepResult] = []

        # Build a sanitized mapping for string.format to avoid brace errors
        _format_inputs = _SafeDefaultDict({k: _escape_braces(v) for k, v in initial_inputs.items()})
        formatted_workflow_task = self.task.format_map(_format_inputs) if self.task else None

        # Create expanded steps list when output schemas are defined
        expanded_steps = []
        for step in self.steps:
            # Add original step
            expanded_steps.append(step)

            # If step has output schema, add IO step
            if step.output and isinstance(step.output.data, type) and issubclass(step.output.data, BaseModel):
                output_tool = StructuredTool.from_pydantic(
                    model=step.output.data,
                    name=step.output.data.__name__.lower(),
                    description=f"Structure output for {step.name} step"
                )

                io_step = WorkflowStep(
                    name=f"{step.name}_io",
                    environment_name=step.environment_name,
                    tools=[output_tool],
                    subtask=f"Structure the workflow results including previous step's output using the {output_tool.name} schema.",
                    sequential_tools=False,
                    run_full_episode=False
                )
                expanded_steps.append(io_step)

        previous_result = None
        for i, step in enumerate(expanded_steps):
            logger.debug(f"Executing step '{step.name}' with inputs: {state}")

            # Add previous step result to inputs if available
            if i > 0 and previous_result is not None:
                state["previous_step_result"] = previous_result

            step_result = await step.execute(
                agent=agent,
                inputs=state,
                mcp_servers=self.mcp_servers,
                workflow_task=formatted_workflow_task
            )

            if step_result.status == "completed":
                # Handle both episode and single step results
                if step_result.is_episode:
                    action_step = step_result.result.get_step_result("action")
                    if action_step:
                        previous_result = _extract_text(action_step.content)
                else:
                    previous_result = _extract_text(step_result.result.content)
                if previous_result is not None:
                    # Make previous result and name available to the next step
                    state["previous_step_result"] = previous_result
                    state["previous_step_name"] = step.name

                    # Store result under the step's own name for template access: {<stepname>}
                    state[step.name] = previous_result

                    # Maintain a generic results map and a history list for downstream use
                    results_map = state.get("_results", {})
                    results_map[step.name] = previous_result
                    state["_results"] = results_map

                    history_list = state.get("_result_history", [])
                    history_list.append({"step": step.name, "content": previous_result})
                    state["_result_history"] = history_list
                # keep state JSON-safe between steps
                state = _json_safe(state)
                try:
                    _p = state.get("previous_step_result")
                    if _p is not None:
                        logger.debug(f"[Workflow] state.previous_step_result preview -> {str(_p)[:200]}{'…' if len(str(_p)) > 200 else ''}")
                except Exception as _e:
                    logger.debug(f"[Workflow] preview state logging failed: {_e}")
                results.append(step_result)
                logger.debug(f"Updated state after {step.name}: {state}")
            else:
                logger.error(f"Step '{step.name}' failed: {step_result.error}")
                results.append(step_result)

        # Collect tool history only from available MCP servers
        tool_history = {}
        for env_name, env in self.mcp_servers.items():
            if isinstance(env, MCPServerEnvironment) and hasattr(env, 'mechanism') and hasattr(env.mechanism, 'tool_history'):
                tool_history[env_name] = env.mechanism.tool_history

        # Defensive logging of the final state before returning result object
        try:
            logger.debug(f"[Workflow] final_state preview -> {str(state)[:200]}{'…' if len(str(state)) > 200 else ''}")
        except Exception as _e:
            logger.debug(f"[Workflow] final_state preview logging failed: {_e}")
        return WorkflowExecutionResult(
            workflow_id=self.name,
            final_state=_json_safe(state),
            step_results=results,
            tool_history=tool_history
        )