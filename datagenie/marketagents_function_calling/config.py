"""Pipeline configuration models."""

import yaml
from enum import Enum
from pathlib import Path
from typing import List, Optional, Dict, Any
from pydantic import BaseModel, Field

from minference.lite.models import LLMClient


class GenerationMode(str, Enum):
    """Pipeline generation mode."""
    CURRICULUM = "curriculum"
    HUGGINGFACE = "huggingface"


class LLMClientType(str, Enum):
    """LLM client type."""
    LITELLM = "litellm"
    OPENAI = "openai"
    ANTHROPIC = "anthropic"


class AgentLLMConfig(BaseModel):
    """LLM configuration for a single agent."""
    client: LLMClientType = Field(default=LLMClientType.OPENAI)
    model: str = Field(default="gpt-4o")
    temperature: float = Field(default=0.3)
    max_tokens: int = Field(default=4096)
    
    def get_llm_client(self) -> LLMClient:
        """Convert to minference LLMClient."""
        mapping = {
            LLMClientType.LITELLM: LLMClient.litellm,
            LLMClientType.OPENAI: LLMClient.openai,
            LLMClientType.ANTHROPIC: LLMClient.anthropic,
        }
        return mapping.get(self.client, LLMClient.openai)


class AgentsConfig(BaseModel):
    """Configuration for all agents."""
    tool_generator: AgentLLMConfig = Field(default_factory=lambda: AgentLLMConfig(
        client=LLMClientType.ANTHROPIC, model="claude-3-5-sonnet-20241022", temperature=0.4
    ))
    query_generator: AgentLLMConfig = Field(default_factory=lambda: AgentLLMConfig(
        client=LLMClientType.OPENAI, model="gpt-4o", temperature=0.6
    ))
    docstring_generator: AgentLLMConfig = Field(default_factory=lambda: AgentLLMConfig(
        client=LLMClientType.ANTHROPIC, model="claude-3-5-sonnet-20241022", temperature=0.2
    ))
    schema_generator: AgentLLMConfig = Field(default_factory=lambda: AgentLLMConfig(
        client=LLMClientType.ANTHROPIC, model="claude-3-5-sonnet-20241022", temperature=0.2
    ))
    results_generator: AgentLLMConfig = Field(default_factory=lambda: AgentLLMConfig(
        client=LLMClientType.ANTHROPIC, model="claude-3-5-sonnet-20241022", temperature=0.4
    ))
    followup_generator: AgentLLMConfig = Field(default_factory=lambda: AgentLLMConfig(
        client=LLMClientType.OPENAI, model="gpt-4o", temperature=0.6
    ))
    clarification_agent: AgentLLMConfig = Field(default_factory=lambda: AgentLLMConfig(
        client=LLMClientType.OPENAI, model="gpt-4o", temperature=0.6
    ))
    tool_calling: AgentLLMConfig = Field(default_factory=lambda: AgentLLMConfig(
        client=LLMClientType.OPENAI, model="gpt-4o", temperature=0.2
    ))
    analysis_followup: AgentLLMConfig = Field(default_factory=lambda: AgentLLMConfig(
        client=LLMClientType.OPENAI, model="gpt-4o", temperature=0.6
    ))
    
    @classmethod
    def from_yaml(cls, yaml_path: str) -> "AgentsConfig":
        """Load agents config from YAML file."""
        path = Path(yaml_path)
        if not path.exists():
            # Try relative to module
            module_dir = Path(__file__).parent
            path = module_dir / yaml_path
        
        if not path.exists():
            print(f"Agents config not found at {yaml_path}, using defaults")
            return cls()
        
        with open(path, 'r') as f:
            data = yaml.safe_load(f)
        
        # Convert dict to AgentsConfig
        agents_dict = {}
        for agent_name, agent_config in data.items():
            if isinstance(agent_config, dict):
                agents_dict[agent_name] = AgentLLMConfig(**agent_config)
        
        return cls(**agents_dict)


class PipelineConfig(BaseModel):
    """Configuration for the function calling data generation pipeline."""
    
    # Mode settings
    mode: GenerationMode = Field(default=GenerationMode.CURRICULUM)
    
    # Curriculum settings
    curriculum_file: str = Field(default="configs/curriculum/function_calling.csv")
    curriculum_categories: List[str] = Field(default_factory=list)
    curriculum_subcategories: List[str] = Field(default_factory=list)
    
    # HuggingFace dataset settings
    dataset_name: str = Field(default="Salesforce/xlam-function-calling-60k")
    dataset_split: str = Field(default="train")
    
    # Generation settings
    batch_size: int = Field(default=8)
    max_recursion_depth: int = Field(default=3, description="Max tool call recursion depth (3 is usually enough for most tasks)")
    per_task_timeout: int = Field(default=300)
    save_partial_results: bool = Field(default=True, description="Save results even if max recursion is hit")
    
    # Docstring generation settings
    generate_docstrings: bool = Field(default=True)
    
    # Analysis follow-up settings
    generate_analysis_followup: bool = Field(
        default=False, 
        description="Generate non-tool-calling follow-up that analyzes previous tool results"
    )
    
    # Reasoning generation settings
    generate_reasoning: bool = Field(
        default=False,
        description="Generate reasoning tokens within <think></think> tags before tool calls and responses"
    )
    validate_reasoning: bool = Field(
        default=True,
        description="Validate that <think> blocks are properly formatted when generate_reasoning is enabled"
    )
    
    # Output settings
    output_dir: str = Field(default="outputs/function_calling")
    output_sharegpt: bool = Field(default=True)
    
    # Debug settings
    debug_print_messages: bool = Field(
        default=False,
        description="Pretty print messages and responses with colors for debugging"
    )
    
    # Validation settings
    validate_tool_calls: bool = Field(default=True)
    require_matching_arguments: bool = Field(default=True)
    require_tool_call_on_first_turn: bool = Field(default=False, description="Require tool call on first assistant turn, otherwise fail")
    allow_clarification_flow: bool = Field(default=True, description="Allow clarification requests and follow-up with details")
    
    # Agents config (loaded from YAML)
    agents_config_file: str = Field(default="configs/agents_config.yaml")
    agents: AgentsConfig = Field(default_factory=AgentsConfig)
    
    def load_agents_config(self) -> None:
        """Load agents configuration from YAML file."""
        self.agents = AgentsConfig.from_yaml(self.agents_config_file)
    
    @classmethod
    def from_yaml(cls, yaml_path: str) -> "PipelineConfig":
        """Load pipeline config from YAML file."""
        with open(yaml_path, 'r') as f:
            data = yaml.safe_load(f)
        
        # Flatten nested config
        config_dict = {
            "mode": data.get("mode", "curriculum"),
            "curriculum_file": data.get("curriculum", {}).get("file", "configs/curriculum/function_calling.csv"),
            "curriculum_categories": data.get("curriculum", {}).get("categories", []),
            "curriculum_subcategories": data.get("curriculum", {}).get("subcategories", []),
            "dataset_name": data.get("huggingface", {}).get("dataset_name", "Salesforce/xlam-function-calling-60k"),
            "dataset_split": data.get("huggingface", {}).get("split", "train"),
            "batch_size": data.get("generation", {}).get("batch_size", 8),
            "max_recursion_depth": data.get("generation", {}).get("max_recursion_depth", 3),
            "per_task_timeout": data.get("generation", {}).get("per_task_timeout", 300),
            "save_partial_results": data.get("generation", {}).get("save_partial_results", True),
            "generate_docstrings": data.get("docstrings", {}).get("generate", True),
            "generate_analysis_followup": data.get("generation", {}).get("generate_analysis_followup", False),
            "generate_reasoning": data.get("generation", {}).get("generate_reasoning", False),
            "validate_reasoning": data.get("generation", {}).get("validate_reasoning", True),
            "output_dir": data.get("output", {}).get("dir", "outputs/function_calling"),
            "output_sharegpt": data.get("output", {}).get("sharegpt", True),
            "debug_print_messages": data.get("output", {}).get("debug_print_messages", False),
            "validate_tool_calls": data.get("validation", {}).get("validate_tool_calls", True),
            "require_matching_arguments": data.get("validation", {}).get("require_matching_arguments", True),
            "require_tool_call_on_first_turn": data.get("validation", {}).get("require_tool_call_on_first_turn", True),
            "allow_clarification_flow": data.get("validation", {}).get("allow_clarification_flow", False),
            "agents_config_file": data.get("agents_config", "configs/agents_config.yaml"),
        }
        
        config = cls(**config_dict)
        config.load_agents_config()
        return config
