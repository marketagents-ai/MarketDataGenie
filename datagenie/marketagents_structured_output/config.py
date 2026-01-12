"""Pipeline configuration models for structured output generation."""

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
    schema_generator: AgentLLMConfig = Field(default_factory=lambda: AgentLLMConfig(
        client=LLMClientType.ANTHROPIC, model="claude-3-5-sonnet-20241022", temperature=0.3
    ))
    query_generator: AgentLLMConfig = Field(default_factory=lambda: AgentLLMConfig(
        client=LLMClientType.OPENAI, model="gpt-4o", temperature=0.6
    ))
    structured_output: AgentLLMConfig = Field(default_factory=lambda: AgentLLMConfig(
        client=LLMClientType.LITELLM, model="Hermes-4-405B", temperature=0.4, max_tokens=8192
    ))
    followup_generator: AgentLLMConfig = Field(default_factory=lambda: AgentLLMConfig(
        client=LLMClientType.OPENAI, model="gpt-4o", temperature=0.6
    ))
    analysis_followup: AgentLLMConfig = Field(default_factory=lambda: AgentLLMConfig(
        client=LLMClientType.OPENAI, model="gpt-4o", temperature=0.6
    ))
    clarification_agent: AgentLLMConfig = Field(default_factory=lambda: AgentLLMConfig(
        client=LLMClientType.OPENAI, model="gpt-4o", temperature=0.6
    ))
    
    @classmethod
    def from_yaml(cls, yaml_path: str) -> "AgentsConfig":
        """Load agents config from YAML file."""
        path = Path(yaml_path)
        if not path.exists():
            module_dir = Path(__file__).parent
            path = module_dir / yaml_path
        
        if not path.exists():
            print(f"Agents config not found at {yaml_path}, using defaults")
            return cls()
        
        with open(path, 'r') as f:
            data = yaml.safe_load(f)
        
        agents_dict = {}
        for agent_name, agent_config in data.items():
            if isinstance(agent_config, dict):
                agents_dict[agent_name] = AgentLLMConfig(**agent_config)
        
        return cls(**agents_dict)


class PipelineConfig(BaseModel):
    """Configuration for the structured output data generation pipeline."""
    
    # Mode settings
    mode: GenerationMode = Field(default=GenerationMode.CURRICULUM)
    
    # Curriculum settings
    curriculum_file: str = Field(default="configs/curriculum/json_mode.csv")
    curriculum_categories: List[str] = Field(default_factory=list)
    curriculum_subcategories: List[str] = Field(default_factory=list)
    
    # HuggingFace dataset settings
    dataset_name: str = Field(default="")
    dataset_split: str = Field(default="train")
    
    # Generation settings
    batch_size: int = Field(default=8)
    max_turns: int = Field(default=5, description="Max conversation turns")
    per_task_timeout: int = Field(default=300)
    
    # Schema generation settings
    generate_schema: bool = Field(
        default=True, 
        description="Generate schema from task if not provided in curriculum"
    )
    
    # Follow-up settings
    generate_followup: bool = Field(
        default=True,
        description="Generate follow-up turns for multi-turn conversations"
    )
    generate_analysis_followup: bool = Field(
        default=False, 
        description="Generate non-structured follow-up that analyzes previous output"
    )
    
    # Clarification settings
    allow_clarification_flow: bool = Field(
        default=True,
        description="Allow clarification flow when model asks for more info instead of generating JSON"
    )
    require_json_on_first_turn: bool = Field(
        default=False,
        description="Require JSON output on first turn (fail if clarification needed)"
    )
    max_clarification_turns: int = Field(
        default=2,
        description="Maximum clarification turns before failing"
    )
    
    # Reasoning generation settings
    generate_reasoning: bool = Field(
        default=False,
        description="Generate reasoning tokens within <think></think> tags"
    )
    validate_reasoning: bool = Field(
        default=True,
        description="Validate that <think> blocks are properly formatted"
    )
    
    # Output settings
    output_dir: str = Field(default="outputs/structured_output")
    output_sharegpt: bool = Field(default=True)
    
    # Debug settings
    debug_print_messages: bool = Field(default=False)
    
    # Agents config
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
        
        config_dict = {
            "mode": data.get("mode", "curriculum"),
            "curriculum_file": data.get("curriculum", {}).get("file", "configs/curriculum/json_mode.csv"),
            "curriculum_categories": data.get("curriculum", {}).get("categories", []),
            "curriculum_subcategories": data.get("curriculum", {}).get("subcategories", []),
            "dataset_name": data.get("huggingface", {}).get("dataset_name", ""),
            "dataset_split": data.get("huggingface", {}).get("split", "train"),
            "batch_size": data.get("generation", {}).get("batch_size", 8),
            "max_turns": data.get("generation", {}).get("max_turns", 5),
            "per_task_timeout": data.get("generation", {}).get("per_task_timeout", 300),
            "generate_schema": data.get("generation", {}).get("generate_schema", True),
            "generate_followup": data.get("generation", {}).get("generate_followup", True),
            "generate_analysis_followup": data.get("generation", {}).get("generate_analysis_followup", False),
            "allow_clarification_flow": data.get("generation", {}).get("allow_clarification_flow", True),
            "require_json_on_first_turn": data.get("generation", {}).get("require_json_on_first_turn", False),
            "max_clarification_turns": data.get("generation", {}).get("max_clarification_turns", 2),
            "generate_reasoning": data.get("generation", {}).get("generate_reasoning", False),
            "validate_reasoning": data.get("generation", {}).get("validate_reasoning", True),
            "output_dir": data.get("output", {}).get("dir", "outputs/structured_output"),
            "output_sharegpt": data.get("output", {}).get("sharegpt", True),
            "debug_print_messages": data.get("output", {}).get("debug_print_messages", False),
            "agents_config_file": data.get("agents_config", "configs/agents_config.yaml"),
        }
        
        config = cls(**config_dict)
        config.load_agents_config()
        return config
