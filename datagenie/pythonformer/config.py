"""Configuration for Pythonformer Dataset Generation."""

from enum import Enum
from pathlib import Path
from typing import List, Dict, Any, Optional
from dataclasses import dataclass, field
import yaml


class EnvironmentType(Enum):
    """Task environment types."""
    MATH_PYTHON = "math-python"
    OOLONG = "oolong"  # Long-context
    CODE = "code"      # Code generation
    CUSTOM = "custom"


class LLMClientType(Enum):
    """Supported LLM clients."""
    OPENAI = "openai"
    ANTHROPIC = "anthropic"
    LITELLM = "litellm"


@dataclass
class REPLConfig:
    """Configuration for the Python REPL sandbox."""
    server_url: str = "http://localhost:5003"
    max_output_chars: int = 8192
    max_output_lines: int = 500
    timeout_seconds: int = 120
    max_turns: int = 20
    
    # Pre-installed packages
    packages: List[str] = field(default_factory=lambda: [
        "numpy", "pandas", "sympy", "scipy", "json", "re", 
        "collections", "math", "itertools", "functools",
        "datetime", "pathlib", "os"
    ])
    
    # Filesystem access
    enable_filesystem: bool = True


@dataclass 
class SubLLMConfig:
    """Configuration for sub-LLM calls (llm_query in sandbox)."""
    model: str = "gpt-4o-mini"
    client: LLMClientType = LLMClientType.OPENAI
    max_tokens: int = 4096
    temperature: float = 0.7
    max_parallel: int = 8
    timeout_seconds: int = 60


@dataclass
class DatasetConfig:
    """Configuration for dataset generation."""
    environment: EnvironmentType = EnvironmentType.CODE
    
    # HuggingFace dataset settings
    dataset_name: Optional[str] = None  # e.g., "nvidia/OpenCodeInstruct"
    dataset_config: Optional[str] = None  # e.g., "train" for OpenCodeInstruct
    dataset_split: str = "train"
    
    # Field mapping - map dataset fields to our expected fields
    field_mapping: Dict[str, str] = field(default_factory=lambda: {
        "id": "id",
        "prompt": "input",
        "expected_answer": "output",
        "context": None,  # Optional field
    })
    
    # Context processor for complex nested fields
    # Options: null, "hotpotqa", "oolong"
    context_processor: Optional[str] = None
    
    # Generation settings
    batch_size: int = 4
    limit: Optional[int] = None
    
    # Tips in system prompt
    include_tips: bool = True
    
    # Output settings
    output_dir: str = "outputs/pythonformer"
    output_sharegpt: bool = True
    mask_observations: bool = False  # If True, add loss_weight=0 to observations


@dataclass
class PythonformerConfig:
    """Main configuration."""
    repl: REPLConfig = field(default_factory=REPLConfig)
    sub_llm: SubLLMConfig = field(default_factory=SubLLMConfig)
    dataset: DatasetConfig = field(default_factory=DatasetConfig)
    
    # Main model
    main_model: str = "gpt-4o"
    main_client: LLMClientType = LLMClientType.OPENAI
    main_temperature: float = 0.7
    main_max_tokens: int = 8192
    
    debug: bool = False
    
    @classmethod
    def from_yaml(cls, path: str) -> "PythonformerConfig":
        """Load config from YAML file."""
        with open(path, 'r') as f:
            data = yaml.safe_load(f)
        
        config = cls()
        
        if 'repl' in data:
            for k, v in data['repl'].items():
                if hasattr(config.repl, k):
                    setattr(config.repl, k, v)
        
        if 'sub_llm' in data:
            for k, v in data['sub_llm'].items():
                if k == 'client':
                    config.sub_llm.client = LLMClientType(v)
                elif hasattr(config.sub_llm, k):
                    setattr(config.sub_llm, k, v)
        
        if 'dataset' in data:
            for k, v in data['dataset'].items():
                if k == 'environment':
                    config.dataset.environment = EnvironmentType(v)
                elif k == 'field_mapping':
                    config.dataset.field_mapping = v
                elif hasattr(config.dataset, k):
                    setattr(config.dataset, k, v)
        
        # Main model settings
        if 'main_model' in data:
            config.main_model = data['main_model']
        if 'main_client' in data:
            config.main_client = LLMClientType(data['main_client'])
        if 'main_temperature' in data:
            config.main_temperature = data['main_temperature']
        if 'main_max_tokens' in data:
            config.main_max_tokens = data['main_max_tokens']
        if 'debug' in data:
            config.debug = data['debug']
        
        return config
    
    def get_llm_client(self) -> 'LLMClient':
        """Get minference LLMClient enum for main model."""
        from minference.lite.models import LLMClient
        return {
            LLMClientType.OPENAI: LLMClient.openai,
            LLMClientType.ANTHROPIC: LLMClient.anthropic,
            LLMClientType.LITELLM: LLMClient.litellm,
        }[self.main_client]
    
    def get_sub_llm_client(self) -> 'LLMClient':
        """Get minference LLMClient enum for sub-LLM."""
        from minference.lite.models import LLMClient
        return {
            LLMClientType.OPENAI: LLMClient.openai,
            LLMClientType.ANTHROPIC: LLMClient.anthropic,
            LLMClientType.LITELLM: LLMClient.litellm,
        }[self.sub_llm.client]


# Environment-specific tips
ENV_TIPS = {
    EnvironmentType.MATH_PYTHON: """
## Tips for Math Problems
- Use sympy for symbolic math
- Verify calculations with multiple approaches
- Show your work step by step in code
""",

    EnvironmentType.OOLONG: """
## Tips for Long Context
- Save the context to a file if it's very large
- Use string methods or regex to search efficiently
- Process in chunks if needed
- Use `llm_query()` for semantic analysis of chunks
""",

    EnvironmentType.CODE: """
## Tips for Code Tasks
- Write clean, testable functions
- Include test cases to verify correctness
- Handle edge cases
- Use the REPL to iterate and debug
""",
}
