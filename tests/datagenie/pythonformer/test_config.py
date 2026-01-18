"""
Tests for pythonformer configuration.

Tests config loading, validation, and environment types.
"""

import pytest
import tempfile
from pathlib import Path

from datagenie.pythonformer.config import (
    PythonformerConfig,
    EnvironmentType,
    LLMClientType,
    REPLConfig,
    SubLLMConfig,
    DatasetConfig,
)


class TestEnvironmentType:
    """Test environment type enum."""
    
    def test_all_environment_types(self):
        """Test all environment types are defined."""
        assert EnvironmentType.MATH_PYTHON.value == "math-python"
        assert EnvironmentType.OOLONG.value == "oolong"
        assert EnvironmentType.HOTPOTQA.value == "hotpotqa"
        assert EnvironmentType.SWE.value == "swe"
        assert EnvironmentType.CODE.value == "code"
        assert EnvironmentType.CUSTOM.value == "custom"
    
    def test_environment_type_from_string(self):
        """Test creating environment type from string."""
        assert EnvironmentType("swe") == EnvironmentType.SWE
        assert EnvironmentType("oolong") == EnvironmentType.OOLONG


class TestLLMClientType:
    """Test LLM client type enum."""
    
    def test_all_client_types(self):
        """Test all client types are defined."""
        assert LLMClientType.OPENAI.value == "openai"
        assert LLMClientType.ANTHROPIC.value == "anthropic"
        assert LLMClientType.LITELLM.value == "litellm"


class TestREPLConfig:
    """Test REPL configuration."""
    
    def test_default_config(self):
        """Test default REPL configuration."""
        config = REPLConfig()
        
        assert config.server_url == "http://localhost:5003"
        assert config.max_output_chars == 8192
        assert config.max_output_lines == 500
        assert config.timeout_seconds == 120
        assert config.max_turns == 20
        assert config.enable_filesystem is True
        assert "numpy" in config.packages
        assert "pandas" in config.packages
    
    def test_custom_config(self):
        """Test custom REPL configuration."""
        config = REPLConfig(
            server_url="http://custom:8000",
            max_output_chars=16384,
            max_turns=30,
        )
        
        assert config.server_url == "http://custom:8000"
        assert config.max_output_chars == 16384
        assert config.max_turns == 30


class TestSubLLMConfig:
    """Test sub-LLM configuration."""
    
    def test_default_config(self):
        """Test default sub-LLM configuration."""
        config = SubLLMConfig()
        
        assert config.model == "gpt-4o-mini"
        assert config.client == LLMClientType.OPENAI
        assert config.max_tokens == 4096
        assert config.temperature == 0.7
    
    def test_custom_config(self):
        """Test custom sub-LLM configuration."""
        config = SubLLMConfig(
            model="Hermes-4-70B",
            client=LLMClientType.LITELLM,
            temperature=0.3,
        )
        
        assert config.model == "Hermes-4-70B"
        assert config.client == LLMClientType.LITELLM
        assert config.temperature == 0.3


class TestDatasetConfig:
    """Test dataset configuration."""
    
    def test_default_config(self):
        """Test default dataset configuration."""
        config = DatasetConfig()
        
        assert config.environment == EnvironmentType.CODE
        assert config.batch_size == 4
        assert config.output_sharegpt is True
        assert config.enable_rewards is True
        assert config.reward_function == "simple"
    
    def test_swe_config(self):
        """Test SWE-specific dataset configuration."""
        config = DatasetConfig(
            environment=EnvironmentType.SWE,
            dataset_name="SWE-bench/SWE-smith-py",
            field_mapping={
                "id": "instance_id",
                "prompt": "problem_statement",
                "expected_answer": "patch",
            }
        )
        
        assert config.environment == EnvironmentType.SWE
        assert config.dataset_name == "SWE-bench/SWE-smith-py"
        assert config.field_mapping["id"] == "instance_id"


class TestPythonformerConfig:
    """Test main pythonformer configuration."""
    
    def test_default_config(self):
        """Test default configuration."""
        config = PythonformerConfig()
        
        assert config.main_model == "gpt-4o"
        assert config.main_client == LLMClientType.OPENAI
        assert config.debug is False
        assert isinstance(config.repl, REPLConfig)
        assert isinstance(config.sub_llm, SubLLMConfig)
        assert isinstance(config.dataset, DatasetConfig)
    
    def test_from_yaml(self):
        """Test loading configuration from YAML."""
        yaml_content = """
main_model: "Hermes-4-405B"
main_client: "litellm"
main_temperature: 0.7
debug: true

repl:
  server_url: "http://localhost:5003"
  max_turns: 30

sub_llm:
  model: "Hermes-4-70B"
  client: "litellm"
  temperature: 0.3

dataset:
  environment: "swe"
  dataset_name: "SWE-bench/SWE-smith-py"
  batch_size: 2
"""
        
        with tempfile.NamedTemporaryFile(mode='w', suffix='.yaml', delete=False) as f:
            f.write(yaml_content)
            temp_path = f.name
        
        try:
            config = PythonformerConfig.from_yaml(temp_path)
            
            assert config.main_model == "Hermes-4-405B"
            assert config.main_client == LLMClientType.LITELLM
            assert config.debug is True
            assert config.repl.max_turns == 30
            assert config.sub_llm.model == "Hermes-4-70B"
            assert config.dataset.environment == EnvironmentType.SWE
            assert config.dataset.batch_size == 2
        finally:
            Path(temp_path).unlink()
    
    def test_get_llm_client(self):
        """Test getting LLM client enum."""
        config = PythonformerConfig()
        config.main_client = LLMClientType.LITELLM
        
        from minference.lite.models import LLMClient
        client = config.get_llm_client()
        assert client == LLMClient.litellm
    
    def test_get_sub_llm_client(self):
        """Test getting sub-LLM client enum."""
        config = PythonformerConfig()
        config.sub_llm.client = LLMClientType.ANTHROPIC
        
        from minference.lite.models import LLMClient
        client = config.get_sub_llm_client()
        assert client == LLMClient.anthropic


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
