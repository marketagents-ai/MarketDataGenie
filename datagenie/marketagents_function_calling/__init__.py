"""
MarketAgents Function Calling Dataset Generation Pipeline

A modular pipeline for generating multi-turn function calling datasets using
minference + market_agents SDK.

Supports two modes:
- Curriculum-based generation: Generate tools/queries from task descriptions
- HuggingFace augmentation: Extend existing datasets to multi-turn
"""

from datagenie.marketagents_function_calling.config import (
    PipelineConfig, 
    GenerationMode, 
    LLMClientType,
    AgentLLMConfig,
    AgentsConfig,
)
from datagenie.marketagents_function_calling.pipeline import FunctionCallingPipeline

__all__ = [
    "PipelineConfig", 
    "GenerationMode", 
    "LLMClientType",
    "AgentLLMConfig",
    "AgentsConfig",
    "FunctionCallingPipeline",
]
