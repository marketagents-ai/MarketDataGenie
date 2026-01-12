"""
MarketAgents Structured Output Dataset Generation Pipeline.

A modular pipeline for generating structured output / JSON mode datasets
with chain-of-thought reasoning using the `minference` + `market_agents` SDK.
"""

from datagenie.marketagents_structured_output.pipeline import StructuredOutputPipeline
from datagenie.marketagents_structured_output.config import PipelineConfig, GenerationMode

__all__ = [
    "StructuredOutputPipeline",
    "PipelineConfig",
    "GenerationMode",
]
