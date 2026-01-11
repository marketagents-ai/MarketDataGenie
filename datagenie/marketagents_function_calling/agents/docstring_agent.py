"""Docstring Generator Agent - Enhances tool definitions with descriptions."""

import json
from typing import List, Dict, Any

from minference.lite.models import (
    LLMConfig, LLMClient, ResponseFormat, StructuredTool
)
from minference.lite.inference import InferenceOrchestrator

from market_agents.agents.base_agent.agent import Agent as MarketAgent
from market_agents.agents.personas.persona import Persona
from market_agents.agents.base_agent.prompter import PromptManager

from datagenie.marketagents_function_calling.schemas import ToolDocStrings


def create_docstring_agent(
    tools: List[Dict[str, Any]],
    orchestrator: InferenceOrchestrator,
    model: str = "claude-3-5-sonnet-20241022",
    llm_client: LLMClient = LLMClient.anthropic,
    temperature: float = 0.2,
    max_tokens: int = 2048
) -> MarketAgent:
    """
    Create agent for generating tool docstrings.
    
    Args:
        tools: List of tool definitions in OpenAI format
        orchestrator: Inference orchestrator instance
        model: LLM model to use
        llm_client: LLM client to use
        temperature: Sampling temperature
        max_tokens: Maximum tokens for response
        
    Returns:
        Configured MarketAgent for docstring generation
    """
    persona = Persona(
        role="API Documentation Specialist",
        persona=(
            "You are an expert at writing clear, comprehensive API documentation. "
            "You understand function signatures and can infer purpose from names and parameters."
        ),
        objectives=["Generate accurate, helpful docstrings for function signatures"],
        skills=["Technical writing", "API design", "JSON schema understanding"]
    )
    
    output_tool = StructuredTool.from_pydantic(
        model=ToolDocStrings,
        name="tool_docstrings",
        description="Return docstrings for each tool function"
    )
    
    tools_json = json.dumps(tools, indent=2)
    task = f"""Generate clear, helpful docstrings for each of these tool functions.
Each docstring should explain what the function does, its parameters, and expected return value.

Tools to document:
{tools_json}

You MUST call the tool_docstrings function with your response."""
    
    agent = MarketAgent(
        name="docstring-generator",
        persona=persona,
        task=task,
        tools=[output_tool],
        llm_config=LLMConfig(
            client=llm_client,
            model=model,
            temperature=temperature,
            max_tokens=max_tokens,
            response_format=ResponseFormat.tool  # Force tool use
        ),
        llm_orchestrator=orchestrator,
        prompt_manager=PromptManager()
    )
    
    # Set forced_output for ResponseFormat.tool to work correctly
    agent.chat_thread.forced_output = output_tool
    
    return agent
