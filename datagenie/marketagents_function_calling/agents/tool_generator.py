"""Tool Generator Agent - Creates tool definitions from curriculum tasks."""

import json
from typing import List, Dict, Any

from minference.lite.models import (
    LLMConfig, LLMClient, ResponseFormat, StructuredTool
)
from minference.lite.inference import InferenceOrchestrator

from market_agents.agents.base_agent.agent import Agent as MarketAgent
from market_agents.agents.personas.persona import Persona
from market_agents.agents.base_agent.prompter import PromptManager

from datagenie.marketagents_function_calling.schemas import GeneratedTools


def create_tool_generator_agent(
    task_description: str,
    category: str,
    subcategory: str,
    orchestrator: InferenceOrchestrator,
    model: str = "claude-3-5-sonnet-20241022",
    llm_client: LLMClient = LLMClient.anthropic,
    temperature: float = 0.4,
    max_tokens: int = 8192
) -> MarketAgent:
    """
    Create agent for generating tool definitions from curriculum task.
    
    Args:
        task_description: The task to generate tools for
        category: Task category (e.g., "Use Apps")
        subcategory: Task subcategory (e.g., "Food delivery apps")
        orchestrator: Inference orchestrator instance
        model: LLM model to use
        llm_client: LLM client to use
        temperature: Sampling temperature
        max_tokens: Maximum tokens for response
        
    Returns:
        Configured MarketAgent for tool generation
    """
    persona = Persona(
        role="API Designer",
        persona=(
            "You are an expert API designer who creates well-structured function definitions. "
            "You understand how to design intuitive APIs with clear parameters and return types."
        ),
        objectives=["Design realistic, useful API functions for the given task"],
        skills=["API design", "JSON Schema", "Function signatures", "Parameter modeling"]
    )
    
    output_tool = StructuredTool.from_pydantic(
        model=GeneratedTools,
        name="generated_tools",
        description="Return the generated tool definitions"
    )
    
    task = f"""Design 1-3 realistic API functions/tools that would be needed to accomplish this task.

Category: {category}
Subcategory: {subcategory}
Task: {task_description}

For each tool, provide:
- A clear snake_case function name
- A helpful description of what it does
- A JSON Schema for the parameters (with types, descriptions, required fields)

Make the tools realistic and practical. Include appropriate parameters that a real API would need.

You MUST call the generated_tools function with your response."""
    
    agent = MarketAgent(
        name="tool-generator",
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
