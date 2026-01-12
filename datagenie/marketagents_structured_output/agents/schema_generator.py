"""Schema Generator Agent - Creates JSON schemas from task descriptions."""

import json
from typing import Dict, Any

from minference.lite.models import (
    LLMConfig, LLMClient, ResponseFormat, StructuredTool
)
from minference.lite.inference import InferenceOrchestrator

from market_agents.agents.base_agent.agent import Agent as MarketAgent
from market_agents.agents.personas.persona import Persona
from market_agents.agents.base_agent.prompter import PromptManager

from datagenie.marketagents_structured_output.schemas import GeneratedSchema


def create_schema_generator_agent(
    task_description: str,
    category: str,
    subcategory: str,
    orchestrator: InferenceOrchestrator,
    model: str = "claude-3-5-sonnet-20241022",
    llm_client: LLMClient = LLMClient.anthropic,
    temperature: float = 0.3,
    max_tokens: int = 4096
) -> MarketAgent:
    """
    Create agent for generating JSON schemas from curriculum task.
    
    Args:
        task_description: The task to generate schema for
        category: Task category (e.g., "JSON Schema")
        subcategory: Task subcategory (e.g., "Address Schema")
        orchestrator: Inference orchestrator instance
        model: LLM model to use
        llm_client: LLM client to use
        temperature: Sampling temperature
        max_tokens: Maximum tokens for response
        
    Returns:
        Configured MarketAgent for schema generation
    """
    persona = Persona(
        role="JSON Schema Designer",
        persona=(
            "You are an expert JSON Schema designer who creates well-structured, "
            "valid JSON schemas following the JSON Schema specification (draft 2020-12). "
            "You understand data modeling, type constraints, and schema validation."
        ),
        objectives=["Design realistic, valid JSON schemas for the given task"],
        skills=["JSON Schema", "Data modeling", "Type systems", "Validation rules"]
    )
    
    output_tool = StructuredTool.from_pydantic(
        model=GeneratedSchema,
        name="generated_schema",
        description="Return the generated JSON schema"
    )
    
    task = f"""Design a JSON Schema for the following task.

Category: {category}
Subcategory: {subcategory}
Task: {task_description}

Requirements:
1. Create a valid JSON Schema (draft 2020-12 compatible)
2. Include appropriate types, descriptions, and constraints
3. Use required fields where appropriate
4. Include realistic property names and types
5. Add format constraints where applicable (email, date-time, uri, etc.)
6. Use nested objects/arrays if the data structure requires it

The schema should be practical and represent real-world data structures.

You MUST call the generated_schema function with your response."""
    
    agent = MarketAgent(
        name="schema-generator",
        persona=persona,
        task=task,
        tools=[output_tool],
        llm_config=LLMConfig(
            client=llm_client,
            model=model,
            temperature=temperature,
            max_tokens=max_tokens,
            response_format=ResponseFormat.tool
        ),
        llm_orchestrator=orchestrator,
        prompt_manager=PromptManager()
    )
    
    agent.chat_thread.forced_output = output_tool
    
    return agent
