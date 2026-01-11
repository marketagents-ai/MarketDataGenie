"""Schema Generator Agent - Creates JSON schemas for tool results."""

import json
from typing import List, Dict, Any

from minference.lite.models import (
    LLMConfig, LLMClient, ResponseFormat, StructuredTool
)
from minference.lite.inference import InferenceOrchestrator

from market_agents.agents.base_agent.agent import Agent as MarketAgent
from market_agents.agents.personas.persona import Persona
from market_agents.agents.base_agent.prompter import PromptManager

from datagenie.marketagents_function_calling.schemas import ContentSchemas


def create_schema_agent(
    tool_calls: List[Dict[str, Any]],
    orchestrator: InferenceOrchestrator,
    model: str = "claude-3-5-sonnet-20241022",
    llm_client: LLMClient = LLMClient.anthropic,
    temperature: float = 0.2,
    max_tokens: int = 2048
) -> MarketAgent:
    """
    Create agent for generating content schemas for tool results.
    
    Supports parallel tool calls - generates schemas for ALL tool calls,
    including multiple calls to the same function.
    
    Args:
        tool_calls: List of tool calls to generate schemas for
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
            "You are an expert at designing JSON schemas for API responses. "
            "You understand data types and can create appropriate schemas for tool results."
        ),
        objectives=["Create valid JSON schemas for tool execution results"],
        skills=["JSON Schema", "API design", "Data modeling"]
    )
    
    output_tool = StructuredTool.from_pydantic(
        model=ContentSchemas,
        name="content_schemas",
        description="Return JSON schemas for each tool's result"
    )
    
    # Build detailed tool call info
    tool_call_details = []
    unique_functions = set()
    for tc in tool_calls:
        func = tc.get('function', {})
        func_name = func.get('name', '')
        unique_functions.add(func_name)
        
        args = func.get('arguments', '{}')
        if isinstance(args, str):
            try:
                args = json.loads(args)
            except json.JSONDecodeError:
                args = {}
        
        tool_call_details.append({
            "name": func_name,
            "arguments": args
        })
    
    calls_json = json.dumps(tool_call_details, indent=2)
    
    # For schemas, we only need one schema per unique function name
    task = f"""Design JSON schemas for the results of these tool calls.
Each schema should define the structure of data the tool would return.

Tool calls:
{calls_json}

Note: If the same function is called multiple times with different arguments,
you only need to provide ONE schema for that function name (the schema structure
should be the same regardless of arguments).

Generate schemas for these unique functions: {list(unique_functions)}

You MUST call the content_schemas function with your response."""
    
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
            response_format=ResponseFormat.tool  # Force tool use
        ),
        llm_orchestrator=orchestrator,
        prompt_manager=PromptManager()
    )
    
    # Set forced_output for ResponseFormat.tool to work correctly
    agent.chat_thread.forced_output = output_tool
    
    return agent
