"""Results Generator Agent - Synthesizes realistic tool execution results."""

import json
from typing import List, Dict, Any

from minference.lite.models import (
    LLMConfig, LLMClient, ResponseFormat, StructuredTool
)
from minference.lite.inference import InferenceOrchestrator

from market_agents.agents.base_agent.agent import Agent as MarketAgent
from market_agents.agents.personas.persona import Persona
from market_agents.agents.base_agent.prompter import PromptManager

from datagenie.marketagents_function_calling.schemas import ToolMessages


def create_results_agent(
    tool_calls: List[Dict[str, Any]],
    schemas: List[Dict[str, Any]],
    user_query: str,
    orchestrator: InferenceOrchestrator,
    model: str = "claude-3-5-sonnet-20241022",
    llm_client: LLMClient = LLMClient.anthropic,
    temperature: float = 0.4,
    max_tokens: int = 4096
) -> MarketAgent:
    """
    Create agent for generating synthetic tool results.
    
    Supports parallel tool calls - generates results for ALL tool calls,
    including multiple calls to the same function with different arguments.
    
    Args:
        tool_calls: List of tool calls to generate results for
        schemas: JSON schemas for expected result structure
        user_query: Original user query for context
        orchestrator: Inference orchestrator instance
        model: LLM model to use
        llm_client: LLM client to use
        temperature: Sampling temperature
        max_tokens: Maximum tokens for response
        
    Returns:
        Configured MarketAgent for results generation
    """
    persona = Persona(
        role="API Response Simulator",
        persona=(
            "You are an expert at simulating realistic API responses. "
            "You generate plausible data that matches schemas and user context."
        ),
        objectives=["Generate realistic, contextually appropriate tool results"],
        skills=["Data generation", "API simulation", "Context understanding"]
    )
    
    output_tool = StructuredTool.from_pydantic(
        model=ToolMessages,
        name="tool_messages",
        description="Return simulated tool execution results"
    )
    
    # Build detailed tool call info including arguments for each call
    tool_call_details = []
    for i, tc in enumerate(tool_calls):
        func = tc.get('function', {})
        args = func.get('arguments', '{}')
        if isinstance(args, str):
            try:
                args = json.loads(args)
            except json.JSONDecodeError:
                args = {}
        
        tool_call_details.append({
            "index": i,
            "id": tc.get('id'),
            "name": func.get('name'),
            "arguments": args
        })
    
    calls_json = json.dumps(tool_call_details, indent=2)
    schemas_json = json.dumps(schemas, indent=2)
    
    task = f"""Generate realistic results for these tool calls based on the user's query.

User Query: {user_query}

Tool Calls (generate ONE result for EACH call, in order):
{calls_json}

Expected Result Schemas:
{schemas_json}

CRITICAL REQUIREMENTS:
1. Generate EXACTLY {len(tool_calls)} result messages - one for each tool call above
2. Each message MUST have role="tool" (exactly this string)
3. Each message MUST have the exact tool_call_id from the corresponding tool call
4. Each message MUST have the tool name in the "name" field
5. Results should be DIFFERENT for each call based on the arguments
6. Content should be realistic data matching the schema

For example, if there are 2 calls to the same function with different arguments,
you must return 2 separate result messages with different content.

You MUST call the tool_messages function with your response."""
    
    agent = MarketAgent(
        name="results-generator",
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
