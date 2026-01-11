"""Query Generator Agent - Creates user queries for tools."""

import json
from typing import List, Dict, Any

from minference.lite.models import (
    LLMConfig, LLMClient, ResponseFormat, StructuredTool
)
from minference.lite.inference import InferenceOrchestrator

from market_agents.agents.base_agent.agent import Agent as MarketAgent
from market_agents.agents.personas.persona import Persona
from market_agents.agents.base_agent.prompter import PromptManager

from datagenie.marketagents_function_calling.schemas import GeneratedQuery


def create_query_generator_agent(
    tools: List[Dict[str, Any]],
    task_description: str,
    orchestrator: InferenceOrchestrator,
    model: str = "gpt-4o",
    llm_client: LLMClient = LLMClient.openai,
    temperature: float = 0.6,
    max_tokens: int = 1024
) -> MarketAgent:
    """
    Create agent for generating user queries that would invoke the tools.
    
    Args:
        tools: List of tool definitions in OpenAI format
        task_description: Original task description for context
        orchestrator: Inference orchestrator instance
        model: LLM model to use
        llm_client: LLM client to use
        temperature: Sampling temperature
        max_tokens: Maximum tokens for response
        
    Returns:
        Configured MarketAgent for query generation
    """
    persona = Persona(
        role="User Simulator",
        persona=(
            "You are simulating a user who needs help with a task. "
            "You ask natural, realistic questions that would require using specific tools."
        ),
        objectives=["Generate natural user queries that would invoke the available tools"],
        skills=["Natural language", "User behavior simulation", "Query formulation"]
    )
    
    output_tool = StructuredTool.from_pydantic(
        model=GeneratedQuery,
        name="generated_query",
        description="Return the generated user query"
    )
    
    # Include full tool schemas so the query can include all required params
    tools_full = json.dumps(tools, indent=2)
    
    # Extract required parameters from tools for emphasis
    required_params_info = []
    for tool in tools:
        func = tool.get('function', tool)
        params = func.get('parameters', {})
        required = params.get('required', [])
        props = params.get('properties', {})
        if required:
            param_details = [f"  - {p}: {props.get(p, {}).get('description', props.get(p, {}).get('type', 'any'))}" for p in required]
            required_params_info.append(f"Tool '{func.get('name')}' REQUIRES:\n" + "\n".join(param_details))
    
    required_params_str = "\n\n".join(required_params_info) if required_params_info else "Check tool schemas for required parameters."
    
    task = f"""Generate a natural user query that would require using one or more of these tools.

Task context: {task_description}

Available tools (with full parameter schemas):
{tools_full}

REQUIRED PARAMETERS TO INCLUDE:
{required_params_str}

CRITICAL REQUIREMENTS:
1. Your query MUST include specific values for ALL required parameters listed above
2. Use realistic fake data (addresses, names, IDs, quantities, dates, etc.)
3. The query should be complete enough that an assistant can IMMEDIATELY call the tool without asking follow-up questions
4. Write naturally as a real user would, but include all necessary details

EXAMPLES:
BAD: "Order me some food" (missing restaurant, items, address, payment)
BAD: "Book a flight" (missing origin, destination, dates, passengers)
BAD: "Send a message" (missing recipient, content)

GOOD: "Order 2 pepperoni pizzas and a Caesar salad from Mario's Pizza (ID: marios_sf_01), deliver to 123 Main St Apt 4B, San Francisco CA 94102. Use my saved card ending in 4242."
GOOD: "Book a round-trip flight from SFO to JFK for 2 adults, departing March 15 2024 and returning March 22 2024. Economy class, window seats preferred."
GOOD: "Send a text message to John Smith at +1-555-123-4567 saying 'Running 10 minutes late for our 3pm meeting'"

Create a realistic, natural-sounding request with ALL necessary details included.

You MUST call the generated_query function with your response."""
    
    agent = MarketAgent(
        name="query-generator",
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
