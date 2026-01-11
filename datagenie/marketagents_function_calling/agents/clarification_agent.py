"""Clarification Agent - Provides missing details when assistant asks for clarification."""

import json
from typing import List, Dict, Any

from minference.lite.models import (
    LLMConfig, LLMClient, ResponseFormat, StructuredTool
)
from minference.lite.inference import InferenceOrchestrator

from market_agents.agents.base_agent.agent import Agent as MarketAgent
from market_agents.agents.personas.persona import Persona
from market_agents.agents.base_agent.prompter import PromptManager

from datagenie.marketagents_function_calling.schemas import ClarificationResponse


def create_clarification_agent(
    original_query: str,
    assistant_response: str,
    tools: List[Dict[str, Any]],
    orchestrator: InferenceOrchestrator,
    model: str = "gpt-4o",
    llm_client: LLMClient = LLMClient.openai,
    temperature: float = 0.6,
    max_tokens: int = 1024
) -> MarketAgent:
    """
    Create agent for providing clarification details when assistant asks for them.
    
    This agent simulates a user providing the missing information that the assistant
    requested, ensuring the conversation can proceed to a tool call.
    
    Args:
        original_query: The user's original (incomplete) query
        assistant_response: The assistant's clarification request
        tools: Available tool definitions (to understand what params are needed)
        orchestrator: Inference orchestrator instance
        model: LLM model to use
        llm_client: LLM client to use
        temperature: Sampling temperature
        max_tokens: Maximum tokens for response
        
    Returns:
        Configured MarketAgent for clarification generation
    """
    persona = Persona(
        role="Helpful User",
        persona=(
            "You are a user who asked for help and the assistant needs more details. "
            "You provide the specific information requested in a natural, conversational way."
        ),
        objectives=["Provide all the missing details the assistant asked for"],
        skills=["Clear communication", "Providing specific details", "Natural conversation"]
    )
    
    output_tool = StructuredTool.from_pydantic(
        model=ClarificationResponse,
        name="clarification_response",
        description="Return the user's clarification with all requested details"
    )
    
    # Include full tool schemas so the agent knows what parameters are needed
    tools_json = json.dumps(tools, indent=2)
    
    task = f"""The assistant asked for more details. Generate a natural user response that provides ALL the missing information.

ORIGINAL USER QUERY:
{original_query}

ASSISTANT'S CLARIFICATION REQUEST:
{assistant_response}

AVAILABLE TOOLS (showing what parameters are needed):
{tools_json}

YOUR TASK:
Generate a user response that provides ALL the specific details the assistant asked for.

REQUIREMENTS:
- Include realistic, plausible values for ALL requested information
- Use fake but realistic data (addresses, names, IDs, quantities, etc.)
- The response should be natural and conversational (not a list of parameters)
- After this response, the assistant should have everything needed to call the tools
- Do NOT ask questions back - just provide the information

EXAMPLE:
If assistant asked for: "I need your delivery address, restaurant choice, and items to order"
Good response: "Sure! I'd like to order from Bella Italia (restaurant ID: bella_123). Please get me 2 margherita pizzas and 1 tiramisu. Deliver to 456 Oak Street, Apt 12B, San Francisco, CA 94102. Use my saved card ending in 4242."

You MUST call the clarification_response function with your response."""
    
    agent = MarketAgent(
        name="clarification-agent",
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
    
    # Set forced_output for ResponseFormat.tool to work correctly
    agent.chat_thread.forced_output = output_tool
    
    return agent
