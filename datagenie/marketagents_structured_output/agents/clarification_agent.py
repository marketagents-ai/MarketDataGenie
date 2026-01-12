"""Clarification Agent - Generates user clarification when assistant asks for more info."""

import json
from typing import List, Dict, Any

from minference.lite.models import (
    LLMConfig, LLMClient, ResponseFormat, StructuredTool
)
from minference.lite.inference import InferenceOrchestrator

from market_agents.agents.base_agent.agent import Agent as MarketAgent
from market_agents.agents.personas.persona import Persona
from market_agents.agents.base_agent.prompter import PromptManager

from datagenie.marketagents_structured_output.schemas import ClarificationResponse


def create_clarification_agent(
    original_query: str,
    assistant_response: str,
    json_schema: Dict[str, Any],
    orchestrator: InferenceOrchestrator,
    model: str = "gpt-4o",
    llm_client: LLMClient = LLMClient.openai,
    temperature: float = 0.6,
    max_tokens: int = 1024
) -> MarketAgent:
    """
    Create agent for generating user clarification responses.
    
    When the assistant asks for clarification instead of generating JSON,
    this agent simulates the user providing the missing information.
    
    Args:
        original_query: The user's original query
        assistant_response: The assistant's clarification request
        json_schema: The JSON schema being used
        orchestrator: Inference orchestrator instance
        model: LLM model to use
        llm_client: LLM client to use
        temperature: Sampling temperature
        max_tokens: Maximum tokens for response
        
    Returns:
        Configured MarketAgent for clarification generation
    """
    persona = Persona(
        role="User Simulator",
        persona=(
            "You are simulating a user who needs to provide additional information "
            "when asked for clarification. You provide specific, concrete details "
            "that would allow the assistant to generate the requested JSON output."
        ),
        objectives=["Provide missing information to enable JSON generation"],
        skills=["Natural language", "Data provision", "Clarification responses"]
    )
    
    output_tool = StructuredTool.from_pydantic(
        model=ClarificationResponse,
        name="clarification_response",
        description="Return the user's clarification response"
    )
    
    schema_str = json.dumps(json_schema, indent=2)
    
    task = f"""The assistant asked for clarification. Generate a realistic user response that provides the missing information.

ORIGINAL USER QUERY:
{original_query}

ASSISTANT'S CLARIFICATION REQUEST:
{assistant_response}

TARGET JSON SCHEMA (the data we need to generate):
{schema_str}

Your task:
1. Identify what information the assistant is asking for
2. Generate realistic, specific values that would satisfy the schema
3. Respond naturally as a user would, providing the requested details

EXAMPLES of good clarification responses:
- "The email should be john.doe@example.com and the phone is +1-555-123-4567"
- "Use 'Premium' for the subscription tier and set the start date to January 15, 2024"
- "The product name is 'Widget Pro' with SKU WP-2024-001"

Make the response natural and conversational, providing concrete values.

You MUST call the clarification_response function with your response."""
    
    agent = MarketAgent(
        name="clarification-generator",
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
