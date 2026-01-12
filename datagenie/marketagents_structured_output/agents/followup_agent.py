"""Follow-up Agent - Generates follow-up queries for multi-turn structured output."""

import json
from typing import List, Dict, Any

from minference.lite.models import (
    LLMConfig, LLMClient, ResponseFormat, StructuredTool
)
from minference.lite.inference import InferenceOrchestrator

from market_agents.agents.base_agent.agent import Agent as MarketAgent
from market_agents.agents.personas.persona import Persona
from market_agents.agents.base_agent.prompter import PromptManager

from datagenie.marketagents_structured_output.schemas import FollowUpQuery


def create_followup_agent(
    messages: List[Dict[str, Any]],
    json_schema: Dict[str, Any],
    orchestrator: InferenceOrchestrator,
    model: str = "gpt-4o",
    llm_client: LLMClient = LLMClient.openai,
    temperature: float = 0.6,
    max_tokens: int = 1024
) -> MarketAgent:
    """
    Create agent for generating follow-up queries that modify structured output.
    
    Args:
        messages: Conversation history
        json_schema: The JSON schema being used
        orchestrator: Inference orchestrator instance
        model: LLM model to use
        llm_client: LLM client to use
        temperature: Sampling temperature
        max_tokens: Maximum tokens for response
        
    Returns:
        Configured MarketAgent for follow-up generation
    """
    persona = Persona(
        role="User Simulator",
        persona=(
            "You are simulating a user who wants to modify or update previously "
            "generated structured data. You ask natural follow-up questions."
        ),
        objectives=["Generate natural follow-up queries to modify structured output"],
        skills=["Natural language", "Data modification requests", "Iterative refinement"]
    )
    
    output_tool = StructuredTool.from_pydantic(
        model=FollowUpQuery,
        name="followup_query",
        description="Return the follow-up query"
    )
    
    # Format conversation history
    history_str = ""
    for msg in messages:
        role = msg.get("role", "unknown")
        content = msg.get("content", "")
        if len(content) > 500:
            content = content[:500] + "..."
        history_str += f"\n[{role.upper()}]: {content}\n"
    
    schema_str = json.dumps(json_schema, indent=2)
    
    task = f"""Generate a natural follow-up query to modify the previously generated structured output.

Conversation so far:
{history_str}

Target Schema:
{schema_str}

Generate a follow-up that does ONE of:
1. UPDATE: Change specific field values
2. ADD: Add new optional fields or array items
3. REMOVE: Remove optional fields or array items
4. CLARIFY: Ask for clarification or more details

EXAMPLES:
- "Change the email to jane@company.com"
- "Add a phone number: +1-555-123-4567"
- "Remove the middle name field"
- "Update the address to 456 Oak Ave, Suite 100"

Make it natural and realistic.

You MUST call the followup_query function with your response."""
    
    agent = MarketAgent(
        name="followup-generator",
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
