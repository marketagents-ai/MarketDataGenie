"""Follow-up Query Agent - Creates follow-up user queries."""

import json
from typing import List, Dict, Any

from minference.lite.models import (
    LLMConfig, LLMClient, ResponseFormat, StructuredTool
)
from minference.lite.inference import InferenceOrchestrator

from market_agents.agents.base_agent.agent import Agent as MarketAgent
from market_agents.agents.personas.persona import Persona
from market_agents.agents.base_agent.prompter import PromptManager

from datagenie.marketagents_function_calling.schemas import FollowUpQuery


def create_followup_agent(
    messages: List[Dict[str, Any]],
    tools: List[Dict[str, Any]],
    orchestrator: InferenceOrchestrator,
    model: str = "gpt-4o",
    llm_client: LLMClient = LLMClient.openai,
    temperature: float = 0.6,
    max_tokens: int = 1024
) -> MarketAgent:
    """
    Create agent for generating follow-up user queries.
    
    Args:
        messages: Conversation history
        tools: Available tool definitions
        orchestrator: Inference orchestrator instance
        model: LLM model to use
        llm_client: LLM client to use
        temperature: Sampling temperature
        max_tokens: Maximum tokens for response
        
    Returns:
        Configured MarketAgent for follow-up generation
    """
    persona = Persona(
        role="Curious User",
        persona=(
            "You are a user who has received helpful information and wants to learn more. "
            "You ask relevant follow-up questions that build on the conversation."
        ),
        objectives=["Generate natural, contextually relevant follow-up questions"],
        skills=["Question formulation", "Context understanding", "Curiosity"]
    )
    
    output_tool = StructuredTool.from_pydantic(
        model=FollowUpQuery,
        name="followup_query",
        description="Return a follow-up user question"
    )
    
    history_json = json.dumps(messages, indent=2)
    tools_json = json.dumps([t.get('function', {}).get('name', t.get('name', '')) for t in tools], indent=2)
    task = f"""Based on this conversation, generate a natural follow-up question.
The question should be relevant to the available tools and build on the previous exchange.

Conversation History:
{history_json}

Available Tools: {tools_json}

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
            response_format=ResponseFormat.tool  # Force tool use
        ),
        llm_orchestrator=orchestrator,
        prompt_manager=PromptManager()
    )
    
    # Set forced_output for ResponseFormat.tool to work correctly
    agent.chat_thread.forced_output = output_tool
    
    return agent
