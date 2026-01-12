"""Analysis Follow-up Agent - Generates non-structured follow-up Q&A."""

import json
from typing import List, Dict, Any

from minference.lite.models import (
    LLMConfig, LLMClient, ResponseFormat, StructuredTool
)
from minference.lite.inference import InferenceOrchestrator

from market_agents.agents.base_agent.agent import Agent as MarketAgent
from market_agents.agents.personas.persona import Persona
from market_agents.agents.base_agent.prompter import PromptManager

from datagenie.marketagents_structured_output.schemas import AnalysisFollowUp


def create_analysis_followup_agent(
    messages: List[Dict[str, Any]],
    orchestrator: InferenceOrchestrator,
    model: str = "gpt-4o",
    llm_client: LLMClient = LLMClient.openai,
    temperature: float = 0.6,
    max_tokens: int = 2048,
    generate_reasoning: bool = False
) -> MarketAgent:
    """
    Create agent for generating analysis follow-up Q&A.
    
    This generates a user question that analyzes the structured output
    (not requesting new structured data), along with the assistant's response.
    
    Args:
        messages: Conversation history with structured outputs
        orchestrator: Inference orchestrator instance
        model: LLM model to use
        llm_client: LLM client to use
        temperature: Sampling temperature
        max_tokens: Maximum tokens for response
        generate_reasoning: Whether to include <think> blocks
        
    Returns:
        Configured MarketAgent for analysis follow-up
    """
    persona = Persona(
        role="Conversation Analyst",
        persona=(
            "You generate realistic follow-up Q&A where the user asks about "
            "the structured data that was generated, and the assistant provides "
            "analysis or explanation without generating new structured output."
        ),
        objectives=["Generate analysis-focused follow-up conversations"],
        skills=["Data analysis", "Explanation", "Q&A generation"]
    )
    
    output_tool = StructuredTool.from_pydantic(
        model=AnalysisFollowUp,
        name="analysis_followup",
        description="Return the analysis follow-up Q&A"
    )
    
    # Format conversation history
    history_str = ""
    for msg in messages:
        role = msg.get("role", "unknown")
        content = msg.get("content", "")
        if len(content) > 800:
            content = content[:800] + "..."
        history_str += f"\n[{role.upper()}]: {content}\n"
    
    reasoning_instruction = ""
    if generate_reasoning:
        reasoning_instruction = """
IMPORTANT: The assistant's response MUST start with a <think> block containing reasoning:
<think>
[Your reasoning about how to answer the question using the existing data]
</think>

[Then provide the actual response]
"""
    
    task = f"""Generate a follow-up Q&A where the user asks an ANALYSIS question about the structured output.

Conversation so far:
{history_str}

Requirements:
1. The user question should ask about the DATA that was generated (not request new data)
2. The question should require REASONING over the existing output
3. The assistant response should ANALYZE or EXPLAIN (not generate new structured output)
{reasoning_instruction}
EXAMPLE ANALYSIS QUESTIONS:
- "Why did you choose that format for the date?"
- "What's the significance of the required fields?"
- "How does this schema handle edge cases?"
- "Can you explain the relationship between these fields?"
- "What validation would this data pass/fail?"

You MUST call the analysis_followup function with your response."""
    
    agent = MarketAgent(
        name="analysis-followup",
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
