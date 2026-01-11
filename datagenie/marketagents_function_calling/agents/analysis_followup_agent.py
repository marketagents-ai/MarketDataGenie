"""Analysis Follow-up Agent - Creates follow-up Q&A that analyzes existing tool results."""

import json
from typing import List, Dict, Any

from minference.lite.models import (
    LLMConfig, LLMClient, ResponseFormat, StructuredTool
)
from minference.lite.inference import InferenceOrchestrator

from market_agents.agents.base_agent.agent import Agent as MarketAgent
from market_agents.agents.personas.persona import Persona
from market_agents.agents.base_agent.prompter import PromptManager

from datagenie.marketagents_function_calling.schemas import AnalysisFollowUp


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
    
    This generates a follow-up question that requires ANALYSIS of existing tool results
    (not new tool calls), along with the assistant's response.
    
    The follow-up should be answerable using ONLY information already available
    from previous tool results - no new data fetching required.
    
    Args:
        messages: Conversation history with tool calls and results
        orchestrator: Inference orchestrator instance
        model: LLM model to use
        llm_client: LLM client to use
        temperature: Sampling temperature
        max_tokens: Maximum tokens for response
        generate_reasoning: If True, response must include <think></think> tags
        
    Returns:
        Configured MarketAgent for analysis follow-up generation
    """
    persona = Persona(
        role="Conversation Extender",
        persona=(
            "You are an expert at extending tool-calling conversations with meaningful follow-up turns. "
            "You generate realistic user follow-up questions that require analysis of existing data, "
            "along with helpful assistant responses that use the available context."
        ),
        objectives=[
            "Generate natural follow-up questions that analyze existing tool results",
            "Create assistant responses that synthesize and interpret available data"
        ],
        skills=["Question formulation", "Data analysis", "Context synthesis", "Conversation flow"]
    )
    
    output_tool = StructuredTool.from_pydantic(
        model=AnalysisFollowUp,
        name="analysis_followup",
        description="Return a follow-up question and assistant response that analyzes existing tool results"
    )
    
    # Extract conversation components for the prompt
    original_question = ""
    tool_summary = []
    previous_response = ""
    
    for msg in messages:
        role = msg.get('role', '')
        content = msg.get('content', '')
        
        if role == 'user' and not original_question:
            original_question = content
        elif role == 'assistant':
            if msg.get('tool_calls'):
                for tc in msg['tool_calls']:
                    func = tc.get('function', {})
                    tool_summary.append(f"- {func.get('name', 'unknown')}({func.get('arguments', '{}')})")
            if content and not msg.get('tool_calls'):
                previous_response = content
        elif role == 'tool':
            tool_name = msg.get('name', 'unknown')
            tool_content = msg.get('content', '{}')
            # Truncate long results
            if len(tool_content) > 500:
                tool_content = tool_content[:500] + "..."
            tool_summary.append(f"  Result from {tool_name}: {tool_content}")
    
    tool_summary_str = "\n".join(tool_summary) if tool_summary else "No tool calls recorded"
    
    # Add reasoning instructions if required
    reasoning_instruction = ""
    if generate_reasoning:
        reasoning_instruction = """
REASONING REQUIREMENT:
The assistant's response MUST start with a <think></think> block containing the reasoning process.
Format the response field as:
<think>
[Your reasoning about how to answer the follow-up using existing context]
</think>

[Your actual response to the user]

This is REQUIRED - responses without <think> tags will be rejected.
"""
    
    task = f"""Generate a follow-up conversation turn that extends this tool-calling interaction.

CONVERSATION SO FAR:
- User's original question: {original_question}
- Tools called and results:
{tool_summary_str}
- Assistant's response: {previous_response}

YOUR TASK:
Generate a realistic follow-up question from the user AND the assistant's response to it.
{reasoning_instruction}
CRITICAL CONSTRAINT: The follow-up MUST be answerable using ONLY the information already available
from the previous tool results. Do NOT generate follow-ups that would require calling additional tools
or fetching new data. The assistant should reason over existing context, not make new tool calls.

Good follow-up types (answerable from existing context):
- "Why is X higher/lower than Y?" - comparing values from tool results
- "What does that mean for...?" - asking for interpretation of existing data
- "Can you explain why...?" - asking for clarification about the response
- "How does X relate to Y?" - connecting information already retrieved
- "Based on that, would you recommend...?" - asking for advice using existing info

Bad follow-up types (would require new tool calls - AVOID THESE):
- "What about [different item/entity]?" - needs new data lookup
- "Can you also check...?" - requests additional tool calls
- "What's the current [X] of...?" - needs fresh data retrieval

IMPORTANT: The follow-up should require the assistant to USE INFORMATION FROM THE PREVIOUS TURN.
This creates a multi-hop reasoning chain where context must be tracked across turns.

You MUST call the analysis_followup function with your response."""
    
    agent = MarketAgent(
        name="analysis-followup-generator",
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
