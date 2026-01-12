"""Query Generator Agent - Creates user queries with actual data for structured output."""

import json
from typing import Dict, Any, List, Optional

from minference.lite.models import (
    LLMConfig, LLMClient, ResponseFormat, StructuredTool
)
from minference.lite.inference import InferenceOrchestrator

from market_agents.agents.base_agent.agent import Agent as MarketAgent
from market_agents.agents.personas.persona import Persona
from market_agents.agents.base_agent.prompter import PromptManager

from datagenie.marketagents_structured_output.schemas import GeneratedQuery


def create_query_generator_agent(
    json_schema: Dict[str, Any],
    task_description: str,
    orchestrator: InferenceOrchestrator,
    model: str = "gpt-4o",
    llm_client: LLMClient = LLMClient.openai,
    temperature: float = 0.7,
    max_tokens: int = 2048
) -> MarketAgent:
    """
    Create agent for generating user queries with actual data for structured output.
    
    The agent generates realistic queries that include actual data values in various
    formats (plain text, CSV, markdown tables, lists, etc.) that need to be converted
    to structured JSON output.
    
    Args:
        json_schema: The JSON schema the output should conform to
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
        role="Data Entry Specialist",
        persona=(
            "You are simulating a user who has raw data in various formats and needs "
            "it converted to structured JSON. You provide actual data values in your "
            "requests, using formats like plain text descriptions, CSV, markdown tables, "
            "bullet lists, or mixed formats that real users would use."
        ),
        objectives=[
            "Generate realistic user queries with actual data values",
            "Use diverse data formats (text, CSV, tables, lists)",
            "Create data that matches the target schema structure"
        ],
        skills=["Data formatting", "Natural language", "User simulation"]
    )
    
    output_tool = StructuredTool.from_pydantic(
        model=GeneratedQuery,
        name="generated_query",
        description="Return the generated user query with actual data"
    )
    
    schema_str = json.dumps(json_schema, indent=2)
    
    # Extract key fields from schema for guidance
    properties = json_schema.get("properties", {})
    required = json_schema.get("required", [])
    field_info = []
    for prop, details in properties.items():
        prop_type = details.get("type", "any")
        prop_desc = details.get("description", "")
        prop_format = details.get("format", "")
        req_marker = " (required)" if prop in required else ""
        format_hint = f" [format: {prop_format}]" if prop_format else ""
        field_info.append(f"  - {prop}: {prop_type}{req_marker}{format_hint} - {prop_desc}")
    
    fields_str = "\n".join(field_info) if field_info else "See schema for fields"
    
    task = f"""Generate a realistic user query that includes ACTUAL DATA VALUES to be converted to structured JSON.

Task context: {task_description}

Target JSON Schema:
{schema_str}

Fields to populate:
{fields_str}

CRITICAL REQUIREMENTS:
1. Include ACTUAL DATA VALUES in your query - not placeholders or vague descriptions
2. Generate realistic, diverse data that matches the schema field types
3. Use one of these data formats (vary your choice):

FORMAT OPTIONS:
a) Plain text with embedded data:
   "Convert this to JSON: John Smith, born March 15 1990, email john.smith@email.com, 
   works as Senior Developer at TechCorp since 2018"

b) CSV format:
   "Parse this CSV into JSON:
   name,age,email,city
   Sarah Connor,35,sarah@future.com,Los Angeles"

c) Markdown table:
   "Convert this table to JSON:
   | Field | Value |
   |-------|-------|
   | title | Introduction to Machine Learning |
   | author | Dr. Jane Wilson |
   | pages | 342 |"

d) Bullet list:
   "Structure this data as JSON:
   - Product: Wireless Headphones XR-500
   - Price: $149.99
   - Category: Electronics
   - In Stock: Yes
   - Rating: 4.5/5"

e) Natural paragraph with data:
   "I need to store this blog post: The title is 'Understanding Neural Networks', 
   written by Alex Chen on January 15, 2024. The content discusses deep learning 
   fundamentals. Tags should include AI, machine learning, and tutorial."

f) Mixed format:
   "Create a JSON record from this order info:
   Customer: Maria Garcia (maria.g@shop.com)
   Items ordered:
   - Blue T-Shirt (M) x2 @ $25.00
   - Running Shoes (Size 8) x1 @ $89.99
   Shipping: 123 Oak Street, Austin, TX 78701"

GUIDELINES:
- Generate REALISTIC data values (real-looking names, valid emails, plausible dates, etc.)
- Match data types to schema (strings, numbers, booleans, arrays, nested objects)
- For date/time fields, use appropriate formats (ISO 8601, common date formats)
- For email fields, use realistic email patterns
- For arrays, include 2-5 items typically
- For nested objects, include all required nested fields
- Vary the complexity and format across different queries

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
            response_format=ResponseFormat.tool
        ),
        llm_orchestrator=orchestrator,
        prompt_manager=PromptManager()
    )
    
    agent.chat_thread.forced_output = output_tool
    
    return agent
