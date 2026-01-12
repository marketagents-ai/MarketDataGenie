"""Pydantic output schemas for structured output pipeline agents."""

from typing import List, Dict, Any, Optional
from pydantic import BaseModel, Field


class GeneratedSchema(BaseModel):
    """Generated JSON schema from task description."""
    name: str = Field(..., description="Schema name (snake_case)")
    description: str = Field(..., description="What the schema represents")
    json_schema: Dict[str, Any] = Field(..., description="The JSON Schema definition")


class GeneratedQuery(BaseModel):
    """Generated user query with actual data for structured output."""
    query: str = Field(..., description="Natural language user query containing actual data values")
    data_format: str = Field(
        default="plain_text",
        description="Format of data in query: plain_text, csv, markdown_table, bullet_list, mixed"
    )
    expected_fields: List[str] = Field(
        default_factory=list, 
        description="Expected fields in the output"
    )


class StructuredResponse(BaseModel):
    """Structured output response."""
    output: Dict[str, Any] = Field(..., description="The structured JSON output")
    reasoning: Optional[str] = Field(
        default=None, 
        description="Reasoning for the output (if generate_reasoning=True)"
    )


class FollowUpQuery(BaseModel):
    """Follow-up user query for multi-turn."""
    role: str = Field(default="user")
    content: str = Field(..., description="Follow-up question")
    modification_type: str = Field(
        default="update",
        description="Type of modification: update, add, remove, clarify"
    )


class AnalysisFollowUp(BaseModel):
    """Follow-up Q&A that analyzes existing structured output."""
    followup_question: str = Field(
        ..., 
        description="User's follow-up question about the structured output"
    )
    response: str = Field(
        ..., 
        description="Assistant's response analyzing the output"
    )


class ClarificationResponse(BaseModel):
    """User's response providing clarification/missing information."""
    content: str = Field(
        ..., 
        description="The user's clarification response with specific values"
    )
    provided_fields: List[str] = Field(
        default_factory=list,
        description="List of fields/values being provided in the clarification"
    )
