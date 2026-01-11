"""Pydantic output schemas for agent responses."""

from typing import List, Dict, Any
from pydantic import BaseModel, Field


# ============================================================
# Tool Generation Schemas (Curriculum Mode)
# ============================================================

class GeneratedTool(BaseModel):
    """Generated tool definition."""
    name: str = Field(..., description="Function name (snake_case)")
    description: str = Field(..., description="What the function does")
    parameters: Dict[str, Any] = Field(..., description="JSON Schema for parameters")


class GeneratedTools(BaseModel):
    """Collection of generated tools."""
    tools: List[GeneratedTool] = Field(default_factory=list)


class GeneratedQuery(BaseModel):
    """Generated user query for tools."""
    query: str = Field(..., description="Natural language user query")
    expected_tool_calls: List[str] = Field(
        default_factory=list, 
        description="Expected tool names to be called"
    )


# ============================================================
# Docstring Schemas
# ============================================================

class ToolDocString(BaseModel):
    """Single tool docstring."""
    name: str = Field(..., description="Tool function name")
    doc_string: str = Field(..., description="Generated documentation string")


class ToolDocStrings(BaseModel):
    """Collection of tool docstrings."""
    doc_strings: List[ToolDocString] = Field(default_factory=list)


# ============================================================
# Schema Generation Schemas
# ============================================================

class ContentSchema(BaseModel):
    """JSON schema for tool result content."""
    name: str = Field(..., description="Tool function name")
    json_schema: Dict[str, Any] = Field(..., description="JSON schema for result", alias="schema")


class ContentSchemas(BaseModel):
    """Collection of content schemas."""
    content_schemas: List[ContentSchema] = Field(default_factory=list)


# ============================================================
# Results Generation Schemas
# ============================================================

class ToolResultMessage(BaseModel):
    """Tool result message in OpenAI format."""
    role: str = Field(default="tool")
    name: str = Field(..., description="Tool function name")
    tool_call_id: str = Field(..., description="ID of the tool call")
    content: Any = Field(..., description="Tool execution result")


class ToolMessages(BaseModel):
    """Collection of tool result messages."""
    messages: List[ToolResultMessage] = Field(default_factory=list)


# ============================================================
# Follow-up Query Schema
# ============================================================

class FollowUpQuery(BaseModel):
    """Follow-up user query."""
    role: str = Field(default="user")
    content: str = Field(..., description="Follow-up question")


# ============================================================
# Clarification Response Schema
# ============================================================

class ClarificationResponse(BaseModel):
    """User's response providing clarification details."""
    content: str = Field(..., description="User's response with all requested details")
    details_provided: List[str] = Field(
        default_factory=list,
        description="List of specific details provided (for validation)"
    )


# ============================================================
# Analysis Follow-up Schema
# ============================================================

class AnalysisFollowUp(BaseModel):
    """
    Follow-up Q&A that analyzes existing tool results.
    
    This is for generating a user follow-up question that requires
    ANALYSIS of previous tool results (not new tool calls), along
    with the assistant's response.
    """
    followup_question: str = Field(
        ..., 
        description="User's follow-up question that analyzes existing tool results"
    )
    response: str = Field(
        ..., 
        description="Assistant's response answering the follow-up using existing context"
    )
