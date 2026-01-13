"""
Sub-LLM orchestration for RLM.

Handles parallel LLM calls via llm_batch() with optional tool access.
Sub-LLMs are fresh instances that don't share context with the main RLM.
"""

import asyncio
from typing import List, Dict, Any, Optional, Callable
from dataclasses import dataclass

from minference.lite.models import (
    LLMConfig, LLMClient, ResponseFormat,
    ChatThread, ChatMessage, MessageRole,
    SystemPrompt, StructuredTool
)
from minference.lite.inference import InferenceOrchestrator


@dataclass
class SubLLMResult:
    """Result from a sub-LLM call."""
    prompt: str
    response: str
    tool_calls: Optional[List[Dict[str, Any]]] = None
    success: bool = True
    error: Optional[str] = None
    tokens_used: int = 0


class SubLLMOrchestrator:
    """
    Orchestrates parallel sub-LLM calls for the RLM.
    
    Key features:
    - Parallel execution via llm_batch()
    - Tool routing (tools only available to sub-LLMs, not main RLM)
    - Fresh context for each sub-LLM (no context bleeding)
    """
    
    def __init__(
        self,
        model: str = "gpt-4o-mini",
        max_parallel: int = 8,
        timeout_seconds: int = 60,
        max_tokens: int = 4096,
        temperature: float = 0.7,
        llm_client: LLMClient = LLMClient.openai,
    ):
        self.model = model
        self.max_parallel = max_parallel
        self.timeout_seconds = timeout_seconds
        self.max_tokens = max_tokens
        self.temperature = temperature
        self.llm_client = llm_client
        
        self.orchestrator = InferenceOrchestrator()
        
        # Available tools for sub-LLMs
        self._tools: Dict[str, Callable] = {}
        self._tool_schemas: Dict[str, StructuredTool] = {}
    
    def register_tool(
        self,
        name: str,
        func: Callable,
        description: str,
        parameters: Dict[str, Any]
    ) -> None:
        """Register a tool that sub-LLMs can use."""
        self._tools[name] = func
        self._tool_schemas[name] = StructuredTool(
            name=name,
            description=description,
            json_schema=parameters
        )
    
    async def llm_batch(
        self,
        prompts: List[str],
        tools: Optional[List[str]] = None,
        system_prompt: Optional[str] = None,
    ) -> List[str]:
        """
        Execute multiple LLM calls in parallel.
        
        This is the main interface exposed to the RLM via the sandbox.
        
        Args:
            prompts: List of prompts to process
            tools: Optional list of tool names to make available
            system_prompt: Optional system prompt for all sub-LLMs
            
        Returns:
            List of responses (same order as prompts)
        """
        if not prompts:
            return []
        
        # Build tool list
        available_tools = []
        if tools:
            for tool_name in tools:
                if tool_name in self._tool_schemas:
                    available_tools.append(self._tool_schemas[tool_name])
        
        # Default system prompt for sub-LLMs
        if system_prompt is None:
            system_prompt = (
                "You are a helpful assistant. Provide concise, accurate responses. "
                "If you use tools, extract and summarize the key information."
            )
        
        # Create chat threads for each prompt
        chat_threads = []
        for i, prompt in enumerate(prompts):
            thread = ChatThread(
                name=f"sub-llm-{i}",
                system_prompt=SystemPrompt(name="sys", content=system_prompt),
                llm_config=LLMConfig(
                    client=self.llm_client,
                    model=self.model,
                    temperature=self.temperature,
                    max_tokens=self.max_tokens,
                    response_format=ResponseFormat.text
                ),
                tools=available_tools if available_tools else [],
                history=[],
                new_message=prompt
            )
            chat_threads.append(thread)
        
        # Process in batches to respect max_parallel
        results = []
        for i in range(0, len(chat_threads), self.max_parallel):
            batch = chat_threads[i:i + self.max_parallel]
            try:
                batch_outputs = await asyncio.wait_for(
                    self.orchestrator.run_parallel_ai_completion(batch),
                    timeout=self.timeout_seconds
                )
                
                for output in batch_outputs:
                    if output and output.content:
                        results.append(output.content)
                    else:
                        results.append("[No response from sub-LLM]")
                        
            except asyncio.TimeoutError:
                # Fill remaining with timeout errors
                for _ in range(len(batch)):
                    results.append(f"[Sub-LLM timeout after {self.timeout_seconds}s]")
            except Exception as e:
                for _ in range(len(batch)):
                    results.append(f"[Sub-LLM error: {str(e)}]")
        
        return results
    
    async def llm_single(
        self,
        prompt: str,
        tools: Optional[List[str]] = None,
        system_prompt: Optional[str] = None,
    ) -> str:
        """Execute a single sub-LLM call."""
        results = await self.llm_batch([prompt], tools, system_prompt)
        return results[0] if results else "[No response]"


# Default tool implementations for common use cases

async def search_tool(query: str) -> str:
    """
    Search the web using Serper API.
    
    This is a placeholder - implement with actual Serper/Google API.
    """
    # TODO: Implement actual search
    return f"[Search results for: {query}]\n1. Result 1\n2. Result 2\n3. Result 3"


async def open_url_tool(url: str) -> str:
    """
    Open and extract content from a URL.
    
    This is a placeholder - implement with actual web scraping.
    """
    # TODO: Implement actual URL fetching
    return f"[Content from {url}]\n[Truncated content...]"


def create_default_sub_llm_orchestrator(
    model: str = "gpt-4o-mini",
    include_web_tools: bool = True,
) -> SubLLMOrchestrator:
    """Create a SubLLMOrchestrator with default configuration."""
    orchestrator = SubLLMOrchestrator(model=model)
    
    if include_web_tools:
        # Register search tool
        orchestrator.register_tool(
            name="search",
            func=search_tool,
            description="Search the web for information. Returns a list of search results.",
            parameters={
                "type": "object",
                "properties": {
                    "query": {
                        "type": "string",
                        "description": "The search query"
                    }
                },
                "required": ["query"]
            }
        )
        
        # Register open tool
        orchestrator.register_tool(
            name="open",
            func=open_url_tool,
            description="Open a URL and extract its content.",
            parameters={
                "type": "object",
                "properties": {
                    "url": {
                        "type": "string",
                        "description": "The URL to open"
                    }
                },
                "required": ["url"]
            }
        )
    
    return orchestrator
