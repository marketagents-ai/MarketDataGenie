"""ShareGPT format conversion utilities with XML tags for tool calls."""

import json
from typing import List, Dict, Any, Optional


def _format_tools_xml(tools: List[Dict[str, Any]]) -> str:
    """Format tools as XML for system prompt."""
    tools_json = json.dumps(tools, indent=2)
    return f"<tools>\n{tools_json}\n</tools>"


def _format_tool_call_xml(tool_call: Dict[str, Any]) -> str:
    """Format a single tool call with XML tags."""
    func = tool_call.get('function', {})
    name = func.get('name', '')
    arguments = func.get('arguments', '{}')
    
    # Parse arguments if string
    if isinstance(arguments, str):
        try:
            args_dict = json.loads(arguments)
        except json.JSONDecodeError:
            args_dict = {}
    else:
        args_dict = arguments
    
    tool_call_obj = {"name": name, "arguments": args_dict}
    return f"<tool_call>\n{json.dumps(tool_call_obj)}\n</tool_call>"


def _format_tool_response_xml(name: str, content: Any) -> str:
    """Format a tool response with XML tags."""
    if isinstance(content, str):
        try:
            content_obj = json.loads(content)
        except json.JSONDecodeError:
            content_obj = content
    else:
        content_obj = content
    
    response_obj = {"name": name, "content": content_obj}
    return f"<tool_response>\n{json.dumps(response_obj)}\n</tool_response>"


def to_sharegpt_format(
    messages: List[Dict[str, Any]],
    tools: List[Dict[str, Any]],
    metadata: Optional[Dict[str, Any]] = None,
    include_reasoning_instructions: bool = False
) -> Dict[str, Any]:
    """
    Convert messages to ShareGPT format with XML tags for tool calls.
    
    Format follows the Hermes function calling format:
    - System prompt includes tools in <tools></tools> XML tags
    - Assistant tool calls use <tool_call></tool_call> XML tags
    - Tool responses use <tool_response></tool_response> XML tags
    - When reasoning is enabled, assistant should use <think></think> tags
    
    Output structure for HuggingFace:
    - id: Sample identifier (top-level for HF column)
    - source: Source dataset name (top-level for HF column)
    - task: Task description (top-level for HF column)
    - category: Category (top-level for HF column)
    - subcategory: Subcategory (top-level for HF column)
    - tools: Tool definitions as JSON string (top-level for HF column)
    - conversations: ShareGPT conversation format
    
    Args:
        messages: Conversation in OpenAI format
        tools: Tool definitions
        metadata: Optional metadata (should contain 'id', 'source', 'task', 'category', 'subcategory')
        include_reasoning_instructions: If True, use deep thinking system prompt (Hermes format)
        
    Returns:
        ShareGPT formatted conversation with top-level fields for HuggingFace
        Returns None if conversation is invalid (e.g., consecutive same-role messages)
    """
    conversations = []
    
    # Build system prompt with tools
    tools_xml = _format_tools_xml(tools)
    
    if include_reasoning_instructions:
        # Deep thinking system prompt (matches Hermes chat template)
        # This is the format used when generate_reasoning=True
        system_prompt = (
            "You are a deep thinking AI, you may use extremely long chains of thought to deeply "
            "consider the problem and deliberate with yourself via systematic reasoning processes "
            "to help come to a correct solution prior to answering. You should enclose your thoughts "
            "and internal monologue inside <think> </think> tags, and then provide your solution "
            "or response to the problem.\n\n"
            "You are a function calling AI model. You may call one or more functions to assist "
            "with the user query. Don't make assumptions about what values to plug into functions.\n\n"
            f"{tools_xml}\n"
            "For each function call return a json object with function name and arguments "
            "within <tool_call> </tool_call> tags with the following schema:\n"
            "<tool_call>\n"
            "{\"name\": <function-name>, \"arguments\": <args-dict>}\n"
            "</tool_call>"
        )
    else:
        # Standard function calling system prompt (no reasoning)
        system_prompt = (
            "You are a function calling AI model. You are provided with function signatures "
            "within <tools> </tools> XML tags. You may call one or more functions to assist "
            "with the user query. Don't make assumptions about what values to plug into functions.\n"
            f"{tools_xml}\n"
            "For each function call return a json object with function name and arguments "
            "within <tool_call> </tool_call> tags with the following schema:\n"
            "<tool_call>\n"
            "{\"name\": <function-name>, \"arguments\": <args-dict>}\n"
            "</tool_call>"
        )
    
    conversations.append({"from": "system", "value": system_prompt})
    
    # Track pending tool responses to group them
    pending_tool_responses = []
    
    # Track last role to detect consecutive same-role messages
    last_role = "system"
    
    # Extract first user query for task (fallback)
    first_user_query = ""
    for msg in messages:
        if msg.get('role') == 'user':
            first_user_query = msg.get('content', '')
            break
    
    for msg in messages:
        role = msg.get('role')
        content = msg.get('content', '')
        
        if role == 'system':
            # Already handled above
            continue
        
        elif role == 'user':
            # Flush any pending tool responses before user message
            if pending_tool_responses:
                tool_responses_str = "\n".join(pending_tool_responses)
                conversations.append({"from": "tool", "value": tool_responses_str})
                pending_tool_responses = []
                last_role = "tool"
            
            # Check for consecutive user messages - skip duplicates or merge
            if last_role == "human":
                # Consecutive user messages - merge them
                if conversations and conversations[-1]["from"] == "human":
                    conversations[-1]["value"] += "\n\n" + content
                    continue
            
            conversations.append({"from": "human", "value": content})
            last_role = "human"
        
        elif role == 'assistant':
            # Flush any pending tool responses before assistant message
            if pending_tool_responses:
                tool_responses_str = "\n".join(pending_tool_responses)
                conversations.append({"from": "tool", "value": tool_responses_str})
                pending_tool_responses = []
                last_role = "tool"
            
            # Format tool calls with XML tags
            if msg.get('tool_calls'):
                # Check if content already contains <tool_call> tags (reasoning mode)
                # If so, don't add them again - just use the content as-is
                content_has_tool_calls = content and '<tool_call>' in content.lower()
                
                if content_has_tool_calls:
                    # Content already has tool calls embedded (reasoning mode)
                    # Just use the content directly
                    value = content
                else:
                    # Native tool calling mode - append formatted tool calls
                    tool_calls_xml = []
                    for tc in msg['tool_calls']:
                        tool_calls_xml.append(_format_tool_call_xml(tc))
                    
                    tool_calls_str = "\n".join(tool_calls_xml)
                    
                    if content:
                        value = f"{content}\n{tool_calls_str}"
                    else:
                        value = tool_calls_str
            else:
                value = content
            
            conversations.append({"from": "gpt", "value": value})
            last_role = "gpt"
        
        elif role == 'tool':
            # Accumulate tool responses to group them together
            tool_name = msg.get('name', 'unknown')
            tool_content = msg.get('content', '')
            pending_tool_responses.append(_format_tool_response_xml(tool_name, tool_content))
            # Note: last_role will be updated when tool responses are flushed
    
    # Flush any remaining tool responses
    if pending_tool_responses:
        tool_responses_str = "\n".join(pending_tool_responses)
        conversations.append({"from": "tool", "value": tool_responses_str})
    
    # Validate conversation - check for consecutive same-role messages (except tool)
    prev_role = None
    for conv in conversations:
        curr_role = conv["from"]
        if curr_role == prev_role and curr_role in ("human", "gpt"):
            print(f"Warning: Consecutive {curr_role} messages detected in conversation")
            # Don't return None - we've already merged consecutive human messages above
        prev_role = curr_role
    
    # Final validation: ensure conversation doesn't end with a human message
    if conversations and conversations[-1]["from"] == "human":
        print(f"Warning: Conversation ends with human message - removing it")
        conversations.pop()
    
    # Ensure we have at least one exchange
    has_human = any(c["from"] == "human" for c in conversations)
    has_gpt = any(c["from"] == "gpt" for c in conversations)
    if not (has_human and has_gpt):
        print(f"Warning: Incomplete conversation - missing human or gpt turn")
        return None
    
    # Build result with top-level fields for HuggingFace columns
    # Order: id, conversations, tools, task, category, subcategory, source
    result = {
        "id": metadata.get("id", "") if metadata else "",
        "conversations": conversations,
        "tools": json.dumps(tools),  # JSON string for HF column
        "task": metadata.get("task", first_user_query) if metadata else first_user_query,
        "category": metadata.get("category", "") if metadata else "",
        "subcategory": metadata.get("subcategory", "") if metadata else "",
        "source": metadata.get("source", "") if metadata else ""
    }
    
    return result


def from_sharegpt_format(
    sharegpt_data: Dict[str, Any]
) -> tuple[List[Dict[str, Any]], List[Dict[str, Any]]]:
    """
    Convert ShareGPT format with XML tags back to OpenAI message format.
    
    Args:
        sharegpt_data: ShareGPT formatted conversation
        
    Returns:
        Tuple of (messages, tools)
    """
    import re
    
    conversations = sharegpt_data.get('conversations', [])
    tools = []
    messages = []
    
    for conv in conversations:
        from_role = conv.get('from', '')
        value = conv.get('value', '')
        
        if from_role == 'system':
            # Extract tools from <tools></tools> tags
            tools_match = re.search(r'<tools>\s*(.*?)\s*</tools>', value, re.DOTALL)
            if tools_match:
                try:
                    tools = json.loads(tools_match.group(1))
                except json.JSONDecodeError:
                    tools = []
            
            # Extract the base system content (before tools instruction)
            base_content = re.sub(r'You are a function calling AI model.*', '', value).strip()
            if not base_content:
                base_content = "You are a helpful assistant with access to tools."
            
            messages.append({'role': 'system', 'content': base_content})
        
        elif from_role == 'human':
            messages.append({'role': 'user', 'content': value})
        
        elif from_role == 'gpt':
            # Extract tool calls from <tool_call></tool_call> tags
            tool_call_matches = re.findall(r'<tool_call>\s*(.*?)\s*</tool_call>', value, re.DOTALL)
            
            if tool_call_matches:
                tool_calls = []
                for tc_str in tool_call_matches:
                    try:
                        tc_obj = json.loads(tc_str)
                        tool_calls.append({
                            'id': f"call_{len(tool_calls)}",
                            'type': 'function',
                            'function': {
                                'name': tc_obj.get('name', ''),
                                'arguments': json.dumps(tc_obj.get('arguments', {}))
                            }
                        })
                    except json.JSONDecodeError:
                        continue
                
                # Extract content without tool calls
                content = re.sub(r'<tool_call>.*?</tool_call>', '', value, flags=re.DOTALL).strip()
                
                messages.append({
                    'role': 'assistant',
                    'content': content,
                    'tool_calls': tool_calls
                })
            else:
                messages.append({'role': 'assistant', 'content': value})
        
        elif from_role == 'tool':
            # Extract tool responses from <tool_response></tool_response> tags
            response_matches = re.findall(r'<tool_response>\s*(.*?)\s*</tool_response>', value, re.DOTALL)
            
            for resp_str in response_matches:
                try:
                    resp_obj = json.loads(resp_str)
                    messages.append({
                        'role': 'tool',
                        'name': resp_obj.get('name', 'unknown'),
                        'content': json.dumps(resp_obj.get('content', '')) if isinstance(resp_obj.get('content'), dict) else str(resp_obj.get('content', ''))
                    })
                except json.JSONDecodeError:
                    messages.append({'role': 'tool', 'content': resp_str})
    
    return messages, tools
