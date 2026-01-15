"""
Pythonformer REPL Server for Inference.

Provides a REST API for executing Python code in a sandboxed environment.
Use this when running inference with a trained model that needs to
interact with a REPL in a request/response loop.

Usage:
    # Start server (as module from project root)
    python -m datagenie.pythonformer.python_server.server --port 5003
    
    # Or standalone (in Docker)
    python server.py --port 5003
    
    # Or with Docker for isolation
    docker-compose up --build

API:
    POST /session/create  - Create a new sandbox session
    POST /session/{id}/execute - Execute code in session
    GET  /session/{id}/state - Get session state (variables, files)
    POST /session/{id}/reset - Reset session
    DELETE /session/{id} - Delete session
    
    POST /execute - Stateless single execution (creates temp session)
"""

import os
import uuid
import time
import threading
from typing import Dict, Optional
from dataclasses import dataclass
from flask import Flask, jsonify, request

# Support both module and standalone imports
try:
    from datagenie.pythonformer.python_server.sandbox import PythonSandbox, ExecutionResult
except ImportError:
    from sandbox import PythonSandbox, ExecutionResult


app = Flask(__name__)

# Session storage
sessions: Dict[str, "REPLSession"] = {}
sessions_lock = threading.Lock()

# Config
SESSION_TIMEOUT = 3600  # 1 hour
MAX_SESSIONS = 100
MAX_OUTPUT_CHARS = 8192


@dataclass
class REPLSession:
    """A REPL session with persistent state."""
    id: str
    sandbox: PythonSandbox
    created_at: float
    last_accessed: float
    execution_count: int = 0


def cleanup_old_sessions():
    """Remove sessions that haven't been accessed recently."""
    now = time.time()
    with sessions_lock:
        expired = [
            sid for sid, session in sessions.items()
            if now - session.last_accessed > SESSION_TIMEOUT
        ]
        for sid in expired:
            sessions[sid].sandbox.cleanup()
            del sessions[sid]


def get_session(session_id: str) -> Optional[REPLSession]:
    """Get a session by ID."""
    with sessions_lock:
        session = sessions.get(session_id)
        if session:
            session.last_accessed = time.time()
        return session


# ============================================================
# Session Management Endpoints
# ============================================================

@app.route("/session/create", methods=["POST"])
def create_session():
    """Create a new REPL session."""
    cleanup_old_sessions()
    
    if len(sessions) >= MAX_SESSIONS:
        return jsonify({"error": "Too many active sessions"}), 503
    
    data = request.json or {}
    context = data.get("context")
    packages = data.get("packages", ["numpy", "pandas", "sympy", "json", "re", "math"])
    max_output = data.get("max_output_chars", MAX_OUTPUT_CHARS)
    max_iterations = data.get("max_iterations", 30)
    
    # Sub-agent config (passed from pipeline config)
    sub_agent_config = data.get("sub_agent_config", {})
    
    # RL reward config (optional)
    reward_config = data.get("reward_config", {})
    
    session_id = str(uuid.uuid4())[:8]
    
    sandbox = PythonSandbox(
        max_output_chars=max_output,
        packages=packages,
        enable_filesystem=True,
        reward_on_success=reward_config.get("on_success", 1.0),
        reward_on_iteration=reward_config.get("on_iteration", 0.0),
        reward_on_error=reward_config.get("on_error", -0.05),
        reward_on_failure=reward_config.get("on_failure", -0.1),
    )
    
    # Set max iterations
    sandbox._max_iterations = max_iterations
    
    # Store sub-agent config in sandbox for later use
    sandbox.sub_agent_config = sub_agent_config
    
    if context:
        sandbox._namespace["context"] = context
        context_file = sandbox.workspace_dir / "context.txt"
        context_file.write_text(context)
    
    session = REPLSession(
        id=session_id,
        sandbox=sandbox,
        created_at=time.time(),
        last_accessed=time.time(),
    )
    
    with sessions_lock:
        sessions[session_id] = session
    
    return jsonify({
        "session_id": session_id,
        "workspace": str(sandbox.workspace_dir),
        "max_iterations": max_iterations,
        "available_functions": [
            # Sub-agent
            "sub_agent(task, system_prompt=None, context=None) - Invoke sub-agent for semantic analysis",
            "sub_agent_batch(tasks, system_prompt=None) - Batch sub-agent calls",
            # File I/O (auto-detects json/csv)
            "save_to_file(filename, content) - Save to workspace (auto-serializes json/csv)",
            "read_file(filename, lines=N, raw=False) - Read file (auto-parses json/csv)",
            "list_files(pattern) - List workspace files",
            "file_exists(filename) - Check if file exists",
            # Enhanced file ops
            "get_file_info(filename) - Get file metadata (size, lines, type)",
            "search_files(query, pattern='*', max_results=20) - Search content in files",
            # Scratch vs output organization
            "save_scratch(filename, content) - Save to scratch/ (temporary)",
            "save_output(filename, content) - Save to output/ (artifacts)",
            "list_scratch() - List scratch files",
            "list_output() - List output files",
            # Finalization
            "FINAL(value) - Signal task completion with final answer",
            "FINAL_VAR('var_name') - Signal completion with variable value",
            "answer dict - Set answer['content'] and answer['ready']=True when done",
        ]
    })


@app.route("/session/<session_id>/execute", methods=["POST"])
def execute_in_session(session_id: str):
    """Execute code in an existing session."""
    session = get_session(session_id)
    if not session:
        return jsonify({"error": f"Session {session_id} not found"}), 404
    
    data = request.json or {}
    code = data.get("code", "")
    
    if not code:
        return jsonify({"error": "No code provided"}), 400
    
    result = session.sandbox.execute(code)
    session.execution_count += 1
    
    state_snapshot = session.sandbox.get_state_snapshot()
    state_formatted = session.sandbox.format_state_for_context()
    episode_state = session.sandbox.get_episode_state()
    
    return jsonify({
        "success": result.success,
        "output": result.output,
        "error": result.error,
        "truncated": result.truncated,
        "execution_time_ms": result.execution_time_ms,
        "answer": result.answer_state,
        "files_created": result.files_created,
        "variables": session.sandbox.get_namespace_summary(),
        "state": state_snapshot,
        "state_formatted": state_formatted,
        "execution_count": session.execution_count,
        "sub_agent_calls": [
            {"task": c.task, "system_prompt": c.system_prompt, "response": c.response}
            for c in result.sub_agent_calls
        ],
        # RLM-style fields
        "done": result.done,
        "final_answer": result.final_answer,
        "iteration": result.iteration,
        "reward": result.reward,
        "episode_state": episode_state,
    })


@app.route("/session/<session_id>/state", methods=["GET"])
def get_session_state(session_id: str):
    """Get current session state."""
    session = get_session(session_id)
    if not session:
        return jsonify({"error": f"Session {session_id} not found"}), 404
    
    state_snapshot = session.sandbox.get_state_snapshot()
    state_formatted = session.sandbox.format_state_for_context()
    episode_state = session.sandbox.get_episode_state()
    
    return jsonify({
        "session_id": session_id,
        "answer": session.sandbox.get_answer(),
        "variables": session.sandbox.get_namespace_summary(),
        "state": state_snapshot,
        "state_formatted": state_formatted,
        "files": session.sandbox._list_files("*"),
        "execution_count": session.execution_count,
        "created_at": session.created_at,
        "last_accessed": session.last_accessed,
        # RLM-style fields
        "done": session.sandbox.done,
        "final_answer": session.sandbox.final_answer,
        "iteration": session.sandbox.iteration,
        "episode_state": episode_state,
    })


@app.route("/session/<session_id>/reset", methods=["POST"])
def reset_session(session_id: str):
    """Reset session to initial state."""
    session = get_session(session_id)
    if not session:
        return jsonify({"error": f"Session {session_id} not found"}), 404
    
    session.sandbox.reset()
    session.execution_count = 0
    
    return jsonify({"status": "reset", "session_id": session_id})


@app.route("/session/<session_id>", methods=["DELETE"])
def delete_session(session_id: str):
    """Delete a session and cleanup resources."""
    with sessions_lock:
        session = sessions.pop(session_id, None)
    
    if not session:
        return jsonify({"error": f"Session {session_id} not found"}), 404
    
    session.sandbox.cleanup()
    return jsonify({"status": "deleted", "session_id": session_id})


# ============================================================
# LLM Query Endpoint (for sub-LLM calls from sandbox)
# ============================================================

# ============================================================
# Sub-Agent Endpoint (for async LLM calls from sandbox)
# ============================================================

# Default sub-agent config - can be overridden per-request or per-session
DEFAULT_SUB_MODEL = os.environ.get("SUB_LLM_MODEL", "gpt-4o-mini")
DEFAULT_SUB_TEMPERATURE = float(os.environ.get("SUB_LLM_TEMPERATURE", "0.3"))
DEFAULT_SUB_MAX_TOKENS = int(os.environ.get("SUB_LLM_MAX_TOKENS", "2048"))

@app.route("/sub_agent", methods=["POST"])
def sub_agent():
    """
    Invoke a sub-agent for semantic analysis.
    
    This endpoint is called by the sandbox's sub_agent() function.
    Uses minference orchestrator for async LLM calls.
    
    Request body:
        {
            "task": "Count the dice rolls in this text...",
            "system_prompt": "You are a helpful assistant.",  # optional
            "context": "...",  # optional, will be wrapped in <file> tags
            "model": "Hermes-4-405B",  # optional, from config
            "client": "litellm",  # optional, from config
            "max_tokens": 2048,  # optional
            "temperature": 0.3  # optional
        }
    
    Returns:
        {"response": "The count is 84.", "model": "..."}
    """
    data = request.json or {}
    task = data.get("task", "")
    system_prompt = data.get("system_prompt", "You are a helpful assistant. Be concise and accurate.")
    context = data.get("context")
    
    # Get config from request (passed from pipeline) or use defaults
    model = data.get("model", DEFAULT_SUB_MODEL)
    client = data.get("client", "litellm")
    max_tokens = data.get("max_tokens", DEFAULT_SUB_MAX_TOKENS)
    temperature = data.get("temperature", DEFAULT_SUB_TEMPERATURE)
    
    if not task:
        return jsonify({"error": "No task provided"}), 400
    
    # Build user message with optional context in <file> tags
    user_message = task
    if context:
        user_message = f"<file name=\"context\">\n{context}\n</file>\n\n{task}"
    
    try:
        # Try minference orchestrator first (supports litellm, openai, anthropic)
        try:
            from minference.lite.inference import InferenceOrchestrator
            from minference.lite.models import (
                LLMConfig, LLMClient, ResponseFormat,
                ChatThread, ChatMessage, MessageRole, SystemPrompt,
                EntityRegistry
            )
            import logging
            
            # Initialize logger if needed
            if not hasattr(EntityRegistry, '_logger') or EntityRegistry._logger is None:
                EntityRegistry._logger = logging.getLogger("minference")
            
            # Map client string to LLMClient enum
            client_map = {
                "openai": LLMClient.openai,
                "anthropic": LLMClient.anthropic,
                "litellm": LLMClient.litellm,
            }
            llm_client = client_map.get(client, LLMClient.litellm)
            
            orchestrator = InferenceOrchestrator()
            
            thread = ChatThread(
                name="sub_agent",
                system_prompt=SystemPrompt(name="sys", content=system_prompt),
                llm_config=LLMConfig(
                    client=llm_client,
                    model=model,
                    temperature=temperature,
                    max_tokens=max_tokens,
                    response_format=ResponseFormat.text
                ),
                tools=[],
                history=[],
                new_message=user_message
            )
            
            # Run synchronously (we're in Flask)
            import asyncio
            loop = asyncio.new_event_loop()
            asyncio.set_event_loop(loop)
            try:
                outputs = loop.run_until_complete(orchestrator.run_parallel_ai_completion([thread]))
                response_text = outputs[0].content if outputs and outputs[0] else "[No response]"
            finally:
                loop.close()
            
            return jsonify({
                "response": response_text,
                "model": model,
                "client": client,
            })
            
        except ImportError:
            # Fallback to openai client directly (for OpenAI-compatible endpoints)
            from openai import OpenAI
            
            # Use LITELLM_ENDPOINT if set (strip /chat/completions if present)
            api_base = os.environ.get("LITELLM_ENDPOINT", "")
            if api_base.endswith("/chat/completions"):
                api_base = api_base.rsplit("/chat/completions", 1)[0]
            if not api_base.endswith("/v1"):
                api_base = api_base.rstrip("/") + "/v1"
            
            api_key = os.environ.get("LITELLM_API_KEY")
            
            client = OpenAI(api_key=api_key, base_url=api_base)
            
            response = client.chat.completions.create(
                model=model,
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_message}
                ],
                max_tokens=max_tokens,
                temperature=temperature,
            )
            
            return jsonify({
                "response": response.choices[0].message.content,
                "model": model,
            })
            
    except Exception as e:
        import traceback
        print(f"Sub-agent error: {e}")
        traceback.print_exc()
        return jsonify({
            "error": str(e),
            "response": f"[Sub-agent call failed: {str(e)}]"
        }), 500


# Legacy endpoint alias
@app.route("/llm_query", methods=["POST"])
def llm_query_legacy():
    """Legacy endpoint - redirects to sub_agent."""
    data = request.json or {}
    # Map old format to new format
    new_data = {
        "task": data.get("prompt", ""),
        "system_prompt": data.get("system_prompt"),
        "max_tokens": data.get("max_tokens"),
        "temperature": data.get("temperature"),
    }
    # Forward to sub_agent
    with app.test_request_context(json=new_data):
        return sub_agent()


# ============================================================
# Stateless Execution
# ============================================================

@app.route("/execute", methods=["POST"])
def execute_stateless():
    """Execute code without session management."""
    data = request.json or {}
    code = data.get("code", "")
    context = data.get("context")
    
    if not code:
        return jsonify({"error": "No code provided"}), 400
    
    sandbox = PythonSandbox(max_output_chars=MAX_OUTPUT_CHARS, enable_filesystem=True)
    
    try:
        if context:
            sandbox._namespace["context"] = context
        
        result = sandbox.execute(code)
        
        return jsonify({
            "success": result.success,
            "output": result.output,
            "error": result.error,
            "execution_time_ms": result.execution_time_ms,
            "answer": result.answer_state,
        })
    finally:
        sandbox.cleanup()


# ============================================================
# Health Check
# ============================================================

@app.route("/health", methods=["GET"])
def health():
    """Health check endpoint."""
    return jsonify({
        "status": "healthy",
        "active_sessions": len(sessions),
        "max_sessions": MAX_SESSIONS,
    })


# ============================================================
# Main
# ============================================================

def main():
    import argparse
    parser = argparse.ArgumentParser(description="Pythonformer REPL Server")
    parser.add_argument("--port", type=int, default=5003, help="Port to listen on")
    parser.add_argument("--host", default="0.0.0.0", help="Host to bind to")
    parser.add_argument("--debug", action="store_true", help="Debug mode")
    args = parser.parse_args()
    
    print(f"Starting Pythonformer REPL Server on {args.host}:{args.port}")
    print(f"Max sessions: {MAX_SESSIONS}, Session timeout: {SESSION_TIMEOUT}s")
    app.run(host=args.host, port=args.port, debug=args.debug)


if __name__ == "__main__":
    main()
