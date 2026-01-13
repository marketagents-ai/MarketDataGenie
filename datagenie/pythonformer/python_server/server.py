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
    
    session_id = str(uuid.uuid4())[:8]
    
    sandbox = PythonSandbox(
        max_output_chars=max_output,
        packages=packages,
        enable_filesystem=True,
    )
    
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
        "available_functions": [
            "llm_query(prompt) - Query sub-LLM (if configured)",
            "save_to_file(filename, content) - Save to workspace",
            "read_file(filename, lines=N) - Read from workspace",
            "list_files(pattern) - List workspace files",
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
    })


@app.route("/session/<session_id>/state", methods=["GET"])
def get_session_state(session_id: str):
    """Get current session state."""
    session = get_session(session_id)
    if not session:
        return jsonify({"error": f"Session {session_id} not found"}), 404
    
    state_snapshot = session.sandbox.get_state_snapshot()
    state_formatted = session.sandbox.format_state_for_context()
    
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
