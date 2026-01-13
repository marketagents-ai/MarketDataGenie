"""
REPL Client for interacting with the Pythonformer server.

Used by both:
- Pipeline (dataset generation)
- Inference (running trained models)
"""

import requests
from typing import Dict, Any, Optional, List
from dataclasses import dataclass


@dataclass
class ExecutionResult:
    """Result from code execution."""
    success: bool
    output: str
    error: Optional[str] = None
    truncated: bool = False
    execution_time_ms: int = 0
    answer: Optional[Dict[str, Any]] = None
    files_created: List[str] = None
    variables: Dict[str, str] = None
    state: Dict[str, Any] = None  # Full state snapshot
    state_formatted: str = None   # Formatted state string
    
    def __post_init__(self):
        if self.files_created is None:
            self.files_created = []
        if self.variables is None:
            self.variables = {}
        if self.answer is None:
            self.answer = {"content": "", "ready": False}
        if self.state is None:
            self.state = {"variables": {}, "functions": {}, "classes": {}, "modules": []}
        if self.state_formatted is None:
            self.state_formatted = "(empty state)"


class REPLClient:
    """
    Client for the Pythonformer REPL server.
    
    Manages sessions and executes code via HTTP API.
    """
    
    def __init__(self, server_url: str = "http://localhost:5003"):
        self.server_url = server_url.rstrip("/")
        self.session_id: Optional[str] = None
    
    def health_check(self) -> bool:
        """Check if server is healthy."""
        try:
            resp = requests.get(f"{self.server_url}/health", timeout=5)
            return resp.status_code == 200
        except:
            return False
    
    def create_session(
        self,
        context: Optional[str] = None,
        packages: Optional[List[str]] = None,
        max_output_chars: int = 8192,
    ) -> str:
        """
        Create a new REPL session.
        
        Args:
            context: Optional large context to make available
            packages: Python packages to pre-import
            max_output_chars: Max output truncation limit
            
        Returns:
            Session ID
        """
        payload = {"max_output_chars": max_output_chars}
        if context:
            payload["context"] = context
        if packages:
            payload["packages"] = packages
        
        resp = requests.post(
            f"{self.server_url}/session/create",
            json=payload,
            timeout=30
        )
        resp.raise_for_status()
        data = resp.json()
        self.session_id = data["session_id"]
        return self.session_id
    
    def execute(self, code: str, session_id: Optional[str] = None) -> ExecutionResult:
        """
        Execute code in a session.
        
        Args:
            code: Python code to execute
            session_id: Session ID (uses current session if not provided)
            
        Returns:
            ExecutionResult with output, errors, state
        """
        sid = session_id or self.session_id
        if not sid:
            raise ValueError("No session ID. Call create_session() first.")
        
        resp = requests.post(
            f"{self.server_url}/session/{sid}/execute",
            json={"code": code},
            timeout=120
        )
        resp.raise_for_status()
        data = resp.json()
        
        return ExecutionResult(
            success=data.get("success", False),
            output=data.get("output", ""),
            error=data.get("error"),
            truncated=data.get("truncated", False),
            execution_time_ms=data.get("execution_time_ms", 0),
            answer=data.get("answer", {"content": "", "ready": False}),
            files_created=data.get("files_created", []),
            variables=data.get("variables", {}),
            state=data.get("state", {"variables": {}, "functions": {}, "classes": {}, "modules": []}),
            state_formatted=data.get("state_formatted", "(empty state)"),
        )
    
    def get_state(self, session_id: Optional[str] = None) -> Dict[str, Any]:
        """Get current session state."""
        sid = session_id or self.session_id
        if not sid:
            raise ValueError("No session ID.")
        
        resp = requests.get(
            f"{self.server_url}/session/{sid}/state",
            timeout=30
        )
        resp.raise_for_status()
        return resp.json()
    
    def reset_session(self, session_id: Optional[str] = None) -> None:
        """Reset session to initial state."""
        sid = session_id or self.session_id
        if not sid:
            return
        
        resp = requests.post(
            f"{self.server_url}/session/{sid}/reset",
            timeout=30
        )
        resp.raise_for_status()
    
    def delete_session(self, session_id: Optional[str] = None) -> None:
        """Delete session and cleanup resources."""
        sid = session_id or self.session_id
        if not sid:
            return
        
        try:
            requests.delete(
                f"{self.server_url}/session/{sid}",
                timeout=30
            )
        except:
            pass
        
        if sid == self.session_id:
            self.session_id = None
    
    def execute_stateless(
        self,
        code: str,
        context: Optional[str] = None
    ) -> ExecutionResult:
        """
        Execute code without session management.
        
        Creates a temporary sandbox, executes, and cleans up.
        Use for one-off executions.
        """
        payload = {"code": code}
        if context:
            payload["context"] = context
        
        resp = requests.post(
            f"{self.server_url}/execute",
            json=payload,
            timeout=120
        )
        resp.raise_for_status()
        data = resp.json()
        
        return ExecutionResult(
            success=data.get("success", False),
            output=data.get("output", ""),
            error=data.get("error"),
            execution_time_ms=data.get("execution_time_ms", 0),
            answer=data.get("answer", {"content": "", "ready": False}),
        )
    
    def __enter__(self):
        """Context manager entry."""
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        """Context manager exit - cleanup session."""
        self.delete_session()
        return False
