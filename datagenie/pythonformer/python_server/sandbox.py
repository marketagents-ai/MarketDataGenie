"""
Sandboxed Python REPL with Filesystem Access.

Provides isolated code execution with:
- State persistence across turns
- Filesystem access for dynamic context management
- Output truncation
- Pre-injected answer variable and llm_query function

Inspired by Cursor's "dynamic context discovery" pattern where
files are the primitive for context management.
"""

import ast
import sys
import io
import os
import tempfile
import shutil
import traceback
import asyncio
from typing import Dict, Any, Optional, Callable, List, Tuple
from dataclasses import dataclass, field
from contextlib import redirect_stdout, redirect_stderr
from pathlib import Path


@dataclass
class SubAgentCall:
    """Record of a sub-agent invocation."""
    task: str
    system_prompt: Optional[str]
    response: str


@dataclass
class ExecutionResult:
    """Result of code execution in sandbox."""
    success: bool
    output: str
    error: Optional[str] = None
    answer_state: Optional[Dict[str, Any]] = None
    truncated: bool = False
    execution_time_ms: int = 0
    files_created: List[str] = field(default_factory=list)
    files_modified: List[str] = field(default_factory=list)
    sub_agent_calls: List[SubAgentCall] = field(default_factory=list)


class PythonSandbox:
    """
    Sandboxed Python execution environment with filesystem access.
    
    Features:
    - Persistent state across multiple executions
    - Workspace filesystem for dynamic context management
    - Pre-injected `answer` variable: {"content": "", "ready": False}
    - Pre-injected `llm_query()` function for sub-LLM calls
    - Output truncation to prevent context explosion
    
    Filesystem Design (inspired by Cursor's dynamic context discovery):
    - Long outputs are written to files instead of bloating context
    - Agent can read/write files to manage working memory
    - Files persist across turns for multi-step reasoning
    """
    
    def __init__(
        self,
        max_output_chars: int = 8192,
        max_output_lines: int = 500,
        timeout_seconds: int = 120,
        llm_batch_callback: Optional[Callable] = None,
        packages: Optional[List[str]] = None,
        workspace_dir: Optional[str] = None,
        enable_filesystem: bool = True,
    ):
        self.max_output_chars = max_output_chars
        self.max_output_lines = max_output_lines
        self.timeout_seconds = timeout_seconds
        self.llm_batch_callback = llm_batch_callback
        self.enable_filesystem = enable_filesystem
        
        # Setup workspace directory
        if workspace_dir:
            self.workspace_dir = Path(workspace_dir)
            self.workspace_dir.mkdir(parents=True, exist_ok=True)
            self._owns_workspace = False
        else:
            self.workspace_dir = Path(tempfile.mkdtemp(prefix="pythonformer_"))
            self._owns_workspace = True
        
        # Track file changes
        self._initial_files: set = set()
        self._scan_workspace()
        
        # Track sub-agent calls during execution
        self._sub_agent_calls: List[SubAgentCall] = []
        
        # Initialize sandbox namespace
        self._namespace: Dict[str, Any] = {}
        self._init_namespace(packages or [])
        
        # Execution history
        self.history: List[Tuple[str, ExecutionResult]] = []
    
    def _scan_workspace(self) -> set:
        """Scan workspace for existing files."""
        if not self.enable_filesystem:
            return set()
        files = set()
        for f in self.workspace_dir.rglob("*"):
            if f.is_file():
                files.add(str(f.relative_to(self.workspace_dir)))
        self._initial_files = files
        return files
    
    def _get_file_changes(self) -> Tuple[List[str], List[str]]:
        """Get files created and modified since last scan."""
        if not self.enable_filesystem:
            return [], []
        
        current_files = set()
        for f in self.workspace_dir.rglob("*"):
            if f.is_file():
                current_files.add(str(f.relative_to(self.workspace_dir)))
        
        created = list(current_files - self._initial_files)
        modified = []
        
        return created, modified
    
    def _init_namespace(self, packages: List[str]) -> None:
        """Initialize the sandbox namespace with builtins and helpers."""
        import builtins
        
        safe_builtins = {
            'abs': abs, 'all': all, 'any': any, 'ascii': ascii,
            'bin': bin, 'bool': bool, 'bytearray': bytearray, 'bytes': bytes,
            'callable': callable, 'chr': chr, 'complex': complex,
            'dict': dict, 'divmod': divmod, 'enumerate': enumerate,
            'filter': filter, 'float': float, 'format': format,
            'frozenset': frozenset, 'getattr': getattr, 'hasattr': hasattr,
            'hash': hash, 'hex': hex, 'int': int, 'isinstance': isinstance,
            'issubclass': issubclass, 'iter': iter, 'len': len, 'list': list,
            'map': map, 'max': max, 'min': min, 'next': next,
            'oct': oct, 'ord': ord, 'pow': pow, 'print': print,
            'range': range, 'repr': repr, 'reversed': reversed, 'round': round,
            'set': set, 'slice': slice, 'sorted': sorted, 'str': str,
            'sum': sum, 'tuple': tuple, 'type': type, 'zip': zip,
            'True': True, 'False': False, 'None': None,
            '__import__': __import__,
            'open': self._sandboxed_open if self.enable_filesystem else open,
            'input': input,
        }
        
        self._namespace['__builtins__'] = safe_builtins
        self._namespace['answer'] = {"content": "", "ready": False}
        # Sub-agent for additional analysis (async LLM calls)
        self._namespace['sub_agent'] = self._sub_agent_wrapper
        self._namespace['sub_agent_batch'] = self._sub_agent_batch_wrapper
        # Legacy alias
        self._namespace['llm_query'] = self._sub_agent_wrapper
        
        if self.enable_filesystem:
            self._namespace['workspace'] = str(self.workspace_dir)
            self._namespace['save_to_file'] = self._save_to_file
            self._namespace['read_file'] = self._read_file
            self._namespace['list_files'] = self._list_files
            self._namespace['file_exists'] = self._file_exists
        
        self._import_packages(packages)
    
    def _sandboxed_open(self, path, mode='r', *args, **kwargs):
        """Sandboxed open() that restricts to workspace directory."""
        path = Path(path)
        if not path.is_absolute():
            path = self.workspace_dir / path
        
        try:
            path.resolve().relative_to(self.workspace_dir.resolve())
        except ValueError:
            raise PermissionError(f"Access denied: {path} is outside workspace")
        
        if 'w' in mode or 'a' in mode:
            path.parent.mkdir(parents=True, exist_ok=True)
        
        return open(path, mode, *args, **kwargs)
    
    def _save_to_file(self, filename: str, content: str) -> str:
        """Save content to a file in the workspace."""
        filepath = self.workspace_dir / filename
        filepath.parent.mkdir(parents=True, exist_ok=True)
        filepath.write_text(content)
        return str(filepath)
    
    def _read_file(self, filename: str, lines: Optional[int] = None) -> str:
        """Read content from a file in the workspace."""
        filepath = self.workspace_dir / filename
        if not filepath.exists():
            raise FileNotFoundError(f"File not found: {filename}")
        
        content = filepath.read_text()
        if lines:
            content_lines = content.split('\n')
            content = '\n'.join(content_lines[-lines:])
        return content
    
    def _list_files(self, pattern: str = "*") -> List[str]:
        """List files in workspace matching pattern."""
        files = []
        for f in self.workspace_dir.rglob(pattern):
            if f.is_file():
                files.append(str(f.relative_to(self.workspace_dir)))
        return sorted(files)
    
    def _file_exists(self, filename: str) -> bool:
        """Check if file exists in workspace."""
        return (self.workspace_dir / filename).exists()
    
    def _import_packages(self, packages: List[str]) -> None:
        """Import pre-approved packages into namespace."""
        import_map = {
            'numpy': 'np', 'pandas': 'pd', 'sympy': 'sympy',
            'scipy': 'scipy', 'json': 'json', 're': 're',
            'collections': 'collections', 'math': 'math',
            'itertools': 'itertools', 'functools': 'functools',
            'datetime': 'datetime', 'pathlib': 'pathlib', 'os': 'os',
        }
        
        for pkg in packages:
            try:
                module = __import__(pkg)
                alias = import_map.get(pkg, pkg)
                self._namespace[alias] = module
            except ImportError:
                pass
    
    def _sub_agent_wrapper(
        self, 
        task: str, 
        system_prompt: Optional[str] = None,
        context: Optional[str] = None
    ) -> str:
        """
        Invoke a sub-agent for semantic analysis.
        
        Args:
            task: The task/question for the sub-agent
            system_prompt: Optional system prompt (defaults to helpful assistant)
            context: Optional context to include (will be wrapped in <file> tags)
        
        Returns:
            Sub-agent response string
        """
        import requests
        
        # Get the server URL from environment or use default
        server_url = os.environ.get("LLM_QUERY_URL", "http://localhost:5003/sub_agent")
        
        # Get sub-agent config from sandbox (set during session creation)
        sub_config = getattr(self, 'sub_agent_config', {})
        
        response_text = ""
        try:
            payload = {
                "task": task,
                "model": sub_config.get("model", "Hermes-4-70B"),
                "client": sub_config.get("client", "litellm"),
                "max_tokens": sub_config.get("max_tokens", 2048),
                "temperature": sub_config.get("temperature", 0.3),
            }
            if system_prompt:
                payload["system_prompt"] = system_prompt
            if context:
                payload["context"] = context
            
            resp = requests.post(server_url, json=payload, timeout=120)
            
            if resp.status_code == 200:
                data = resp.json()
                response_text = data.get("response", "[No response]")
            else:
                response_text = f"[Sub-agent call failed: HTTP {resp.status_code}]"
                
        except requests.exceptions.ConnectionError:
            # If server endpoint not available, return placeholder
            response_text = f"[sub_agent not available - task: {task[:100]}...]"
        except Exception as e:
            response_text = f"[sub_agent error: {str(e)}]"
        
        # Record the call for observation formatting
        self._sub_agent_calls.append(SubAgentCall(
            task=task[:200] + "..." if len(task) > 200 else task,
            system_prompt=system_prompt,
            response=response_text
        ))
        
        return response_text
    
    def _sub_agent_batch_wrapper(
        self, 
        tasks: List[str], 
        system_prompt: Optional[str] = None
    ) -> List[str]:
        """Invoke sub-agents with multiple tasks (sequential for now)."""
        results = []
        for task in tasks:
            result = self._sub_agent_wrapper(task, system_prompt)
            results.append(result)
        return results
    
    def _truncate_output(self, output: str) -> Tuple[str, bool]:
        """Truncate output to configured limits."""
        truncated = False
        
        lines = output.split('\n')
        if len(lines) > self.max_output_lines:
            lines = lines[:self.max_output_lines]
            lines.append(f"\n... [truncated: {len(lines)} lines shown]")
            output = '\n'.join(lines)
            truncated = True
        
        if len(output) > self.max_output_chars:
            output = output[:self.max_output_chars]
            output += f"\n... [truncated at {self.max_output_chars} chars]"
            truncated = True
        
        return output, truncated
    
    def execute(self, code: str) -> ExecutionResult:
        """Execute Python code in the sandbox."""
        import time
        start_time = time.time()
        
        # Clear sub-agent calls from previous execution
        self._sub_agent_calls = []
        
        stdout_capture = io.StringIO()
        stderr_capture = io.StringIO()
        
        result = ExecutionResult(
            success=False, output="",
            answer_state=self._namespace.get('answer', {}).copy()
        )
        
        original_cwd = os.getcwd()
        if self.enable_filesystem:
            os.chdir(self.workspace_dir)
        
        try:
            ast.parse(code)
            with redirect_stdout(stdout_capture), redirect_stderr(stderr_capture):
                exec(code, self._namespace)
            result.success = True
        except SyntaxError as e:
            result.error = f"SyntaxError: {e}"
        except Exception as e:
            result.error = f"{type(e).__name__}: {e}\n{traceback.format_exc()}"
        finally:
            if self.enable_filesystem:
                os.chdir(original_cwd)
        
        stdout_output = stdout_capture.getvalue()
        stderr_output = stderr_capture.getvalue()
        
        combined_output = stdout_output
        if stderr_output:
            combined_output += f"\n[stderr]: {stderr_output}"
        if result.error:
            combined_output += f"\n[error]: {result.error}"
        
        result.output, result.truncated = self._truncate_output(combined_output)
        
        # Safely get answer state (model might have overwritten the answer variable)
        answer_val = self._namespace.get('answer', {})
        if isinstance(answer_val, dict):
            result.answer_state = answer_val.copy()
        else:
            # Model overwrote answer with a non-dict value
            result.answer_state = {"content": str(answer_val), "ready": True}
        
        result.execution_time_ms = int((time.time() - start_time) * 1000)
        
        if self.enable_filesystem:
            result.files_created, result.files_modified = self._get_file_changes()
        
        # Include sub-agent calls made during this execution
        result.sub_agent_calls = self._sub_agent_calls.copy()
        
        self.history.append((code, result))
        return result
    
    def get_answer(self) -> Dict[str, Any]:
        """Get current state of the answer variable."""
        return self._namespace.get('answer', {"content": "", "ready": False}).copy()
    
    def is_ready(self) -> bool:
        """Check if answer["ready"] is True."""
        answer = self._namespace.get('answer', {})
        return answer.get('ready', False)
    
    def reset(self) -> None:
        """Reset the sandbox to initial state."""
        self._namespace.clear()
        self._init_namespace([])
        self.history.clear()
        
        if self._owns_workspace and self.workspace_dir.exists():
            for f in self.workspace_dir.iterdir():
                if f.is_file():
                    f.unlink()
                elif f.is_dir():
                    shutil.rmtree(f)
    
    def cleanup(self) -> None:
        """Clean up resources (call when done)."""
        if self._owns_workspace and self.workspace_dir.exists():
            shutil.rmtree(self.workspace_dir)
    
    def get_namespace_summary(self) -> Dict[str, str]:
        """Get a summary of variables in the namespace."""
        summary = {}
        skip_keys = {'__builtins__', 'sub_agent', 'sub_agent_batch', 'llm_query',
                     'save_to_file', 'read_file', 'list_files', 'file_exists', 'workspace'}
        
        for key, value in self._namespace.items():
            if key in skip_keys:
                continue
            try:
                type_name = type(value).__name__
                if isinstance(value, (str, int, float, bool, type(None))):
                    summary[key] = f"{type_name}: {repr(value)[:100]}"
                elif isinstance(value, (list, dict, set, tuple)):
                    summary[key] = f"{type_name}: len={len(value)}"
                else:
                    summary[key] = f"{type_name}"
            except:
                summary[key] = "<?>"
        return summary
    
    def get_state_snapshot(self) -> Dict[str, Any]:
        """Get a comprehensive state snapshot."""
        import inspect
        import types
        
        skip_keys = {'__builtins__', 'sub_agent', 'sub_agent_batch', 'llm_query',
                     'save_to_file', 'read_file', 'list_files', 'file_exists', 'workspace', 'context'}
        
        variables, functions, classes, modules = {}, {}, {}, []
        
        for key, value in self._namespace.items():
            if key in skip_keys or key.startswith('_'):
                continue
            
            try:
                if isinstance(value, types.FunctionType):
                    try:
                        sig = str(inspect.signature(value))
                        functions[key] = f"def {key}{sig}"
                    except (ValueError, TypeError):
                        functions[key] = f"def {key}(...)"
                elif isinstance(value, type):
                    methods = [m for m in dir(value) if not m.startswith('_') and callable(getattr(value, m, None))]
                    classes[key] = {"name": key, "methods": methods[:10],
                                   "bases": [b.__name__ for b in value.__bases__ if b.__name__ != 'object']}
                elif isinstance(value, types.ModuleType):
                    modules.append(key)
                else:
                    type_name = type(value).__name__
                    if isinstance(value, (str, int, float, bool, type(None))):
                        val_repr = repr(value)[:80]
                        variables[key] = {"type": type_name, "value": val_repr}
                    elif isinstance(value, (list, tuple)):
                        variables[key] = {"type": type_name, "len": len(value)}
                    elif isinstance(value, dict):
                        variables[key] = {"type": type_name, "len": len(value), "keys": list(value.keys())[:5]}
                    elif isinstance(value, set):
                        variables[key] = {"type": type_name, "len": len(value)}
                    elif hasattr(value, 'shape'):
                        variables[key] = {"type": type_name, "shape": str(value.shape)}
                    else:
                        variables[key] = {"type": type_name}
            except Exception:
                variables[key] = {"type": "unknown"}
        
        return {"variables": variables, "functions": functions, "classes": classes, "modules": modules}
    
    def format_state_for_context(self) -> str:
        """Format state snapshot as a concise string for context injection."""
        state = self.get_state_snapshot()
        parts = []
        
        if state["modules"]:
            parts.append(f"imports: {', '.join(state['modules'])}")
        if state["functions"]:
            parts.append(f"functions: {'; '.join(state['functions'].values())}")
        if state["classes"]:
            class_strs = []
            for name, info in state["classes"].items():
                bases = f"({', '.join(info['bases'])})" if info['bases'] else ""
                methods = f" [{', '.join(info['methods'][:5])}]" if info['methods'] else ""
                class_strs.append(f"class {name}{bases}{methods}")
            parts.append(f"classes: {'; '.join(class_strs)}")
        if state["variables"]:
            var_strs = []
            for name, info in state["variables"].items():
                if "value" in info:
                    var_strs.append(f"{name}={info['value']}")
                elif "shape" in info:
                    var_strs.append(f"{name}: {info['type']}({info['shape']})")
                elif "len" in info:
                    var_strs.append(f"{name}: {info['type']}[{info['len']}]")
                else:
                    var_strs.append(f"{name}: {info['type']}")
            parts.append(f"vars: {', '.join(var_strs)}")
        
        # Add workspace files
        if self.enable_filesystem:
            files = self._list_files("*")
            if files:
                parts.append(f"files: {', '.join(files)}")
        
        return " | ".join(parts) if parts else "(empty state)"


class AsyncPythonSandbox(PythonSandbox):
    """Async version of PythonSandbox with timeout support."""
    
    async def execute_async(self, code: str) -> ExecutionResult:
        """Execute code asynchronously with timeout."""
        try:
            result = await asyncio.wait_for(
                asyncio.get_event_loop().run_in_executor(None, self.execute, code),
                timeout=self.timeout_seconds
            )
            return result
        except asyncio.TimeoutError:
            return ExecutionResult(
                success=False, output="",
                error=f"Execution timed out after {self.timeout_seconds} seconds",
                answer_state=self.get_answer()
            )
