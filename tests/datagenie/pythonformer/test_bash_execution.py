"""
Tests for bash execution in pythonformer sandbox.

Tests the new bash execution feature added for SWE environment.
"""

import pytest
import sys
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent.parent.parent.parent
sys.path.insert(0, str(project_root))

from datagenie.pythonformer.python_server.sandbox import PythonSandbox, ExecutionResult


class TestBashExecution:
    """Test bash command execution in sandbox."""
    
    def setup_method(self):
        """Create a fresh sandbox for each test."""
        self.sandbox = PythonSandbox(enable_filesystem=True, enable_bash=True)
    
    def teardown_method(self):
        """Cleanup sandbox after each test."""
        self.sandbox.cleanup()
    
    def test_bash_disabled_by_default(self):
        """Test that bash is disabled by default."""
        sandbox_no_bash = PythonSandbox(enable_filesystem=True)
        result = sandbox_no_bash.execute_bash("echo test")
        
        assert result.success is False
        assert "not enabled" in result.error.lower()
        
        sandbox_no_bash.cleanup()
    
    def test_simple_bash_command(self):
        """Test basic bash command execution."""
        result = self.sandbox.execute_bash("echo 'Hello World'")
        
        assert result.success is True
        assert "Hello World" in result.output
        assert result.error is None
    
    def test_bash_pwd_in_workspace(self):
        """Test that bash commands run in workspace directory."""
        result = self.sandbox.execute_bash("pwd")
        
        assert result.success is True
        assert str(self.sandbox.workspace_dir) in result.output
    
    def test_bash_file_creation(self):
        """Test creating files with bash."""
        result = self.sandbox.execute_bash("echo 'test content' > test.txt && cat test.txt")
        
        assert result.success is True
        assert "test content" in result.output
        assert "test.txt" in result.files_created
    
    def test_bash_error_handling(self):
        """Test bash command error handling."""
        result = self.sandbox.execute_bash("nonexistent_command")
        
        assert result.success is False
        assert result.error is not None
    
    def test_bash_timeout(self):
        """Test bash command timeout."""
        result = self.sandbox.execute_bash("sleep 10", timeout=1)
        
        assert result.success is False
        assert "timed out" in result.error.lower()
    
    def test_bash_multiline_commands(self):
        """Test multiline bash commands."""
        result = self.sandbox.execute_bash("""
            echo "Line 1"
            echo "Line 2"
            echo "Line 3"
        """)
        
        assert result.success is True
        assert "Line 1" in result.output
        assert "Line 2" in result.output
        assert "Line 3" in result.output
    
    def test_bash_python_interop(self):
        """Test bash and python working together."""
        # Create file with bash
        bash_result = self.sandbox.execute_bash("echo 'data' > data.txt")
        assert bash_result.success is True
        
        # Read file with python
        python_result = self.sandbox.execute("""
with open('data.txt', 'r') as f:
    content = f.read()
print(content)
""")
        assert python_result.success is True
        assert "data" in python_result.output
    
    def test_bash_exit_code(self):
        """Test bash exit code handling."""
        # Success (exit 0)
        result = self.sandbox.execute_bash("exit 0")
        assert result.success is True
        
        # Failure (exit 1)
        result = self.sandbox.execute_bash("exit 1")
        assert result.success is False
        assert "Exit code: 1" in result.error


class TestBashPythonIntegration:
    """Test bash and python code block integration."""
    
    def setup_method(self):
        """Create a fresh sandbox for each test."""
        self.sandbox = PythonSandbox(enable_filesystem=True, enable_bash=True)
    
    def teardown_method(self):
        """Cleanup sandbox after each test."""
        self.sandbox.cleanup()
    
    def test_swe_workflow_simulation(self):
        """Simulate a typical SWE workflow: explore, read, fix, test."""
        # Step 1: Create buggy file with bash
        result1 = self.sandbox.execute_bash("""
            echo 'def add(a, b):' > calculator.py
            echo '    return a - b  # BUG: should be addition' >> calculator.py
            cat calculator.py
        """)
        assert result1.success is True
        assert "def add" in result1.output
        
        # Step 2: Read file with python
        result2 = self.sandbox.execute("""
with open('calculator.py', 'r') as f:
    content = f.read()
print(content)
""")
        assert result2.success is True
        assert "return a - b" in result2.output
        
        # Step 3: Fix bug with python
        result3 = self.sandbox.execute("""
with open('calculator.py', 'r') as f:
    content = f.read()

fixed = content.replace('return a - b', 'return a + b')

with open('calculator.py', 'w') as f:
    f.write(fixed)

print('Fixed!')
""")
        assert result3.success is True
        assert "Fixed!" in result3.output
        
        # Step 4: Verify with bash
        result4 = self.sandbox.execute_bash("cat calculator.py")
        assert result4.success is True
        assert "return a + b" in result4.output
        assert "return a - b" not in result4.output


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
