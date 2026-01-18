"""
Tests for pythonformer sandbox.

Tests Python execution, filesystem operations, and state management.
"""

import pytest
import tempfile
from pathlib import Path

from datagenie.pythonformer.python_server.sandbox import PythonSandbox, ExecutionResult


class TestPythonExecution:
    """Test Python code execution."""
    
    def setup_method(self):
        """Create a fresh sandbox for each test."""
        self.sandbox = PythonSandbox(enable_filesystem=True)
    
    def teardown_method(self):
        """Cleanup sandbox after each test."""
        self.sandbox.cleanup()
    
    def test_simple_execution(self):
        """Test simple Python execution."""
        result = self.sandbox.execute("print('Hello World')")
        
        assert result.success is True
        assert "Hello World" in result.output
        assert result.error is None
    
    def test_syntax_error(self):
        """Test syntax error handling."""
        result = self.sandbox.execute("print('unclosed")
        
        assert result.success is False
        assert result.error is not None
        assert "SyntaxError" in result.error
    
    def test_runtime_error(self):
        """Test runtime error handling."""
        result = self.sandbox.execute("1 / 0")
        
        assert result.success is False
        assert "ZeroDivisionError" in result.error
    
    def test_state_persistence(self):
        """Test that state persists across executions."""
        # Set variable
        result1 = self.sandbox.execute("x = 42")
        assert result1.success is True
        
        # Use variable
        result2 = self.sandbox.execute("print(x)")
        assert result2.success is True
        assert "42" in result2.output
    
    def test_imports(self):
        """Test that pre-imported packages are available."""
        result = self.sandbox.execute("""
import numpy as np
import pandas as pd
print(f"numpy: {np.__version__}")
print(f"pandas: {pd.__version__}")
""")
        
        assert result.success is True
        assert "numpy:" in result.output
        assert "pandas:" in result.output


class TestFilesystemOperations:
    """Test filesystem operations."""
    
    def setup_method(self):
        """Create a fresh sandbox for each test."""
        self.sandbox = PythonSandbox(enable_filesystem=True)
    
    def teardown_method(self):
        """Cleanup sandbox after each test."""
        self.sandbox.cleanup()
    
    def test_save_to_file(self):
        """Test saving content to file."""
        result = self.sandbox.execute("""
save_to_file('test.txt', 'Hello World')
print('Saved!')
""")
        
        assert result.success is True
        assert "test.txt" in result.files_created
    
    def test_read_file(self):
        """Test reading file."""
        # Create file
        result1 = self.sandbox.execute("save_to_file('data.txt', 'test data')")
        assert result1.success is True
        
        # Read file
        result2 = self.sandbox.execute("""
content = read_file('data.txt')
print(content)
""")
        assert result2.success is True
        assert "test data" in result2.output
    
    def test_list_files(self):
        """Test listing files."""
        # Create files
        self.sandbox.execute("save_to_file('file1.txt', 'data1')")
        self.sandbox.execute("save_to_file('file2.txt', 'data2')")
        
        # List files
        result = self.sandbox.execute("""
files = list_files('*.txt')
print(files)
""")
        
        assert result.success is True
        assert "file1.txt" in result.output
        assert "file2.txt" in result.output
    
    def test_file_exists(self):
        """Test checking file existence."""
        result = self.sandbox.execute("""
print(file_exists('nonexistent.txt'))
save_to_file('exists.txt', 'data')
print(file_exists('exists.txt'))
""")
        
        assert result.success is True
        assert "False" in result.output
        assert "True" in result.output
    
    def test_json_auto_serialization(self):
        """Test automatic JSON serialization."""
        result = self.sandbox.execute("""
data = {'key': 'value', 'number': 42}
save_to_file('data.json', data)
loaded = read_file('data.json')
print(loaded)
""")
        
        assert result.success is True
        assert "'key': 'value'" in result.output or '"key": "value"' in result.output


class TestAnswerVariable:
    """Test answer variable and finalization."""
    
    def setup_method(self):
        """Create a fresh sandbox for each test."""
        self.sandbox = PythonSandbox(enable_filesystem=True)
    
    def teardown_method(self):
        """Cleanup sandbox after each test."""
        self.sandbox.cleanup()
    
    def test_answer_variable_exists(self):
        """Test that answer variable is pre-injected."""
        result = self.sandbox.execute("print(answer)")
        
        assert result.success is True
        assert "content" in result.output
        assert "ready" in result.output
    
    def test_answer_ready_flag(self):
        """Test setting answer ready flag."""
        result = self.sandbox.execute("""
answer['content'] = '42'
answer['ready'] = True
""")
        
        assert result.success is True
        assert result.answer_state['content'] == '42'
        assert result.answer_state['ready'] is True
        assert result.done is True
    
    def test_final_helper(self):
        """Test FINAL() helper function."""
        result = self.sandbox.execute("FINAL('The answer is 42')")
        
        assert result.success is True
        assert result.done is True
        assert result.final_answer == "The answer is 42"
    
    def test_final_var_helper(self):
        """Test FINAL_VAR() helper function."""
        result = self.sandbox.execute("""
result = 42
FINAL_VAR('result')
""")
        
        assert result.success is True
        assert result.done is True
        assert result.final_answer == "42"


class TestIterationTracking:
    """Test iteration and episode tracking."""
    
    def setup_method(self):
        """Create a fresh sandbox for each test."""
        self.sandbox = PythonSandbox(enable_filesystem=True)
    
    def teardown_method(self):
        """Cleanup sandbox after each test."""
        self.sandbox.cleanup()
    
    def test_iteration_counter(self):
        """Test iteration counter increments."""
        assert self.sandbox.iteration == 0
        
        self.sandbox.execute("x = 1")
        assert self.sandbox.iteration == 1
        
        self.sandbox.execute("x = 2")
        assert self.sandbox.iteration == 2
    
    def test_max_iterations(self):
        """Test max iterations limit."""
        self.sandbox._max_iterations = 3
        
        self.sandbox.execute("x = 1")
        self.sandbox.execute("x = 2")
        result = self.sandbox.execute("x = 3")
        
        assert result.done is True
        assert self.sandbox.iteration == 3
    
    def test_episode_state(self):
        """Test getting episode state."""
        self.sandbox.execute("x = 42")
        
        state = self.sandbox.get_episode_state()
        
        assert state['iteration'] > 0
        assert state['done'] is False
        assert 'available_variables' in state


class TestStateManagement:
    """Test namespace state management."""
    
    def setup_method(self):
        """Create a fresh sandbox for each test."""
        self.sandbox = PythonSandbox(enable_filesystem=True)
    
    def teardown_method(self):
        """Cleanup sandbox after each test."""
        self.sandbox.cleanup()
    
    def test_namespace_summary(self):
        """Test getting namespace summary."""
        self.sandbox.execute("""
x = 42
y = "hello"
data = [1, 2, 3]
""")
        
        summary = self.sandbox.get_namespace_summary()
        
        assert 'x' in summary
        assert 'y' in summary
        assert 'data' in summary
    
    def test_state_snapshot(self):
        """Test getting state snapshot."""
        self.sandbox.execute("""
def my_function(x):
    return x * 2

class MyClass:
    def method(self):
        pass

import json
""")
        
        snapshot = self.sandbox.get_state_snapshot()
        
        assert 'my_function' in snapshot['functions']
        assert 'MyClass' in snapshot['classes']
        assert 'json' in snapshot['modules']
    
    def test_reset(self):
        """Test resetting sandbox."""
        self.sandbox.execute("x = 42")
        assert self.sandbox.iteration > 0
        
        self.sandbox.reset()
        
        assert self.sandbox.iteration == 0
        result = self.sandbox.execute("print(x)")
        assert result.success is False  # x should not exist


class TestOutputTruncation:
    """Test output truncation."""
    
    def test_long_output_truncation(self):
        """Test that long output is truncated."""
        sandbox = PythonSandbox(max_output_chars=100, enable_filesystem=True)
        
        result = sandbox.execute("print('x' * 1000)")
        
        assert result.truncated is True
        assert len(result.output) <= 150  # Some buffer for truncation message
        
        sandbox.cleanup()
    
    def test_many_lines_truncation(self):
        """Test that many lines are truncated."""
        sandbox = PythonSandbox(max_output_lines=5, enable_filesystem=True)
        
        result = sandbox.execute("""
for i in range(100):
    print(i)
""")
        
        assert result.truncated is True
        
        sandbox.cleanup()


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
