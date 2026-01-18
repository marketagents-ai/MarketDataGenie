"""
Tests for pythonformer REPL client.

Tests client-server communication and session management.
"""

import pytest
import requests

from datagenie.pythonformer.repl_client import REPLClient, ExecutionResult


@pytest.fixture(scope="module")
def server_url():
    """Server URL for testing."""
    return "http://localhost:5003"


@pytest.fixture(scope="module")
def check_server(server_url):
    """Check if server is running before tests."""
    try:
        resp = requests.get(f"{server_url}/health", timeout=2)
        if resp.status_code != 200:
            pytest.skip("Server not running")
    except requests.exceptions.ConnectionError:
        pytest.skip("Server not running on localhost:5003")


class TestREPLClient:
    """Test REPL client basic functionality."""
    
    def test_client_initialization(self):
        """Test client can be initialized."""
        client = REPLClient(server_url="http://localhost:5003")
        assert client.server_url == "http://localhost:5003"
        assert client.session_id is None
    
    def test_health_check(self, server_url, check_server):
        """Test health check."""
        client = REPLClient(server_url=server_url)
        assert client.health_check() is True
    
    def test_health_check_bad_server(self):
        """Test health check with bad server."""
        client = REPLClient(server_url="http://localhost:9999")
        assert client.health_check() is False


class TestSessionManagement:
    """Test session creation and management."""
    
    def test_create_session(self, server_url, check_server):
        """Test creating a session."""
        client = REPLClient(server_url=server_url)
        session_id = client.create_session()
        
        assert session_id is not None
        assert len(session_id) > 0
        assert client.session_id == session_id
        
        # Cleanup
        client.delete_session()
    
    def test_create_session_with_context(self, server_url, check_server):
        """Test creating session with context."""
        client = REPLClient(server_url=server_url)
        session_id = client.create_session(context="Test context data")
        
        assert session_id is not None
        
        # Cleanup
        client.delete_session()
    
    def test_create_session_with_bash_enabled(self, server_url, check_server):
        """Test creating session with bash enabled."""
        client = REPLClient(server_url=server_url)
        session_id = client.create_session(enable_bash=True)
        
        assert session_id is not None
        
        # Cleanup
        client.delete_session()
    
    def test_delete_session(self, server_url, check_server):
        """Test deleting a session."""
        client = REPLClient(server_url=server_url)
        client.create_session()
        
        # Should not raise
        client.delete_session()
        assert client.session_id is None
    
    def test_context_manager(self, server_url, check_server):
        """Test using client as context manager."""
        with REPLClient(server_url=server_url) as client:
            session_id = client.create_session()
            assert session_id is not None
        
        # Session should be cleaned up after context exit


class TestCodeExecution:
    """Test code execution via client."""
    
    def test_execute_python(self, server_url, check_server):
        """Test executing Python code."""
        with REPLClient(server_url=server_url) as client:
            client.create_session()
            
            result = client.execute("print('Hello from client')")
            
            assert isinstance(result, ExecutionResult)
            assert result.success is True
            assert "Hello from client" in result.output
    
    def test_execute_with_error(self, server_url, check_server):
        """Test executing code with error."""
        with REPLClient(server_url=server_url) as client:
            client.create_session()
            
            result = client.execute("1 / 0")
            
            assert result.success is False
            assert result.error is not None
            assert "ZeroDivisionError" in result.error
    
    def test_execute_bash(self, server_url, check_server):
        """Test executing bash commands."""
        with REPLClient(server_url=server_url) as client:
            client.create_session(enable_bash=True)
            
            result = client.execute_bash("echo 'Hello from bash'")
            
            assert isinstance(result, ExecutionResult)
            assert result.success is True
            assert "Hello from bash" in result.output
    
    def test_execute_bash_disabled(self, server_url, check_server):
        """Test bash execution when disabled."""
        with REPLClient(server_url=server_url) as client:
            client.create_session(enable_bash=False)
            
            result = client.execute_bash("echo test")
            
            assert result.success is False
            assert "not enabled" in result.error.lower()


class TestStateRetrieval:
    """Test retrieving session state."""
    
    def test_get_state(self, server_url, check_server):
        """Test getting session state."""
        with REPLClient(server_url=server_url) as client:
            client.create_session()
            client.execute("x = 42")
            
            state = client.get_state()
            
            assert 'session_id' in state
            assert 'variables' in state
            assert 'x' in state['variables']
    
    def test_reset_session(self, server_url, check_server):
        """Test resetting session."""
        with REPLClient(server_url=server_url) as client:
            client.create_session()
            client.execute("x = 42")
            
            # Reset
            client.reset_session()
            
            # Variable should be gone
            result = client.execute("print(x)")
            assert result.success is False


class TestStatelessExecution:
    """Test stateless execution."""
    
    def test_execute_stateless(self, server_url, check_server):
        """Test stateless execution."""
        client = REPLClient(server_url=server_url)
        
        result = client.execute_stateless("print('Stateless')")
        
        assert result.success is True
        assert "Stateless" in result.output
    
    def test_execute_stateless_with_context(self, server_url, check_server):
        """Test stateless execution with context."""
        client = REPLClient(server_url=server_url)
        
        result = client.execute_stateless(
            "print(context)",
            context="Test context"
        )
        
        assert result.success is True
        assert "Test context" in result.output


if __name__ == "__main__":
    pytest.main([__file__, "-v", "-s"])
