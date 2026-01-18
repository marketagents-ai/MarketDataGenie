"""
Integration tests for bash execution server endpoints.

Tests the REST API endpoints for bash execution.
"""

import pytest
import requests
import time


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


@pytest.fixture
def session(server_url, check_server):
    """Create a test session with bash enabled."""
    resp = requests.post(f"{server_url}/session/create", json={"enable_bash": True})
    assert resp.status_code == 200
    session_id = resp.json()["session_id"]
    
    yield session_id
    
    # Cleanup
    try:
        requests.delete(f"{server_url}/session/{session_id}")
    except:
        pass


class TestBashEndpoint:
    """Test bash execution endpoint."""
    
    def test_bash_endpoint_exists(self, server_url, session):
        """Test that bash endpoint is registered."""
        resp = requests.post(
            f"{server_url}/session/{session}/execute_bash",
            json={"code": "echo test"}
        )
        assert resp.status_code == 200
    
    def test_bash_simple_command(self, server_url, session):
        """Test simple bash command."""
        resp = requests.post(
            f"{server_url}/session/{session}/execute_bash",
            json={"code": "echo 'Hello from bash'"}
        )
        
        assert resp.status_code == 200
        data = resp.json()
        assert data["success"] is True
        assert "Hello from bash" in data["output"]
    
    def test_bash_with_timeout(self, server_url, session):
        """Test bash command with custom timeout."""
        resp = requests.post(
            f"{server_url}/session/{session}/execute_bash",
            json={"code": "sleep 5", "timeout": 1}
        )
        
        assert resp.status_code == 200
        data = resp.json()
        assert data["success"] is False
        assert "timed out" in data["error"].lower()
    
    def test_bash_file_tracking(self, server_url, session):
        """Test that bash tracks file creation."""
        resp = requests.post(
            f"{server_url}/session/{session}/execute_bash",
            json={"code": "echo 'content' > newfile.txt"}
        )
        
        assert resp.status_code == 200
        data = resp.json()
        assert data["success"] is True
        assert "newfile.txt" in data["files_created"]
    
    def test_bash_error_handling(self, server_url, session):
        """Test bash error handling."""
        resp = requests.post(
            f"{server_url}/session/{session}/execute_bash",
            json={"code": "invalid_command_xyz"}
        )
        
        assert resp.status_code == 200
        data = resp.json()
        assert data["success"] is False
        assert data["error"] is not None


class TestBashPythonInterop:
    """Test bash and python interoperability via API."""
    
    def test_bash_then_python(self, server_url, session):
        """Test bash creating file, python reading it."""
        # Bash: create file
        resp1 = requests.post(
            f"{server_url}/session/{session}/execute_bash",
            json={"code": "echo 'test data' > data.txt"}
        )
        assert resp1.status_code == 200
        assert resp1.json()["success"] is True
        
        # Python: read file
        resp2 = requests.post(
            f"{server_url}/session/{session}/execute",
            json={"code": "with open('data.txt', 'r') as f:\n    print(f.read())"}
        )
        assert resp2.status_code == 200
        assert "test data" in resp2.json()["output"]
    
    def test_python_then_bash(self, server_url, session):
        """Test python creating file, bash reading it."""
        # Python: create file
        resp1 = requests.post(
            f"{server_url}/session/{session}/execute",
            json={"code": "with open('output.txt', 'w') as f:\n    f.write('python output')"}
        )
        assert resp1.status_code == 200
        assert resp1.json()["success"] is True
        
        # Bash: read file
        resp2 = requests.post(
            f"{server_url}/session/{session}/execute_bash",
            json={"code": "cat output.txt"}
        )
        assert resp2.status_code == 200
        assert "python output" in resp2.json()["output"]
    
    def test_swe_workflow(self, server_url, session):
        """Test complete SWE workflow."""
        # 1. Bash: explore
        resp1 = requests.post(
            f"{server_url}/session/{session}/execute_bash",
            json={"code": "echo 'def bug(): return 1 - 1' > code.py && cat code.py"}
        )
        assert resp1.json()["success"] is True
        
        # 2. Python: read and analyze
        resp2 = requests.post(
            f"{server_url}/session/{session}/execute",
            json={"code": "with open('code.py', 'r') as f:\n    print(f.read())"}
        )
        assert "def bug" in resp2.json()["output"]
        
        # 3. Python: fix
        resp3 = requests.post(
            f"{server_url}/session/{session}/execute",
            json={"code": """
with open('code.py', 'r') as f:
    content = f.read()
fixed = content.replace('1 - 1', '1 + 1')
with open('code.py', 'w') as f:
    f.write(fixed)
print('Fixed!')
"""}
        )
        assert resp3.json()["success"] is True
        
        # 4. Bash: verify
        resp4 = requests.post(
            f"{server_url}/session/{session}/execute_bash",
            json={"code": "cat code.py"}
        )
        assert "1 + 1" in resp4.json()["output"]


if __name__ == "__main__":
    pytest.main([__file__, "-v", "-s"])
