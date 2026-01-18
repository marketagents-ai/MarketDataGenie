# Pythonformer Tests

Comprehensive test suite for the pythonformer module.

## Test Structure

```
tests/datagenie/pythonformer/
├── conftest.py                  # Pytest configuration and fixtures
├── test_config.py               # Configuration tests
├── test_sandbox.py              # Sandbox unit tests
├── test_bash_execution.py       # Bash execution tests
├── test_prompts.py              # Prompt validation tests
├── test_client.py               # Client integration tests
├── test_server_bash.py          # Server bash endpoint tests
└── README.md                    # This file
```

## Running Tests

### All Tests
```bash
pytest tests/datagenie/pythonformer/ -v
```

### By Category

**Unit Tests** (no external dependencies):
```bash
pytest tests/datagenie/pythonformer/test_config.py -v
pytest tests/datagenie/pythonformer/test_sandbox.py -v
pytest tests/datagenie/pythonformer/test_prompts.py -v
```

**Integration Tests** (requires server):
```bash
# Start server first
conda run -n datagen python -m datagenie.pythonformer.python_server.server --port 5003

# Run tests
pytest tests/datagenie/pythonformer/test_client.py -v
pytest tests/datagenie/pythonformer/test_server_bash.py -v
pytest tests/datagenie/pythonformer/test_bash_execution.py::TestBashExecution -v
```

### By Test File
```bash
pytest tests/datagenie/pythonformer/test_config.py -v
pytest tests/datagenie/pythonformer/test_sandbox.py -v
pytest tests/datagenie/pythonformer/test_bash_execution.py -v
```

### Specific Test
```bash
pytest tests/datagenie/pythonformer/test_sandbox.py::TestPythonExecution::test_simple_execution -v
```

### With Coverage
```bash
pytest tests/datagenie/pythonformer/ \
    --cov=datagenie.pythonformer \
    --cov-report=html \
    --cov-report=term
```

## Test Categories

### 1. Configuration Tests (`test_config.py`)
Tests configuration loading, validation, and environment types.

**Coverage:**
- Environment type enum
- LLM client type enum
- REPL configuration
- Sub-LLM configuration
- Dataset configuration
- YAML loading
- Client enum conversion

**Run:** `pytest tests/datagenie/pythonformer/test_config.py -v`

### 2. Sandbox Tests (`test_sandbox.py`)
Tests core Python execution, filesystem operations, and state management.

**Coverage:**
- Python code execution
- Error handling (syntax, runtime)
- State persistence
- Pre-imported packages
- Filesystem operations (save, read, list, exists)
- JSON auto-serialization
- Answer variable and finalization
- FINAL() and FINAL_VAR() helpers
- Iteration tracking
- Episode state management
- Namespace summary
- State snapshots
- Reset functionality
- Output truncation

**Run:** `pytest tests/datagenie/pythonformer/test_sandbox.py -v`

### 3. Bash Execution Tests (`test_bash_execution.py`)
Tests bash command execution and Python-bash integration.

**Coverage:**
- Bash disabled by default
- Simple bash commands
- Working directory
- File creation tracking
- Error handling
- Timeout handling
- Multiline commands
- Python-bash interoperability
- Exit code handling
- SWE workflow simulation

**Run:** `pytest tests/datagenie/pythonformer/test_bash_execution.py -v`

### 4. Prompt Tests (`test_prompts.py`)
Tests that all environment-specific prompts are properly defined.

**Coverage:**
- Prompt existence (base, oolong, hotpotqa, swe)
- Prompt content validation
- Tag presence (<python>, <bash>, <final_answer>)
- Environment-specific mentions
- Formatting capabilities
- Workflow guidance
- Tool descriptions

**Run:** `pytest tests/datagenie/pythonformer/test_prompts.py -v`

### 5. Client Tests (`test_client.py`)
Tests REPL client functionality and server communication.

**Coverage:**
- Client initialization
- Health checks
- Session creation
- Session deletion
- Context manager usage
- Python code execution
- Bash execution
- Error handling
- State retrieval
- Session reset
- Stateless execution

**Run:** `pytest tests/datagenie/pythonformer/test_client.py -v` (requires server)

### 6. Server Bash Tests (`test_server_bash.py`)
Tests bash execution endpoints and API-level integration.

**Coverage:**
- Bash endpoint existence
- Simple bash commands via API
- Custom timeouts
- File tracking
- Error responses
- Python-bash interoperability via API
- Complete SWE workflow

**Run:** `pytest tests/datagenie/pythonformer/test_server_bash.py -v` (requires server)

## Prerequisites

### For Unit Tests
No prerequisites - tests create their own sandboxes.

### For Integration Tests
Server must be running:
```bash
conda run -n datagen python -m datagenie.pythonformer.python_server.server --port 5003
```

Tests will automatically skip if server is not available.

## Test Markers

Tests are marked with pytest markers for selective running:

- `@pytest.mark.unit` - Unit tests (no external dependencies)
- `@pytest.mark.integration` - Integration tests (requires server)
- `@pytest.mark.slow` - Slow-running tests

Run only unit tests:
```bash
pytest tests/datagenie/pythonformer/ -m unit -v
```

Run only integration tests:
```bash
pytest tests/datagenie/pythonformer/ -m integration -v
```

## Coverage Goals

- **Overall**: 85%+ coverage
- **Sandbox**: 90%+ coverage
- **Client**: 85%+ coverage
- **Config**: 95%+ coverage
- **Prompts**: 100% coverage (validation only)

## Continuous Integration

### GitHub Actions Example
```yaml
name: Pythonformer Tests

on: [push, pull_request]

jobs:
  unit-tests:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v2
      - name: Set up Python
        uses: actions/setup-python@v2
        with:
          python-version: '3.10'
      - name: Install dependencies
        run: |
          pip install -r requirements.txt
          pip install pytest pytest-cov
      - name: Run unit tests
        run: |
          pytest tests/datagenie/pythonformer/test_config.py -v
          pytest tests/datagenie/pythonformer/test_sandbox.py -v
          pytest tests/datagenie/pythonformer/test_prompts.py -v
  
  integration-tests:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v2
      - name: Set up Python
        uses: actions/setup-python@v2
        with:
          python-version: '3.10'
      - name: Install dependencies
        run: |
          pip install -r requirements.txt
          pip install pytest
      - name: Start server
        run: |
          python -m datagenie.pythonformer.python_server.server --port 5003 &
          sleep 5
      - name: Run integration tests
        run: |
          pytest tests/datagenie/pythonformer/test_client.py -v
          pytest tests/datagenie/pythonformer/test_server_bash.py -v
```

## Troubleshooting

### Import Errors
Ensure you're in the project root:
```bash
cd /path/to/MarketDataGenie
pytest tests/datagenie/pythonformer/ -v
```

### Server Not Running
Start the server:
```bash
conda run -n datagen python -m datagenie.pythonformer.python_server.server --port 5003
```

### Port Already in Use
Kill existing server:
```bash
lsof -ti:5003 | xargs kill -9
```

### Module Not Found
Install in development mode:
```bash
pip install -e .
```

## Adding New Tests

### Test Template
```python
"""
Tests for new feature.
"""

import pytest
from datagenie.pythonformer.your_module import YourClass


class TestYourFeature:
    """Test your feature."""
    
    def setup_method(self):
        """Setup before each test."""
        self.instance = YourClass()
    
    def teardown_method(self):
        """Cleanup after each test."""
        pass
    
    def test_basic_functionality(self):
        """Test basic functionality."""
        result = self.instance.method()
        assert result is not None


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
```

## Test Coverage Report

Generate HTML coverage report:
```bash
pytest tests/datagenie/pythonformer/ \
    --cov=datagenie.pythonformer \
    --cov-report=html

# Open in browser
open htmlcov/index.html
```

## Quick Test Commands

```bash
# Run all tests
pytest tests/datagenie/pythonformer/ -v

# Run with coverage
pytest tests/datagenie/pythonformer/ --cov=datagenie.pythonformer --cov-report=term

# Run specific test file
pytest tests/datagenie/pythonformer/test_sandbox.py -v

# Run specific test
pytest tests/datagenie/pythonformer/test_sandbox.py::TestPythonExecution::test_simple_execution -v

# Run with output
pytest tests/datagenie/pythonformer/ -v -s

# Run failed tests only
pytest tests/datagenie/pythonformer/ --lf

# Run in parallel (requires pytest-xdist)
pytest tests/datagenie/pythonformer/ -n auto
```
