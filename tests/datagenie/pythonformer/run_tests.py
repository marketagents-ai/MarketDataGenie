#!/usr/bin/env python3
"""
Simple test runner for pythonformer tests.

Usage:
    python tests/datagenie/pythonformer/run_tests.py [option]
    
Options:
    all          - Run all tests (default)
    unit         - Run unit tests only
    integration  - Run integration tests (requires server)
    bash         - Run bash-related tests
    coverage     - Run with coverage report
"""

import sys
import subprocess
from pathlib import Path


def run_pytest(test_path, description):
    """Run pytest on a specific test file."""
    print(f"\n{'='*60}")
    print(f"Running {description}...")
    print(f"{'='*60}\n")
    
    cmd = ["pytest", str(test_path), "-v", "--tb=short"]
    result = subprocess.run(cmd, cwd=Path.cwd())
    
    if result.returncode == 0:
        print(f"\n✓ {description} passed\n")
        return True
    else:
        print(f"\n✗ {description} failed\n")
        return False


def check_server():
    """Check if server is running."""
    try:
        import requests
        resp = requests.get("http://localhost:5003/health", timeout=2)
        return resp.status_code == 200
    except:
        return False


def main():
    option = sys.argv[1] if len(sys.argv) > 1 else "all"
    
    test_dir = Path("tests/datagenie/pythonformer")
    
    if not test_dir.exists():
        print(f"Error: Test directory not found: {test_dir}")
        sys.exit(1)
    
    results = []
    
    if option in ["all", "unit"]:
        print("\n" + "="*60)
        print("UNIT TESTS")
        print("="*60)
        
        results.append(run_pytest(test_dir / "test_config.py", "Config Tests"))
        results.append(run_pytest(test_dir / "test_sandbox.py", "Sandbox Tests"))
        results.append(run_pytest(test_dir / "test_prompts.py", "Prompt Tests"))
    
    if option in ["all", "integration"]:
        if not check_server():
            print("\n⚠️  Server not running on localhost:5003")
            print("Skipping integration tests")
            print("Start server with:")
            print("  conda run -n datagen python -m datagenie.pythonformer.python_server.server --port 5003")
        else:
            print("\n" + "="*60)
            print("INTEGRATION TESTS")
            print("="*60)
            
            results.append(run_pytest(test_dir / "test_client.py", "Client Tests"))
            results.append(run_pytest(test_dir / "test_server_bash.py", "Server Bash Tests"))
            results.append(run_pytest(test_dir / "test_bash_execution.py", "Bash Execution Tests"))
    
    if option == "bash":
        results.append(run_pytest(test_dir / "test_bash_execution.py", "Bash Execution Tests"))
    
    if option == "coverage":
        print("\n" + "="*60)
        print("RUNNING WITH COVERAGE")
        print("="*60 + "\n")
        
        cmd = [
            "pytest",
            str(test_dir),
            "--cov=datagenie.pythonformer",
            "--cov-report=html",
            "--cov-report=term",
            "-v"
        ]
        result = subprocess.run(cmd, cwd=Path.cwd())
        print("\n✓ Coverage report generated in htmlcov/index.html")
        sys.exit(result.returncode)
    
    # Summary
    print("\n" + "="*60)
    print("TEST SUMMARY")
    print("="*60)
    
    passed = sum(results)
    total = len(results)
    
    print(f"\nPassed: {passed}/{total}")
    
    if passed == total:
        print("\n✅ All tests passed!")
        sys.exit(0)
    else:
        print(f"\n❌ {total - passed} test(s) failed")
        sys.exit(1)


if __name__ == "__main__":
    main()
