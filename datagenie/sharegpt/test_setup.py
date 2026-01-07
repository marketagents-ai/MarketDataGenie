#!/usr/bin/env python3
"""
Test script to verify ShareGPT conversion setup.
"""

import os
import sys
from pathlib import Path

def test_imports():
    """Test if all required modules can be imported."""
    try:
        import yaml
        print("✓ PyYAML imported successfully")
    except ImportError:
        print("✗ PyYAML not found. Install with: pip install pyyaml")
        return False
    
    try:
        from datasets import load_dataset
        print("✓ HuggingFace datasets imported successfully")
    except ImportError:
        print("✗ HuggingFace datasets not found. Install with: pip install datasets")
        return False
    
    try:
        from sharegpt_formatter import DatasetToShareGPTConverter
        print("✓ ShareGPT formatter imported successfully")
    except ImportError:
        print("✗ ShareGPT formatter not found")
        return False
    
    return True

def test_config_file():
    """Test if configuration file exists and is valid."""
    config_path = "dataset_config.yml"
    
    if not os.path.exists(config_path):
        print(f"✗ Configuration file {config_path} not found")
        return False
    
    try:
        import yaml
        with open(config_path, 'r', encoding='utf-8') as f:
            config = yaml.safe_load(f)
        
        if 'repositories' in config:
            print(f"✓ Configuration file loaded successfully")
            print(f"  Found {len(config['repositories'])} configured datasets")
            
            # Check for target repository configurations
            has_target_repos = False
            for repo_name, repo_config in config['repositories'].items():
                target_repo = repo_config.get('target_repo', 'Not specified')
                print(f"    - {repo_name}")
                print(f"      Target Repo: {target_repo}")
                if target_repo != 'Not specified':
                    has_target_repos = True
            
            if has_target_repos:
                print("  ✓ Target repositories configured")
            else:
                print("  ⚠ No target repositories configured (uploads will use auto-generated names)")
            
            return True
        else:
            print("✗ Configuration file missing 'repositories' section")
            return False
            
    except Exception as e:
        print(f"✗ Error loading configuration file: {e}")
        return False

def test_converter_initialization():
    """Test if converter can be initialized."""
    try:
        from sharegpt_formatter import DatasetToShareGPTConverter
        converter = DatasetToShareGPTConverter("dataset_config.yml")
        print("✓ Converter initialized successfully")
        return True
    except Exception as e:
        print(f"✗ Error initializing converter: {e}")
        return False

def test_target_repo_functionality():
    """Test target repository functionality from the runner script."""
    try:
        # Import the function from run_sharegpt.py
        sys.path.append('.')
        from run_sharegpt import get_target_repo
        
        # Test with sample config
        test_config = {
            'repositories': {
                'test/repo': {
                    'target_repo': 'username/test-repo-sharegpt'
                }
            },
            'defaults': {
                'target_repo_prefix': 'testuser'
            }
        }
        
        # Test explicit target repo
        target = get_target_repo(test_config, 'test/repo')
        if target == 'username/test-repo-sharegpt':
            print("✓ Target repository function works correctly")
            return True
        else:
            print(f"✗ Target repository function returned unexpected value: {target}")
            return False
            
    except Exception as e:
        print(f"✗ Error testing target repository functionality: {e}")
        return False

def test_list_datasets():
    """Test the --list functionality."""
    try:
        import subprocess
        result = subprocess.run([
            sys.executable, "run_sharegpt.py", "--list"
        ], capture_output=True, text=True, cwd=os.getcwd())
        
        if result.returncode == 0:
            print("✓ --list command works successfully")
            # Check if target repos are shown in output
            if "Target Repo:" in result.stdout:
                print("  ✓ Target repository information displayed")
            else:
                print("  ⚠ Target repository information not displayed")
            return True
        else:
            print(f"✗ --list command failed: {result.stderr}")
            return False
    except Exception as e:
        print(f"✗ Error testing --list command: {e}")
        return False

def main():
    """Run all tests."""
    print("Testing ShareGPT conversion setup...\n")
    
    tests = [
        ("Import tests", test_imports),
        ("Configuration file", test_config_file),
        ("Converter initialization", test_converter_initialization),
        ("Target repository functionality", test_target_repo_functionality),
        ("List datasets command", test_list_datasets),
    ]
    
    passed = 0
    total = len(tests)
    
    for test_name, test_func in tests:
        print(f"Running {test_name}...")
        if test_func():
            passed += 1
        print()
    
    print(f"Tests completed: {passed}/{total} passed")
    
    if passed == total:
        print("✓ All tests passed! Setup is ready.")
        print("\nYou can now run:")
        print("  python run_sharegpt.py --list")
        print("  python run_sharegpt.py \"your-username/your-dataset\"")
        print("  python run_sharegpt.py \"your-username/your-dataset\" --upload")
    else:
        print("✗ Some tests failed. Please fix the issues above.")
        return 1
    
    return 0

if __name__ == "__main__":
    sys.exit(main())
