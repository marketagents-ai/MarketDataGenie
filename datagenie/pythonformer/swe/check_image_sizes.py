"""Check Docker image sizes for all SWE-smith repositories."""

import json
import requests
from pathlib import Path
from typing import Dict, List, Optional
import time

def get_image_size(repo: str) -> Optional[int]:
    """
    Get Docker image size from Docker Hub API.
    
    Args:
        repo: Repository name (e.g., "swesmith/pytest-dev__iniconfig.16793ead")
        
    Returns:
        Image size in bytes, or None if not found
    """
    # Convert repo name to image name format
    # swesmith/pytest-dev__iniconfig.16793ead -> swebench/swesmith.x86_64.pytest-dev_1776_iniconfig.16793ead
    repo_part = repo.replace("swesmith/", "")
    image_name = f"swesmith.x86_64.{repo_part.replace('__', '_1776_')}"
    
    url = f"https://hub.docker.com/v2/repositories/swebench/{image_name}/tags/latest"
    
    try:
        response = requests.get(url, timeout=10)
        if response.status_code == 200:
            data = response.json()
            return data.get('full_size')
        else:
            print(f"  ⚠️  Failed to fetch {repo}: HTTP {response.status_code}")
            return None
    except Exception as e:
        print(f"  ⚠️  Error fetching {repo}: {e}")
        return None


def format_size(size_bytes: int) -> str:
    """Format bytes to human-readable size."""
    for unit in ['B', 'KB', 'MB', 'GB']:
        if size_bytes < 1024.0:
            return f"{size_bytes:.2f} {unit}"
        size_bytes /= 1024.0
    return f"{size_bytes:.2f} TB"


def main():
    # Load repo list
    repos_file = Path("datagenie/pythonformer/top-repos-by-problem-count.json")
    with open(repos_file) as f:
        repos_data = json.load(f)
    
    print(f"Checking Docker image sizes for {len(repos_data)} repositories...")
    print("This will take a few minutes...\n")
    
    results = []
    
    for i, repo_info in enumerate(repos_data, 1):
        repo = repo_info['repo']
        problem_count = repo_info['problem_count']
        
        print(f"[{i}/{len(repos_data)}] {repo} ({problem_count} tasks)...", end=" ")
        
        size_bytes = get_image_size(repo)
        
        if size_bytes:
            size_str = format_size(size_bytes)
            print(f"✓ {size_str}")
            
            results.append({
                'repo': repo,
                'problem_count': problem_count,
                'size_bytes': size_bytes,
                'size_human': size_str
            })
        else:
            print("✗ Not found")
        
        # Rate limit: be nice to Docker Hub
        time.sleep(0.5)
    
    # Sort by size
    results.sort(key=lambda x: x['size_bytes'])
    
    # Save results
    output_file = Path("datagenie/pythonformer/repo-sizes.json")
    with open(output_file, 'w') as f:
        json.dump(results, f, indent=2)
    
    print(f"\n{'='*80}")
    print("SMALLEST DOCKER IMAGES (by size):")
    print(f"{'='*80}\n")
    
    for i, result in enumerate(results[:20], 1):
        print(f"{i:2d}. {result['size_human']:>10s} | {result['problem_count']:>4d} tasks | {result['repo']}")
    
    print(f"\n{'='*80}")
    print("LARGEST DOCKER IMAGES (by size):")
    print(f"{'='*80}\n")
    
    for i, result in enumerate(results[-10:], 1):
        print(f"{i:2d}. {result['size_human']:>10s} | {result['problem_count']:>4d} tasks | {result['repo']}")
    
    print(f"\nResults saved to: {output_file}")
    print(f"Total repos checked: {len(results)}")
    
    if results:
        avg_size = sum(r['size_bytes'] for r in results) / len(results)
        print(f"Average image size: {format_size(avg_size)}")
        print(f"Smallest: {results[0]['size_human']} ({results[0]['repo']})")
        print(f"Largest: {results[-1]['size_human']} ({results[-1]['repo']})")


if __name__ == "__main__":
    main()
