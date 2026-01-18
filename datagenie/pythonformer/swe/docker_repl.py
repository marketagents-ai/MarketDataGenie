"""Docker-based REPL session management for SWE tasks."""

import time
import docker
from pathlib import Path
from typing import Optional
import logging

logger = logging.getLogger(__name__)


class DockerREPLSession:
    """Manages a Docker container running the Python REPL server for SWE tasks."""
    
    def __init__(
        self,
        image_name: str,
        instance_id: str,
        server_port: int = 5003,
        timeout: int = 300,
        cleanup: bool = True
    ):
        """
        Initialize Docker REPL session.
        
        Args:
            image_name: Docker image name (e.g., "swebench/swesmith.x86_64.oauthlib...")
            instance_id: Unique task instance ID
            server_port: Port for REPL server inside container
            timeout: Container timeout in seconds
            cleanup: Whether to remove container after stopping
        """
        self.image_name = image_name
        self.instance_id = instance_id
        self.server_port = server_port
        self.timeout = timeout
        self.cleanup = cleanup
        
        self.client = docker.from_env()
        self.container = None
        self.host_port = None
        
    def start(self) -> str:
        """
        Start Docker container with REPL server.
        
        Returns:
            URL of the REPL server (e.g., "http://localhost:32768")
        """
        logger.info(f"Starting Docker container for {self.instance_id}")
        
        # Pull image if not exists
        try:
            self.client.images.get(self.image_name)
            logger.info(f"Using cached image: {self.image_name}")
            print(f"[Docker] Using cached image: {self.image_name}")
        except docker.errors.ImageNotFound:
            logger.info(f"Pulling image: {self.image_name}")
            print(f"[Docker] Pulling image: {self.image_name}")
            print(f"[Docker] This may take 5-10 minutes (~3.5GB download)...")
            print(f"[Docker] Progress:")
            
            # Pull with progress
            for line in self.client.api.pull(self.image_name, stream=True, decode=True):
                if 'status' in line:
                    status = line['status']
                    if 'id' in line:
                        layer_id = line['id']
                        progress = line.get('progress', '')
                        print(f"[Docker]   {layer_id}: {status} {progress}", end='\r')
                    elif 'Pulling' in status or 'Downloaded' in status or 'Extracting' in status:
                        print(f"[Docker]   {status}")
            
            print(f"\n[Docker] Pull complete!")
        
        # Get pythonformer code path (project root)
        # docker_repl.py is in: datagenie/pythonformer/swe/docker_repl.py
        # We need: MarketDataGenie/ (4 levels up)
        pythonformer_path = Path(__file__).parent.parent.parent.parent
        
        # Install pythonformer dependencies and start REPL server
        # Run server.py directly (not as module) to avoid heavy imports
        startup_script = f"""
set -e
echo "Installing REPL server dependencies..."
pip install -q flask 2>&1 | grep -v "already satisfied" || true
echo "Starting REPL server on port {self.server_port}..."
cd /pythonformer/datagenie/pythonformer/python_server
python server.py --port {self.server_port}
"""
        
        # Start container
        self.container = self.client.containers.run(
            image=self.image_name,
            command=["/bin/bash", "-c", startup_script],
            detach=True,
            ports={f'{self.server_port}/tcp': None},  # Random host port
            remove=False,  # Keep for debugging
            name=f"swe-{self.instance_id}",
            # Mount pythonformer code into container
            volumes={
                str(pythonformer_path): {
                    'bind': '/pythonformer',
                    'mode': 'ro'
                }
            },
            environment={
                'PYTHONPATH': '/pythonformer'
            },
            # Resource limits
            cpu_count=2,
            mem_limit='4g'
        )
        
        # Get mapped host port
        self.container.reload()
        port_mapping = self.container.ports[f'{self.server_port}/tcp']
        if not port_mapping:
            raise RuntimeError("Failed to get port mapping")
        self.host_port = int(port_mapping[0]['HostPort'])
        
        logger.info(f"Container started: {self.container.short_id}, port: {self.host_port}")
        
        # Wait for server to be ready
        self._wait_for_server()
        
        return f"http://localhost:{self.host_port}"
    
    def _wait_for_server(self, max_wait: int = 60):
        """Wait for REPL server to be ready (increased timeout for dependency installation)."""
        import requests
        
        url = f"http://localhost:{self.host_port}/health"
        start_time = time.time()
        
        while time.time() - start_time < max_wait:
            try:
                response = requests.get(url, timeout=1)
                if response.status_code == 200:
                    logger.info("REPL server is ready")
                    return
            except requests.exceptions.RequestException:
                pass
            time.sleep(0.5)
        
        raise TimeoutError(f"REPL server did not start within {max_wait}s")
    
    def stop(self):
        """Stop and optionally remove container."""
        if not self.container:
            return
        
        logger.info(f"Stopping container: {self.container.short_id}")
        
        try:
            self.container.stop(timeout=10)
            if self.cleanup:
                self.container.remove()
                logger.info("Container removed")
        except Exception as e:
            logger.error(f"Error stopping container: {e}")
    
    def get_logs(self, tail: int = 100) -> str:
        """Get container logs."""
        if not self.container:
            return ""
        return self.container.logs(tail=tail).decode('utf-8')
    
    def __enter__(self):
        """Context manager entry."""
        return self.start()
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        """Context manager exit."""
        self.stop()


class LocalRepoManager:
    """Manages local repository clones for development/testing without Docker."""
    
    def __init__(self, cache_dir: str = "/tmp/swe-repos"):
        """
        Initialize local repo manager.
        
        Args:
            cache_dir: Directory to cache cloned repositories
        """
        self.cache_dir = Path(cache_dir)
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        
    def setup_repo(self, repo: str, commit: Optional[str] = None) -> Path:
        """
        Clone repository and optionally checkout specific commit.
        
        Args:
            repo: Repository name (e.g., "oauthlib/oauthlib")
            commit: Optional commit hash to checkout
            
        Returns:
            Path to cloned repository
        """
        import subprocess
        
        repo_name = repo.replace('/', '_')
        repo_path = self.cache_dir / repo_name
        
        if not repo_path.exists():
            logger.info(f"Cloning {repo} to {repo_path}")
            subprocess.run([
                "git", "clone", 
                f"https://github.com/{repo}.git",
                str(repo_path)
            ], check=True, capture_output=True)
        else:
            logger.info(f"Using cached repo: {repo_path}")
        
        if commit:
            logger.info(f"Checking out commit: {commit}")
            subprocess.run(
                ["git", "checkout", commit],
                cwd=repo_path,
                check=True,
                capture_output=True
            )
        
        # Install package in development mode
        if (repo_path / "setup.py").exists() or (repo_path / "pyproject.toml").exists():
            logger.info("Installing package in development mode")
            subprocess.run(
                ["pip", "install", "-e", "."],
                cwd=repo_path,
                check=True,
                capture_output=True
            )
        
        return repo_path
    
    def create_testbed_symlink(self, repo_path: Path):
        """
        Create /testbed symlink to repository.
        
        Note: Requires sudo on most systems.
        
        Args:
            repo_path: Path to repository
        """
        import subprocess
        
        testbed = Path("/testbed")
        
        # Remove existing symlink/directory
        if testbed.exists() or testbed.is_symlink():
            subprocess.run(["sudo", "rm", "-rf", str(testbed)], check=True)
        
        # Create symlink
        subprocess.run(
            ["sudo", "ln", "-s", str(repo_path), str(testbed)],
            check=True
        )
        
        logger.info(f"Created symlink: /testbed -> {repo_path}")


def create_repl_session(
    mode: str,
    image_name: Optional[str] = None,
    instance_id: Optional[str] = None,
    repo: Optional[str] = None,
    commit: Optional[str] = None,
    **kwargs
):
    """
    Factory function to create appropriate REPL session.
    
    Args:
        mode: "docker" or "local"
        image_name: Docker image name (required for docker mode)
        instance_id: Task instance ID (required for docker mode)
        repo: Repository name (required for local mode)
        commit: Commit hash (optional for local mode)
        **kwargs: Additional arguments passed to session constructor
        
    Returns:
        DockerREPLSession or LocalRepoManager
    """
    if mode == "docker":
        if not image_name or not instance_id:
            raise ValueError("image_name and instance_id required for docker mode")
        return DockerREPLSession(image_name, instance_id, **kwargs)
    
    elif mode == "local":
        if not repo:
            raise ValueError("repo required for local mode")
        manager = LocalRepoManager(**kwargs)
        repo_path = manager.setup_repo(repo, commit)
        manager.create_testbed_symlink(repo_path)
        return manager
    
    else:
        raise ValueError(f"Unknown mode: {mode}")


# Example usage
if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    
    # Docker mode example
    print("=== Docker Mode Example ===")
    with DockerREPLSession(
        image_name="python:3.11-slim",  # Use a simple image for testing
        instance_id="test-001"
    ) as url:
        print(f"REPL server URL: {url}")
        # Now you can connect with REPLClient(url)
    
    # Local mode example
    print("\n=== Local Mode Example ===")
    manager = LocalRepoManager()
    repo_path = manager.setup_repo("oauthlib/oauthlib")
    print(f"Repository cloned to: {repo_path}")
    # manager.create_testbed_symlink(repo_path)  # Requires sudo
