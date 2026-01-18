"""SWE (Software Engineering) utilities for Pythonformer.

This module contains utilities for working with SWE-bench datasets:
- Docker container management for isolated repository environments
- Repository metadata and image size information
"""

from datagenie.pythonformer.swe.docker_repl import (
    DockerREPLSession,
    LocalRepoManager,
    create_repl_session,
)

__all__ = [
    "DockerREPLSession",
    "LocalRepoManager", 
    "create_repl_session",
]
