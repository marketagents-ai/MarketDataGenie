"""
Task Manager package for MarketAgents.

This package provides task management functionality for curriculum-based orchestration,
including loading tasks from CSV/JSON files, managing textbook chunks, and creating
workflow configurations.
"""

from .task import (
    Task,
    TextbookChunk,
    WorkflowConfig,
    WorkflowStepConfig,
    TaskExecutionResult,
    TaskBatch
)

from .task_manager import TaskManager

__all__ = [
    "Task",
    "TextbookChunk", 
    "WorkflowConfig",
    "WorkflowStepConfig",
    "TaskExecutionResult",
    "TaskBatch",
    "TaskManager"
]

__version__ = "0.1.0"
