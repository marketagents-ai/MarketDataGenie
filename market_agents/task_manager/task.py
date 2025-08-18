import csv
import json
import uuid
from typing import Dict, List, Any, Optional, Union
from pydantic import BaseModel, Field
from pathlib import Path
from datetime import datetime

class Task(BaseModel):
    """Task representation for curriculum-based orchestration"""
    id: str = Field(..., description="Unique task identifier")
    category: str = Field(..., description="Task category")
    subcategory: str = Field(..., description="Task subcategory")
    description: str = Field(..., description="Task description")
    metadata: Dict[str, Any] = Field(default_factory=dict)
    context: Dict[str, Any] = Field(default_factory=dict)
    output_schema: Optional[str] = Field(default=None, description="Expected output schema")
    workflow_config: Optional[Dict[str, Any]] = Field(default=None, description="Workflow configuration for this task")
    
    @classmethod
    def from_csv_row(cls, row: Dict[str, str], task_id: str) -> 'Task':
        """Create task from CSV row (datagenie style)"""
        return cls(
            id=task_id,
            category=row.get('Category', ''),
            subcategory=row.get('SubCategory', ''),
            description=row.get('Task', ''),
            metadata={'source': 'csv', 'original_row': row}
        )
    
    def to_workflow_inputs(self) -> Dict[str, Any]:
        """Convert task to workflow inputs"""
        return {
            "task_id": self.id,
            "category": self.category,
            "subcategory": self.subcategory,
            "description": self.description,
            "metadata": self.metadata,
            "context": self.context,
            "output_schema": self.output_schema
        }
    
    def get_workflow_config(self) -> Dict[str, Any]:
        """Get workflow configuration for this task"""
        if self.workflow_config:
            return self.workflow_config
        
        # Default workflow configuration based on task category
        default_configs = {
            "function_calling": {
                "workflow_type": "function_calling",
                "orchestration_mode": "dependency",
                "steps": [
                    {"name": "function_signature_generator", "dependencies": []},
                    {"name": "function_caller", "dependencies": ["function_signature_generator"]}
                ]
            },
            "textbook_qa": {
                "workflow_type": "textbook_qa",
                "orchestration_mode": "dependency",
                "steps": [
                    {"name": "questioner", "dependencies": []},
                    {"name": "reader", "dependencies": []},
                    {"name": "answerer", "dependencies": ["reader"]},
                    {"name": "judge", "dependencies": ["questioner", "reader", "answerer"]},
                    {"name": "coordinator", "dependencies": ["questioner", "reader", "answerer", "judge"]}
                ]
            }
        }
        
        return default_configs.get(self.category, {
            "workflow_type": "generic",
            "orchestration_mode": "sequential",
            "steps": []
        })

class TextbookChunk(BaseModel):
    """Textbook chunk from textbook dataset"""
    id: str = Field(..., description="Chunk identifier")
    text: str = Field(..., description="Text content")
    source: str = Field(..., description="Source information")
    subject: str = Field(..., description="Subject area")
    grade: int = Field(..., description="Grade level")
    chapter_index: int = Field(..., description="Chapter index")
    chapter_title: str = Field(..., description="Chapter title")
    metadata: Dict[str, Any] = Field(default_factory=dict)
    
    def to_workflow_inputs(self) -> Dict[str, Any]:
        """Convert chunk to workflow inputs"""
        return {
            "chunk_id": self.id,
            "text": self.text,
            "source": self.source,
            "subject": self.subject,
            "grade": self.grade,
            "chapter_index": self.chapter_index,
            "chapter_title": self.chapter_title,
            "metadata": self.metadata
        }
    
    def get_workflow_config(self) -> Dict[str, Any]:
        """Get workflow configuration for textbook QA"""
        return {
            "workflow_type": "textbook_qa",
            "orchestration_mode": "dependency",
            "environment": "textbook_qa",
            "steps": [
                {
                    "name": "questioner",
                    "environment_name": "textbook_qa",
                    "tools": ["question_generation_tool"],
                    "subtask": "Generate initial question based on topic and grade level",
                    "dependencies": [],
                    "run_full_episode": False
                },
                {
                    "name": "reader",
                    "environment_name": "textbook_qa",
                    "tools": ["evidence_retrieval_tool"],
                    "subtask": "Retrieve evidence from source text for the question",
                    "dependencies": [],
                    "run_full_episode": False
                },
                {
                    "name": "answerer",
                    "environment_name": "textbook_qa",
                    "tools": ["answer_generation_tool"],
                    "subtask": "Generate answer using provided rationale",
                    "dependencies": ["reader"],
                    "run_full_episode": False
                },
                {
                    "name": "judge",
                    "environment_name": "textbook_qa",
                    "tools": ["quality_evaluation_tool"],
                    "subtask": "Evaluate answer quality and coherence",
                    "dependencies": ["questioner", "reader", "answerer"],
                    "run_full_episode": False
                },
                {
                    "name": "coordinator",
                    "environment_name": "textbook_qa",
                    "tools": ["conversation_management_tool"],
                    "subtask": "Orchestrate conversation and format output",
                    "dependencies": ["questioner", "reader", "answerer", "judge"],
                    "run_full_episode": False
                }
            ]
        }

class WorkflowStepConfig(BaseModel):
    """Configuration for a workflow step"""
    name: str = Field(..., description="Step name")
    environment_name: str = Field(..., description="Environment name")
    tools: List[str] = Field(default_factory=list, description="Tools to use")
    subtask: str = Field(..., description="Subtask description")
    dependencies: List[str] = Field(default_factory=list, description="Step dependencies")
    run_full_episode: bool = Field(default=False, description="Whether to run full episode")
    input_schema: Optional[Dict[str, Any]] = Field(default=None, description="Input schema")
    output_schema: Optional[Dict[str, Any]] = Field(default=None, description="Output schema")

class WorkflowConfig(BaseModel):
    """Configuration for a workflow"""
    workflow_type: str = Field(..., description="Type of workflow")
    orchestration_mode: str = Field(default="sequential", description="Orchestration mode")
    environment: str = Field(..., description="Primary environment")
    steps: List[WorkflowStepConfig] = Field(default_factory=list, description="Workflow steps")
    task: str = Field(default="", description="Workflow task description")
    max_turns: int = Field(default=5, description="Maximum turns")
    quality_threshold: float = Field(default=0.7, description="Quality threshold")

class TaskExecutionResult(BaseModel):
    """Result from task execution"""
    task_id: str = Field(..., description="Task identifier")
    status: str = Field(..., description="Execution status")
    result: Optional[Dict[str, Any]] = Field(default=None, description="Execution result")
    workflow_result: Optional[Dict[str, Any]] = Field(default=None, description="Workflow execution result")
    error: Optional[str] = Field(default=None, description="Error message if failed")
    execution_time: float = Field(default=0.0, description="Execution time in seconds")
    timestamp: str = Field(default_factory=lambda: datetime.now().isoformat())

class TaskBatch(BaseModel):
    """Batch of tasks for execution"""
    batch_id: str = Field(default_factory=lambda: str(uuid.uuid4()), description="Batch identifier")
    tasks: List[Task] = Field(default_factory=list, description="Tasks in batch")
    workflow_config: Optional[WorkflowConfig] = Field(default=None, description="Workflow configuration")
    metadata: Dict[str, Any] = Field(default_factory=dict, description="Batch metadata")
    created_at: str = Field(default_factory=lambda: datetime.now().isoformat())
    
    def add_task(self, task: Task):
        """Add task to batch"""
        self.tasks.append(task)
    
    def get_workflow_config(self) -> WorkflowConfig:
        """Get workflow configuration for this batch"""
        if self.workflow_config:
            return self.workflow_config
        
        # Use first task's workflow config as template
        if self.tasks:
            first_task_config = self.tasks[0].get_workflow_config()
            return WorkflowConfig(**first_task_config)
        
        # Default config
        return WorkflowConfig(
            workflow_type="generic",
            orchestration_mode="sequential",
            environment="default",
            steps=[],
            task="Execute tasks in batch"
        )
