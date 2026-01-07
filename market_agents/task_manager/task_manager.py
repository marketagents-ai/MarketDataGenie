import csv
import json
import logging
from typing import Dict, List, Any, Optional, Union
from pathlib import Path
from datetime import datetime
import uuid

from .task import (
    Task, 
    TextbookChunk, 
    WorkflowConfig, 
    WorkflowStepConfig,
    TaskExecutionResult,
    TaskBatch
)

logger = logging.getLogger(__name__)

class TaskManager:
    """Manager for loading and handling tasks and textbook chunks"""
    
    def __init__(self, base_path: str = "configs/curriculum"):
        self.base_path = Path(base_path)
        self.base_path.mkdir(parents=True, exist_ok=True)
        self.execution_history: List[TaskExecutionResult] = []


    def _coerce_int(self, v: Any, default: int = 0) -> int:
        try:
            if isinstance(v, bool):
                return default
            if isinstance(v, (int, float)):
                return int(v)
            if isinstance(v, str):
                s = v.strip()
                if s.isdigit() or (s.startswith('-') and s[1:].isdigit()):
                    return int(s)
            return default
        except Exception:
            return default
    
    def load_tasks_from_csv(self, filename: str, num_tasks: Optional[int] = None) -> List[Task]:
        """Load tasks from CSV file (datagenie style)"""
        file_path = self.base_path / filename
        
        if not file_path.exists():
            raise FileNotFoundError(f"Task file not found: {file_path}")
        
        tasks = []
        with open(file_path, 'r', encoding='utf-8') as csv_file:
            reader = csv.DictReader(csv_file, skipinitialspace=True)
            
            for i, row in enumerate(reader):
                if num_tasks and i >= num_tasks:
                    break
                
                task_id = f"{filename}_{i+1}"
                task = Task.from_csv_row(row, task_id)
                tasks.append(task)
        
        logger.info(f"Loaded {len(tasks)} tasks from {filename}")
        return tasks
    
    def load_tasks_from_json(self, filename: str) -> List[Task]:
        """Load tasks from JSON file"""
        file_path = self.base_path / filename
        
        if not file_path.exists():
            raise FileNotFoundError(f"Task file not found: {file_path}")
        
        with open(file_path, 'r', encoding='utf-8') as json_file:
            data = json.load(json_file)
        
        tasks = []
        if isinstance(data, list):
            for i, item in enumerate(data):
                task_id = item.get('id', f"{filename}_{i+1}")
                task = Task(id=task_id, **item)
                tasks.append(task)
        else:
            task_id = data.get('id', f"{filename}_1")
            task = Task(id=task_id, **data)
            tasks.append(task)
        
        logger.info(f"Loaded {len(tasks)} tasks from {filename}")
        return tasks
    
    def load_textbook_chunks_from_csv(self, filename: str, num_chunks: Optional[int] = None) -> List[TextbookChunk]:
        """Load textbook chunks from CSV file (HuggingFace dataset format) with pragmatic column resolution"""
        file_path = self.base_path / filename
        
        if not file_path.exists():
            raise FileNotFoundError(f"Textbook chunks file not found: {file_path}")
        
        chunks = []
        with open(file_path, 'r', encoding='utf-8') as csv_file:
            reader = csv.DictReader(csv_file, skipinitialspace=True)
            available = reader.fieldnames or []
            text_col    = self._resolve_column(available, 'text',    ['content','body','passage','context','raw_text'])
            subject_col = self._resolve_column(available, 'subject', ['course','topic','subject_name'])
            grade_col   = self._resolve_column(available, 'grade',   ['class','std','level','grade_level'])
            source_col  = self._resolve_column(available, 'source',  ['book','dataset','corpus'])
            id_col      = self._resolve_column(available, 'id',      ['_id','uuid','doc_id','row_id'])
            chap_idx_col= self._resolve_column(available, 'chapter_index', ['seg_index','segment_index','chunk_index','page_index'])
            chap_title_col = self._resolve_column(available, 'chapter_title', ['chapter','unit','unit_title','lesson','topic_title','heading'])
            logger.info(f"CSV column mapping: id={id_col}, text={text_col}, subject={subject_col}, grade={grade_col}, chapter_title={chap_title_col}, chapter_index={chap_idx_col}, source={source_col}")
            for i, row in enumerate(reader):
                if num_chunks and i >= num_chunks:
                    break
                chunk = TextbookChunk(
                    id=row.get(id_col) or f"chunk_{i+1}",
                    text=row.get(text_col, '') if text_col else '',
                    source=row.get(source_col, '') if source_col else '',
                    subject=row.get(subject_col, '') if subject_col else '',
                    grade=self._coerce_int(row.get(grade_col)) if grade_col else 0,
                    chapter_index=self._coerce_int(row.get(chap_idx_col)) if chap_idx_col else 0,
                    chapter_title=row.get(chap_title_col, '') if chap_title_col else '',
                    metadata={'source': 'csv', 'original_row': row}
                )
                chunks.append(chunk)
        
        logger.info(f"Loaded {len(chunks)} textbook chunks from {filename}")
        return chunks
    
    def load_textbook_chunks_from_huggingface(
        self,
        dataset_name: Optional[str] = None,
        num_chunks: Optional[int] = None,
        subjects: Optional[List[str]] = None,
        grades: Optional[List[int]] = None,
        split: str = "train",
        schema: Optional[Dict[str, str]] = None
    ) -> List[TextbookChunk]:
        """Load textbook chunks from HuggingFace dataset, using only the provided schema mapping (no fallback)."""
        if not dataset_name:
            raise ValueError("dataset_name is required")
        try:
            from datasets import load_dataset
        except ImportError:
            raise ImportError("HuggingFace datasets library not installed. Install with: pip install datasets")

        logger.info(f"Loading HuggingFace dataset: {dataset_name} (split={split})")

        dataset = load_dataset(dataset_name, split=split)
        available = list(dataset.column_names)
        mapping = dict(schema or {})
        logger.info(f"HF column mapping (config-only): {mapping} | available={available}")

        def col(row: Dict[str, Any], key: str, default: Any = None):
            src = mapping.get(key)
            return row.get(src, default) if src else default

        chunks: List[TextbookChunk] = []
        for i, row in enumerate(dataset):
            if num_chunks and len(chunks) >= num_chunks:
                break
            try:
                text_val = str(col(row, 'text', ''))
                subject_val = str(col(row, 'subject', ''))
                grade_val_raw = col(row, 'grade', 0)
                grade_val = self._coerce_int(grade_val_raw, 0)
                chapter_title_val = str(col(row, 'chapter_title', ''))
                chapter_index_val = self._coerce_int(col(row, 'chapter_index', 0), 0)
                source_val = str(col(row, 'source', ''))
                id_val = str(col(row, 'id', f"chunk_{i+1}"))
                if subjects and subject_val not in subjects:
                    continue
                if grades and grade_val not in grades:
                    continue
                chunk = TextbookChunk(
                    id=id_val,
                    text=text_val,
                    source=source_val,
                    subject=subject_val,
                    grade=grade_val,
                    chapter_index=chapter_index_val,
                    chapter_title=chapter_title_val,
                    metadata={
                        'dataset_source': dataset_name,
                        'dataset_split': split,
                        'schema_mapping': mapping,
                        'available_columns': available
                    }
                )
            except Exception as e:
                logger.warning(f"Skipping row {i} due to schema conversion error: {e}")
                continue
            chunks.append(chunk)
        logger.info(f"Loaded {len(chunks)} textbook chunks from HuggingFace dataset")
        return chunks
    
    def filter_chunks_by_criteria(
        self, 
        chunks: List[TextbookChunk], 
        subject: Optional[str] = None,
        grade: Optional[int] = None,
        chapter_index: Optional[int] = None,
        source: Optional[str] = None
    ) -> List[TextbookChunk]:
        """Filter chunks by specific criteria"""
        filtered_chunks = chunks
        
        if subject:
            filtered_chunks = [c for c in filtered_chunks if c.subject == subject]
        
        if grade:
            filtered_chunks = [c for c in filtered_chunks if c.grade == grade]
        
        if chapter_index:
            filtered_chunks = [c for c in filtered_chunks if c.chapter_index == chapter_index]
        
        if source:
            filtered_chunks = [c for c in filtered_chunks if c.source == source]
        
        logger.info(f"Filtered {len(chunks)} chunks to {len(filtered_chunks)} chunks")
        return filtered_chunks
    
    def validate_tasks(self, tasks: List[Task]) -> Dict[str, List[str]]:
        """Validate tasks and return errors"""
        errors = {}
        
        for task in tasks:
            task_errors = []
            if not task.description:
                task_errors.append("Task description is required")
            if not task.category:
                task_errors.append("Task category is required")
            
            if task_errors:
                errors[task.id] = task_errors
        
        if errors:
            logger.warning(f"Found validation errors in {len(errors)} tasks")
        else:
            logger.info("All tasks passed validation")
        
        return errors
    
    def create_task_batch(
        self, 
        tasks: List[Task], 
        workflow_config: Optional[WorkflowConfig] = None,
        metadata: Optional[Dict[str, Any]] = None
    ) -> TaskBatch:
        """Create a batch of tasks for execution"""
        batch = TaskBatch(
            tasks=tasks,
            workflow_config=workflow_config,
            metadata=metadata or {}
        )
        
        logger.info(f"Created task batch with {len(tasks)} tasks")
        return batch
    
    def create_textbook_qa_batch(
        self, 
        chunks: List[TextbookChunk],
        max_turns: int = 5,
        quality_threshold: float = 0.7
    ) -> TaskBatch:
        """Create a batch for textbook QA generation"""
        # Convert chunks to tasks
        tasks = []
        for chunk in chunks:
            task = Task(
                id=f"textbook_qa_{chunk.id}",
                category="textbook_qa",
                subcategory=chunk.subject,
                description=f"Generate QA conversation for {chunk.subject} grade {chunk.grade} chapter {chunk.chapter_index}",
                metadata={
                    "chunk_id": chunk.id,
                    "source": chunk.source,
                    "subject": chunk.subject,
                    "grade": chunk.grade,
                    "chapter_index": chunk.chapter_index,
                    "chapter_title": chunk.chapter_title
                },
                context={"chunk": chunk.to_workflow_inputs()}
            )
            tasks.append(task)
        
        # Create workflow config
        workflow_config = WorkflowConfig(
            workflow_type="textbook_qa",
            orchestration_mode="dependency",
            environment="textbook_qa",
            task="Generate high-quality textbook QA conversations",
            max_turns=max_turns,
            quality_threshold=quality_threshold,
            steps=[
                WorkflowStepConfig(
                    name="questioner",
                    environment_name="textbook_qa",
                    tools=["question_generation_tool"],
                    subtask="Generate initial question based on topic and grade level",
                    dependencies=[],
                    run_full_episode=False
                ),
                WorkflowStepConfig(
                    name="rephraser",
                    environment_name="textbook_qa",
                    tools=["evidence_retrieval_tool"],
                    subtask="Retrieve evidence from source text for the question",
                    dependencies=[],
                    run_full_episode=False
                ),
                WorkflowStepConfig(
                    name="answerer",
                    environment_name="textbook_qa",
                    tools=["answer_generation_tool"],
                    subtask="Generate answer using provided rationale",
                    dependencies=["reader"],
                    run_full_episode=False
                ),
                WorkflowStepConfig(
                    name="judge",
                    environment_name="textbook_qa",
                    tools=["quality_evaluation_tool"],
                    subtask="Evaluate answer quality and coherence",
                    dependencies=["questioner", "reader", "answerer"],
                    run_full_episode=False
                ),
                WorkflowStepConfig(
                    name="coordinator",
                    environment_name="textbook_qa",
                    tools=["conversation_management_tool"],
                    subtask="Orchestrate conversation and format output",
                    dependencies=["questioner", "reader", "answerer", "judge"],
                    run_full_episode=False
                )
            ]
        )
        
        batch = TaskBatch(
            tasks=tasks,
            workflow_config=workflow_config,
            metadata={
                "batch_type": "textbook_qa",
                "num_chunks": len(chunks),
                "max_turns": max_turns,
                "quality_threshold": quality_threshold
            }
        )
        
        logger.info(f"Created textbook QA batch with {len(tasks)} tasks")
        return batch
    
    def create_function_calling_batch(
        self, 
        tasks: List[Task],
        max_iterations: int = 3
    ) -> TaskBatch:
        """Create a batch for function calling dataset generation"""
        # Create workflow config
        workflow_config = WorkflowConfig(
            workflow_type="function_calling",
            orchestration_mode="dependency",
            environment="function_calling",
            task="Generate function calling dataset",
            max_turns=max_iterations,
            steps=[
                WorkflowStepConfig(
                    name="function_signature_generator",
                    environment_name="function_calling",
                    tools=["signature_generation_tool"],
                    subtask="Generate function signatures based on task description",
                    dependencies=[],
                    run_full_episode=False
                ),
                WorkflowStepConfig(
                    name="function_caller",
                    environment_name="function_calling",
                    tools=["function_calling_tool"],
                    subtask="Call functions using generated signatures",
                    dependencies=["function_signature_generator"],
                    run_full_episode=False
                )
            ]
        )
        
        batch = TaskBatch(
            tasks=tasks,
            workflow_config=workflow_config,
            metadata={
                "batch_type": "function_calling",
                "num_tasks": len(tasks),
                "max_iterations": max_iterations
            }
        )
        
        logger.info(f"Created function calling batch with {len(tasks)} tasks")
        return batch
    
    def save_execution_result(self, result: TaskExecutionResult):
        """Save task execution result to history"""
        self.execution_history.append(result)
        
        # Save to file
        history_file = self.base_path / "execution_history.jsonl"
        with open(history_file, 'a', encoding='utf-8') as f:
            f.write(result.model_dump_json() + '\n')
    
    def get_execution_history(
        self, 
        task_id: Optional[str] = None,
        status: Optional[str] = None,
        limit: Optional[int] = None
    ) -> List[TaskExecutionResult]:
        """Get execution history with optional filtering"""
        history = self.execution_history
        
        if task_id:
            history = [h for h in history if h.task_id == task_id]
        
        if status:
            history = [h for h in history if h.status == status]
        
        if limit:
            history = history[-limit:]
        
        return history
    
    def get_execution_stats(self) -> Dict[str, Any]:
        """Get execution statistics"""
        if not self.execution_history:
            return {
                "total_executions": 0,
                "successful_executions": 0,
                "failed_executions": 0,
                "average_execution_time": 0.0
            }
        
        total = len(self.execution_history)
        successful = len([h for h in self.execution_history if h.status == "completed"])
        failed = total - successful
        avg_time = sum(h.execution_time for h in self.execution_history) / total
        
        return {
            "total_executions": total,
            "successful_executions": successful,
            "failed_executions": failed,
            "average_execution_time": avg_time,
            "success_rate": successful / total if total > 0 else 0.0
        }
    
    def export_batch_to_json(self, batch: TaskBatch, filename: str):
        """Export task batch to JSON file"""
        file_path = self.base_path / filename
        with open(file_path, 'w', encoding='utf-8') as f:
            json.dump(batch.model_dump(), f, indent=2, ensure_ascii=False)
        
        logger.info(f"Exported batch to {file_path}")
    
    def load_batch_from_json(self, filename: str) -> TaskBatch:
        """Load task batch from JSON file"""
        file_path = self.base_path / filename
        
        if not file_path.exists():
            raise FileNotFoundError(f"Batch file not found: {file_path}")
        
        with open(file_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
        
        batch = TaskBatch(**data)
        logger.info(f"Loaded batch with {len(batch.tasks)} tasks from {filename}")
        return batch
