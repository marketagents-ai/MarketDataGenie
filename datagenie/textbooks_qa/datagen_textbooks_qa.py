"""
Textbook QA Data Generation Pipeline

This script orchestrates the generation of QA datasets from textbook chunks
using the HuggingFace dataset format and the MarketAgents workflow system.

Schema (prompt variables):
- text: content chunk
- subject: subject label
- grade: class/grade
- chapter_title: chapter/unit name

This pipeline expects a YAML config that provides the dataset path (HuggingFace name + split) and a schema mapping for the four prompt variables: text, subject, grade, chapter_title.
"""


import json
import ast
import logging
import asyncio
from typing import Dict, List, Any, Optional, Union
from datetime import datetime
from pathlib import Path
import re

# Additional imports for pydantic tool schema
from pydantic import BaseModel, Field
from minference.lite.models import StructuredTool

# YAML config support
try:
    import yaml
except ImportError:
    yaml = None



# MarketAgents imports
from market_agents.agents.market_agent import MarketAgent
from market_agents.environments.mechanisms.mcp_server import MCPServerEnvironment
from market_agents.environments.environment import MultiAgentEnvironment
from market_agents.workflows.market_agent_workflow import Workflow, WorkflowStep

from market_agents.task_manager import (
    TaskManager,
    TextbookChunk
)


# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Pydantic model for judge metrics (all fields REQUIRED)
class JudgeMetrics(BaseModel):
    grounded_in_context: float = Field(..., ge=0.0, le=10.0, description="Whether answer is grounded in the context of the provided text")
    context_query_relevance: float = Field(..., ge=0.0, le=10.0, description="How relevant the query is to the provided text")
    answer_query_relevance: float = Field(..., ge=0.0, le=10.0, description="How relevant the answer is to the provided text")
    factual_correctness: float = Field(..., ge=0.0, le=10.0, description="How factually correct is the answer?")
    language_quality: float = Field(..., ge=0.0, le=10.0, description="Grammar and language quality")
    feedback: str = Field(..., description="Short feedback on the textbook Q/A pair")

class TextbookQAGenerator:
    """
    Main orchestrator for textbook QA data generation.
    """

    def __init__(
        self,
        dataset_name: Optional[str] = None,
        output_dir: str = "outputs/textbook_qa",
        num_samples: Optional[int] = None,
        subjects: Optional[List[str]] = None,
        grades: Optional[List[int]] = None,
        config_path: Optional[str] = None,
        schema_mapping: Optional[Dict[str, str]] = None
    ):
        self.dataset_name = dataset_name
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.num_samples = num_samples
        self.subjects = subjects
        self.grades = grades

        # Config-driven overrides
        self.schema_mapping: Dict[str, str] = schema_mapping or {}
        self.dataset_split: str = "train"

        if config_path:
            if yaml is None:
                raise ImportError("PyYAML is required to use config files. Install with: pip install pyyaml")
            config_path_p = Path(config_path)
            if not config_path_p.exists():
                raise FileNotFoundError(f"Config file not found: {config_path}")
            with open(config_path_p, 'r', encoding='utf-8') as f:
                cfg = yaml.safe_load(f) or {}

            # Top-level keys
            ds_cfg = (cfg.get('dataset') or {})
            schema_cfg = (cfg.get('schema') or {})
            gen_cfg = (cfg.get('generation') or {})

            # Dataset settings
            self.dataset_name = ds_cfg.get('name', self.dataset_name)
            self.dataset_split = ds_cfg.get('split', self.dataset_split)

            # Filters and limits
            self.num_samples = gen_cfg.get('num_samples', self.num_samples)
            self.subjects = gen_cfg.get('subjects', self.subjects)
            self.grades = gen_cfg.get('grades', self.grades)

            # Parallel rollout controls
            self.batch_size = int(gen_cfg.get('batch_size', 1))
            self.per_task_timeout = int(gen_cfg.get('per_task_timeout', 300))

            # Output dir
            self.output_dir = Path(cfg.get('output_dir', output_dir))
            self.output_dir.mkdir(parents=True, exist_ok=True)

            # Schema mapping for prompt variables only: text, subject, grade, chapter_title
            self.schema_mapping.update(schema_cfg)

        if not self.dataset_name:
            raise ValueError("dataset_name must be provided either as an argument or via YAML config under dataset.name")

        # Ensure defaults for parallel rollout controls if not set
        if not hasattr(self, 'batch_size'):
            self.batch_size = 1
        if not hasattr(self, 'per_task_timeout'):
            self.per_task_timeout = 300

        # Initialize components
        self.task_manager = TaskManager()
        self.mcp_servers = self._setup_mcp_servers()

        # Results tracking
        self.results = []
        self.execution_stats = {}

    def _extract_qa_from_workflow(self, wf_result: Dict[str, Any]) -> Dict[str, Any]:
        """Extracts question/analysis/answer/judge/coordinator text from workflow_results."""
        def _get_step_content(step_results: List[Dict[str, Any]], step_id: str) -> str:
            for s in step_results or []:
                if s.get("step_id") == step_id and s.get("status") == "completed":
                    try:
                        return s.get("result", {}).get("content", "") or ""
                    except Exception:
                        return ""
            return ""

        step_results = (wf_result or {}).get("step_results", [])
        question = _get_step_content(step_results, "questioner")
        answer = _get_step_content(step_results, "answerer")
        judge_text = _get_step_content(step_results, "judge")

        # Feedback text variable for downstream use
        feedback_text = ""

        # Enhanced: Try JSON metrics first; compute overall score as average of 5 metrics
        metrics = None
        quality_score = None
        metric_keys = [
            "grounded_in_context",
            "context_query_relevance",
            "answer_query_relevance",
            "factual_correctness",
            "language_quality",
        ]

        if judge_text:
            # Case 1: tool already returned a dict
            if isinstance(judge_text, dict):
                metrics = judge_text
            # Case 2: string that might be JSON or a Python dict repr with single quotes
            elif isinstance(judge_text, str):
                parsed = None
                # First try strict JSON
                try:
                    parsed = json.loads(judge_text)
                except Exception:
                    # Then try a safe Python literal parse to handle single-quoted dicts
                    try:
                        parsed = ast.literal_eval(judge_text)
                    except Exception:
                        parsed = None
                if isinstance(parsed, dict):
                    metrics = parsed

            # Compute average if we have metrics
            if isinstance(metrics, dict):
                vals = []
                for k in metric_keys:
                    v = metrics.get(k)
                    try:
                        if v is not None:
                            vals.append(float(v))
                    except Exception:
                        continue
                if vals:
                    quality_score = sum(vals) / len(vals)
                feedback_text = str(metrics.get("feedback", ""))
            else:
                # Fallback: attempt regex extraction like "9/10" if some legacy judges return prose
                m = re.search(r"(\d+(?:\.\d+)?)\s*/\s*10", str(judge_text))
                if m:
                    try:
                        quality_score = float(m.group(1))
                    except Exception:
                        quality_score = None
                feedback_text = str(judge_text).strip()

        # Warn if any required metrics are missing
        if metrics is not None:
            missing = [k for k in metric_keys if k not in metrics]
            if missing:
                logger.warning(f"Judge metrics missing fields: {missing}")

        return {
            "question": question.strip(),
            "answer": answer.strip(),
            "feedback": feedback_text,
            "quality_score": quality_score,
            "metrics": metrics,
        }
    
    def _setup_mcp_servers(self) -> Dict[str, Union[MCPServerEnvironment, MultiAgentEnvironment]]:
        """Setup environments for textbook QA workflow"""
        # Create a simple chat environment for LLM-only workflows
        from market_agents.environments.mechanisms.chat import ChatMechanism
        from market_agents.environments.environment import MultiAgentEnvironment
        
        # Create a simple chat mechanism for QA generation
        chat_mechanism = ChatMechanism()
        
        # Create a default environment
        default_env = MultiAgentEnvironment(
            name="default",
            mechanism=chat_mechanism
        )
        
        # Return the simple chat environment
        return {"default": default_env}
    
    def load_huggingface_dataset(self) -> List[TextbookChunk]:
        """Load textbook chunks from HuggingFace dataset using optional schema mapping and split"""
        return self.task_manager.load_textbook_chunks_from_huggingface(
            dataset_name=self.dataset_name,
            num_chunks=self.num_samples,
            subjects=self.subjects,
            grades=self.grades,
            split=self.dataset_split,
            schema=self.schema_mapping
        )
    
    def create_textbook_qa_workflow(self) -> Workflow:
        """Create the textbook QA workflow with 5-step dependency chain"""

        # Create judge tool from Pydantic model
        judge_tool = StructuredTool.from_pydantic(
            model=JudgeMetrics,
            name="judge_metrics",
            description="Return JSON metrics for the educational content quality evaluation"
        )
        logger.info("Judge step uses StructuredTool; ensure the agent's LLM config supports tool use (e.g., ResponseFormat.auto_tools)")

        # Define workflow steps using chat environment
        steps = [
            WorkflowStep(
                name="questioner",
                environment_name="default",
                tools=[],
                subtask="""
                Generate an educational question based on the provided textbook chunk.
                - The question should be in the same language as the text chunk.
                - The query should be relevant to the subject, chapter and grade level.
                - The question should not make direct reference to the textbook or chapter used as reference.
                - The question should be drafted as if no reference material was provided.
                - For text chunks containing problem sets or example solutions such as in maths, recreate the complete question.
                - Provide the question directly without filler text such as Here is the generated question...

                Context:
                - Subject: {subject}
                - Grade: {grade}
                - Chapter: {chapter_title}
                - Text: {text}

                Generate a question that:
                1. Is appropriate for the grade level
                2. Tests understanding of the key concepts
                3. Can be answered using the provided text
                4. Is clear and unambiguous
                """,
                run_full_episode=False
            ),
            WorkflowStep(
                name="answerer",
                environment_name="default",
                tools=[],
                subtask="""
                Generate a comprehensive answer based on the provided text chunk from a chapter.
                - You may use the text as reference but you should answer as if you are the author of the textbook.
                - You should answer in the same language as the text query and text chunk.
                - The answer should be relevant to the subject, chapter and grade level.
                - The answer should not make direct reference to the textbook or chapter.
                - The answer should be drafted as if no reference material was provided while using the context from the text.
                - For problem sets, provide complete correct solutions.
                - Provide the answer directly without filler text such as Here is the generated answer...


                Context:
                - Subject: {subject}
                - Grade: {grade}
                - Chapter: {chapter_title}
                - Text: {text}

                Create an answer that:
                1. Demonstrates understanding of the key concepts
                2. Uses evidence from the text but no need to cite the source
                3. Is appropriate for the grade level
                4. Provides clear explanations
                5. Includes relevant examples

                Generate a sample educational answer that could be used to test student understanding.

                Here's the generated question based on the chapter chunk:
                - {questioner}
                """,
                run_full_episode=False
            ),
            WorkflowStep(
                name="judge",
                environment_name="default",
                tools=[judge_tool],
                subtask="""
                Evaluate the quality and educational value of the generated content.
                - Use the provided context and the previously generated question and answer (available in the chat history)
                - Produce structured metrics by calling the `judge_metrics` tool exactly once.

                Context:
                - Subject: {subject}
                - Grade: {grade}
                - Chapter: {chapter_title}
                - Text: {text}
                Generated Question: {questioner}
                Generated Answer: {answerer}

                Evaluation dimensions (0-10 each):
                - grounded_in_context
                - context_query_relevance
                - answer_query_relevance
                - factual_correctness
                - language_quality

                You MUST return all five metric values and `feedback` via the `judge_metrics` tool in a single call. All five metrics are required and must be numbers between 0 and 10.
                """,
                run_full_episode=False
            )
        ]

        # For now, let's use a simple workflow without MCP servers
        # We'll create a basic workflow that can execute the steps

        return Workflow(
            name="textbook_qa_workflow",
            task="""
            Generate high-quality educational QA conversations from provided textbook chunks.

            For each textbook chunk, create a complete QA conversation that:
            - Is educationally valuable and grade-appropriate
            - Tests understanding of key concepts
            - Uses evidence from the source text
            - Maintains high quality standards
            - Is suitable for educational applications

            Subject: {subject}
            Grade: {grade}
            Chapter: {chapter_title}
            Text: {text}
            """,
            steps=steps,
            mcp_servers=self.mcp_servers  # Use the chat environment we created
        )
    
    async def generate_qa_for_chunk(
        self,
        chunk: TextbookChunk,
        agent: MarketAgent
    ) -> Dict[str, Any]:
        """Generate QA conversation for a single textbook chunk using workflow"""

        logger.info(f"Generating QA for chunk {chunk.id}: {chunk.subject} - {chunk.chapter_title}")

        # Create workflow
        workflow = self.create_textbook_qa_workflow()

        # Prepare inputs
        inputs = chunk.to_workflow_inputs()

        try:
            # Execute workflow
            start_time = datetime.now()
            workflow_result = await workflow.execute(agent, inputs)
            execution_time = (datetime.now() - start_time).total_seconds()

            # Build structured qa_conversation from workflow outputs
            wf_dump = workflow_result.model_dump(mode='json')
            qa_struct = self._extract_qa_from_workflow(wf_dump)

            result = {
                "chunk_id": chunk.id,
                "subject": chunk.subject,
                "grade": chunk.grade,
                "chapter_index": chunk.chapter_index,
                "chapter_title": chunk.chapter_title,
                "source": chunk.source,
                "original_text": chunk.text,
                "qa_conversation": qa_struct,
                "workflow_results": wf_dump,
                "execution_time": execution_time,
                "timestamp": datetime.now().isoformat(),
                "metadata": chunk.metadata
            }

            logger.info(f"Successfully generated QA for chunk {chunk.id}")
            return result

        except Exception as e:
            logger.error(f"Failed to generate QA for chunk {chunk.id}: {e}")
            return {
                "chunk_id": chunk.id,
                "error": str(e),
                "timestamp": datetime.now().isoformat()
            }
    
    async def generate_dataset(
        self,
        agent: MarketAgent,
        max_workers: int = 1
    ) -> List[Dict[str, Any]]:
        """Generate QA dataset from all loaded chunks with bounded parallelism"""
        # Load chunks
        chunks = self.load_huggingface_dataset()
        # Hard cap to exactly the requested number of samples
        if self.num_samples is not None:
            chunks = chunks[: self.num_samples]
        total = len(chunks)
        workers = max(1, int(max_workers or getattr(self, 'batch_size', 1)))
        logger.info(f"Starting QA generation for {total} chunks (parallel batch_size={workers})")

        import asyncio
        from datetime import datetime
        from minference.lite.inference import InferenceOrchestrator
        from market_agents.agents.market_agent import MarketAgent as _MA

        sem = asyncio.Semaphore(workers)

        def spawn_agent() -> MarketAgent:
            llm_orch = InferenceOrchestrator()
            return _MA(
                name=getattr(agent, 'name', 'textbook_qa_worker'),
                persona=getattr(agent, 'persona', None),
                llm_config=getattr(agent, 'llm_config', None),
                llm_orchestrator=llm_orch,
                task=getattr(agent, 'task', ''),
                memory_enabled=getattr(agent, 'memory_enabled', False),
                verbose=getattr(agent, 'verbose', False)
            )

        async def process_one(idx: int, chunk) -> Dict[str, Any]:
            async with sem:
                local_agent = spawn_agent()
                try:
                    coro = self.generate_qa_for_chunk(chunk, local_agent)
                    timeout = max(1, int(getattr(self, 'per_task_timeout', 300)))
                    return await asyncio.wait_for(coro, timeout=timeout)
                except asyncio.TimeoutError:
                    logger.error(f"Timeout while generating QA for chunk {getattr(chunk, 'id', idx)} (> {getattr(self, 'per_task_timeout', 300)}s)")
                    return {"chunk_id": getattr(chunk, 'id', idx), "error": f"timeout > {getattr(self, 'per_task_timeout', 300)}s", "timestamp": datetime.now().isoformat()}
                except Exception as e:
                    logger.error(f"Failed to generate QA for chunk {getattr(chunk, 'id', idx)}: {e}")
                    return {"chunk_id": getattr(chunk, 'id', idx), "error": str(e), "timestamp": datetime.now().isoformat()}

        tasks = [asyncio.create_task(process_one(i, c)) for i, c in enumerate(chunks)]
        results = await asyncio.gather(*tasks)

        # Save final results
        self.results = results
        self.save_results(results)

        logger.info(f"Completed QA generation. Generated {len(results)} QA conversations")
        return results
    
    def save_intermediate_results(self, results: List[Dict[str, Any]], filename: str):
        """Save intermediate results to file"""
        filepath = self.output_dir / filename
        with open(filepath, 'w', encoding='utf-8') as f:
            json.dump(results, f, indent=2, ensure_ascii=False)
        logger.info(f"Saved intermediate results to {filepath}")
    
    def save_results(self, results: List[Dict[str, Any]]):
        """Save final results to file"""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        
        # Save full results
        full_results_file = self.output_dir / f"textbook_qa_results_{timestamp}.json"
        with open(full_results_file, 'w', encoding='utf-8') as f:
            json.dump(results, f, indent=2, ensure_ascii=False)
        
        # Save summary
        summary = {
            "total_chunks": len(results),
            "successful_generations": len([r for r in results if "error" not in r]),
            "failed_generations": len([r for r in results if "error" in r]),
            "subjects": list(set(r.get("subject", "") for r in results if "subject" in r)),
            "grades": list(set(r.get("grade", 0) for r in results if "grade" in r)),
            "generation_timestamp": timestamp,
            "execution_stats": self.execution_stats
        }
        
        summary_file = self.output_dir / f"textbook_qa_summary_{timestamp}.json"
        with open(summary_file, 'w', encoding='utf-8') as f:
            json.dump(summary, f, indent=2, ensure_ascii=False)
        
        logger.info(f"Saved results to {full_results_file}")
        logger.info(f"Saved summary to {summary_file}")
        
        # Also save in ShareGPT format for multi-turn QA datasets
        sharegpt_file = self.output_dir / f"textbook_qa_sharegpt_{timestamp}.json"
        self.save_sharegpt_dataset(results, sharegpt_file.name)
    
    def save_sharegpt_dataset(self, results: List[Dict[str, Any]], filename: str = None):
        """Save results in ShareGPT format for multi-turn QA datasets"""
        if not filename:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = f"textbook_qa_sharegpt_{timestamp}.json"

        filepath = self.output_dir / filename

        # Convert to ShareGPT format
        sharegpt_data = []
        for result in results:
            if 'qa_conversation' in result and isinstance(result['qa_conversation'], dict):
                qa = result['qa_conversation']
                question = qa.get('question') or '"प्रश्न प्रस्तावत हुन्छ?"'
                answer = qa.get('answer') or ''
                metrics = qa.get('metrics') or ''
                feedback = qa.get('feedback') or ''
                quality_score = qa.get('quality_score')

                # Build minimal two-turn conversation per ShareGPT: human asks, assistant answers.
                # Context text and other fields are stored in metadata.
                conversation = {
                    "id": result['chunk_id'],
                    "conversations": [
                        {
                            "from": "human",
                            "value": question
                        },
                        {
                            "from": "gpt",
                            "value": answer
                        }
                    ],
                    "metadata": {
                        "subject": result['subject'],
                        "grade": result['grade'],
                        "chapter_title": result['chapter_title'],
                        "source": result['source'],
                        "execution_time": result.get('execution_time', 0),
                        "timestamp": result.get('timestamp', ''),
                        "context_text": result.get('original_text', ''),
                        "feedback": feedback,
                        "llm_judge_metrics": metrics,
                        "average_score": quality_score
                    }
                }
                sharegpt_data.append(conversation)

        with open(filepath, 'w', encoding='utf-8') as f:
            json.dump(sharegpt_data, f, ensure_ascii=False, indent=2)

        logger.info(f"Saved ShareGPT dataset to {filepath}")
        return filepath
    
    def print_summary(self):
        """Print generation summary"""
        print("\n" + "="*50)
        print("TEXTBOOK QA GENERATION SUMMARY")
        print("="*50)
        print(f"Total chunks processed: {self.execution_stats.get('total_executions', 0)}")
        print(f"Successful generations: {self.execution_stats.get('successful_executions', 0)}")
        print(f"Failed generations: {self.execution_stats.get('failed_executions', 0)}")
        print(f"Success rate: {self.execution_stats.get('success_rate', 0):.2%}")
        print(f"Average execution time: {self.execution_stats.get('average_execution_time', 0):.2f}s")
        print(f"Output directory: {self.output_dir}")
        print("="*50)


