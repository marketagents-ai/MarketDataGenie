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
from datasets import load_dataset

import yaml
import re
import hashlib
import random
import tiktoken

# Additional imports for pydantic tool schema
from pydantic import BaseModel, Field
from minference.lite.models import StructuredTool

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


# Validated few shot examples to add to prompt from recently generated batch
def _pick_dynamic_example(batch_results: List[Dict[str, Any]]) -> Optional[Dict[str, str]]:
    """
    From a batch of `generate_qa_for_chunk` results, pick the highest-quality valid example.
    Returns dict with 'question' and 'answer' keys (and ignores if invalid).
    """
    candidates = []
    for r in batch_results:
        try:
            qa = r.get("qa_conversation") or {}
            q = (qa.get("question") or "").strip()
            a = (qa.get("answer") or "").strip()
            rp = (qa.get("rephrased_text") or "").strip()
            score = qa.get("quality_score")
            score_val = float(score) if score is not None else 0.0
            candidates.append((score_val, {"question": q, "answer": a}))
        except Exception:
            continue
    if not candidates:
        return None
    # Pick by highest score, fall back to first if ties
    candidates.sort(key=lambda x: x[0], reverse=True)
    return candidates[0][1]

# Pydantic model for judge metrics (all fields REQUIRED)
class JudgeMetrics(BaseModel):
    grounded_in_context: float = Field(..., ge=0.0, le=10.0, description="Whether answer is grounded in the context of the provided text")
    context_query_relevance: float = Field(..., ge=0.0, le=10.0, description="How relevant the query is to the provided text")
    answer_query_relevance: float = Field(..., ge=0.0, le=10.0, description="How relevant the answer is to the provided text")
    factual_correctness: float = Field(..., ge=0.0, le=10.0, description="How factually correct is the answer?")
    language_quality: float = Field(..., ge=0.0, le=10.0, description="Grammar and language quality")

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

        # Optional: previously generated HF dataset to skip duplicates
        self.generated_dataset_name: Optional[str] = None
        self.generated_dataset_split: str = "train"

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

            # Previously generated dataset (for dedup/filtering)
            gen_ds_cfg = (cfg.get('generated_dataset') or {})
            self.generated_dataset_name = gen_ds_cfg.get('name') or self.generated_dataset_name
            self.generated_dataset_split = gen_ds_cfg.get('split', self.generated_dataset_split)

            # Filters and limits
            self.num_samples = gen_cfg.get('num_samples', self.num_samples)
            self.subjects = gen_cfg.get('subjects', self.subjects)
            self.grades = gen_cfg.get('grades', self.grades)

            # Parallel rollout controls
            self.batch_size = int(gen_cfg.get('batch_size', 1))
            self.per_task_timeout = int(gen_cfg.get('per_task_timeout', 300))

            # Minimal retry-until-complete controller
            self.retry_until_complete: bool = bool(gen_cfg.get('retry_until_complete', False))
            self.completed_state_file: str = str(gen_cfg.get('completed_state_file', str(Path(self.output_dir) / "completed_state.json")))

            # Tokenization & truncation controls
            self.model_encoding: str = str(gen_cfg.get("model_encoding", "cl100k_base"))
            # Hard token cap for the raw input text fed into the workflow (rephraser step)
            self.max_input_tokens: int = int(gen_cfg.get("max_input_tokens", 12000))

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
        if not hasattr(self, 'retry_until_complete'):
            self.retry_until_complete = False
        if not hasattr(self, 'completed_state_file'):
            self.completed_state_file = str(Path(self.output_dir) / "completed_state.json")

        # Ensure safe defaults for tokenization/truncation controls
        if not hasattr(self, "model_encoding"):
            self.model_encoding = "cl100k_base"
        if not hasattr(self, "max_input_tokens"):
            self.max_input_tokens = 12000

        # Initialize components
        self.task_manager = TaskManager()
        self.mcp_servers = self._setup_mcp_servers()

        # Results tracking
        self.results = []
        self.execution_stats = {}

        # Per-run bookkeeping for incremental saves
        self.run_timestamp: Optional[str] = None
        self.results_jsonl_path: Optional[Path] = None
        self.sharegpt_jsonl_path: Optional[Path] = None
        self._io_lock: Optional[asyncio.Lock] = None

        # Optional: few-shot example for question/answer style guidance
        self.example_qa: Optional[Dict[str, str]] = None
        try:
            example_path = Path("configs/example.json")
            if example_path.exists():
                with open(example_path, "r", encoding="utf-8") as ef:
                    data = json.load(ef) or {}
                q = (data.get("question") or "").strip()
                a = (data.get("answer") or "").strip()
                if q and a:
                    self.example_qa = {"question": q, "answer": a}
                    logger.info("Loaded example Q/A from configs/example.json for few-shot guidance.")
                else:
                    logger.warning("configs/example.json found but missing non-empty 'question'/'answer'. Ignoring.")
        except Exception as ex:
            logger.warning(f"Failed to load configs/example.json: {ex}")

        # Optional: banned phrases (chapter-anchoring leaks) from YAML or JSON
        self.banned_phrases: List[str] = []
        try:
            banned_yml = Path("configs/banned_phrases.yml")
            banned_json = Path("configs/banned_phrases.json")
            phrases: List[str] = []
            if banned_yml.exists() and yaml is not None:
                with open(banned_yml, "r", encoding="utf-8") as bf:
                    data = yaml.safe_load(bf) or {}
                phrases = list(data.get("banned_phrases") or [])
            elif banned_json.exists():
                with open(banned_json, "r", encoding="utf-8") as bf:
                    data = json.load(bf) or {}
                phrases = list(data.get("banned_phrases") or [])
            # normalize + dedupe
            norm = []
            seen = set()
            for p in phrases:
                if not isinstance(p, str):
                    continue
                s = p.strip()
                if not s or s in seen:
                    continue
                seen.add(s)
                norm.append(s)
            self.banned_phrases = norm
            if self.banned_phrases:
                logger.info(f"Loaded {len(self.banned_phrases)} banned phrases for leakage prevention.")
        except Exception as ex:
            logger.warning(f"Failed to load banned phrases: {ex}")

    @staticmethod
    def _normalize_for_hash(text: str) -> str:
        # Minimal, language-agnostic normalization: trim, collapse whitespace
        t = (text or "").strip()
        t = re.sub(r"\s+", " ", t)
        return t

    @staticmethod
    def _hash_text(text: str) -> str:
        norm = TextbookQAGenerator._normalize_for_hash(text)
        return hashlib.sha256(norm.encode("utf-8")).hexdigest()

    def _get_encoder(self):
        """
        Return a tiktoken encoding. Falls back to cl100k_base if model-specific encoding is unavailable.
        If tiktoken is missing, returns None and token functions will degrade gracefully.
        """
        if tiktoken is None:
            logger.warning("tiktoken not installed; token counting will be disabled.")
            return None
        try:
            return tiktoken.get_encoding(self.model_encoding)
        except Exception:
            try:
                return tiktoken.get_encoding("cl100k_base")
            except Exception:
                logger.warning("Failed to acquire tiktoken encoder; token counting disabled.")
                return None

    def _count_tokens(self, text: str) -> int:
        enc = self._get_encoder()
        if enc is None:
            # Rough fallback: 4 chars per token heuristic
            return max(1, (len(text or "") // 4))
        try:
            return len(enc.encode(text or ""))
        except Exception:
            return max(1, (len(text or "") // 4))

    def _truncate_by_tokens(self, text: str, max_tokens: int) -> str:
        """
        Truncate a string to at most `max_tokens` tokens using tiktoken.
        If tiktoken is unavailable, fall back to a rough character cut.
        """
        if not isinstance(text, str):
            return ""
        if max_tokens <= 0:
            return ""
        enc = self._get_encoder()
        if enc is None:
            approx_chars = max_tokens * 4
            return text[:approx_chars]
        toks = enc.encode(text)
        if len(toks) <= max_tokens:
            return text
        truncated = enc.decode(toks[:max_tokens])
        return truncated

    def _load_existing_hashes(self) -> Optional[set]:
        """
        Load hashes of context_text from the previously generated HF dataset.
        Returns a set of sha256 hashes, or None if not configured/available.
        """
        if not self.generated_dataset_name:
            return None
        try:
            logger.info(f"Loading existing generated dataset for dedup: {self.generated_dataset_name} [{self.generated_dataset_split}]")
            # Stream to avoid loading entire dataset into memory
            ds_iterable = load_dataset(self.generated_dataset_name, split=self.generated_dataset_split, streaming=True)
            seen = set()
            count = 0
            for ex in ds_iterable:
                ctx = ex.get("context_text") or ""
                if ctx:
                    h = self._hash_text(ctx)
                    seen.add(h)
                    count += 1
            logger.info(f"Prepared {len(seen)} existing-context hashes from generated dataset (scanned {count} rows).")
            return seen
        except Exception as e:
            logger.warning(f"Failed to load existing generated dataset for dedup: {e}")
            return None

    def _ensure_run_files(self):
        """Ensure per-run jsonl file paths are initialized and files exist."""
        if not self.run_timestamp:
            self.run_timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        if not self.results_jsonl_path:
            self.results_jsonl_path = self.output_dir / f"textbook_qa_results_{self.run_timestamp}.jsonl"
        if not self.sharegpt_jsonl_path:
            self.sharegpt_jsonl_path = self.output_dir / f"textbook_qa_sharegpt_{self.run_timestamp}.jsonl"
        # Touch files
        self.results_jsonl_path.parent.mkdir(parents=True, exist_ok=True)
        self.results_jsonl_path.touch(exist_ok=True)
        self.sharegpt_jsonl_path.touch(exist_ok=True)
        if self._io_lock is None:
            self._io_lock = asyncio.Lock()

    def _completed_state_path(self) -> Path:
        return Path(self.completed_state_file)

    def _done_key(self, chunk_id: str, chapter_title: Optional[str]) -> str:
        base = f"{chunk_id}||{(chapter_title or '').strip()}"
        return hashlib.sha1(base.encode('utf-8')).hexdigest()

    def _load_completed(self) -> set:
        p = self._completed_state_path()
        try:
            if p.exists():
                with open(p, 'r', encoding='utf-8') as f:
                    data = json.load(f)
                if isinstance(data, list):
                    return set(data)
                if isinstance(data, set):
                    return data
        except Exception as e:
            logger.warning(f"Failed to load completed state from {p}: {e}")
        return set()

    def _save_completed(self, completed: set):
        p = self._completed_state_path()
        try:
            p.parent.mkdir(parents=True, exist_ok=True)
            with open(p, 'w', encoding='utf-8') as f:
                json.dump(sorted(list(completed)), f, ensure_ascii=False, indent=2)
        except Exception as e:
            logger.error(f"Failed to save completed state to {p}: {e}")

    async def _append_jsonl(self, path: Path, obj: Dict[str, Any]):
        """Append a single JSON object as a line to a JSONL file with async lock to avoid interleaving."""
        if self._io_lock is None:
            self._io_lock = asyncio.Lock()
        line = json.dumps(obj, ensure_ascii=False)
        async with self._io_lock:
            with open(path, 'a', encoding='utf-8') as f:
                f.write(line)
                f.write("\n")

    def _to_sharegpt_item(self, result: Dict[str, Any]) -> Dict[str, Any]:
        """Convert a per-chunk result to one ShareGPT item, mirroring save_sharegpt_dataset formatting."""
        qa = result.get('qa_conversation') or {}
        question = qa.get('question') or ''
        answer = qa.get('answer') or ''
        rephrased_text = qa.get('rephrased_text') or ''
        # Populate nulls when judge didn't run
        metrics = qa.get('metrics') if qa.get('metrics') is not None else None
        feedback = qa.get('feedback') or ''
        quality_score = qa.get('quality_score') if qa.get('quality_score') is not None else None
        item = {
            "id": result['chunk_id'],
            "conversations": [
                {"from": "human", "value": question},
                {"from": "gpt", "value": answer}
            ],
            "metadata": {
                "subject": result.get('subject'),
                "grade": result.get('grade'),
                "chapter_title": result.get("chapter_title", ""),
                "source": (result.get('source') or (result.get('metadata', {}) or {}).get('source') or ""),
                "execution_time": result.get('execution_time', 0),
                "timestamp": result.get('timestamp', ''),
                "context_text": result.get('original_text', ''),
                "rephrased_text": rephrased_text,
                "feedback": feedback,
                "llm_judge_metrics": metrics,
                "average_score": quality_score
            }
        }
        return item

    def _is_valid_sharegpt(self, item: Dict[str, Any]) -> bool:
        """
        Validate minimal requirements for ShareGPT saving.
        Rule: question, answer, and rephrased_text must be present and non-empty.
        Exception: judge metrics may be missing.
        """
        try:
            if not isinstance(item, dict):
                return False
            conv = item.get("conversations") or []
            if not isinstance(conv, list) or len(conv) < 2:
                return False
            q = (conv[0] or {}).get("value") if isinstance(conv[0], dict) else None
            a = (conv[1] or {}).get("value") if isinstance(conv[1], dict) else None
            if not isinstance(q, str) or not q.strip():
                return False
            if not isinstance(a, str) or not a.strip():
                return False
            md = item.get("metadata") or {}
            r = md.get("rephrased_text")
            if not isinstance(r, str) or not r.strip():
                return False
            # subject, grade, chapter_title, source should exist but we won't fail hard if absent here
            return True
        except Exception:
            return False

    def _is_frontmatter(self, title: Optional[str]) -> bool:
        t = (title or "").strip().lower()
        if not t:
            return False
        # Match "Chapter 0", "Chapter0", or localized/case-insensitive variants, and preface/frontmatter
        if re.search(r"\bchapter\s*0\b", t):
            return True
        if "preface" in t:
            return True
        if "frontmatter" in t or "front matter" in t:
            return True
        # Explicit exact string provided as example
        if t == "chapter 0: preface/frontmatter":
            return True
        return False

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
        rephrased_text = _get_step_content(step_results, "rephraser")
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
            "rephrased_text": rephrased_text.strip(),
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
        """Load textbook chunks from HuggingFace dataset using optional schema mapping and split, skipping front matter."""
        chunks = self.task_manager.load_textbook_chunks_from_huggingface(
            dataset_name=self.dataset_name,
            num_chunks=self.num_samples,
            subjects=self.subjects,
            grades=self.grades,
            split=self.dataset_split,
            schema=self.schema_mapping
        )
        # Filter out preface/frontmatter or Chapter 0
        before = len(chunks)
        chunks = [c for c in chunks if not self._is_frontmatter(getattr(c, "chapter_title", ""))]
        skipped = before - len(chunks)
        if skipped:
            logger.info(f"Skipped {skipped} front-matter chunks (Chapter 0 / Preface / Frontmatter).")
        # Optional: filter out chunks that already exist in previously generated dataset
        existing_hashes = self._load_existing_hashes()
        if existing_hashes:
            before_dedup = len(chunks)
            filtered = []
            for c in chunks:
                try:
                    h = self._hash_text(getattr(c, "text", "") or "")
                    if h not in existing_hashes:
                        filtered.append(c)
                except Exception:
                    # If hashing fails for any reason, keep the sample to avoid accidental drops
                    filtered.append(c)
            dropped = before_dedup - len(filtered)
            if dropped > 0:
                logger.info(f"Dedup filter: skipped {dropped} chunks already present in generated dataset.")
            chunks = filtered
        return chunks
    
    def create_textbook_qa_workflow(self) -> Workflow:
        """Create the textbook QA workflow with rephraser step."""

        # Create judge tool from Pydantic model
        judge_tool = StructuredTool.from_pydantic(
            model=JudgeMetrics,
            name="judge_metrics",
            description="Return JSON metrics for the educational content quality evaluation"
        )
        logger.info("Judge step uses StructuredTool; ensure the agent's LLM config supports tool use (e.g., ResponseFormat.auto_tools)")

        example_block = ""
        if getattr(self, "example_qa", None):
            ex_q = self.example_qa.get("question", "")
            ex_a = self.example_qa.get("answer", "")
            example_block = f"""
                Few-shot style guidance (do NOT copy; only mirror clarity and structure):
                - Example Question: {ex_q}
                - Example Answer: {ex_a}
            """.strip()

        banned_block = ""
        if getattr(self, "banned_phrases", None):
            # Render as a comma-separated inline list to keep prompts compact
            banned_inline = ", ".join(self.banned_phrases)
            banned_block = (
                "Strictly avoid references to specific chapters/lessons/units and do NOT use any of these phrases: "
                f"{banned_inline}."
            )

        steps = [
            WorkflowStep(
                name="rephraser",
                environment_name="default",
                tools=[],
                subtask=f"""
                Rephrase the provided textbook chunk in an word for word, information-dense, pedagogically clear and standalone textbook content.

                Guidelines for rephrasing:
                - Paraphrase in the same language (with interleaved English) as the original text.
                - Preserve only the factual content from the original text without adding external facts or hallucinations.
                - Do not mention paricular chapter number, title or author and focus on the educational content.
                - Increase information density without missing important facts: remove boilerplate, redundancies, asides etc.
                - Maintain the original language of the text chunk and keep terminology consistent with the subject and grade.
                - Align to instructional and conversational use: write in a crisp, explanatory tone that is easy to ask/answer questions from.
                - Define key terms and provide examples; keep symbols, formulas, and units intact.
                - Avoid references to diagrams or images such as in maths and sciences since this is text only rephrasing.
                - Fix any OCR junk by correcting the details, math equations, notations or the text in original language.
                - The text chunk is split from the whole chapter so make the rephrased content as coherent as possible.
                - Directly output ONLY the rephrased text without opening text such as: Here is the rephrased...
                {banned_block if banned_block else ""}

                Context:
                - Subject: {{subject}}
                - Grade: {{grade}}
                - Chapter: {{chapter_title}}
                - Original Text: {{text}}

                Return rephrased text in the same langauge with high quality educational content:
                """,
                run_full_episode=False
            ),
            WorkflowStep(
                name="questioner",
                environment_name="default",
                tools=[],
                subtask=f"""
                Generate a problem set or exam question based on the REPHRASED summary of textbook chapter chunk.
                - Use the rephrased text as the sole knowledge source to generate a closed book exam question.
                - The question must be answerable even without any additional context.
                - The question should be in the same language as the rephrased text.
                - The query should be relevant to the topics and informational content only.
                - Do not metion chapter number, title, author, grade and subject in your question.
                - For fictitious text such as chunk of story, poems etc, do not ask about the characters or the exact content.
                - For maths and problem set chunks, recreate a complete solvable question with necessary equations and details.
                - Questions should not depend on a diagrams such as in sets, geometry, organic chemistry etc.
                - Directly start with the question without any preamble.
                - Do not answer the question in this step.
                {example_block if example_block else ""}
                {banned_block if banned_block else ""}

                Context:
                - Subject: {{subject}}
                - Grade: {{grade}}
                - Chapter: {{chapter_title}}
                - Rephrased Text: {{rephraser}}

                Generate a question that:
                1. Is appropriate for the grade level
                2. Tests understanding of the key concepts
                3. Can be answered using the rephrased text
                4. Is clear and unambiguous
                5. Can be answered independent of a particular textbook/chapter

                Generate a complete question in the same langauge without the answer:
                """,
                run_full_episode=False
            ),
            WorkflowStep(
                name="answerer",
                environment_name="default",
                tools=[],
                subtask=f"""
                Provide a comprehensive answer to the generated question using ONLY the rephrased text as your knowledge source.
                - Answer in the same language as the question/rephrased text.
                - Do not reference that a rephrased text exists and write as if it's an exam question.
                - For problem sets, provide a complete correct solution.
                - Do not metion the chapter number or title, grade, author and subject in your answer.
                - Your answer should be as descriptive and informative as possible.
                - If the question is not answerable without additional context please reply with exact phrase: "Question Not Answerable"
                - Directly start with the answer; no preamble.
                {example_block if example_block else ""}
                {banned_block if banned_block else ""}

                Context:
                - Subject: {{subject}}
                - Grade: {{grade}}
                - Chapter: {{chapter_title}}
                - Rephrased Text: {{rephraser}}

                Create an answer that:
                1. Demonstrates understanding of the key concepts
                2. Is grounded in the rephrased text (no external facts)
                3. Is appropriate for the grade level
                4. Provides clear explanations
                5. Includes relevant examples or steps

                Here's the generated question you should answer correctly:
                - {{questioner}}
                """,
                run_full_episode=False
            ),
            WorkflowStep(
                name="judge",
                environment_name="default",
                tools=[judge_tool],
                subtask="""
                Evaluate the quality and educational value of the generated Q/A. 
                - Use the REPHRASED text as the primary grounding source.
                - You MUST call the `judge_metrics` tool exactly once, returning all required fields.
                - A score of 10 for any metric should only be awarded for perfect answers.
                - Please be critical while scoring and evaluate properly.

                Primary Grounding:
                - Rephrased Text: {rephraser}

                Generated Question: {questioner}
                Generated Answer: {answerer}

                Evaluation dimensions (0-10 each):
                - grounded_in_context
                - context_query_relevance
                - answer_query_relevance
                - factual_correctness
                - language_quality

                Return all five numeric metrics via a single `judge_metrics` tool call.
                """,
                run_full_episode=False
            )
        ]

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
            mcp_servers=self._setup_mcp_servers()
        )
    
    async def generate_qa_for_chunk(
        self,
        chunk: TextbookChunk,
        agent: MarketAgent
    ) -> Dict[str, Any]:
        """Generate QA conversation for a single textbook chunk using workflow"""

        logger.info(f"Generating QA for chunk {chunk.id}: {chunk.subject} - {chunk.chapter_title}")

        # Ensure no global tools bleed into steps; judge tool should be step-scoped only
        try:
            ct = getattr(agent, "chat_thread", None)
            if ct is not None:
                # Clear any lingering tools from previous runs/batches
                if hasattr(ct, "tools"):
                    ct.tools = []
                # Reset any step index bookkeeping if present
                if hasattr(ct, "workflow_step"):
                    ct.workflow_step = None
        except Exception:
            pass

        # Create workflow
        workflow = self.create_textbook_qa_workflow()

        # Prepare inputs
        inputs = chunk.to_workflow_inputs()

        # Token-aware truncation for the main text to avoid context overflows
        original_text = inputs.get("text", "") or ""
        orig_tokens = self._count_tokens(original_text)
        truncated_text = self._truncate_by_tokens(original_text, self.max_input_tokens)
        trunc_tokens = self._count_tokens(truncated_text)
        if trunc_tokens < orig_tokens:
            logger.info(
                f"Chunk {chunk.id}: text tokens {orig_tokens} > max_input_tokens={self.max_input_tokens}. "
                f"Truncated to {trunc_tokens} tokens."
            )
        else:
            logger.info(f"Chunk {chunk.id}: text tokens {orig_tokens} within limit {self.max_input_tokens}.")

        inputs["text"] = truncated_text

        # Minimal propagation of chapter_title
        inputs["chapter_title"] = getattr(chunk, "chapter_title", "")
        # Propagate source into workflow inputs
        inputs["source"] = getattr(chunk, "source", "")

        try:
            # Execute workflow
            start_time = datetime.now()
            workflow_result = await workflow.execute(agent, inputs)
            execution_time = (datetime.now() - start_time).total_seconds()

            # Build structured qa_conversation from workflow outputs
            wf_dump = workflow_result.model_dump(mode='json')
            qa_struct = self._extract_qa_from_workflow(wf_dump)

            try:
                # Post-hoc token diagnostics for generated fields
                q_tok = self._count_tokens(qa_struct.get("question", ""))
                a_tok = self._count_tokens(qa_struct.get("answer", ""))
                r_tok = self._count_tokens(qa_struct.get("rephrased_text", ""))
                logger.info(f"Chunk {chunk.id}: tokens — rephrased={r_tok}, question={q_tok}, answer={a_tok}")
            except Exception:
                pass

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
                "metadata": {
                    **(chunk.metadata or {}),
                    "chapter_title": chunk.chapter_title
                },
                "sharegpt_valid": False,
            }

            # Incremental persistence: append raw result and ShareGPT item as they complete
            try:
                self._ensure_run_files()
                sharegpt_item = self._to_sharegpt_item(result)
                await self._append_jsonl(self.results_jsonl_path, result)
                if self._is_valid_sharegpt(sharegpt_item):
                    await self._append_jsonl(self.sharegpt_jsonl_path, sharegpt_item)
                    logger.info(f"Appended JSONL for chunk {chunk.id} -> {self.results_jsonl_path.name}, {self.sharegpt_jsonl_path.name}")
                    result["sharegpt_valid"] = True
                else:
                    logger.warning(f"Invalid ShareGPT item for chunk {chunk.id}; skipping append (missing question/answer/rephrased_text).")
            except Exception as io_err:
                logger.error(f"Failed incremental save for chunk {chunk.id}: {io_err}")

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
        max_workers: int = 1,
        agent_factory: Optional[callable] = None
    ) -> List[Dict[str, Any]]:
        """Generate QA dataset from all loaded chunks with bounded parallelism"""
        # Initial load of chunks
        all_chunks = self.load_huggingface_dataset()
        if self.num_samples is not None:
            all_chunks = all_chunks[: self.num_samples]
        total = len(all_chunks)

        # Prepare per-run jsonl files for incremental saves
        self._ensure_run_files()
        logger.info(f"Incremental save paths: results={self.results_jsonl_path.name}, sharegpt={self.sharegpt_jsonl_path.name}")

        # Completed tracker
        completed_keys = self._load_completed()
        logger.info(f"Loaded {len(completed_keys)} completed keys from state.")

        # Helper to filter pending work
        def pending_chunks():
            out = []
            for c in all_chunks:
                key = self._done_key(getattr(c, 'id', ''), getattr(c, 'chapter_title', ''))
                if key not in completed_keys:
                    out.append(c)
            return out

        pass_idx = 0
        results: List[Dict[str, Any]] = []
        workers_cfg = max(1, int(max_workers or getattr(self, 'batch_size', 1)))

        while True:
            pending = pending_chunks()
            if not pending:
                logger.info("No pending chunks remain. All done.")
                break

            pass_idx += 1
            logger.info(f"Starting pass {pass_idx}: {len(pending)}/{total} pending (configured batch_size={workers_cfg})")

            workers = workers_cfg
            sem = asyncio.Semaphore(workers)

            # Agent factory
            def new_agent() -> MarketAgent:
                if agent_factory is not None:
                    return agent_factory()
                from minference.lite.inference import InferenceOrchestrator
                from market_agents.agents.market_agent import MarketAgent as _MA
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
                    local_agent = new_agent()
                    try:
                        # small jitter to avoid synchronized spikes
                        await asyncio.sleep(random.uniform(0.0, 0.15))
                        coro = self.generate_qa_for_chunk(chunk, local_agent)
                        timeout = max(1, int(getattr(self, 'per_task_timeout', 300)))
                        return await asyncio.wait_for(coro, timeout=timeout)
                    except asyncio.TimeoutError:
                        logger.error(f"Timeout while generating QA for chunk {getattr(chunk, 'id', idx)} (> {getattr(self, 'per_task_timeout', 300)}s)")
                        return {"chunk_id": getattr(chunk, 'id', idx), "error": f"timeout > {getattr(self, 'per_task_timeout', 300)}s", "timestamp": datetime.now().isoformat()}
                    except Exception as e:
                        logger.error(f"Failed to generate QA for chunk {getattr(chunk, 'id', idx)}: {e}")
                        return {"chunk_id": getattr(chunk, 'id', idx), "error": str(e), "timestamp": datetime.now().isoformat()}

            batch_results: List[Dict[str, Any]] = []
            # Process this pass in batches of size `workers`
            for start in range(0, len(pending), workers):
                end = min(start + workers, len(pending))
                batch_chunks = pending[start:end]
                logger.info(f"Pass {pass_idx} - processing chunks {start}..{end-1} of {len(pending)}")
                tasks = [asyncio.create_task(process_one(i, c)) for i, c in enumerate(batch_chunks, start=start)]
                out = await asyncio.gather(*tasks)
                batch_results.extend(out)

                # Try to update few-shot example for next mini-batch
                dyn_ex = _pick_dynamic_example(out)
                if dyn_ex:
                    self.example_qa = dyn_ex

                # Mark successful ones as completed and persist state
                newly_completed = 0
                for res, ch in zip(out, batch_chunks):
                    try:
                        if res.get("sharegpt_valid"):
                            key = self._done_key(getattr(ch, 'id', ''), getattr(ch, 'chapter_title', ''))
                            if key not in completed_keys:
                                completed_keys.add(key)
                                newly_completed += 1
                    except Exception:
                        continue
                if newly_completed:
                    self._save_completed(completed_keys)
                    logger.info(f"Pass {pass_idx}: marked {newly_completed} newly completed samples (total completed {len(completed_keys)}/{total}).")

            results.extend(batch_results)

            # Stop after one pass if retry_until_complete is false
            if not self.retry_until_complete:
                logger.info("retry_until_complete is disabled. Stopping after first pass.")
                break

            # If this pass made no progress, bail to avoid infinite loops
            if not any(r.get("sharegpt_valid") for r in batch_results):
                logger.warning("No successful samples in this pass. Stopping to avoid infinite retries.")
                break

        # Save final results summary
        self.results = results
        self.save_results(results)
        logger.info(f"Completed QA generation over {pass_idx} pass(es). Generated {sum(1 for r in results if r.get('sharegpt_valid'))} valid ShareGPT conversations.")
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
        logger.info(f"Incremental JSONL files already contain per-sample records: {self.results_jsonl_path}, {self.sharegpt_jsonl_path}")
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
                question = qa.get('question') or ''
                answer = qa.get('answer') or ''
                rephrased_text = qa.get('rephrased_text') or ''
                # Populate nulls when judge didn't run so JSON shows `null`
                metrics = qa.get('metrics') if qa.get('metrics') is not None else None
                feedback = qa.get('feedback') or ''
                quality_score = qa.get('quality_score') if qa.get('quality_score') is not None else None

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
                        "chapter_title": result.get("chapter_title", ""),
                        "source": (result.get('source') or (result.get('metadata', {}) or {}).get('source') or ""),
                        "execution_time": result.get('execution_time', 0),
                        "timestamp": result.get('timestamp', ''),
                        "context_text": result.get('original_text', ''),
                        "rephrased_text": rephrased_text,
                        "feedback": feedback,
                        "llm_judge_metrics": metrics,
                        "average_score": quality_score
                    }
                }
                if self._is_valid_sharegpt(conversation):
                    sharegpt_data.append(conversation)
                else:
                    logger.warning(f"Invalid ShareGPT item for chunk {result.get('chunk_id')}; excluded from final dataset.")

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


