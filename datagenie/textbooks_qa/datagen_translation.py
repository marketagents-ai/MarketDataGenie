import json
import logging
import asyncio
from typing import Dict, List, Any, Optional, Union
from datetime import datetime
from pathlib import Path

import yaml
import re
import hashlib
import random
import tiktoken # Assuming tiktoken is available for token counting, though not strictly required for translation itself

from datasets import load_dataset

from market_agents.agents.market_agent import MarketAgent
from market_agents.environments.mechanisms.mcp_server import MCPServerEnvironment
from market_agents.environments.environment import MultiAgentEnvironment
from market_agents.workflows.market_agent_workflow import Workflow, WorkflowStep

from market_agents.task_manager import (
    TaskManager
)
from minference.lite.models import LLMClient, LLMConfig

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


class TranslationGenerator:
    """
    Main orchestrator for generic translation data generation.
    """

    def __init__(
        self,
        dataset_name: Optional[str] = None,
        output_dir: str = "outputs/translations", # Generic output directory
        num_samples: Optional[int] = None,
        config_path: Optional[str] = None,
        # Removed schema_mapping from constructor as it's now handled dynamically
        source_languages: Optional[List[str]] = None,
        target_language: Optional[str] = None,
        id_field: Optional[str] = None,
    ):
        self.dataset_name = dataset_name
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.num_samples = num_samples

        # New fields for dynamic configuration
        self.source_languages: List[str] = source_languages or []
        self.target_language: str = target_language or "english" # Default target language
        self.id_field: str = id_field or "id" # Default field for unique ID
        
        self.dataset_split: str = "train" # Default split, can be configured

        if config_path:
            if yaml is None:
                raise ImportError("PyYAML is required to use config files. Install with: pip install pyyaml")
            config_path_p = Path(config_path)
            if not config_path_p.exists():
                raise FileNotFoundError(f"Config file not found: {config_path}")
            with open(config_path_p, 'r', encoding='utf-8') as f:
                cfg = yaml.safe_load(f) or {}

            ds_cfg = (cfg.get('dataset') or {})
            gen_cfg = (cfg.get('generation') or {})
            lang_cfg = (cfg.get('languages') or {}) # New config section for languages

            self.dataset_name = ds_cfg.get('name', self.dataset_name)
            self.dataset_split = ds_cfg.get('split', self.dataset_split)
            self.id_field = ds_cfg.get('id_field', self.id_field) # ID field from dataset config
            
            self.num_samples = gen_cfg.get('num_samples', self.num_samples)

            self.source_languages = lang_cfg.get('source_languages', self.source_languages)
            self.target_language = lang_cfg.get('target_language', self.target_language)

            self.batch_size = int(gen_cfg.get('batch_size', 1))
            self.per_task_timeout = int(gen_cfg.get('per_task_timeout', 300))
            self.retry_until_complete: bool = bool(gen_cfg.get('retry_until_complete', False))
            self.completed_state_file: str = str(gen_cfg.get('completed_state_file', str(Path(self.output_dir) / "completed_state.json")))\

            self.output_dir = Path(cfg.get('output_dir', output_dir))
            self.output_dir.mkdir(parents=True, exist_ok=True)
        
        if not self.dataset_name:
            raise ValueError("dataset_name must be provided either as an argument or via YAML config under dataset.name")
        if not self.source_languages:
            raise ValueError("source_languages must be provided either as an argument or via YAML config under languages.source_languages")
        if not self.target_language:
            raise ValueError("target_language must be provided either as an argument or via YAML config under languages.target_language")
        
        # Ensure defaults for parallel rollout controls if not set via config
        if not hasattr(self, 'batch_size'):
            self.batch_size = 1
        if not hasattr(self, 'per_task_timeout'):
            self.per_task_timeout = 300
        if not hasattr(self, 'retry_until_complete'):
            self.retry_until_complete = False
        if not hasattr(self, 'completed_state_file'):
            self.completed_state_file = str(Path(self.output_dir) / "completed_state.json")

        self.task_manager = TaskManager()
        self.mcp_servers = self._setup_mcp_servers()

        self.results = []
        self.execution_stats = {}

        self.run_timestamp: Optional[str] = None
        self.results_jsonl_path: Optional[Path] = None
        self._io_lock: Optional[asyncio.Lock] = None
    
    @staticmethod
    def _normalize_for_hash(text: str) -> str:
        # Minimal, language-agnostic normalization: trim, collapse whitespace
        t = (text or "").strip()
        t = re.sub(r"\\s+", " ", t)
        return t

    def _done_key(self, record: Dict[str, Any]) -> str:
        # Using the configurable ID field for uniqueness
        record_id = record.get(self.id_field, str(record)) # Fallback to full record string if id_field not found
        base = f"{record_id}"
        return hashlib.sha1(base.encode('utf-8')).hexdigest()

    def _load_completed(self) -> set:
        p = Path(self.completed_state_file)
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
        p = Path(self.completed_state_file)
        try:
            p.parent.mkdir(parents=True, exist_ok=True)
            with open(p, 'w', encoding='utf-8') as f:
                json.dump(sorted(list(completed)), f, ensure_ascii=False, indent=2)
        except Exception as e:
            logger.error(f"Failed to save completed state to {p}: {e}")

    def _ensure_run_files(self):
        """Ensure per-run jsonl file paths are initialized and files exist."""
        if not self.run_timestamp:
            self.run_timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        if not self.results_jsonl_path:
            self.results_jsonl_path = self.output_dir / f"translations_{self.run_timestamp}.jsonl"
        
        self.results_jsonl_path.parent.mkdir(parents=True, exist_ok=True)
        self.results_jsonl_path.touch(exist_ok=True)
        if self._io_lock is None:
            self._io_lock = asyncio.Lock()

    async def _append_jsonl(self, path: Path, obj: Dict[str, Any]):
        """Append a single JSON object as a line to a JSONL file with async lock to avoid interleaving."""
        if self._io_lock is None:
            self._io_lock = asyncio.Lock()
        line = json.dumps(obj, ensure_ascii=False)
        async with self._io_lock:
            with open(path, 'a', encoding='utf-8') as f:
                f.write(line)
                f.write("\n")

    def _setup_mcp_servers(self) -> Dict[str, Union[MCPServerEnvironment, MultiAgentEnvironment]]:
        """Setup environments for translation workflow"""
        from market_agents.environments.mechanisms.chat import ChatMechanism
        
        chat_mechanism = ChatMechanism()
        
        default_env = MultiAgentEnvironment(
            name="default",
            mechanism=chat_mechanism
        )
        return {"default": default_env}

    def load_huggingface_dataset(self) -> List[Dict[str, Any]]: # Changed return type to List[Dict[str, Any]]
        """Load records from HuggingFace dataset."""
        logger.info(f"Loading dataset {self.dataset_name} (split: {self.dataset_split}) from HuggingFace...")
        
        try:
            dataset = load_dataset(self.dataset_name, split=self.dataset_split)
        except Exception as e:
            logger.error(f"Failed to load HuggingFace dataset {self.dataset_name}: {e}")
            raise

        records: List[Dict[str, Any]] = []
        for i, row in enumerate(dataset):
            if self.num_samples is not None and i >= self.num_samples:
                break
            
            # Ensure an 'id' field exists for consistent tracking
            if self.id_field not in row:
                row[self.id_field] = f"generated_id_{i}" # Generate a simple ID if not present

            records.append(row)
        
        logger.info(f"Loaded {len(records)} records from the dataset.")
        return records

    def create_translation_workflow(self) -> Workflow:
        """Create the translation workflow with dynamic language prompts."""
        context_lines = []
        for lang in self.source_languages:
            context_lines.append(f"- {lang.capitalize()}: {{{lang}}}")
        context_str = "\\n".join(context_lines)

        subtask_prompt = f"""
        Translate the following text from {', '.join(self.source_languages).capitalize()} into {self.target_language.capitalize()}.
        Preserve the original meaning and context. The output should be only the {self.target_language.capitalize()} translation.

        Context:
        {context_str}

        Translate to {self.target_language.capitalize()}:
        """

        steps = [
            WorkflowStep(
                name="translator",
                environment_name="default",
                tools=[],
                subtask=subtask_prompt,
                run_full_episode=False
            )
        ]

        return Workflow(
            name="generic_translation_workflow",
            task=f"""
            Translate records from {', '.join(self.source_languages).capitalize()} into {self.target_language.capitalize()}.
            """,
            steps=steps,
            mcp_servers=self._setup_mcp_servers()
        )

    async def generate_translation_for_chunk(
        self,
        record: Dict[str, Any], # Changed type to Dict[str, Any]
        agent: MarketAgent
    ) -> Dict[str, Any]:
        """Generate translation for a single record using workflow"""

        record_id = record.get(self.id_field, "N/A_ID")
        logger.info(f"Translating record {record_id}")

        try:
            ct = getattr(agent, "chat_thread", None)
            if ct is not None:
                if hasattr(ct, "tools"):
                    ct.tools = []
                if hasattr(ct, "workflow_step"):
                    ct.workflow_step = None
        except Exception:
            pass

        workflow = self.create_translation_workflow()

        # Prepare inputs dynamically from the record dictionary
        inputs = {lang: record.get(lang) for lang in self.source_languages}
        # Add the ID field to inputs for context in prompt if desired (optional)
        inputs[self.id_field] = record.get(self.id_field)
        
        # Add other potential metadata like chapter if they exist and are relevant
        if 'chapter' in record:
            inputs['chapter'] = record['chapter']


        try:
            start_time = datetime.now()
            workflow_result = await workflow.execute(agent, inputs)
            execution_time = (datetime.now() - start_time).total_seconds()

            wf_dump = workflow_result.model_dump(mode='json')
            
            # Extract the translated text from the workflow result
            translated_text = ""
            for s in wf_dump.get("step_results", []):
                if s.get("step_id") == "translator" and s.get("status") == "completed":
                    translated_text = s.get("result", {}).get("content", "") or ""
                    break

            result = {
                "id": record_id,
                **record, # Include all original fields from the record
                f"{self.target_language}_translation": translated_text.strip(), # Dynamic key for translation
                "workflow_results": wf_dump,
                "execution_time": execution_time,
                "timestamp": datetime.now().isoformat(),
            }

            self._ensure_run_files()
            await self._append_jsonl(self.results_jsonl_path, result)
            logger.info(f"Appended JSONL for record {record_id} -> {self.results_jsonl_path.name}")
            
            logger.info(f"Successfully translated record {record_id}")
            return result

        except Exception as e:
            logger.error(f"Failed to translate record {record_id}: {e}")
            return {
                "id": record_id,
                "error": str(e),
                "timestamp": datetime.now().isoformat()
            }

    async def generate_dataset(
        self,
        agent: MarketAgent,
        max_workers: int = 1,
        agent_factory: Optional[callable] = None
    ) -> List[Dict[str, Any]]:
        """Generate translation dataset from all loaded records with bounded parallelism"""
        all_records = self.load_huggingface_dataset()
        if self.num_samples is not None:
            all_records = all_records[: self.num_samples]
        total = len(all_records) # Changed variable name

        self._ensure_run_files()
        logger.info(f"Incremental save path: results={self.results_jsonl_path.name}")

        completed_keys = self._load_completed()
        logger.info(f"Loaded {len(completed_keys)} completed keys from state.")

        def pending_records(): 
            out = []
            for r in all_records: 
                key = self._done_key(r) # Passed the record object directly
                if key not in completed_keys:
                    out.append(r)
            return out

        pass_idx = 0
        results: List[Dict[str, Any]] = []
        workers_cfg = max(1, int(max_workers or getattr(self, 'batch_size', 1)))

        while True:
            pending = pending_records() # Changed function call
            if not pending:
                logger.info("No pending records remain. All done.") # Changed message
                break

            pass_idx += 1
            logger.info(f"Starting pass {pass_idx}: {len(pending)}/{total} pending (configured batch_size={workers_cfg})") # Changed message

            workers = workers_cfg
            sem = asyncio.Semaphore(workers)

            def new_agent() -> MarketAgent:
                if agent_factory is not None:
                    return agent_factory()
                from minference.lite.inference import InferenceOrchestrator
                from market_agents.agents.market_agent import MarketAgent as _MA
                llm_orch = InferenceOrchestrator()
                return _MA(
                    name=getattr(agent, 'name', 'translation_worker'),
                    persona=getattr(agent, 'persona', None),
                    llm_config=getattr(agent, 'llm_config', None),
                    llm_orchestrator=llm_orch,
                    task=getattr(agent, 'task', ''),
                    memory_enabled=getattr(agent, 'memory_enabled', False),
                    verbose=getattr(agent, 'verbose', False)
                )

            async def process_one(idx: int, record_obj: Dict[str, Any]) -> Dict[str, Any]: # Changed parameter name and type
                async with sem:
                    local_agent = new_agent()
                    try:
                        await asyncio.sleep(random.uniform(0.0, 0.15))
                        coro = self.generate_translation_for_chunk(record_obj, local_agent) # Changed parameter name
                        timeout = max(1, int(getattr(self, 'per_task_timeout', 300)))
                        return await asyncio.wait_for(coro, timeout=timeout)
                    except asyncio.TimeoutError:
                        logger.error(f"Timeout while translating record {record_obj.get(self.id_field, idx)} (> {getattr(self, 'per_task_timeout', 300)}s)")
                        return {"id": record_obj.get(self.id_field, idx), "error": f"timeout > {getattr(self, 'per_task_timeout', 300)}s", "timestamp": datetime.now().isoformat()}
                    except Exception as e:
                        logger.error(f"Failed to translate record {record_obj.get(self.id_field, idx)}: {e}")
                        return {"id": record_obj.get(self.id_field, idx), "error": str(e), "timestamp": datetime.now().isoformat()}

            batch_results: List[Dict[str, Any]] = []
            for start in range(0, len(pending), workers):
                end = min(start + workers, len(pending))
                batch_records = pending[start:end] # Changed variable name
                logger.info(f"Pass {pass_idx} - processing records {start}..{end-1} of {len(pending)}") # Changed message
                tasks = [asyncio.create_task(process_one(i, r)) for i, r in enumerate(batch_records, start=start)] # Changed parameter name
                out = await asyncio.gather(*tasks)
                batch_results.extend(out)

                newly_completed = 0
                for res, r_obj in zip(out, batch_records): # Changed variable name
                    try:
                        if f"{self.target_language}_translation" in res and res[f"{self.target_language}_translation"].strip():
                            key = self._done_key(r_obj) # Passed the record object directly
                            if key not in completed_keys:
                                completed_keys.add(key)
                                newly_completed += 1
                    except Exception:
                        continue
                if newly_completed:
                    self._save_completed(completed_keys)
                    logger.info(f"Pass {pass_idx}: marked {newly_completed} newly completed samples (total completed {len(completed_keys)}/{total}).")
            
            results.extend(batch_results)

            if not self.retry_until_complete:
                logger.info("retry_until_complete is disabled. Stopping after first pass.")
                break

            if not any(f"{self.target_language}_translation" in r and r[f"{self.target_language}_translation"].strip() for r in batch_results):
                logger.warning("No successful samples in this pass. Stopping to avoid infinite retries.")
                break

        self.results = results
        self.save_results(results)
        logger.info(f"Completed translation over {pass_idx} pass(es). Generated {sum(1 for r in results if f'{self.target_language}_translation' in r and r[f'{self.target_language}_translation'].strip())} valid translations.")
        return results

    def save_results(self, results: List[Dict[str, Any]]):
        """Save final results to file"""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        
        full_results_file = self.output_dir / f"translation_results_{timestamp}.json"
        with open(full_results_file, 'w', encoding='utf-8') as f:
            json.dump(results, f, indent=2, ensure_ascii=False)
        
        summary = {
            "total_records": len(results),
            "successful_translations": len([r for r in results if "error" not in r and f"{self.target_language}_translation" in r and r[f"{self.target_language}_translation"].strip()]),
            "failed_translations": len([r for r in results if "error" in r or not (f"{self.target_language}_translation" in r and r[f"{self.target_language}_translation"].strip())]),
            "generation_timestamp": timestamp,
            "source_languages": self.source_languages,
            "target_language": self.target_language,
            "execution_stats": self.execution_stats
        }
        
        summary_file = self.output_dir / f"translation_summary_{timestamp}.json"
        with open(summary_file, 'w', encoding='utf-8') as f:
            json.dump(summary, f, indent=2, ensure_ascii=False)
        
        logger.info(f"Saved results to {full_results_file}")
        logger.info(f"Saved summary to {summary_file}")
        logger.info(f"Incremental JSONL file already contains per-sample records: {self.results_jsonl_path}")


    def print_summary(self):
        """Print generation summary"""
        print("\n" + "="*50)
        print("GENERIC TRANSLATION SUMMARY")
        print("="*50)
        print(f"Total records processed: {self.execution_stats.get('total_executions', 0)}")
        print(f"Successful translations: {self.execution_stats.get('successful_executions', 0)}")
        print(f"Failed translations: {self.execution_stats.get('failed_executions', 0)}")
        print(f"Success rate: {self.execution_stats.get('success_rate', 0):.2%}")
        print(f"Source Languages: {', '.join(self.source_languages)}")
        print(f"Target Language: {self.target_language}")
        print(f"Output directory: {self.output_dir}")
        print("="*50)

# Main execution block for direct script invocation
if __name__ == "__main__":
    import argparse
    import re
    import hashlib
    import random
    from market_agents.agents.market_agent import MarketAgent
    from market_agents.agents.personas.persona import Persona
    from minference.lite.models import LLMConfig, LLMClient
    from minference.lite.inference import InferenceOrchestrator

    parser = argparse.ArgumentParser(description="Generate generic translations.")
    parser.add_argument("--config_path", type=str, default="datagenie/textbooks_qa/configs/translation_config.yaml", # Changed default config path
                        help="Path to the YAML configuration file for dataset and generation settings.")
    parser.add_argument("--dataset_name", type=str, default="seed_dataset",
                        help="HuggingFace dataset name (e.g., 'some_user/seed_dataset').")
    parser.add_argument("--output_dir", type=str, default="outputs/translations", # Generic output directory
                        help="Output directory for generated translations.")
    parser.add_argument("--num_samples", type=int, default=None,
                        help="Number of samples to generate (optional, overrides config).")
    parser.add_argument("--max_workers", type=int, default=1,
                        help="Number of parallel workers for translation.")
    
    # New arguments for dynamic language configuration
    parser.add_argument("--source_languages", nargs='+', default=[],
                        help="List of source languages (column names in dataset) to translate from.")
    parser.add_argument("--target_language", type=str, default="english",
                        help="Target language for translation.")
    parser.add_argument("--id_field", type=str, default="id",
                        help="Field in the dataset to use as a unique identifier for records.")

    args = parser.parse_args()

    async def main():
        # Initialize the generator with dynamic language configurations
        generator = TranslationGenerator(
            dataset_name=args.dataset_name,
            output_dir=args.output_dir,
            num_samples=args.num_samples,
            config_path=args.config_path,
            source_languages=args.source_languages,
            target_language=args.target_language,
            id_field=args.id_field,
        )

        # Removed schema_mapping update as it's no longer directly used for chunk fields

        # Set up a default agent for translation
        llm_config = LLMConfig(
            client=LLMClient.litellm,
            model="gpt-4.1", # You might want to use a specific translation model if available
            temperature=0.7,
            max_tokens=4096
        )
        llm_orchestrator = InferenceOrchestrator()
        # Dynamic persona instructions based on configured languages
        persona_instructions = (
            f"You are a highly skilled linguist specializing in multiple languages. Your task is to accurately translate text from "
            f"{', '.join(generator.source_languages).capitalize()} into {generator.target_language.capitalize()}, "
            f"preserving the original meaning, poetic nuance, and context. Produce only the {generator.target_language.capitalize()} translation."
        )
        translation_persona = Persona(
            role="Linguist",
            persona=persona_instructions,
            objectives=[f"Accurately translate text to {generator.target_language.capitalize()}", "Preserve meaning and context"],
            skills=["Multilingual translation", "Contextual understanding", "Linguistic analysis"],
        )
        translation_agent = MarketAgent(
            name="TranslationAgent", # Changed agent name
            persona=translation_persona,
            llm_config=llm_config,
            llm_orchestrator=llm_orchestrator,
            task=f"Translate text from {', '.join(generator.source_languages).capitalize()} to {generator.target_language.capitalize()}.",
            memory_enabled=False,
            verbose=False
        )

        logger.info(f"Starting generic translation pipeline for dataset: {generator.dataset_name}")
        
        try:
            results = await generator.generate_dataset(translation_agent, max_workers=args.max_workers)
            # You can add further processing of 'results' here if needed
            generator.print_summary()
        except Exception as e:
            logger.critical(f"An error occurred during dataset generation: {e}", exc_info=True)

    asyncio.run(main())