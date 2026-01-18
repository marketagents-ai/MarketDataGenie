"""CLI entry point for Pythonformer dataset generation."""

import asyncio
import argparse
from pathlib import Path

from dotenv import load_dotenv
load_dotenv()

from datagenie.pythonformer.config import PythonformerConfig, EnvironmentType, LLMClientType
from datagenie.pythonformer.pipeline import PythonformerPipeline


def main():
    parser = argparse.ArgumentParser(
        description="Generate interleaved reasoning + code training data"
    )
    
    parser.add_argument(
        "--config",
        default=None,
        help="Config YAML file (e.g., configs/default_config.yaml)"
    )
    parser.add_argument(
        "--env",
        choices=["math-python", "oolong", "code"],
        default=None,
        help="Task environment (overrides config)"
    )
    parser.add_argument(
        "--dataset",
        default=None,
        help="HuggingFace dataset name (overrides config)"
    )
    parser.add_argument(
        "--model",
        default=None,
        help="Main model (overrides config)"
    )
    parser.add_argument(
        "--client",
        choices=["openai", "anthropic", "litellm"],
        default=None,
        help="LLM client (overrides config)"
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=None,
        help="Number of tasks (overrides config)"
    )
    parser.add_argument(
        "--max-turns",
        type=int,
        default=None,
        help="Max REPL turns per task (overrides config)"
    )
    parser.add_argument(
        "--server-url",
        default=None,
        help="REPL server URL (overrides config)"
    )
    parser.add_argument(
        "--output-dir",
        default=None,
        help="Output directory (overrides config)"
    )
    parser.add_argument(
        "--mask-observations",
        action="store_true",
        help="Add loss_weight=0 to observations in ShareGPT output"
    )
    parser.add_argument(
        "--debug",
        action="store_true",
        help="Debug mode"
    )
    
    args = parser.parse_args()
    
    # Load config from file or use defaults
    if args.config and Path(args.config).exists():
        print(f"Loading config from: {args.config}")
        config = PythonformerConfig.from_yaml(args.config)
    else:
        config = PythonformerConfig()
    
    # Apply CLI overrides
    env_map = {
        "math-python": EnvironmentType.MATH_PYTHON,
        "oolong": EnvironmentType.OOLONG,
        "code": EnvironmentType.CODE,
    }
    client_map = {
        "openai": LLMClientType.OPENAI,
        "anthropic": LLMClientType.ANTHROPIC,
        "litellm": LLMClientType.LITELLM,
    }
    
    if args.env:
        config.dataset.environment = env_map[args.env]
    if args.dataset:
        config.dataset.dataset_name = args.dataset
    if args.model:
        config.main_model = args.model
    if args.client:
        config.main_client = client_map[args.client]
    if args.limit:
        config.dataset.limit = args.limit
    if args.max_turns:
        config.repl.max_turns = args.max_turns
    if args.server_url:
        config.repl.server_url = args.server_url
    if args.output_dir:
        config.dataset.output_dir = args.output_dir
    if args.mask_observations:
        config.dataset.mask_observations = True
    if args.debug:
        config.debug = True
    
    print("=" * 60)
    print("Pythonformer Dataset Generation")
    print("=" * 60)
    print(f"Environment:  {config.dataset.environment.value}")
    print(f"Dataset:      {config.dataset.dataset_name or 'default'}")
    print(f"Model:        {config.main_model} ({config.main_client.value})")
    print(f"Server:       {config.repl.server_url}")
    print(f"Limit:        {config.dataset.limit or 'all'}")
    print(f"Max turns:    {config.repl.max_turns}")
    print(f"Mask obs:     {config.dataset.mask_observations}")
    print(f"Output:       {config.dataset.output_dir}")
    print("=" * 60)
    
    # Initialize Database if configured
    data_inserter = None
    try:
        import os
        from market_agents.memory.config import AgentStorageConfig
        from market_agents.memory.agent_storage.setup_db import AsyncDatabase
        from datagenie.pythonformer.utils.db_utils import PythonformerDataInserter
        
        # Manually map env vars if they exist
        db_args = {}
        if os.getenv("DB_NAME"): db_args["db_name"] = os.getenv("DB_NAME")
        if os.getenv("DB_USER"): db_args["user"] = os.getenv("DB_USER")
        if os.getenv("DB_PASSWORD"): db_args["password"] = os.getenv("DB_PASSWORD")
        if os.getenv("DB_HOST"): db_args["host"] = os.getenv("DB_HOST")
        if os.getenv("DB_PORT"): db_args["port"] = os.getenv("DB_PORT")

        db_config = AgentStorageConfig(**db_args)
        db = AsyncDatabase(db_config)
        data_inserter = PythonformerDataInserter(db)
        print(f"Database insertion: ENABLED (User: {db_config.user}, DB: {db_config.db_name})")
    except Exception as e:
        print(f"Database insertion: DISABLED ({e})")

    async def run_pipeline():
        if data_inserter:
            await data_inserter.db.initialize()
        
        pipeline = PythonformerPipeline(config, data_inserter=data_inserter)
        await pipeline.run(limit=config.dataset.limit)
    
    asyncio.run(run_pipeline())


if __name__ == "__main__":
    main()
