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
    
    pipeline = PythonformerPipeline(config)
    asyncio.run(pipeline.run(limit=config.dataset.limit))


if __name__ == "__main__":
    main()
