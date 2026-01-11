#!/usr/bin/env python3
"""
CLI entry point for the Function Calling Dataset Generation Pipeline.

Usage:
    # Using config files (recommended)
    python -m datagenie.marketagents_function_calling.run --config configs/pipeline_config.yaml --limit 10
    
    # Curriculum mode with defaults
    python -m datagenie.marketagents_function_calling.run --mode curriculum --limit 10
    
    # HuggingFace mode
    python -m datagenie.marketagents_function_calling.run --mode huggingface --limit 100
"""

import asyncio
import argparse
from pathlib import Path

from dotenv import load_dotenv
load_dotenv()

from datagenie.marketagents_function_calling.config import PipelineConfig, GenerationMode
from datagenie.marketagents_function_calling.pipeline import FunctionCallingPipeline


def parse_args():
    parser = argparse.ArgumentParser(
        description="Function Calling Dataset Generation Pipeline"
    )
    
    # Config file (recommended)
    parser.add_argument(
        "--config",
        default=None,
        help="Path to pipeline config YAML (loads all settings from file)"
    )
    
    # Agents config override
    parser.add_argument(
        "--agents_config",
        default="configs/agents_config.yaml",
        help="Path to agents config YAML (per-agent LLM settings)"
    )
    
    # Mode selection
    parser.add_argument(
        "--mode",
        choices=["curriculum", "huggingface"],
        default="curriculum",
        help="Generation mode: 'curriculum' (default) or 'huggingface'"
    )
    
    # Curriculum mode options
    parser.add_argument(
        "--curriculum",
        default="configs/curriculum/function_calling.csv",
        help="Path to curriculum CSV/JSONL file"
    )
    
    # HuggingFace mode options
    parser.add_argument(
        "--dataset",
        default="Salesforce/xlam-function-calling-60k",
        help="HuggingFace dataset name"
    )
    parser.add_argument(
        "--start",
        type=int,
        default=0,
        help="Starting index in dataset"
    )
    
    # Common options
    parser.add_argument(
        "--limit",
        type=int,
        default=None,
        help="Maximum number of tasks to process"
    )
    parser.add_argument(
        "--batch_size",
        type=int,
        default=8,
        help="Parallel batch size"
    )
    parser.add_argument(
        "--output_dir",
        default="outputs/function_calling",
        help="Output directory"
    )
    parser.add_argument(
        "--no_validate",
        action="store_true",
        help="Disable tool call validation"
    )
    parser.add_argument(
        "--no_docstrings",
        action="store_true",
        help="Disable docstring generation"
    )
    
    return parser.parse_args()


async def main():
    args = parse_args()
    
    # Load from config file if provided
    if args.config:
        config_path = Path(args.config)
        if not config_path.exists():
            # Try relative to module
            module_dir = Path(__file__).parent
            config_path = module_dir / args.config
        
        if config_path.exists():
            print(f"Loading config from: {config_path}")
            config = PipelineConfig.from_yaml(str(config_path))
        else:
            print(f"Config file not found: {args.config}, using CLI args")
            config = PipelineConfig()
    else:
        # Build config from CLI args
        config = PipelineConfig(
            mode=GenerationMode(args.mode),
            curriculum_file=args.curriculum,
            dataset_name=args.dataset,
            batch_size=args.batch_size,
            output_dir=args.output_dir,
            validate_tool_calls=not args.no_validate,
            generate_docstrings=not args.no_docstrings,
            agents_config_file=args.agents_config,
        )
        config.load_agents_config()
    
    # CLI overrides
    if args.limit:
        pass  # limit is passed to run()
    
    print(f"Mode: {config.mode.value}")
    print(f"Agents config: {config.agents_config_file}")
    print(f"Generate reasoning: {config.generate_reasoning}")
    print(f"Generate analysis followup: {config.generate_analysis_followup}")
    print(f"Tool generator: {config.agents.tool_generator.client.value}/{config.agents.tool_generator.model}")
    print(f"Query generator: {config.agents.query_generator.client.value}/{config.agents.query_generator.model}")
    print(f"Tool calling: {config.agents.tool_calling.client.value}/{config.agents.tool_calling.model}")
    
    pipeline = FunctionCallingPipeline(config)
    await pipeline.run(
        start_index=args.start,
        limit=args.limit
    )


if __name__ == "__main__":
    asyncio.run(main())
