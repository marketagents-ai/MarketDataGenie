"""CLI entry point for structured output dataset generation."""

import asyncio
import argparse
from pathlib import Path

from dotenv import load_dotenv
load_dotenv()

from datagenie.marketagents_structured_output.config import PipelineConfig, GenerationMode
from datagenie.marketagents_structured_output.pipeline import StructuredOutputPipeline


def main():
    parser = argparse.ArgumentParser(
        description="Generate structured output / JSON mode datasets"
    )
    
    # Mode selection
    parser.add_argument(
        "--mode", 
        choices=["curriculum", "huggingface"],
        default="curriculum",
        help="Generation mode"
    )
    
    # Curriculum options
    parser.add_argument(
        "--curriculum",
        default="configs/curriculum/json_mode.csv",
        help="Path to curriculum CSV/JSONL file"
    )
    parser.add_argument(
        "--categories",
        nargs="+",
        default=[],
        help="Filter by categories"
    )
    parser.add_argument(
        "--subcategories",
        nargs="+",
        default=[],
        help="Filter by subcategories"
    )
    
    # HuggingFace options
    parser.add_argument(
        "--dataset",
        default="",
        help="HuggingFace dataset name"
    )
    parser.add_argument(
        "--start",
        type=int,
        default=0,
        help="Start index for HuggingFace dataset"
    )
    
    # Generation options
    parser.add_argument(
        "--limit",
        type=int,
        default=None,
        help="Limit number of tasks to process"
    )
    parser.add_argument(
        "--max-turns",
        type=int,
        default=3,
        help="Maximum conversation turns"
    )
    parser.add_argument(
        "--generate-reasoning",
        action="store_true",
        help="Generate <think> reasoning blocks"
    )
    parser.add_argument(
        "--generate-followup",
        action="store_true",
        dest="generate_followup",
        default=None,
        help="Generate follow-up turns (default: enabled)"
    )
    parser.add_argument(
        "--no-followup",
        action="store_false",
        dest="generate_followup",
        help="Disable follow-up generation"
    )
    parser.add_argument(
        "--analysis-followup",
        action="store_true",
        help="Generate analysis follow-up Q&A"
    )
    
    # Clarification options
    parser.add_argument(
        "--allow-clarification",
        action="store_true",
        default=True,
        help="Allow clarification flow when model asks for more info"
    )
    parser.add_argument(
        "--no-clarification",
        action="store_false",
        dest="allow_clarification",
        help="Disable clarification flow"
    )
    parser.add_argument(
        "--require-json-first-turn",
        action="store_true",
        help="Require JSON output on first turn (fail if clarification needed)"
    )
    parser.add_argument(
        "--max-clarification-turns",
        type=int,
        default=2,
        help="Maximum clarification turns before failing"
    )
    
    # Output options
    parser.add_argument(
        "--output-dir",
        default="outputs/structured_output",
        help="Output directory"
    )
    parser.add_argument(
        "--debug",
        action="store_true",
        help="Enable debug printing"
    )
    
    # Config file
    parser.add_argument(
        "--config",
        default=None,
        help="Path to pipeline config YAML"
    )
    
    args = parser.parse_args()
    
    # Load config - try default location if not specified
    default_config_path = Path(__file__).parent / "configs" / "pipeline_config.yaml"
    
    if args.config and Path(args.config).exists():
        config = PipelineConfig.from_yaml(args.config)
        print(f"Loaded config from: {args.config}")
    elif default_config_path.exists():
        config = PipelineConfig.from_yaml(str(default_config_path))
        print(f"Loaded config from: {default_config_path}")
    else:
        config = PipelineConfig()
        print("Using default config")
    
    # Override with CLI args (CLI takes precedence over config file)
    config.mode = GenerationMode(args.mode)
    if args.curriculum != "configs/curriculum/json_mode.csv":  # Only override if explicitly set
        config.curriculum_file = args.curriculum
    config.curriculum_categories = args.categories if args.categories else config.curriculum_categories
    config.curriculum_subcategories = args.subcategories if args.subcategories else config.curriculum_subcategories
    if args.dataset:  # Only override if explicitly set
        config.dataset_name = args.dataset
    if args.max_turns != 3:  # Only override if explicitly set
        config.max_turns = args.max_turns
    if args.generate_reasoning:  # CLI flag explicitly set
        config.generate_reasoning = True
    # generate_followup: None means use config default, otherwise use CLI value
    if args.generate_followup is not None:
        config.generate_followup = args.generate_followup
    if args.analysis_followup:  # CLI flag explicitly set
        config.generate_analysis_followup = True
    if not args.allow_clarification:  # --no-clarification was used
        config.allow_clarification_flow = False
    if args.require_json_first_turn:
        config.require_json_on_first_turn = True
    if args.max_clarification_turns != 2:  # Only override if explicitly set
        config.max_clarification_turns = args.max_clarification_turns
    if args.output_dir != "outputs/structured_output":  # Only override if explicitly set
        config.output_dir = args.output_dir
    if args.debug:
        config.debug_print_messages = True
    
    # Load agents config
    config.load_agents_config()
    
    # Print effective config summary
    print(f"Mode: {config.mode.value}")
    print(f"Generate reasoning: {config.generate_reasoning}")
    print(f"Generate followup: {config.generate_followup}")
    print(f"Generate analysis followup: {config.generate_analysis_followup}")
    print(f"Allow clarification: {config.allow_clarification_flow}")
    
    # Run pipeline
    pipeline = StructuredOutputPipeline(config)
    asyncio.run(pipeline.run(limit=args.limit, start_index=args.start))


if __name__ == "__main__":
    main()
