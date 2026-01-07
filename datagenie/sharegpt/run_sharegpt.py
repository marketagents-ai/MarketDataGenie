#!/usr/bin/env python3
"""
Generic script to convert any HuggingFace dataset to ShareGPT format.
Uses YAML configuration for dataset-specific column mappings and target repositories.
Supports deduplication and appending to existing datasets.
"""

import argparse
import json
import os
import sys
import hashlib
import re
from pathlib import Path
from sharegpt_formatter import DatasetToShareGPTConverter

def list_available_datasets(config_path: str):
    """List all available datasets from the configuration file."""
    converter = DatasetToShareGPTConverter(config_path)
    
    if 'repositories' in converter.config:
        print("Available datasets:")
        for repo_name, config in converter.config['repositories'].items():
            source = config.get('source', repo_name)
            category = config.get('category', 'Q&A')
            columns = config.get('columns', {})
            question_col = columns.get('question', 'Question')
            answer_col = columns.get('answer', 'Answer')
            target_repo = config.get('target_repo', 'Not specified')
            
            print(f"  {repo_name}")
            print(f"    Source: {source}")
            print(f"    Category: {category}")
            print(f"    Columns: {question_col} -> {answer_col}")
            print(f"    Target Repo: {target_repo}")
            print()
    else:
        print("No datasets configured in the config file.")

def get_target_repo(config: dict, repo_name: str, dataset_path: str = None) -> str:
    """Get target repository name from config or generate one."""
    if dataset_path:
        return dataset_path
    
    # Check if target repo is specified in config
    if 'repositories' in config and repo_name in config['repositories']:
        target_repo = config['repositories'][repo_name].get('target_repo')
        if target_repo:
            return target_repo
    
    # Generate default target repo name
    safe_repo_name = repo_name.replace('/', '_')
    prefix = config.get('defaults', {}).get('target_repo_prefix', 'interstellarninja')
    return f"{prefix}/{safe_repo_name}_sharegpt"

def load_existing_dataset(target_repo: str) -> list:
    """Load existing dataset from HuggingFace Hub if it exists."""
    try:
        from datasets import load_dataset
        print(f"Loading existing dataset from: {target_repo}")
        dataset = load_dataset(target_repo, split="train")
        existing_data = dataset.to_list()
        print(f"Loaded {len(existing_data)} existing conversations")
        return existing_data
    except Exception as e:
        print(f"No existing dataset found at {target_repo} or error loading: {e}")
        return []

def validate_conversation(conversation: list, validation_config: dict = None) -> tuple[bool, str]:
    """
    Validate a conversation for quality and filter out low-quality data.
    
    Args:
        conversation: List of conversation messages
        validation_config: Validation configuration from config file
    
    Returns:
        tuple: (is_valid, reason_for_rejection)
    """
    if not conversation or len(conversation) < 2:
        return False, "Invalid conversation structure"
    
    # Get human and gpt messages
    human_msg = None
    gpt_msg = None
    
    for msg in conversation:
        if msg.get('from') == 'human':
            human_msg = msg.get('value', '').strip()
        elif msg.get('from') == 'gpt':
            gpt_msg = msg.get('value', '').strip()
    
    # 1. Check for non-empty values
    if not human_msg:
        return False, "Empty human message"
    if not gpt_msg:
        return False, "Empty gpt message"
    
    # 2. Check that human and gpt values are not exactly the same
    if human_msg == gpt_msg:
        return False, "Human and gpt messages are identical"
    
    # 3. Minimum length requirements
    min_length = validation_config.get('min_length', 10) if validation_config else 10
    
    if len(human_msg) < min_length:
        return False, f"Human message too short ({len(human_msg)} chars < {min_length})"
    if len(gpt_msg) < min_length:
        return False, f"Gpt message too short ({len(gpt_msg)} chars < {min_length})"
    
    # 4. Minimum word count requirements
    min_words = validation_config.get('min_words', 3) if validation_config else 3
    
    human_words = len(human_msg.split())
    gpt_words = len(gpt_msg.split())
    
    if human_words < min_words:
        return False, f"Human message too few words ({human_words} < {min_words})"
    if gpt_words < min_words:
        return False, f"Gpt message too few words ({gpt_words} < {min_words})"
    
    # 5. Check for consecutive repetitive patterns (potential synthetic data)
    if validation_config and validation_config.get('check_repetition', True):
        max_consecutive = validation_config.get('max_consecutive_repetition', 3)
        
        # Check for consecutive repetition in human message
        human_words_list = human_msg.split()
        if len(human_words_list) > 0:
            consecutive_count = 1
            prev_word = human_words_list[0]
            
            for word in human_words_list[1:]:
                if word == prev_word:
                    consecutive_count += 1
                    if consecutive_count > max_consecutive:
                        return False, f"Excessive consecutive repetition in human message: '{word}' repeated {consecutive_count} times in a row"
                else:
                    consecutive_count = 1
                prev_word = word
        
        # Check for consecutive repetition in gpt message
        gpt_words_list = gpt_msg.split()
        if len(gpt_words_list) > 0:
            consecutive_count = 1
            prev_word = gpt_words_list[0]
            
            for word in gpt_words_list[1:]:
                if word == prev_word:
                    consecutive_count += 1
                    if consecutive_count > max_consecutive:
                        return False, f"Excessive consecutive repetition in gpt message: '{word}' repeated {consecutive_count} times in a row"
                else:
                    consecutive_count = 1
                prev_word = word
    
    # 6. Check for common low-quality patterns
    if validation_config and validation_config.get('check_patterns', True):
        # Check for placeholder text
        placeholder_patterns = [
            r'\[.*?\]',  # [placeholder]
            r'\{.*?\}',  # {placeholder}
            r'<.*?>',    # <placeholder>
            r'___+',     # ____
            r'\.\.\.+',  # ...
        ]
        
        for pattern in placeholder_patterns:
            if re.search(pattern, human_msg) or re.search(pattern, gpt_msg):
                return False, f"Contains placeholder pattern: {pattern}"
        
        # Check for very short responses that might be errors
        if len(gpt_msg) < 20 and gpt_msg.lower() in ['yes', 'no', 'ok', 'okay', 'good', 'bad']:
            return False, "Gpt message too short and generic"
    
    return True, "Valid conversation"

def create_conversation_hash(conversation: list, method: str = "conversation") -> str:
    """Create a hash for deduplication based on the specified method."""
    if method == "question":
        # Hash only the question (first human message)
        question = conversation[0]['value'] if conversation and conversation[0]['from'] == 'human' else ""
        return hashlib.md5(question.encode('utf-8')).hexdigest()
    else:
        # Hash the entire conversation
        conversation_str = json.dumps(conversation, sort_keys=True, ensure_ascii=False)
        return hashlib.md5(conversation_str.encode('utf-8')).hexdigest()

def deduplicate_conversations(new_data: list, existing_data: list, method: str = "conversation") -> list:
    """Remove duplicates from new data based on existing data."""
    if not existing_data:
        return new_data
    
    print(f"Deduplicating using method: {method}")
    
    # Create set of existing conversation hashes
    existing_hashes = set()
    for item in existing_data:
        if 'conversations' in item:
            conv_hash = create_conversation_hash(item['conversations'], method)
            existing_hashes.add(conv_hash)
    
    # Filter new data
    deduplicated_data = []
    duplicates_found = 0
    
    for item in new_data:
        if 'conversations' in item:
            conv_hash = create_conversation_hash(item['conversations'], method)
            if conv_hash not in existing_hashes:
                deduplicated_data.append(item)
                existing_hashes.add(conv_hash)  # Add to set to avoid duplicates within new data
            else:
                duplicates_found += 1
    
    print(f"Removed {duplicates_found} duplicate conversations")
    print(f"Kept {len(deduplicated_data)} unique conversations")
    
    return deduplicated_data

def convert_dataset(
    repo_name: str,
    config_path: str = "dataset_config.yml",
    split: str = "train",
    max_samples: int = None,
    output_dir: str = "output",
    upload: bool = False,
    dataset_path: str = None,
    auto_upload: bool = False,
    append: bool = False
):
    """Convert a dataset to ShareGPT format with optional deduplication and appending."""
    
    print(f"Converting dataset: {repo_name}")
    print(f"Using config: {config_path}")
    
    # Initialize converter
    converter = DatasetToShareGPTConverter(config_path)
    
    # Check if dataset is configured
    if 'repositories' not in converter.config or repo_name not in converter.config['repositories']:
        print(f"Error: Dataset '{repo_name}' not found in configuration file.")
        print("Available datasets:")
        if 'repositories' in converter.config:
            for repo in converter.config['repositories'].keys():
                print(f"  - {repo}")
        else:
            print("  No datasets configured")
        return False
    
    # Get dataset configuration
    dataset_config = converter.config['repositories'][repo_name]
    columns = dataset_config.get('columns', {})
    question_column = columns.get('question', 'Question')
    answer_column = columns.get('answer', 'Answer')
    additional_context_column = columns.get('additional_context', None)
    
    print(f"Column mapping: {question_column} -> {answer_column}")
    if additional_context_column:
        print(f"Additional context column: {additional_context_column}")
    
    # Convert dataset
    sharegpt_data = converter.convert_dataset_to_sharegpt(
        repo_name=repo_name,
        question_column=question_column,
        answer_column=answer_column,
        split=split,
        max_samples=max_samples
    )
    
    if not sharegpt_data:
        print("No data was converted. Please check the dataset and column names.")
        return False
    
    # Apply validation and filtering
    validation_config = converter.config.get('options', {}).get('validation', {})
    print(f"Applying validation with config: {validation_config}")
    
    validated_data = []
    validation_stats = {
        'total': len(sharegpt_data),
        'valid': 0,
        'invalid': 0,
        'rejection_reasons': {}
    }
    
    for i, item in enumerate(sharegpt_data):
        if 'conversations' in item:
            is_valid, reason = validate_conversation(item['conversations'], validation_config)
            if is_valid:
                validated_data.append(item)
                validation_stats['valid'] += 1
            else:
                validation_stats['invalid'] += 1
                validation_stats['rejection_reasons'][reason] = validation_stats['rejection_reasons'].get(reason, 0) + 1
        
        if (i + 1) % 1000 == 0:
            print(f"Validated {i + 1}/{len(sharegpt_data)} conversations...")
    
    # Print validation statistics
    print(f"\nValidation Results:")
    print(f"  Total conversations: {validation_stats['total']}")
    print(f"  Valid conversations: {validation_stats['valid']}")
    print(f"  Invalid conversations: {validation_stats['invalid']}")
    print(f"  Validation rate: {validation_stats['valid']/validation_stats['total']*100:.1f}%")
    
    if validation_stats['rejection_reasons']:
        print(f"\nRejection reasons:")
        for reason, count in sorted(validation_stats['rejection_reasons'].items(), key=lambda x: x[1], reverse=True):
            print(f"  {reason}: {count}")
    
    if not validated_data:
        print("No valid conversations found after validation.")
        return False
    
    sharegpt_data = validated_data
    
    # Handle deduplication and appending
    dedup_config = converter.config.get('options', {}).get('deduplication', {})
    dedup_enabled = dedup_config.get('enabled', False)
    dedup_method = dedup_config.get('method', 'conversation')
    
    if append or dedup_enabled:
        # Get target repository
        target_repo = get_target_repo(converter.config, repo_name, dataset_path)
        
        # Load existing data
        existing_data = load_existing_dataset(target_repo)
        
        if existing_data:
            if dedup_enabled:
                # Deduplicate new data against existing data
                sharegpt_data = deduplicate_conversations(sharegpt_data, existing_data, dedup_method)
            
            if append:
                # Combine existing and new data
                combined_data = existing_data + sharegpt_data
                print(f"Combined {len(existing_data)} existing + {len(sharegpt_data)} new = {len(combined_data)} total conversations")
                sharegpt_data = combined_data
    
    # Create output directory
    os.makedirs(output_dir, exist_ok=True)
    
    # Generate output filename
    safe_repo_name = repo_name.replace('/', '_')
    output_file = os.path.join(output_dir, f"{safe_repo_name}_sharegpt.json")
    
    # Save to file
    converter.save_sharegpt_data(sharegpt_data, output_file)
    
    # Print sample
    print("\nSample converted conversation:")
    sample = sharegpt_data[0]
    print(json.dumps(sample, ensure_ascii=False, indent=2))
    
    print(f"\nSuccessfully converted {len(sharegpt_data)} conversations")
    print(f"Output saved to: {output_file}")
    
    # Determine if we should upload
    should_upload = upload or auto_upload or converter.config.get('options', {}).get('auto_upload', False)
    
    if should_upload:
        # Get target repository
        target_repo = get_target_repo(converter.config, repo_name, dataset_path)
        print(f"Uploading to HuggingFace Hub: {target_repo}")
        
        try:
            converter.upload_to_hub(sharegpt_data, target_repo, upload=True)
            print(f"✓ Successfully uploaded to: {target_repo}")
        except Exception as e:
            print(f"✗ Upload failed: {e}")
            print("You can manually upload later using the saved JSON file.")
    else:
        print("Upload skipped. Use --upload flag to enable upload.")
    
    return True

def main():
    parser = argparse.ArgumentParser(
        description="Convert HuggingFace datasets to ShareGPT format using YAML configuration",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # List available datasets
  python run_sharegpt.py --list
  
  # Convert a specific dataset
  python run_sharegpt.py "your_username/your_dataset"
  
  # Convert with custom config and limit samples
  python run_sharegpt.py "your_username/your_dataset" --config custom_config.yml --max-samples 1000
  
  # Convert and upload to HuggingFace Hub (uses target_repo from config)
  python run_sharegpt.py "your_username/your_dataset" --upload
  
  # Convert and append to existing dataset with deduplication
  python run_sharegpt.py "your_username/your_dataset" --upload --append
  
  # Convert and upload to custom repository
  python run_sharegpt.py "your_username/your_dataset" --upload --dataset-path "username/custom-name"
        """
    )
    
    parser.add_argument(
        "repo_name", 
        nargs='?',
        help="HuggingFace dataset repository name (required unless using --list)"
    )
    parser.add_argument(
        "--config", 
        default="dataset_config.yml",
        help="Path to YAML configuration file (default: dataset_config.yml)"
    )
    parser.add_argument(
        "--split", 
        default="train",
        help="Dataset split to use (default: train)"
    )
    parser.add_argument(
        "--max-samples", 
        type=int,
        help="Maximum number of samples to convert"
    )
    parser.add_argument(
        "--output-dir", 
        default="output",
        help="Output directory for converted files (default: output)"
    )
    parser.add_argument(
        "--upload", 
        action="store_true",
        help="Upload to HuggingFace Hub (uses target_repo from config if available)"
    )
    parser.add_argument(
        "--dataset-path",
        help="HuggingFace dataset path for upload (overrides target_repo from config)"
    )
    parser.add_argument(
        "--list", 
        action="store_true",
        help="List all available datasets from configuration file"
    )
    parser.add_argument(
        "--auto-upload",
        action="store_true",
        help="Automatically upload after conversion (overrides config setting)"
    )
    parser.add_argument(
        "--append",
        action="store_true",
        help="Append to existing dataset (combines with existing data)"
    )
    
    args = parser.parse_args()
    
    # Check if config file exists
    if not os.path.exists(args.config):
        print(f"Error: Configuration file '{args.config}' not found.")
        print("Please create a dataset_config.yml file with your dataset configurations.")
        sys.exit(1)
    
    # List available datasets
    if args.list:
        list_available_datasets(args.config)
        return
    
    # Check if repo_name is provided
    if not args.repo_name:
        print("Error: Repository name is required unless using --list")
        parser.print_help()
        sys.exit(1)
    
    # Convert dataset
    success = convert_dataset(
        repo_name=args.repo_name,
        config_path=args.config,
        split=args.split,
        max_samples=args.max_samples,
        output_dir=args.output_dir,
        upload=args.upload,
        dataset_path=args.dataset_path,
        auto_upload=args.auto_upload,
        append=args.append
    )
    
    if not success:
        sys.exit(1)

if __name__ == "__main__":
    main()