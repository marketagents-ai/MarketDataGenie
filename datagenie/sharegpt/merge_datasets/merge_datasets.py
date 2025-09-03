#!/usr/bin/env python3
"""
Script to merge multiple ShareGPT datasets and upload to a single repository.
Adds system messages to conversations that don't have them.
"""

import argparse
import json
import os
import sys
from pathlib import Path
from typing import List, Dict, Any, Optional
import yaml
from datasets import load_dataset

class DatasetMerger:
    def __init__(self, target_repo: str, system_message: str = "You are a helpful AI assistant", source_specific_system_messages: Optional[Dict[str, str]] = None, reasoning_source: Optional[str] = None, reasoning_system_message: Optional[str] = None):
        """
        Initialize the dataset merger.
        
        Args:
            target_repo: Target HuggingFace repository for merged dataset
            system_message: System message to add to conversations without one
        """
        self.target_repo = target_repo
        self.system_message = system_message
        # Backward-compat: single-source override via --reasoning-*
        self.reasoning_source = reasoning_source
        self.reasoning_system_message = reasoning_system_message
        # Preferred: mapping of source -> system_message (from config)
        self.source_specific_system_messages = source_specific_system_messages or {}
        self.merged_data = []
        self.stats = {
            'total_conversations': 0,
            'conversations_with_system': 0,
            'conversations_added_system': 0,
            'datasets_processed': 0,
            'items_skipped_no_conversation': 0
        }
    
    def add_system_message_if_missing(self, conversation: List[Dict[str, str]], system_message: Optional[str] = None) -> List[Dict[str, str]]:
        """
        Add system message to conversation if it doesn't have one.
        
        Args:
            conversation: List of conversation messages
            
        Returns:
            Updated conversation with system message
        """
        # Check if conversation already has a system message
        has_system = any(msg.get('from') == 'system' for msg in conversation)
        
        if has_system:
            self.stats['conversations_with_system'] += 1
            return conversation
        
        # Add system message at the beginning
        if system_message is None:
            system_message = self.system_message
        updated_conversation = [
            {
                "from": "system",
                "value": system_message
            }
        ] + conversation
        self.stats['conversations_added_system'] += 1
        return updated_conversation

    def get_system_message_for_source(self, source: str) -> str:
        """Return the system message to use for a given source."""
        # Config-provided mapping takes precedence
        if source in self.source_specific_system_messages:
            return self.source_specific_system_messages[source]
        # Backward-compat CLI override
        if self.reasoning_source and self.reasoning_system_message and source == self.reasoning_source:
            return self.reasoning_system_message
        return self.system_message

    def _normalize_to_conversations(self, item: Dict[str, Any]) -> Optional[List[Dict[str, str]]]:
        """
        Normalize various dataset schemas to the expected 'conversations' list.
        Supported forms:
          - item['conversations'] already a list
          - item['messages'] list with {'role','content'} -> convert to {'from','value'}
          - item with ('question','answer') or ('instruction','output') -> build 2-turn convo
        Returns None if cannot normalize.
        """
        # 1) conversations already present and is a list
        if 'conversations' in item and isinstance(item.get('conversations'), list):
            return item['conversations']

        # 2) messages schema
        if isinstance(item.get('messages'), list):
            converted: List[Dict[str, str]] = []
            for msg in item['messages']:
                role = msg.get('role') or msg.get('from')
                content = msg.get('content') or msg.get('value')
                if not role or content is None:
                    continue
                if role == 'system':
                    from_value = 'system'
                elif role in ('user', 'human', 'prompt'):
                    from_value = 'human'
                else:
                    from_value = 'gpt'
                converted.append({"from": from_value, "value": str(content)})
            return converted if converted else None

        # 3) simple QA schema
        question = item.get('question') or item.get('input') or item.get('instruction')
        answer = item.get('answer') or item.get('output') or item.get('response')
        if question is not None and answer is not None:
            return [
                {"from": "human", "value": str(question)},
                {"from": "gpt", "value": str(answer)}
            ]

        return None
    
    def load_dataset_from_hub(self, repo_name: str, split: str = "train") -> List[Dict[str, Any]]:
        """
        Load dataset from HuggingFace Hub.
        
        Args:
            repo_name: HuggingFace dataset repository name
            split: Dataset split to use
            
        Returns:
            List of dataset items
        """
        try:
            print(f"Loading dataset: {repo_name}")
            dataset = load_dataset(repo_name, split=split)
            data = dataset.to_list()
            print(f"  Loaded {len(data)} items")
            return data
        except Exception as e:
            print(f"  Error loading {repo_name}: {e}")
            return []
    
    def load_dataset_from_file(self, file_path: str) -> List[Dict[str, Any]]:
        """
        Load dataset from local JSON file.
        
        Args:
            file_path: Path to local JSON file
            
        Returns:
            List of dataset items
        """
        try:
            print(f"Loading dataset from file: {file_path}")
            with open(file_path, 'r', encoding='utf-8') as f:
                data = json.load(f)
            print(f"  Loaded {len(data)} items")
            return data
        except Exception as e:
            print(f"  Error loading {file_path}: {e}")
            return []
    
    def process_dataset(self, source: str, data: List[Dict[str, Any]]) -> None:
        """
        Process a dataset and add to merged data.
        
        Args:
            source: Source identifier for the dataset
            data: Dataset items to process
        """
        print(f"Processing dataset: {source}")
        
        for item in data:
            conversations = self._normalize_to_conversations(item)
            if not conversations:
                self.stats['items_skipped_no_conversation'] += 1
                continue

            # Add system message if missing (with per-source override)
            sys_msg = self.get_system_message_for_source(source)
            updated_conversations = self.add_system_message_if_missing(conversations, sys_msg)

            # Create merged item with only required fields
            merged_item = {
                "conversations": updated_conversations,
                "source": source
            }

            # Add ID if present in original
            if 'id' in item:
                merged_item['id'] = item['id']

            self.merged_data.append(merged_item)
            self.stats['total_conversations'] += 1
        
        self.stats['datasets_processed'] += 1
        print(f"  Processed {len(data)} items")
    
    def add_dataset_from_hub(self, repo_name: str, split: str = "train") -> bool:
        """
        Add dataset from HuggingFace Hub.
        
        Args:
            repo_name: HuggingFace dataset repository name
            split: Dataset split to use
            
        Returns:
            True if successful, False otherwise
        """
        data = self.load_dataset_from_hub(repo_name, split)
        if data:
            self.process_dataset(repo_name, data)
            return True
        return False
    
    def add_dataset_from_file(self, file_path: str, source_name: str = None) -> bool:
        """
        Add dataset from local file.
        
        Args:
            file_path: Path to local JSON file
            source_name: Custom source name (defaults to filename)
            
        Returns:
            True if successful, False otherwise
        """
        data = self.load_dataset_from_file(file_path)
        if data:
            if source_name is None:
                source_name = Path(file_path).stem
            self.process_dataset(source_name, data)
            return True
        return False
    
    def save_merged_dataset(self, output_path: str = "merged_dataset.jsonl") -> None:
        """
        Save merged dataset to local file in JSONL format.
        
        Args:
            output_path: Path to save merged dataset
        """
        try:
            print(f"Saving {len(self.merged_data)} conversations to {output_path}")
            
            with open(output_path, 'w', encoding='utf-8') as f:
                for i, item in enumerate(self.merged_data):
                    f.write(json.dumps(item, ensure_ascii=False) + '\n')
                    
                    # Progress indicator
                    if (i + 1) % 10000 == 0:
                        print(f"  Saved {i + 1}/{len(self.merged_data)} conversations")
            
            print(f"✓ Successfully saved to: {output_path}")
            
        except Exception as e:
            print(f"✗ Error saving merged dataset: {e}")
    
    def upload_jsonl_to_hub(self, jsonl_file: str, target_repo: str) -> bool:
        """
        Upload JSONL file directly to HuggingFace Hub without loading into memory.
        Automatically creates repository if it doesn't exist.
        
        Args:
            jsonl_file: Path to JSONL file
            target_repo: Target HuggingFace repository
            
        Returns:
            True if successful, False otherwise
        """
        try:
            from huggingface_hub import HfApi, create_repo, HfFolder
            
            print(f"Uploading JSONL file to: {target_repo}")
            print("Using streaming upload to avoid memory issues...")
            
            # Resolve auth token from env or local login
            token = os.getenv("HUGGINGFACE_HUB_TOKEN") or os.getenv("HF_TOKEN") or HfFolder.get_token()
            api = HfApi(token=token)
            
            # Check if repository exists, create if it doesn't
            try:
                print(f"Checking if repository {target_repo} exists...")
                api.repo_info(repo_id=target_repo, repo_type='dataset')
                print(f"✓ Repository {target_repo} exists, will append/overwrite")
            except Exception:
                print(f"Repository {target_repo} doesn't exist, creating it...")
                try:
                    create_repo(
                        repo_id=target_repo,
                        repo_type='dataset',
                        exist_ok=True
                    )
                    print(f"✓ Created repository: {target_repo}")
                except Exception as e:
                    print(f"✗ Failed to create repository: {e}")
                    return False
            
            # Upload the JSONL file directly
            print("Starting upload...")
            api.upload_file(
                path_or_fileobj=jsonl_file,
                path_in_repo='merged_dataset.jsonl',
                repo_id=target_repo,
                repo_type='dataset'
            )
            
            print(f"✓ Successfully uploaded to: {target_repo}")
            return True
            
        except Exception as e:
            print(f"✗ Upload failed: {e}")
            return False
    
    def upload_to_hub(self) -> bool:
        """
        Upload merged dataset to HuggingFace Hub using streaming approach.
        Automatically creates repository if needed.
        
        Returns:
            True if successful, False otherwise
        """
        try:
            # Check if JSONL file exists
            jsonl_file = "merged_dataset.jsonl"
            if not os.path.exists(jsonl_file):
                print(f"Error: {jsonl_file} not found. Please run the merge first.")
                return False
            
            # Use streaming upload with automatic repository creation
            return self.upload_jsonl_to_hub(jsonl_file, self.target_repo)
            
        except Exception as e:
            print(f"Error during upload: {e}")
            return False
    
    def print_stats(self) -> None:
        """Print processing statistics."""
        print(f"\n{'='*50}")
        print("PROCESSING STATISTICS")
        print(f"{'='*50}")
        print(f"Datasets processed: {self.stats['datasets_processed']}")
        print(f"Total conversations: {self.stats['total_conversations']}")
        print(f"Conversations with existing system: {self.stats['conversations_with_system']}")
        print(f"Conversations added system: {self.stats['conversations_added_system']}")
        print(f"Items skipped (no conversations): {self.stats['items_skipped_no_conversation']}")
        print(f"{'='*50}")
    
    def get_sample_conversation(self) -> Optional[Dict[str, Any]]:
        """Get a sample conversation from merged data."""
        if self.merged_data:
            return self.merged_data[0]
        return None

def main():
    parser = argparse.ArgumentParser(
        description="Merge multiple ShareGPT datasets and upload to a single repository",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Add datasets from HuggingFace Hub
  python merge_datasets.py --target-repo "username/merged-dataset" \\
    --add-hub "your-username/your-dataset-1" \\
    --add-hub "your-username/your-dataset-2"
  
  # Add datasets from local files
  python merge_datasets.py --target-repo "username/merged-dataset" \\
    --add-file "dataset1.json" "Dataset 1" \\
    --add-file "dataset2.json" "Dataset 2"
  
  # Mix of hub and local files
  python merge_datasets.py --target-repo "username/merged-dataset" \\
    --add-hub "some/dataset" \\
    --add-file "local_dataset.json" "Local Dataset"
        """
    )
    
    parser.add_argument(
        "--target-repo",
        required=True,
        help="Target HuggingFace repository for merged dataset"
    )
    
    parser.add_argument(
        "--system-message",
        default="You are a helpful AI assistant",
        help="System message to add to conversations without one"
    )
    
    parser.add_argument(
        "--add-hub",
        action="append",
        nargs=2,
        metavar=("REPO_NAME", "SPLIT"),
        help="Add dataset from HuggingFace Hub (can be used multiple times)"
    )
    
    parser.add_argument(
        "--add-file",
        action="append",
        nargs=2,
        metavar=("FILE_PATH", "SOURCE_NAME"),
        help="Add dataset from local file (can be used multiple times)"
    )
    
    parser.add_argument(
        "--output",
        default="merged_dataset.jsonl",
        help="Output file path for merged dataset (default: merged_dataset.jsonl)"
    )
    
    parser.add_argument(
        "--upload",
        action="store_true",
        help="Upload merged dataset to HuggingFace Hub"
    )
    
    parser.add_argument(
        "--no-save",
        action="store_true",
        help="Don't save merged dataset to local file"
    )
    
    parser.add_argument(
        "--append-existing",
        action="store_true",
        help="Append to existing merged_dataset.jsonl in target repo if present"
    )

    parser.add_argument(
        "--reasoning-source",
        help="Source name (repo or file source) that should use the reasoning system message"
    )

    parser.add_argument(
        "--reasoning-system-message",
        help="System message to use for the reasoning source"
    )

    parser.add_argument(
        "--config",
        help="Path to YAML config that can list hub_datasets and per-dataset system_message overrides"
    )
    
    args = parser.parse_args()
    
    # Initialize merger
    # Prepare system message and per-source overrides, possibly from config
    system_message_for_run = args.system_message
    per_source_overrides: Dict[str, str] = {}
    config_hub_datasets: List[Dict[str, Any]] = []
    if args.config:
        try:
            with open(args.config, 'r', encoding='utf-8') as cf:
                cfg = yaml.safe_load(cf) or {}
            if isinstance(cfg.get('system_message'), str):
                system_message_for_run = cfg['system_message']
            # Collect hub_datasets and per-dataset system messages
            if isinstance(cfg.get('hub_datasets'), list):
                for item in cfg['hub_datasets']:
                    if not isinstance(item, dict):
                        continue
                    config_hub_datasets.append(item)
                    if 'system_message' in item and isinstance(item['system_message'], str) and 'name' in item:
                        per_source_overrides[item['name']] = item['system_message']
        except Exception as e:
            print(f"Warning: Failed to read config {args.config}: {e}")

    # Create merger with combined settings
    merger = DatasetMerger(
        args.target_repo,
        system_message_for_run,
        source_specific_system_messages=per_source_overrides,
        reasoning_source=args.reasoning_source,
        reasoning_system_message=args.reasoning_system_message
    )
    
    # Aggregate datasets from CLI and config
    add_hub_entries: List[List[str]] = []
    if args.add_hub:
        add_hub_entries.extend(args.add_hub)
    for item in config_hub_datasets:
        name = item.get('name')
        split = item.get('split', 'train')
        if name:
            add_hub_entries.append([name, split])

    # Add datasets from HuggingFace Hub (combined list)
    if add_hub_entries:
        for repo_name, split in add_hub_entries:
            success = merger.add_dataset_from_hub(repo_name, split)
            if not success:
                print(f"Warning: Failed to add dataset from hub: {repo_name}")
    
    # Add datasets from local files
    if args.add_file:
        for file_path, source_name in args.add_file:
            success = merger.add_dataset_from_file(file_path, source_name)
            if not success:
                print(f"Warning: Failed to add dataset from file: {file_path}")
    
    # Check if any datasets were processed
    if merger.stats['datasets_processed'] == 0:
        print("Error: No datasets were successfully processed.")
        sys.exit(1)
    
    # Print statistics
    merger.print_stats()
    
    # Show sample conversation
    sample = merger.get_sample_conversation()
    if sample:
        print("\nSample merged conversation:")
        print(json.dumps(sample, ensure_ascii=False, indent=2))
    
    # Save to local file
    if not args.no_save:
        merger.save_merged_dataset(args.output)
    
    # Upload to HuggingFace Hub
    if args.upload:
        upload_file_path = args.output
        if args.append_existing:
            # Try to download existing merged file from the repo and combine
            try:
                from huggingface_hub import HfApi, hf_hub_download
                api = HfApi()
                print(f"Checking for existing merged_dataset.jsonl in {args.target_repo}...")
                try:
                    api.repo_info(repo_id=args.target_repo, repo_type='dataset')
                    existing_local = None
                    try:
                        existing_local = hf_hub_download(
                            repo_id=args.target_repo,
                            filename='merged_dataset.jsonl',
                            repo_type='dataset'
                        )
                        print("✓ Found existing merged_dataset.jsonl; will append to it")
                    except Exception as e:
                        print("No existing merged_dataset.jsonl found; will upload new data only")
                    if existing_local:
                        # Combine existing + new into a temporary combined file
                        combined_path = str(Path(args.output).with_suffix('.combined.jsonl'))
                        print(f"Combining existing + new into {combined_path}")
                        with open(combined_path, 'w', encoding='utf-8') as out_f:
                            with open(existing_local, 'r', encoding='utf-8') as ex_f:
                                for line in ex_f:
                                    out_f.write(line)
                            with open(args.output, 'r', encoding='utf-8') as new_f:
                                for line in new_f:
                                    out_f.write(line)
                        upload_file_path = combined_path
                except Exception as e:
                    print(f"Warning: Could not check/download existing file: {e}")
            except Exception as e:
                print(f"Warning: Append check failed to initialize: {e}")
        success = merger.upload_jsonl_to_hub(upload_file_path, args.target_repo)
        if not success:
            print("Upload failed. You can manually upload the saved JSONL file later.")
            sys.exit(1)
    else:
        print("\nUpload skipped. Use --upload flag to enable upload.")
        print(f"Target repository: {args.target_repo}")

if __name__ == "__main__":
    main()
