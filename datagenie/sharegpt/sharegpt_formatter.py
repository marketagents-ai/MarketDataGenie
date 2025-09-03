# python script to download a huggingface dataset and convert it to sharegpt format

import argparse
import json
import os
import uuid
import yaml
from datasets import load_dataset
from typing import Dict, List, Any, Optional

class DatasetToShareGPTConverter:
    def __init__(self, config_path: Optional[str] = None):
        """
        Initialize the converter with optional YAML config file.
        
        Args:
            config_path: Path to YAML configuration file
        """
        self.config = self._load_config(config_path) if config_path else {}
    
    def _load_config(self, config_path: str) -> Dict[str, Any]:
        """Load configuration from YAML file."""
        try:
            with open(config_path, 'r', encoding='utf-8') as file:
                return yaml.safe_load(file)
        except Exception as e:
            print(f"Error loading config file {config_path}: {e}")
            return {}
    
    def _get_column_mapping(self, repo_name: str) -> Dict[str, str]:
        """Get column mapping for a specific repository."""
        if 'repositories' in self.config and repo_name in self.config['repositories']:
            return self.config['repositories'][repo_name].get('columns', {})
        return {}
    
    def _get_source_name(self, repo_name: str) -> str:
        """Get source name for a repository."""
        if 'repositories' in self.config and repo_name in self.config['repositories']:
            return self.config['repositories'][repo_name].get('source', repo_name)
        return repo_name
    
    def convert_dataset_to_sharegpt(
        self, 
        repo_name: str, 
        question_column: str = "Question", 
        answer_column: str = "Answer",
        split: str = "train",
        max_samples: Optional[int] = None
    ) -> List[Dict[str, Any]]:
        """
        Convert a HuggingFace dataset to ShareGPT format.
        
        Args:
            repo_name: HuggingFace dataset repository name
            question_column: Name of the question column
            answer_column: Name of the answer column
            split: Dataset split to use (default: "train")
            max_samples: Maximum number of samples to convert (optional)
            
        Returns:
            List of ShareGPT formatted conversations
        """
        print(f"Loading dataset: {repo_name}")
        
        try:
            # Load dataset from HuggingFace
            dataset = load_dataset(repo_name, split=split)
            print(f"Loaded {len(dataset)} samples from {repo_name}")
            
            # Get column mapping from config if available
            column_mapping = self._get_column_mapping(repo_name)
            if column_mapping:
                question_column = column_mapping.get('question', question_column)
                answer_column = column_mapping.get('answer', answer_column)
                additional_context_column = column_mapping.get('additional_context', None)
            else:
                additional_context_column = None
            
            # Get source name
            source_name = self._get_source_name(repo_name)
            
            # Convert to ShareGPT format
            sharegpt_data = []
            
            # Limit samples if specified
            if max_samples:
                dataset = dataset.select(range(min(max_samples, len(dataset))))
            
            for i, sample in enumerate(dataset):
                try:
                    # Extract question and answer
                    question = sample.get(question_column, "")
                    answer = sample.get(answer_column, "")
                    
                    # Handle additional context if specified
                    if additional_context_column:
                        additional_context = sample.get(additional_context_column, "")
                        if additional_context and str(additional_context).strip():
                            # Combine question and additional context
                            human_value = f"{str(question).strip()}\n\n{str(additional_context).strip()}"
                        else:
                            human_value = str(question).strip()
                    else:
                        human_value = str(question).strip()
                    
                    # Skip if either question or answer is empty
                    if not human_value or not answer:
                        print(f"Skipping sample {i}: missing question or answer")
                        continue
                    
                    # Create ShareGPT conversation format
                    conversation = [
                        {
                            "from": "human",
                            "value": human_value
                        },
                        {
                            "from": "gpt", 
                            "value": str(answer).strip()
                        }
                    ]
                    
                    # Create ShareGPT entry
                    sharegpt_entry = {
                        "id": str(uuid.uuid4()),
                        "conversations": conversation,
                        "source": source_name,
                        "category": "Q&A",
                        "task": human_value[:100] + "..." if len(human_value) > 100 else human_value
                    }
                    
                    sharegpt_data.append(sharegpt_entry)
                    
                    if (i + 1) % 100 == 0:
                        print(f"Processed {i + 1} samples...")
                        
                except Exception as e:
                    print(f"Error processing sample {i}: {e}")
                    continue
            
            print(f"Successfully converted {len(sharegpt_data)} samples to ShareGPT format")
            return sharegpt_data
            
        except Exception as e:
            print(f"Error loading dataset {repo_name}: {e}")
            return []
    
    def save_sharegpt_data(self, data: List[Dict[str, Any]], output_path: str):
        """Save ShareGPT data to JSON file."""
        try:
            with open(output_path, 'w', encoding='utf-8') as f:
                json.dump(data, f, ensure_ascii=False, indent=2)
            print(f"ShareGPT data saved to: {output_path}")
        except Exception as e:
            print(f"Error saving data to {output_path}: {e}")
    
    def upload_to_hub(
        self, 
        data: List[Dict[str, Any]], 
        dataset_path: str,
        upload: bool = False
    ):
        """Upload dataset to HuggingFace Hub."""
        if not upload:
            print("Upload disabled. Use --upload flag to enable.")
            return
        
        try:
            from datasets import Dataset
            
            # Create dataset
            dataset = Dataset.from_list(data)
            
            # Upload to hub
            dataset.push_to_hub(dataset_path)
            print(f"Dataset uploaded to HuggingFace Hub: {dataset_path}")
            
        except Exception as e:
            print(f"Error uploading to hub: {e}")

def main():
    parser = argparse.ArgumentParser(description="Convert HuggingFace datasets to ShareGPT format")
    parser.add_argument("repo_name", help="HuggingFace dataset repository name")
    parser.add_argument("--config", help="Path to YAML configuration file")
    parser.add_argument("--question-column", default="Question", help="Name of question column")
    parser.add_argument("--answer-column", default="Answer", help="Name of answer column")
    parser.add_argument("--split", default="train", help="Dataset split to use")
    parser.add_argument("--max-samples", type=int, help="Maximum number of samples to convert")
    parser.add_argument("--output", default="sharegpt_output.json", help="Output JSON file path")
    parser.add_argument("--upload", action="store_true", help="Upload to HuggingFace Hub")
    parser.add_argument("--dataset-path", help="HuggingFace dataset path for upload")
    
    args = parser.parse_args()
    
    # Initialize converter
    converter = DatasetToShareGPTConverter(args.config)
    
    # Convert dataset
    sharegpt_data = converter.convert_dataset_to_sharegpt(
        repo_name=args.repo_name,
        question_column=args.question_column,
        answer_column=args.answer_column,
        split=args.split,
        max_samples=args.max_samples
    )
    
    if sharegpt_data:
        # Save to file
        converter.save_sharegpt_data(sharegpt_data, args.output)
        
        # Upload to hub if requested
        if args.upload:
            dataset_path = args.dataset_path or f"interstellarninja/{args.repo_name.replace('/', '_')}_sharegpt"
            converter.upload_to_hub(sharegpt_data, dataset_path, upload=True)
    else:
        print("No data was converted. Please check the dataset and column names.")

if __name__ == "__main__":
    main()