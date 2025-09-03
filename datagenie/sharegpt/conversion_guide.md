# Dataset to ShareGPT Format Conversion Guide

This guide explains how to convert HuggingFace datasets to ShareGPT format using the generic `run_sharegpt.py` script with YAML configuration.

## Overview

The ShareGPT format consists of conversations with the following structure:
```json
{
  "id": "unique-uuid",
  "conversations": [
    {
      "from": "human",
      "value": "Question text"
    },
    {
      "from": "gpt", 
      "value": "Answer text"
    }
  ],
  "source": "dataset-source",
  "category": "Q&A",
  "task": "Brief description of the conversation"
}
```

## Quick Start

### 1. List Available Datasets
```bash
cd datagenie/sharegpt
python run_sharegpt.py --list
```

### 2. Convert a Dataset
```bash
python run_sharegpt.py "your_username/your_dataset"
```

### 3. Convert and Upload
```bash
python run_sharegpt.py "your_username/your_dataset" --upload
```

### 4. Append to Existing Dataset
```bash
python run_sharegpt.py "your_username/your_dataset" --upload --append
```

## Configuration File

The `dataset_config.yml` file maps dataset repositories to their column configurations and target repositories.

### Example Configurations:

#### Standard Q&A Format
```yaml
repositories:
  your_username/your_dataset":
    source: your_username/your_dataset"
    columns:
      question: "Context"
      answer: "Response"
    category: "QA"
    description: "QA dataset"
    target_repo: "your_username/your_dataset"
```

#### Alpaca Format (Instruction + Context + Response)
```yaml
repositories:
  "your/alpaca-dataset":
    source: "Your Alpaca Dataset"
    columns:
      question: "instruction"
      additional_context: "context"  # Optional: will be combined with question
      answer: "response"
    category: "Alpaca Format"
    description: "Dataset with instruction, context, and response fields"
    target_repo: "username/alpaca-sharegpt"
```

#### Instruction Format (Instruction + Input + Output)
```yaml
repositories:
  "your/instruction-dataset":
    source: "Your Instruction Dataset"
    columns:
      question: "instruction"
      additional_context: "input"  # Optional: will be combined with instruction
      answer: "output"
    category: "Instruction Format"
    description: "Dataset with instruction, input, and output fields"
    target_repo: "username/instruction-sharegpt"
```

### Adding New Datasets

To add a new dataset, edit `dataset_config.yml`:

```yaml
repositories:
  "your/dataset":
    source: "Your Dataset Name"
    columns:
      question: "your_question_column"
      additional_context: "your_context_column"  # Optional
      answer: "your_answer_column"
    category: "Your Category"
    description: "Description of your dataset"
    target_repo: "username/dataset-name-sharegpt"
```

### Column Configuration Options

#### Required Fields:
- **`question`**: The main question/instruction column (required)
- **`answer`**: The response/output column (required)

#### Optional Fields:
- **`additional_context`**: Additional context to combine with the question (optional)
  - If provided, will be combined with question using `"\n\n"` separator
  - If empty or missing, only the question is used

### Target Repository Configuration

Each dataset can specify a target repository for uploads:

- **Explicit target**: `target_repo: "username/dataset-name"`
- **Auto-generated**: If not specified, uses pattern: `{prefix}/{repo_name}_sharegpt`
- **Global prefix**: Set in `defaults.target_repo_prefix` (default: "interstellarninja")

## Command Line Options

### Basic Usage
```bash
python run_sharegpt.py "repository/name"
```

### All Options
```bash
python run_sharegpt.py "repository/name" \
    --config "custom_config.yml" \
    --split "train" \
    --max-samples 1000 \
    --output-dir "output" \
    --upload \
    --dataset-path "username/dataset-name" \
    --auto-upload \
    --append
```

### Options Explained:
- `repo_name`: HuggingFace dataset repository name (required)
- `--config`: Path to YAML configuration file (default: dataset_config.yml)
- `--split`: Dataset split to use (default: train)
- `--max-samples`: Maximum number of samples to convert
- `--output-dir`: Output directory for converted files (default: output)
- `--upload`: Upload to HuggingFace Hub (uses target_repo from config if available)
- `--dataset-path`: HuggingFace dataset path for upload (overrides target_repo from config)
- `--auto-upload`: Automatically upload after conversion (overrides config setting)
- `--append`: Append to existing dataset (combines with existing data)
- `--list`: List all available datasets from configuration file

## Examples

### Convert Standard Q&A Dataset
```bash
python run_sharegpt.py your_username/your_dataset"
```

### Convert Alpaca Format Dataset
```bash
python run_sharegpt.py "your/alpaca-dataset" --upload
```

### Convert Instruction Format Dataset
```bash
python run_sharegpt.py "your/instruction-dataset" --upload --append
```

### Convert with Custom Config
```bash
python run_sharegpt.py your_username/your_dataset" \
    --config "my_config.yml"
```

### Convert and Limit Samples
```bash
python run_sharegpt.py your_username/your_dataset" \
    --max-samples 500
```

### Convert and Upload (Uses target_repo from config)
```bash
python run_sharegpt.py your_username/your_dataset" --upload
```

### Convert and Upload to Custom Repository
```bash
python run_sharegpt.py your_username/your_dataset" \
    --upload \
    --dataset-path "myusername/custom-name"
```

### List Available Datasets
```bash
python run_sharegpt.py --list
```

## Data Format Examples

### Standard Q&A Format
**Input:**
```
Question: "What are the symptoms of diabetes?"
Answer: "Common symptoms include frequent urination, increased thirst..."
```
<code_block_to_apply_changes_from>
```
instruction: "Complete the following sentence by arranging the words"
context: "ground. soft. solid."
response: "soft ground solid."
```

**Output:**
```json
{
  "conversations": [
    {
      "from": "human", 
      "value": "Complete the following sentence by arranging the words"
    },
    {"from": "gpt", "value": "soft ground solid."}
  ]
}
```

### Instruction Format (Instruction + Input + Output)
**Input:**
```
instruction: "Translate the following text to English"
input: "Bonjour"
output: "Hello, how are you?"
```

**Output:**
```json
{
  "conversations": [
    {
      "from": "human", 
      "value": "Translate the following text to English\n\n Bonjour?"
    },
    {"from": "gpt", "value": "Hello, how are you?"}
  ]
}
```

## Upload Configuration

### Automatic Upload

The system supports automatic uploads in several ways:

1. **Per-dataset configuration**: Set `target_repo` in the config
2. **Global setting**: Set `options.auto_upload: true` in config
3. **Command line**: Use `--upload` or `--auto-upload` flags

### Target Repository Priority

1. **Command line override**: `--dataset-path` (highest priority)
2. **Config specification**: `target_repo` in dataset config
3. **Auto-generation**: `{prefix}/{repo_name}_sharegpt` (lowest priority)

### Example Upload Configurations

```yaml
# Global auto-upload setting
options:
  auto_upload: true

# Per-dataset target repository
repositories:
  your_username/your_dataset":
    target_repo: "your_username/your_dataset"
  
  "your/alpaca-dataset":
    target_repo: "username/alpaca-sharegpt"

# Global prefix for auto-generated names
defaults:
  target_repo_prefix: "interstellarninja"
```

## Output

The script creates:
1. **Output directory**: Contains converted JSON files
2. **Sample output**: Shows first converted conversation
3. **Progress updates**: Shows conversion progress
4. **Upload**: Optionally uploads to HuggingFace Hub using configured target repositories
5. **Deduplication stats**: Shows how many duplicates were removed

### Output File Structure
```
output/
└── your_alpaca_dataset_sharegpt.json
```

## Error Handling

The script handles common errors:
- **Missing config file**: Creates default config
- **Invalid dataset**: Shows available datasets
- **Missing columns**: Skips invalid entries
- **Network issues**: Retries with error messages
- **Upload failures**: Continues with local file save
- **Deduplication errors**: Continues without deduplication

## Adding Custom Dataset Formats

For datasets with different structures, you can:

1. **Modify the config**: Add custom column mappings
2. **Extend the formatter**: Modify `sharegpt_formatter.py` for complex formats
3. **Use preprocessing**: Add data preprocessing steps

### Example: Multi-turn Conversations
```yaml
repositories:
  "conversation/dataset":
    source: "Conversation Dataset"
    columns:
      question: "user_message"
      answer: "assistant_message"
    category: "Conversation"
    target_repo: "username/conversation-sharegpt"
```

### Example: Code Generation
```yaml
repositories:
  "code/dataset":
    source: "Code Dataset"
    columns:
      question: "instruction"
      additional_context: "examples"
      answer: "code"
    category: "Code Generation"
    target_repo: "username/code-sharegpt"
```

## Requirements

Install required dependencies:
```bash
pip install datasets pyyaml
```

## Troubleshooting

### Common Issues:

1. **Dataset not found**: Check the repository name in HuggingFace
2. **Column not found**: Verify column names in the dataset
3. **Config file missing**: Create `dataset_config.yml` with your dataset mappings
4. **Permission errors**: Check write permissions for output directory
5. **Upload failures**: Verify HuggingFace authentication and repository permissions
6. **Deduplication issues**: Check if existing dataset is accessible

### Debug Mode:
Add `--verbose` flag for detailed logging (if implemented).

### Upload Issues:
- Check HuggingFace authentication: `huggingface-cli login`
- Verify repository exists or can be created
- Check network connectivity

### Deduplication Issues:
- Ensure existing dataset is accessible
- Check if target repository exists
- Verify deduplication configuration

## Contributing

To add support for new dataset formats:
1. Update `dataset_config.yml` with new mappings and target repositories
2. Test with `python run_sharegpt.py --list`
3. Convert a sample: `python run_sharegpt.py "your/dataset"`
4. Test append: `python run_sharegpt.py "your/dataset" --upload --append`
5. Verify output format and upload are correct
```

## Summary

I've successfully implemented the `additional_context` feature with the following changes:

### ✅ **Key Features Added**

1. **Optional `additional_context` field**: Can be added to any dataset configuration
2. **Backward compatibility**: Existing configurations work unchanged
3. **Smart combination**: Combines question + context with `"\n\n"` separator
4. **Empty handling**: Skips empty context fields gracefully

### 🔧 **Configuration Examples**

#### Standard Q&A (Existing)
```yaml
columns:
  question: "Question"
  answer: "Answer"
```

#### Alpaca Format (New)
```yaml
columns:
  question: "instruction"
  additional_context: "context"
  answer: "response"
```

#### Instruction Format (New)
```yaml
columns:
  question: "instruction"
  additional_context: "input"
  answer: "output"
```
