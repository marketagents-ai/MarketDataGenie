# ShareGPT Dataset Converter

This directory contains tools for converting HuggingFace datasets to ShareGPT format with automatic upload support.

## Files

- `run_sharegpt.py` - Main script for converting datasets
- `sharegpt_formatter.py` - Core conversion logic
- `dataset_config.yml` - Configuration file for dataset mappings and target repositories
- `conversion_guide.md` - Detailed usage guide
- `test_setup.py` - Test script to verify setup
- `README.md` - This file

## Quick Start

1. **Test the setup:**
   ```bash
   python test_setup.py
   ```

2. **List available datasets:**
   ```bash
   python run_sharegpt.py --list
   ```

3. **Convert a dataset:**
   ```bash
   python run_sharegpt.py "your-username/your-dataset"
   ```

4. **Convert and upload:**
   ```bash
   python run_sharegpt.py "your-username/your-dataset" --upload
   ```

## Configuration

Edit `dataset_config.yml` to add new datasets with target repositories:

```yaml
repositories:
  "your/dataset":
    source: "Your Dataset Name"
    columns:
      question: "your_question_column"
      answer: "your_answer_column"
    category: "Your Category"
    target_repo: "username/dataset-name-sharegpt"  # Target repo for upload
```

### Target Repository Options

- **Explicit**: Set `target_repo` in config for each dataset
- **Auto-generated**: Uses pattern `{prefix}/{repo_name}_sharegpt`
- **Global prefix**: Configure in `defaults.target_repo_prefix`

## Output

Converted datasets are saved to the `output/` directory in ShareGPT format:

```json
{
  "id": "unique-uuid",
  "conversations": [
    {"from": "human", "value": "Question"},
    {"from": "gpt", "value": "Answer"}
  ],
  "source": "dataset-source",
  "category": "Q&A"
}
```

## Upload Features

- **Automatic upload** to configured target repositories
- **Config-driven** target repository specification
- **Command line override** with `--dataset-path`
- **Error handling** for upload failures

## Requirements

```bash
pip install datasets pyyaml
huggingface-cli login  # For upload functionality
```

## More Information

See `conversion_guide.md` for detailed usage instructions, examples, and troubleshooting.
