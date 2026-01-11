# MarketAgents Function Calling Dataset Generation Pipeline

A modular pipeline for generating multi-turn function calling datasets using the `minference` + `market_agents` SDK.

## Features

- **Two generation modes**:
  - **Curriculum mode** (default): Generate tools and queries from task descriptions
  - **HuggingFace mode**: Augment existing single-turn datasets to multi-turn

- **Multi-agent architecture**:
  - Tool Generator Agent (curriculum mode)
  - Query Generator Agent (curriculum mode)
  - Docstring Generator Agent (conditional)
  - Schema Generator Agent
  - Results Generator Agent
  - Follow-up Query Agent

- **Conditional docstring generation**: Only generates docstrings when tools are missing descriptions AND `generate_docstrings=True`

## Installation

Ensure you have the required dependencies:

```bash
pip install minference market_agents pydantic datasets tqdm tenacity python-dotenv
```

## Usage

### Curriculum Mode (Default)

Generate from curriculum CSV/JSONL:

```bash
# Basic usage
python -m datagenie.marketagents_function_calling.run --mode curriculum --limit 10

# Filter by category
python -m datagenie.marketagents_function_calling.run --mode curriculum --categories "Use Apps" --limit 50

# Custom curriculum file
python -m datagenie.marketagents_function_calling.run --mode curriculum --curriculum my_tasks.jsonl --limit 100
```

### HuggingFace Mode

Augment existing datasets:

```bash
# Basic usage
python -m datagenie.marketagents_function_calling.run --mode huggingface --limit 100

# Custom dataset
python -m datagenie.marketagents_function_calling.run --mode huggingface --dataset Salesforce/xlam-function-calling-60k --start 0 --limit 100

# Skip docstring generation
python -m datagenie.marketagents_function_calling.run --mode huggingface --no_docstrings --limit 100
```

### CLI Options

| Option | Default | Description |
|--------|---------|-------------|
| `--mode` | `curriculum` | Generation mode: `curriculum` or `huggingface` |
| `--curriculum` | `configs/curriculum/function_calling.csv` | Curriculum file path |
| `--categories` | `[]` | Filter by categories (curriculum mode) |
| `--subcategories` | `[]` | Filter by subcategories (curriculum mode) |
| `--dataset` | `Salesforce/xlam-function-calling-60k` | HuggingFace dataset |
| `--start` | `0` | Starting index (huggingface mode) |
| `--limit` | `None` | Max tasks to process |
| `--batch_size` | `8` | Parallel batch size |
| `--output_dir` | `outputs/function_calling` | Output directory |
| `--tool_model` | `openai/gpt-4o` | Model for tool calling |
| `--agent_model` | `anthropic/claude-3-5-sonnet-20241022` | Model for agents |
| `--no_validate` | `False` | Disable validation (huggingface mode) |
| `--no_docstrings` | `False` | Disable docstring generation |

## Project Structure

```
datagenie/marketagents_function_calling/
├── __init__.py              # Package exports
├── config.py                # PipelineConfig, GenerationMode
├── schemas.py               # Pydantic output schemas
├── pipeline.py              # Main FunctionCallingPipeline class
├── run.py                   # CLI entry point
├── agents/
│   ├── __init__.py
│   ├── tool_generator.py    # Tool generation (curriculum mode)
│   ├── query_generator.py   # Query generation (curriculum mode)
│   ├── docstring_agent.py   # Docstring generation (conditional)
│   ├── schema_agent.py      # Schema generation
│   ├── results_agent.py     # Results generation
│   └── followup_agent.py    # Follow-up query generation
├── utils/
│   ├── __init__.py
│   ├── validation.py        # Tool call validation
│   └── sharegpt.py          # ShareGPT format conversion
└── configs/
    ├── pipeline_config.yaml
    └── curriculum/
        └── function_calling.csv
```

## Output Format

Results are saved in two formats:

1. **Raw JSONL** (`function_calling_results_*.jsonl`): Full task data with metadata
2. **ShareGPT JSONL** (`function_calling_sharegpt_*.jsonl`): Training-ready format

### ShareGPT Example

```json
{
  "conversations": [
    {"from": "system", "value": "You are a helpful assistant..."},
    {"from": "human", "value": "Order pizza from DoorDash"},
    {"from": "gpt", "value": "Tool Calls:\n[{\"name\": \"order_food\", ...}]"},
    {"from": "tool", "value": "[order_food]: {\"order_id\": \"123\", ...}"},
    {"from": "gpt", "value": "I've placed your order..."},
    {"from": "human", "value": "What's the delivery time?"},
    {"from": "gpt", "value": "Tool Calls:\n[{\"name\": \"get_order_status\", ...}]"},
    {"from": "tool", "value": "[get_order_status]: {\"eta\": \"30 mins\", ...}"},
    {"from": "gpt", "value": "Your order will arrive in 30 minutes."}
  ],
  "tools": [...],
  "metadata": {"id": "...", "mode": "curriculum"}
}
```

## Curriculum Format

### CSV Format

```csv
Category,SubCategory,Task
Use Apps,Food delivery apps,Order Food from Doordash
Use Apps,Ride-hailing apps,Call a Uber
```

### JSONL Format

```json
{"category": "Use Apps", "subcategory": "Food delivery apps", "task": "Order Food from Doordash"}
{"category": "Use Apps", "subcategory": "Ride-hailing apps", "task": "Call a Uber"}
```

## Environment Variables

Set these in your `.env` file:

```bash
OPENAI_KEY=your_openai_key
ANTHROPIC_API_KEY=your_anthropic_key
LITELLM_API_KEY=your_litellm_key  # If using LiteLLM proxy
LITELLM_ENDPOINT=http://localhost:8000/v1/chat/completions
```

## Reference

This pipeline replaces the legacy `datagenie/hermes_function_calling/datagen_salesforce.py` which used the old `src/` framework. The original is preserved for reference.
