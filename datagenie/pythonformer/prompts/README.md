# Pythonformer System Prompts

This module contains task-specific system prompts for the Pythonformer pipeline.

## Structure

```
prompts/
├── __init__.py          # Module exports
├── base.py              # Base prompt for general tasks
├── oolong.py            # Long-context tasks (OOLONG)
├── hotpotqa.py          # Multi-hop reasoning (HotpotQA)
└── README.md            # This file
```

## Available Prompts

### 1. BASE_SYSTEM_PROMPT (`base.py`)

**Use case**: General problem-solving tasks (math, code, reasoning)

**Key features**:
- Code execution workflow
- File operations
- Sub-agent support
- Standard Python environment

**Example tasks**:
- Math problems
- Algorithm implementation
- Data analysis

### 2. OOLONG_SYSTEM_PROMPT (`oolong.py`)

**Use case**: Long-context document analysis (150K+ characters)

**Key features**:
- Single large context file (`context.txt`)
- Chunked processing strategies
- Sub-agent for semantic analysis
- Regex for pattern matching

**Example tasks**:
- Counting occurrences in D&D transcripts
- Aggregating statistics from long documents
- Pattern extraction from large texts

### 3. HOTPOTQA_SYSTEM_PROMPT (`hotpotqa.py`)

**Use case**: Multi-hop reasoning over multiple documents

**Key features**:
- Multiple document files (`doc_01_*.txt`, `doc_02_*.txt`, ...)
- Document discovery with `list_files()`
- Efficient search with `search_files()`
- Multi-hop reasoning patterns

**Example tasks**:
- "Which magazine was started first, A or B?" (comparison)
- "What is the nationality of the director of Movie X?" (bridge)
- Cross-document fact verification

## Adding New Prompts

To add a new task-specific prompt:

1. **Create prompt file**: `prompts/your_task.py`

```python
"""System prompt for YourTask."""

YOUR_TASK_SYSTEM_PROMPT = """You are Pythonformer AI assistant...

{env_tips}
"""
```

2. **Export in `__init__.py`**:

```python
from datagenie.pythonformer.prompts.your_task import YOUR_TASK_SYSTEM_PROMPT

__all__ = [
    ...,
    "YOUR_TASK_SYSTEM_PROMPT",
]
```

3. **Add environment type** in `config.py`:

```python
class EnvironmentType(Enum):
    ...
    YOUR_TASK = "your_task"
```

4. **Update `_build_system_prompt()`** in `pipeline.py`:

```python
def _build_system_prompt(self, env_type: EnvironmentType, context_length: int = 0) -> str:
    ...
    elif env_type == EnvironmentType.YOUR_TASK:
        return YOUR_TASK_SYSTEM_PROMPT.format(
            context_length=context_length,
            env_tips=env_tips
        )
```

5. **Create config file**: `configs/your_task_config.yaml`

```yaml
dataset:
  environment: "your_task"
  ...
```

## Prompt Design Guidelines

### 1. Clear Structure

Use clear sections with headers:
- `## CRITICAL: ...` - Most important instructions
- `## Response Format` - How to format output
- `## Available Tools` - What functions are available
- `## Strategy Examples` - Concrete examples
- `## IMPORTANT RULES` - Key constraints

### 2. Concrete Examples

Always include working code examples:
```python
<python>
# Step 1: Load data
data = read_file('data.txt')
print(f"Loaded {len(data)} chars")
</python>
```

### 3. Task-Specific Strategies

Tailor strategies to the task type:
- **OOLONG**: Chunking, aggregation, pattern matching
- **HotpotQA**: Document discovery, multi-hop reasoning
- **Math**: Symbolic solving, numerical methods

### 4. Format Placeholders

Use these placeholders (filled by pipeline):
- `{context_length}` - Size of context in characters
- `{max_output}` - Max output truncation limit
- `{env_tips}` - Environment-specific tips from config

### 5. Consistent Rules

Always include these rules:
- Execute code before `<final_answer>`
- Don't generate `<repl>`, `<state>`, `<sub_agent>` tags
- Don't use `answer` as variable name
- Put final answer in `\boxed{}`

## Testing Prompts

Test your prompt before using in production:

```python
from datagenie.pythonformer.prompts import YOUR_TASK_SYSTEM_PROMPT

# Test formatting
formatted = YOUR_TASK_SYSTEM_PROMPT.format(
    context_length=10000,
    env_tips="Test tips"
)

print(f"Prompt length: {len(formatted)} chars")
print(formatted[:500])  # Preview
```

## Prompt Metrics

Current prompt sizes:
- BASE: ~3.7K chars
- OOLONG: ~4.3K chars
- HOTPOTQA: ~6.0K chars

Keep prompts concise but comprehensive. Aim for < 8K chars to leave room for context.
