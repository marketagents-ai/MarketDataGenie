# Pythonformer Environments

Quick reference for all supported task environments.

## Overview

| Environment | Config | Context | Documents | Strategy | Example Task |
|-------------|--------|---------|-----------|----------|--------------|
| **base** | `default_config.yaml` | Optional | 0-1 | Direct solving | Solve x² - 4 = 0 |
| **oolong** | `oolong_config.yaml` | 150K+ chars | 1 large file | Chunking + map-reduce | Count rolls in transcript |
| **hotpotqa** | `hotpotqa_config.yaml` | 5-10K chars | 10 small files | Document hopping | Which magazine started first? |
| **swe** | `swe_config.yaml` | Repository | /testbed | Code exploration + editing | Fix bug in calculate_sum() |

## Base Environment

**Purpose**: General problem-solving with code

**Context Format**:
- Optional context passed as variable or small file
- No special processing

**Prompt**: `prompts/base.py`

**Tools**:
- Standard Python libraries (numpy, pandas, sympy, scipy)
- File operations (read_file, save_to_file, list_files)
- Sub-agent for semantic analysis

**Example**:
```python
import sympy as sp
x = sp.Symbol('x')
result = sp.solve(x**2 - 4, x)
print(f"Solutions: {result}")
```

**Use Cases**:
- Math problems
- Algorithm implementation
- Data analysis
- Code generation

## OOLONG Environment

**Purpose**: Long-context document analysis

**Context Format**:
- Single `context.txt` file (150K+ characters)
- Saved to workspace at start

**Prompt**: `prompts/oolong.py`

**Tools**:
- `read_file('context.txt')` - Load full context
- `read_file('context.txt', lines=100)` - Read last N lines
- `search_files(query, pattern)` - Search across files
- `sub_agent(task, system_prompt)` - Semantic analysis of chunks

**Strategy**:
1. Load context file
2. Chunk into manageable pieces
3. Use regex for syntactic patterns
4. Use sub-agent for semantic understanding
5. Aggregate results

**Example**:
```python
context = read_file('context.txt')
lines = context.split('\n')
chunk_size = 200
results = []

for i in range(0, len(lines), chunk_size):
    chunk = '\n'.join(lines[i:i+chunk_size])
    count = sub_agent(
        task=f"Count dice rolls in this chunk. Return just the number.\n\n{chunk}",
        system_prompt="You count dice rolls. Return only an integer."
    )
    results.append(int(count) if count.isdigit() else 0)

total = sum(results)
```

**Use Cases**:
- Counting occurrences in long documents
- Aggregating statistics
- Pattern extraction
- Document summarization

**Dataset**: [OOLONG benchmark](https://huggingface.co/datasets/oolongbench/oolong-real)
- D&D transcripts (~150K chars)
- Counting and aggregation tasks

## HotpotQA Environment

**Purpose**: Multi-hop reasoning over multiple documents

**Context Format**:
- Multiple files: `doc_01_Title.txt`, `doc_02_Title.txt`, ...
- Each document is a separate file (10 documents typical)

**Prompt**: `prompts/hotpotqa.py`

**Tools**:
- `list_files("doc_*.txt")` - List all documents
- `read_file(filename)` - Read specific document
- `search_files(query, "doc_*.txt")` - Search across documents
- `get_file_info(filename)` - Get metadata
- `sub_agent(task, system_prompt)` - Extract entities/facts

**Strategy**:
1. List available documents
2. Find relevant documents (filename matching or search)
3. **HOP 1**: Read first document, extract entity/fact
4. **HOP 2**: Find related document, extract answer
5. Compare/combine information

**Example**:
```python
# Step 1: List documents
docs = list_files("doc_*.txt")

# Step 2: Find relevant documents
laleli_files = [d for d in docs if 'Laleli_Mosque' in d]
esma_files = [d for d in docs if 'Esma_Sultan_Mansion' in d]

# Step 3: HOP 1 - Extract from first document
laleli_text = read_file(laleli_files[0])
laleli_neighborhood = sub_agent(
    task=f"What neighborhood is the Laleli Mosque located in?\n\n{laleli_text}",
    system_prompt="Extract the neighborhood name. Return only the name."
)

# Step 4: HOP 2 - Extract from second document
esma_text = read_file(esma_files[0])
esma_neighborhood = sub_agent(
    task=f"What neighborhood is the Esma Sultan Mansion located in?\n\n{esma_text}",
    system_prompt="Extract the neighborhood name. Return only the name."
)

# Step 5: Compare
if laleli_neighborhood.lower() == esma_neighborhood.lower():
    result = "Yes"
else:
    result = "No"
```

**Use Cases**:
- Comparison questions ("Which X was Y first?")
- Bridge questions ("What is the Z of the Y of X?")
- Cross-document fact verification
- Multi-hop reasoning

**Dataset**: [HotpotQA](https://huggingface.co/datasets/hotpotqa/hotpot_qa)
- **Distractor**: 10 documents (2 relevant + 8 distractors)
- **Fullwiki**: Full Wikipedia retrieval (harder)

## SWE Environment

**Purpose**: Software engineering tasks - bug fixing and feature implementation

**Context Format**:
- Repository cloned at `/testbed`
- Pre-configured Docker environment with dependencies installed
- Test suite available

**Prompt**: `prompts/swe.py`

**Tools**:
- **Python blocks** (`<python>`) - For code analysis and file editing
- **Bash blocks** (`<bash>`) - For shell commands
- File operations (read_file, save_to_file)
- Standard Unix tools (grep, find, ls, git, pytest)

**Strategy**:
1. Understand the problem statement
2. Explore codebase structure
3. Locate relevant files
4. Read and analyze code
5. Make necessary changes
6. Run tests to verify fix

**Example**:
```bash
# Step 1: Explore the repository
cd /testbed
find . -name "*.py" | grep -E "(calculator|math)" | head -10
```

```python
# Step 2: Read the buggy file
with open('/testbed/src/calculator.py', 'r') as f:
    content = f.read()
print(content)
```

```bash
# Step 3: Search for the function
grep -rn "def calculate_sum" /testbed/src
```

```python
# Step 4: Fix the bug
with open('/testbed/src/calculator.py', 'r') as f:
    content = f.read()

# Replace the buggy line
new_content = content.replace(
    'return a - b  # BUG: should be addition',
    'return a + b  # Fixed: now correctly adds'
)

with open('/testbed/src/calculator.py', 'w') as f:
    f.write(new_content)

print("✓ Fixed calculate_sum() function")
```

```bash
# Step 5: Run tests
cd /testbed
python -m pytest tests/test_calculator.py -xvs
```

**Use Cases**:
- Bug fixing
- Feature implementation
- Code refactoring
- Test-driven development

**Dataset**: [SWE-bench/SWE-smith-py](https://huggingface.co/datasets/SWE-bench/SWE-smith-py)
- Real-world Python repositories
- Pre-built Docker images with repos at `/testbed`
- Test suites for validation (FAIL_TO_PASS, PASS_TO_PASS)

**Key Features**:
- **Dual execution modes**: Python for precise editing, Bash for exploration
- **Test-based validation**: Solutions verified by running test suites
- **Pre-configured environments**: Docker images with all dependencies
- **Real repositories**: Actual open-source projects (conan, pytest, etc.)

## Comparison

### Context Management

| Environment | Files | Size | Access Pattern |
|-------------|-------|------|----------------|
| base | 0-1 | Small | Direct |
| oolong | 1 | 150K+ chars | Sequential chunks |
| hotpotqa | 10 | 5-10K total | Selective hopping |
| swe | Many | Repository | Exploration + targeted edits |

### Sub-Agent Usage

| Environment | Sub-Agent Purpose | Frequency |
|-------------|-------------------|-----------|
| base | Optional semantic analysis | Rare |
| oolong | Chunk analysis (map-reduce) | High (5-10 calls) |
| hotpotqa | Entity extraction | Medium (2-3 calls) |
| swe | Code understanding (optional) | Low (0-2 calls) |

### Complexity

| Environment | Reasoning Type | Typical Turns | Code Blocks |
|-------------|----------------|---------------|-------------|
| base | Direct | 2-4 | 1-2 |
| oolong | Aggregation | 4-8 | 3-6 |
| hotpotqa | Multi-hop | 2-4 | 1-2 |
| swe | Iterative debugging | 5-15 | 3-10 |

## Running Examples

```bash
# Base: Math problems
python -m datagenie.pythonformer.run \
    --config datagenie/pythonformer/configs/default_config.yaml \
    --limit 10

# OOLONG: Long-context analysis
python -m datagenie.pythonformer.run \
    --config datagenie/pythonformer/configs/oolong_config.yaml \
    --limit 4 \
    --debug

# HotpotQA: Multi-hop reasoning
python -m datagenie.pythonformer.run \
    --config datagenie/pythonformer/configs/hotpotqa_config.yaml \
    --limit 4 \
    --debug

# SWE: Software engineering tasks
python -m datagenie.pythonformer.run \
    --config datagenie/pythonformer/configs/swe_config.yaml \
    --limit 2 \
    --debug
```

## Adding New Environments

To add a new environment:

1. **Create prompt**: `prompts/your_env.py`
2. **Add environment type**: `config.py` → `EnvironmentType.YOUR_ENV`
3. **Create config**: `configs/your_env_config.yaml`
4. **Add context processor**: `pipeline.py` → `_process_context()`
5. **Update prompt builder**: `pipeline.py` → `_build_system_prompt()`

See `prompts/README.md` for detailed instructions.
