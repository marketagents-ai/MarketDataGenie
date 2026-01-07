#!/bin/bash
set -euo pipefail

# === CONFIG ===
HF_PATH="your_username/your-textbook-dataset-multiturn"   # Dataset repo
SPLIT="train"                                   # Dataset split
NUM_TURNS=3                                     # Number of turns per conversation
BATCH_SIZE=32                                   # Parallel requests per batch
REASONING_LANGUAGE="English"                   # Language used for reasoning in <think> blocks
ANSWER_LANGUAGE="English"                      # Language used for final answers
START=""                                        # Starting index
LIMIT=""                                        # Number of rows to process; set to "" for all
SEED_QUESTION=false                             # Set to true to generate a fresh seed question
SKIP_PUBLISHED=true                             # Avoid regenerating existing samples
PUBLISHED_REPO="your_username/your-textbook-dataset-multiturn"
PUBLISHED_SPLIT="train"

# NEW: filter only rows whose top-level `source` equals this exact string
SOURCE="your_username/your-textbook-dataset"

# === ENVIRONMENT ===
echo "[INFO] Activating virtualenv..."
source .venv/bin/activate

# === OUTPUT DIR ===
mkdir -p outputs/multiturn_alpaca   

# === RUN PIPELINE ===
echo "[INFO] Starting dataset generation..."
CMD="python datagenie/textbooks_qa/run_multiturn_qa_parallel_alpaca.py \
    --hf_path ${HF_PATH} \
    --split ${SPLIT} \
    --num_turns ${NUM_TURNS} \
    --batch_size ${BATCH_SIZE} \
    --reasoning_language \"${REASONING_LANGUAGE}\" \
    --answer_language \"${ANSWER_LANGUAGE}\" \
    --published_repo_id ${PUBLISHED_REPO} \
    --published_split ${PUBLISHED_SPLIT}"

if [ -n "${START}" ]; then
    CMD="$CMD --start ${START}"
fi
if [ -n "${LIMIT}" ]; then
    CMD="$CMD --limit ${LIMIT}"
fi

if [ -n "${SOURCE}" ]; then
    CMD="$CMD --source \"${SOURCE}\""
fi

if [ "${SKIP_PUBLISHED}" = true ]; then
    CMD="$CMD --skip_published"
fi
if [ "${SEED_QUESTION}" = true ]; then
    CMD="$CMD --seed_question"
fi

echo "[INFO] Running: $CMD"
nohup bash -c "$CMD" > "outputs/multiturn_alpaca/datagen_$(date +'%Y%m%d_%H%M%S').log" 2>&1 &

echo "[INFO] Data generation started in background!"
echo "[INFO] Check logs with: tail -f outputs/multiturn_alpaca/datagen_*.log"