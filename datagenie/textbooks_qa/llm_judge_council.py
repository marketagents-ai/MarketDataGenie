#!/usr/bin/env python3
"""
council_verify_math.py

Pipeline:
  - Read ShareGPT-style records (system -> human -> gpt ...).
  - Extract the final assistant answer and pull the boxed value: \boxed{...}.
  - Build a compact judge prompt per item.
  - Run a council of LLM judges in parallel via your InferenceOrchestrator.
  - Aggregate verdicts (majority + mean score), save verification block.
  - Write ALL judged items to <out_all.jsonl> (no acceptance filtering).

Notes:
  - We do NOT do deterministic equality here (per your instruction). We only extract
    a normalized boxed payload for judges to reference and store it for downstream use.
  - Judges must call the `judge_tool` and return a structured object; we still defensively parse content if a model leaks text.
  - Keep judges’ outputs tiny: JSON only.

Input JSONL record shape (minimal):
{
  "id": "...",
  "conversations": [
    {"from":"system","value":"..."}?,     # optional
    {"from":"human","value":"<question>"},
    {"from":"gpt","value":"... \\boxed{...} ..."},
    ...
  ],
  "...": "other metadata ok"
}

Output fields added on verified:
  "verification": {
    "boxed": "<raw inside \\boxed{...}>",
    "judges": [{"model":"...", "verdict":"correct|incorrect|unclear", "score":0..1, "reason":"..."}],
    "aggregate": {"method":"majority+mean", "verdict":"...", "score":0.xxx, "tally":{"correct":X,"incorrect":Y,"unclear":Z}}
  }

CLI:
  python llm_judge_council.py \
    --in_repo your_username/your_dataset \
    --split train \
    --subject Math \
    --out_all data/math_judged_all.jsonl \
    --batch_size 128 \
    --judges_per_item 5
"""

import os
import re
import json
import argparse
import random
from pathlib import Path
from typing import Any, Dict, List, Tuple, Optional

import asyncio

from datasets import load_dataset

# Your infra
from minference.lite.inference import InferenceOrchestrator
from minference.enregistry import EntityRegistry
from minference.lite.models import (
    ChatThread,
    LLMConfig,
    ResponseFormat,
    StructuredTool,
    SystemPrompt,
    LLMClient,
)
from pydantic import BaseModel, Field

EntityRegistry()
# ----------------------------
# Config
# ----------------------------

DEFAULT_JUDGE_POOL = [
    "Hermes-4-70B",
    "Hermes-4-405B",
    "Hermes-3-Llama-3.1-70B",
    "DeepHermes-3-Llama-3-8B-Preview",
    "DeepHermes-3-Mistral-24B-Preview",
    "Hermes-3-Llama-3.1-405B",
#    "Qwen/Qwen3-30B-A3B-Instruct-2507",
#    "deepseek-ai/DeepSeek-R1-Distill-Qwen-32B",
#    "mistralai/Ministral-8B-Instruct-2410",
#    "meta-llama/Llama-3.1-8B-Instruct",
#    "google/gemma-3-12b-it",
#    "openai/gpt-oss-20b",
#    "NousResearch/DeepHermes-3-Llama-3-8B-Preview",
    # You can include thinking model only if you trust it to output strict JSON:
    # "Qwen/Qwen3-30B-A3B-Thinking-2507",
]

class MathJudgeVerdict(BaseModel):
    verdict: str = Field(..., pattern="^(correct|incorrect|unclear)$")
    score: float = Field(..., ge=0.0, le=1.0)
    reason: str = Field(..., min_length=1, max_length=160)

judge_tool = StructuredTool.from_pydantic(
    model=MathJudgeVerdict,
    name="judge_tool",
    description="Return a strict math-judge verdict: {verdict, score, reason}"
)

BOX_RE = re.compile(r"\\boxed\\{([^}]*)\\}")

# ----------------------------
# I/O helpers
# ----------------------------

def _ensure_parent(path: str) -> None:
    Path(path).parent.mkdir(parents=True, exist_ok=True)

def read_jsonl(path: str) -> List[Dict[str, Any]]:
    out = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                out.append(json.loads(line))
            except Exception:
                continue
    return out

def append_jsonl(path: str, rec: Dict[str, Any]) -> None:
    _ensure_parent(path)
    with open(path, "a", encoding="utf-8") as f:
        f.write(json.dumps(rec, ensure_ascii=False) + "\n")

# ----------------------------
# Conversation helpers
# ----------------------------

def last_assistant(conversations: List[Dict[str, str]]) -> Optional[str]:
    if not isinstance(conversations, list):
        return None
    for msg in reversed(conversations):
        if msg.get("from") == "gpt":
            val = (msg.get("value") or "").strip()
            if val:
                return val
    return None

def first_human(conversations: List[Dict[str, str]]) -> Optional[str]:
    if not isinstance(conversations, list):
        return None
    for msg in conversations:
        if msg.get("from") == "human":
            val = (msg.get("value") or "").strip()
            if val:
                return val
    return None

def extract_boxed(text: Optional[str]) -> Optional[str]:
    if not text:
        return None
    m = BOX_RE.search(text)
    return m.group(1).strip() if m else None

# ----------------------------
# Judge prompt & parsing
# ----------------------------

def build_judge_user_msg(question: str, answer_text: str, boxed_payload: Optional[str]) -> str:
    # Keep ultra-compact; ask for JSON verdict and short reason. No chain, no steps.
    return (
        "You are a strict math validator.\n"
        "Call the tool `judge_tool` with a STRICT object: {verdict, score, reason}.\n"
        "verdict must be one of: correct, incorrect, unclear. score must be 0..1.\n"
        "If there is no \\boxed{...} (or boxed_payload is NONE), set verdict=unclear and score<=0.3.\n\n"
        f"Question:\n{question}\n\n"
        f"Assistant final answer:\n{answer_text}\n\n"
        f"Extracted boxed payload (if any):\n{boxed_payload if boxed_payload is not None else 'NONE'}\n"
        "Only call the tool. Do not write free-form text."
    )

def safe_json_parse(s: str) -> Dict[str, Any]:
    s = (s or "").strip()
    # Heuristic: find first '{' and last '}' to trim junk
    l, r = s.find("{"), s.rfind("}")
    if l != -1 and r != -1 and r > l:
        s = s[l:r+1]
    try:
        obj = json.loads(s)
        if not isinstance(obj, dict):
            raise ValueError
        return obj
    except Exception:
        # Fallback: super-conservative defaults
        return {"verdict": "unclear", "score": 0.0, "reason": "malformed_json"}

# ----------------------------
# Council orchestration
# ----------------------------

def choose_judges(pool: List[str], k: int, seed: int) -> List[str]:
    rnd = random.Random(seed)
    if k >= len(pool):
        return pool[:]
    return rnd.sample(pool, k)

def new_judge_thread(model_id: str, user_msg: str) -> ChatThread:
    cfg = LLMConfig(
        client=LLMClient.litellm,
        model=model_id,
        temperature=0.0,
        max_tokens=512,
        response_format=ResponseFormat.auto_tools,
    )
    sys = SystemPrompt(name="math_judge", content="Use the judge_tool. Output must be a tool call.")
    th = ChatThread(
        system_prompt=sys,
        llm_config=cfg,
        tools=[judge_tool],
    )
    th.new_message = user_msg
    #th.prefill="<think>"
    return th

async def run_council_batch(
    problems: List[Tuple[str, str, Optional[str], Dict[str, Any]]],  # (qid, question, boxed, record)
    judge_pool: List[str],
    judges_per_item: int,
    batch_size: int,
    orchestrator: Optional[InferenceOrchestrator] = None,
    seed_offset: int = 0,
) -> List[Tuple[Dict[str, Any], Dict[str, Any]]]:
    """
    Returns list of (record, verification_block) aligned to problems.
    verification_block = {
      "boxed": "...",
      "judges": [{"model":..., "verdict":..., "score":..., "reason":...}, ...],
      "aggregate": {...}
    }
    """
    orch = orchestrator or InferenceOrchestrator()
    # Build judge threads for entire slab
    threads: List[ChatThread] = []
    meta: List[Tuple[int, str]] = []  # (idx_in_problems, model)
    for i, (qid, qtext, boxed, rec) in enumerate(problems):
        # Collect last assistant text
        ans_text = (rec.get("assistant_text") or last_assistant(rec.get("conversations") or []) or "")
        user_msg = build_judge_user_msg(qtext, ans_text, boxed)
        judges = choose_judges(judge_pool, judges_per_item, seed=hash(qid) + seed_offset)
        for m in judges:
            th = new_judge_thread(m, user_msg)
            threads.append(th)
            meta.append((i, m))

    # Run in slabs to control memory/latency, batching by samples, not raw judge threads.
    outs_all: List[Any] = []
    threads_per_sample = judges_per_item
    num_samples = len(problems)
    num_slabs = (num_samples + batch_size - 1) // batch_size if batch_size > 0 else 1
    for slab_idx in range(num_slabs):
        sample_start = slab_idx * batch_size
        sample_end = min(sample_start + batch_size, num_samples)
        # Map sample range to contiguous thread indices because we append judges per sample in order
        thread_start = sample_start * threads_per_sample
        thread_end = sample_end * threads_per_sample
        slab = threads[thread_start:thread_end]
        # Optional: helpful logging
        print(f"[council] processing samples {sample_start}:{sample_end} (count={sample_end - sample_start}) × {threads_per_sample} judges = {len(slab)} threads")
        if not slab:
            continue
        outs = await orch.run_parallel_ai_completion(slab)
        outs_all.extend(outs)

    # Collect votes per item (only include successful tool calls)
    votes_per_item: List[List[Dict[str, Any]]] = [[] for _ in problems]
    for (i, model), out in zip(meta, outs_all):
        obj: Optional[Dict[str, Any]] = None

        # 1) Prefer direct tool output
        tool = getattr(out, "tool_output", None)
        if tool is not None and hasattr(tool, "object") and isinstance(tool.object, dict):
            obj = tool.object

        # 2) Try json_object entity
        if obj is None:
            json_obj = getattr(out, "json_object", None)
            if json_obj is not None:
                if hasattr(json_obj, "object") and isinstance(json_obj.object, dict):
                    obj = json_obj.object
                elif isinstance(json_obj, dict):
                    obj = json_obj

        # If we still don't have a valid tool call, skip this judge entirely
        if obj is None or not isinstance(obj, dict):
            continue

        # Extract fields safely
        v = obj.get("verdict")
        sc = obj.get("score")
        rsn = obj.get("reason") or ""

        # Skip judges missing mandatory fields
        if v not in {"correct", "incorrect", "unclear"} or sc is None:
            continue

        try:
            sc = float(sc)
        except Exception:
            continue

        votes_per_item[i].append({
            "model": model,
            "verdict": v,
            "score": max(0.0, min(1.0, sc)),
            "reason": rsn,
        })

    # Aggregate
    results: List[Tuple[Dict[str, Any], Dict[str, Any]]] = []
    for (qid, _qtext, boxed, rec), votes in zip(problems, votes_per_item):
        tally = {"correct": 0.0, "incorrect": 0.0, "unclear": 0.0}
        scores = 0.0
        for v in votes:
            verdict = v["verdict"] if v["verdict"] in tally else "unclear"
            tally[verdict] += 1.0
            scores += v["score"]
        total = float(len(votes))
        avg_score = round(scores / (total or 1.0), 3)

        if total == 0:
            majority = "unclear"
        else:
            majority = max(tally, key=tally.get)
        verification = {
            "boxed": boxed,
            "judges": votes,
            "aggregate": {
                "method": "majority+mean",
                "verdict": majority,
                "score": avg_score,
                "tally": tally,
            },
        }
        results.append((rec, verification))
    return results

# ----------------------------
# Runner
# ----------------------------

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Council-of-Judges math verification over a HuggingFace dataset.")
    p.add_argument("--in_repo", required=True, help="HuggingFace dataset repo id (e.g., your_username/your_dataset)")
    p.add_argument("--split", default="train", help="HF dataset split (default: train)")
    p.add_argument("--subject", default=None, help="Optional subject filter (exact match), e.g., 'Math'")
    p.add_argument("--out_all", required=True, help="Output JSONL for ALL judged items (no acceptance filtering)")
    p.add_argument("--batch_size", type=int, default=256, help="Parallel slab size for judge calls")
    p.add_argument("--judges_per_item", type=int, default=5, help="How many judges per sample")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--max_items", type=int, default=None, help="Cap number of items (for quick tests)")
    p.add_argument("--judge_pool", type=str, default=None, help="Comma-separated list to override default judge pool")
    return p.parse_args()

def is_numeric_math(q: str) -> bool:
    q = (q or "").lower()
    # cheap heuristic; you can replace with tags in metadata
    return any(k in q for k in ["गणना", "जोड", "घटाउ", "गुण", "भाग", "calculate", "solve", "कति", "मान", "value"])

def build_problem_units(records: List[Dict[str, Any]]) -> List[Tuple[str, str, Optional[str], Dict[str, Any]]]:
    problems = []
    for rec in records:
        qid = str(rec.get("id") or "")
        conv = rec.get("conversations") or []

        # Prefer flattened fields when present
        qtext = (rec.get("question") or first_human(conv) or "").strip()
        ans_text = (rec.get("assistant_text") or last_assistant(conv) or "").strip()

        if not qtext or not ans_text:
            continue

        # Require upstream extraction (dataset provides this). Do NOT fallback to reparsing.
        boxed = rec.get("extracted_answer") or  rec.get("boxed_raw")
        if boxed is None:
            # Skip records without a pre-extracted answer to avoid wasting judge calls.
            continue
        boxed = str(boxed).strip()
        if not boxed:
            # Skip empty strings as well.
            continue

        problems.append((qid, qtext, boxed, rec))
    return problems

async def main_async(args: argparse.Namespace) -> None:
    random.seed(args.seed)

    judge_pool = [s.strip() for s in (args.judge_pool.split(",") if args.judge_pool else []) if s.strip()] or DEFAULT_JUDGE_POOL
    print(f"[cfg] judge_pool={judge_pool}")
    print(f"[cfg] judges_per_item={args.judges_per_item} batch_size={args.batch_size}")

    # Load from HF repo and optionally filter by subject
    hf_ds = load_dataset(args.in_repo, split=args.split)
    records = [dict(r) for r in hf_ds]
    if args.subject:
        records = [r for r in records if (r.get("subject") == args.subject)]
    if args.max_items is not None:
        records = records[: args.max_items]
    print(f"[io] loaded {len(records)} records from {args.in_repo}:{args.split} (subject={args.subject})")

    problems = build_problem_units(records)
    print(f"[prep] built {len(problems)} judge inputs")

    # Slab through problems
    processed = 0
    for s in range(0, len(problems), args.batch_size):
        slab = problems[s : s + args.batch_size]
        results = await run_council_batch(
            problems=slab,
            judge_pool=judge_pool,
            judges_per_item=args.judges_per_item,
            batch_size=args.batch_size,
            orchestrator=InferenceOrchestrator(),
            seed_offset=processed,
        )
        for rec, verification in results:
            rec = dict(rec)  # copy
            rec["verification"] = verification
            append_jsonl(args.out_all, rec)
        processed += len(slab)
        print(f"[run] processed {processed}/{len(problems)}")

def main():
    args = parse_args()
    try:
        asyncio.run(main_async(args))
    except KeyboardInterrupt:
        print("\n[run] interrupted")

if __name__ == "__main__":
    main()
