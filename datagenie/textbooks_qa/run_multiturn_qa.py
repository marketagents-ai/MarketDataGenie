"""
Runner script for multi-turn ShareGPT generation on HuggingFace textbook QA dataset
It uses the modular MultiTurnQAPipeline and:
- Seeds a persistent ChatThread per chunk with the rephrased text in the system prompt
- Reuses existing single-turn Q(a)/A(a) from the dataset's `conversations` list as few-shot context (NOT counted toward num_turns)
- Generates follow-up questions via a simple agent persona (unless disabled)
- Answers all turns through ChatThread + InferenceOrchestrator
- Appends JSONL lines to results and sharegpt outputs (safe to resume)

Usage (examples):
  python datagenie/textbooks_qa/run_multiturn_qa.py \
      --num_turns 3 --start 0 --limit 100
"""

import os
import sys
import json
import argparse
from pathlib import Path
from typing import Any, Dict, List, Optional
from datetime import datetime

from datasets import load_dataset

# Pipeline + thread
from datagen_textbooks_qa_multiturn import (
    PipelineConfig,
    MultiTurnQAPipeline,
    SimpleChatThread,   # ChatThread-backed wrapper
)

# Agent bits for follow-up question generation
from market_agents.agents.base_agent.agent import Agent as MarketAgent  # thin Agent wrapper
from market_agents.agents.personas.persona import Persona
from market_agents.agents.base_agent.prompter import PromptManager

from minference.lite.models import (
    LLMConfig, LLMClient, ResponseFormat
)
from minference.lite.inference import InferenceOrchestrator


# -----------------------
# IO helpers
# -----------------------

def _ensure_parent(path_str: str) -> None:
    p = Path(path_str)
    p.parent.mkdir(parents=True, exist_ok=True)


def _append_jsonl_line(path_str: str, record: Dict[str, Any]) -> None:
    _ensure_parent(path_str)
    with open(path_str, "a", encoding="utf-8") as f:
        f.write(json.dumps(record, ensure_ascii=False) + "\n")


def _autogen_output_paths(results_jsonl: Optional[str], sharegpt_jsonl: Optional[str]) -> tuple[str, str]:
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    base_dir = Path("outputs/multiturn")
    base_dir.mkdir(parents=True, exist_ok=True)
    res = results_jsonl or str(base_dir / f"multiturn_results_{ts}.jsonl")
    sgd = sharegpt_jsonl or str(base_dir / f"multiturn_sharegpt_{ts}.jsonl")
    return res, sgd


def _pairwise_conversations_only(conversations: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """
    Keep only valid human->gpt pairs. Drop system entries, orphan gpt turns,
    and any human turn that is followed by a missing/blank gpt value.
    """
    if not isinstance(conversations, list):
        return []
    # keep only human/gpt roles
    msgs = [m for m in conversations if m and m.get("from") in ("human", "gpt")]
    out: List[Dict[str, Any]] = []
    i = 0
    while i < len(msgs):
        m = msgs[i]
        if m.get("from") == "human":
            # must be followed by a gpt with non-empty value
            if i + 1 < len(msgs) and msgs[i + 1].get("from") == "gpt":
                g = msgs[i + 1]
                if (g.get("value") or "").strip():
                    out.append(m)
                    out.append(g)
                # whether blank or not, skip the pair and continue
                i += 2
                continue
            # dangling human without gpt -> drop
            i += 1
            continue
        # orphan gpt without preceding human -> drop
        i += 1
    return out


def _append_sharegpt_filtered(path_str: str, record: Dict[str, Any]) -> None:
    """
    Sanitize ShareGPT output to ensure:
      - We drop any leading seed pair (dataset's initial human/gpt) to keep only generated turns.
      - We only keep complete human->gpt pairs.
      - If nothing remains after filtering, we skip writing to avoid empty ShareGPT lines.
    Heuristics:
      - If there are >= 4 messages, we drop the first 2 (likely the dataset seed Q/A),
        then re-pair what's left. This matches our pipeline: seed Q/A from dataset,
        then generated follow-ups.
    """
    conversations = record.get("conversations") or []

    # Heuristic: if we have at least 2 pairs total, assume the first pair is dataset seed and drop it
    trimmed = conversations
    if isinstance(conversations, list) and len(conversations) >= 4:
        trimmed = conversations[2:]

    paired = _pairwise_conversations_only(trimmed)

    if len(paired) >= 2:
        # Overwrite conversations with filtered pairs only
        record = dict(record)  # shallow copy to avoid side effects
        record["conversations"] = paired
        _append_jsonl_line(path_str, record)
    else:
        # If we still have at least one valid pair in the original list, write that;
        # otherwise skip to avoid empty ShareGPT entries.
        fallback = _pairwise_conversations_only(conversations)
        if len(fallback) >= 2:
            record = dict(record)
            record["conversations"] = fallback
            _append_jsonl_line(path_str, record)
        # else: nothing worth writing


# -----------------------
# Adapter callables for pipeline
#   - We prefer reusing existing conversations for seed Q/A.
#   - Follow-ups: small agent persona that proposes next question.
#   - Answers: provided by SimpleChatThread internally (ChatThread + orchestrator).
# -----------------------

async def rephraser_fn_use_existing(chunk: Dict[str, Any]) -> str:
    """Prefer chunk['rephrased_text'] else fall back to chunk['text']."""
    return chunk.get("rephrased_text") or chunk.get("text") or ""


async def seed_question_from_chunk_or_agent(chunk: Dict[str, Any], rephrased_text: str) -> str:
    """
    If the dataset already provides Q(a) in chunk['conversations'][0]['value'],
    just return it. Otherwise, generate a seed question with a tiny agent.
    """
    conv = chunk.get("conversations") or chunk.get("seed_conversation")
    if isinstance(conv, list) and len(conv) >= 1 and conv[0].get("from") == "human":
        return conv[0]["value"]

    # Fallback: tiny agent to write part (a)
    persona = Persona(
        role="Question Writer",
        persona=(
            "You are an expert in curriculum development and exam question writing."
            "You design problem sets and structured exam questions"
        ),
        objectives=["Produce one concise exam style question"],
        skills=["Question design", "Curriculum alignment"],
    )
    task_prompt = (
        "Context (hidden):\n" + rephrased_text +
        "\n\n##Guidelines:\n"
        "- Write a single concise question in the same language as the context.\n"
        "- Use the provided context as textbook material for question generation.\n"
        "- The question must be fully self-contained and answerable **without** additional context.\n"
        "- Do **not** ask about minor details like character names, authors.\n"
        "- If the context is fiction (stories, poems, etc.), focus on **themes, ideas, or structures** rather than exact character or plot details.\n"
        "- For problem sets (maths, physics, chemistry, etc.), recreate a **complete solvable question** with necessary numbers, equations, or assumptions.\n"
        "- Do **not** generate questions that depend on diagrams or illustrations.\n"
        "- Do not use question numbers and do not include the answer.\n"
        "- Directly output only the question text.\n"
    )
    agent = MarketAgent(
        name="seed-question",
        persona=persona,
        task=task_prompt,
        llm_config=LLMConfig(
            client=LLMClient.openai,
            model=os.environ.get("DATAGEN_MODEL", "gpt-5-mini"),
            temperature=0.2,
            response_format=ResponseFormat.text,
            reasoning_effort="minimal",
            max_completion_tokens=4096
        ),
        llm_orchestrator=InferenceOrchestrator(),
        prompt_manager=PromptManager(),
        tools=[],
    )
    out = await agent.execute()
    return out.strip() if isinstance(out, str) else json.dumps(out, ensure_ascii=False)


async def followup_question_agent(prev_questions: List[str], prev_answers: List[str], label: str, target_level: str) -> str:
    """Agent to generate the next sub-question based on prior Q/A."""
    persona = Persona(
        role="Follow-up Question Writer",
        persona=(
            "You extend a structured exam question with the next part. "
            "Increase difficulty according to the target level."
        ),
        objectives=["Produce the next sub-question only."],
        skills=["Scaffolding difficulty", "Curriculum mapping"],
    )
    history_text = "\n".join(
        [f"Q{i}: {q}\nA{i}: {a}" for i, (q, a) in enumerate(zip(prev_questions, prev_answers))]
    )
    task = (
        f"Previous parts and answers:\n{history_text}\n\n"
        f"Write the next sub-question (part {label}). Difficulty: {target_level}.\n"
        f"\n#Guidelines:\n"
        f"- Use the same language as previous parts. Output only the question text."
        "- The question must be fully self-contained and answerable **without** additional context.\n"
        "- Do **not** ask about minor details like character names, authors.\n"
        "- If the context is fiction (stories, poems, etc.), focus on **themes, ideas, or structures** rather than exact character or plot details.\n"
        "- For problem sets (maths, physics, chemistry, etc.), recreate a **complete solvable question** with necessary numbers, equations, or assumptions.\n"
        "- Do **not** generate questions that depend on diagrams or illustrations.\n"
        "- Do not include the answer.\n"
        "- Directly output only the question text without question numbers"
    )
    agent = MarketAgent(
        name="followup-question",
        persona=persona,
        task=task,
        llm_config=LLMConfig(
            client=LLMClient.openai,
            model=os.environ.get("DATAGEN_MODEL", "gpt-5-nano"),
            temperature=0.3,
            response_format=ResponseFormat.text,
            reasoning_effort="minimal",
            max_completion_tokens=4096
        ),
        llm_orchestrator=InferenceOrchestrator(),
        prompt_manager=PromptManager(),
        tools=[],
    )
    out = await agent.execute()
    return out.strip() if isinstance(out, str) else json.dumps(out, ensure_ascii=False)


def new_chatthread_fn(system_hidden_context: str) -> SimpleChatThread:
    """Construct a ChatThread-backed wrapper that embeds the rephrased context once."""
    return SimpleChatThread(system_hidden_context=system_hidden_context, llm_model=os.environ.get("DATAGEN_MODEL", "gpt-5"))


# -----------------------
# Runner
# -----------------------

def build_pipeline(num_turns: int) -> MultiTurnQAPipeline:
    cfg = PipelineConfig(
        mode="multiturn",
        num_turns=num_turns,
        difficulty_profile=["recall", "application", "analysis"],
        include_context_in_metadata=True,
        include_rephrased_in_metadata=True,
    )
    return MultiTurnQAPipeline(
        cfg=cfg,
        rephraser_fn=rephraser_fn_use_existing,
        seed_question_fn=seed_question_from_chunk_or_agent,
        followup_question_fn=followup_question_agent,
        new_chatthread_fn=new_chatthread_fn,
        append_results_jsonl_fn=lambda rec: _append_jsonl_line(ARGS.results_jsonl, rec),
        append_sharegpt_jsonl_fn=lambda rec: _append_sharegpt_filtered(ARGS.sharegpt_jsonl, rec),
    )


def row_to_chunk(row: Dict[str, Any]) -> Dict[str, Any]:
    """Map HF row fields to our chunk schema."""
    return {
        "id": row.get("id"),
        "text": row.get("context_text"),
        "rephrased_text": row.get("rephrased_text"),
        "subject": row.get("subject"),
        "grade": row.get("grade"),
        "chapter_title": row.get("chapter_title"),
        "source": row.get("source"),
        "conversations": row.get("conversations"),  # seed Q(a)/A(a)
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run multi-turn QA generation on HF dataset.")
    parser.add_argument("--hf_path", default=None)
    parser.add_argument("--split", default="train")
    parser.add_argument("--start", type=int, default=0)
    parser.add_argument("--limit", type=int, default=50, help="Number of rows to process from start")
    parser.add_argument("--num_turns", type=int, default=3, help="Number of GENERATED user turns (seed Q/A is few-shot and NOT counted)")
    parser.add_argument("--results_jsonl", default=None, help="Path to results JSONL. If omitted, auto-generates under outputs/multiturn.")
    parser.add_argument("--sharegpt_jsonl", default=None, help="Path to ShareGPT JSONL. If omitted, auto-generates under outputs/multiturn.")
    return parser.parse_args()


def main() -> None:
    global ARGS
    ARGS = parse_args()

    ARGS.results_jsonl, ARGS.sharegpt_jsonl = _autogen_output_paths(ARGS.results_jsonl, ARGS.sharegpt_jsonl)
    print(f"[runner] Results will be appended to: {ARGS.results_jsonl}")
    print(f"[runner] ShareGPT will be appended to: {ARGS.sharegpt_jsonl}")

    print(f"[runner] Loading HF dataset: {ARGS.hf_path} split={ARGS.split}")
    ds = load_dataset(ARGS.hf_path, split=ARGS.split)

    end = min(len(ds), ARGS.start + ARGS.limit)
    print(f"[runner] Processing rows {ARGS.start}:{end} (num_turns={ARGS.num_turns})")

    pipeline = build_pipeline(ARGS.num_turns)

    processed = 0
    for idx in range(ARGS.start, end):
        row = ds[idx]
        try:
            chunk = row_to_chunk(row)
            # Process one chunk
            # Note: process_chunk is async, so we need to run it via asyncio
            import asyncio
            out = asyncio.run(pipeline.process_chunk(chunk))
            processed += 1
            if processed % 10 == 0:
                print(f"[runner] Processed {processed} items...")
        except Exception as e:
            print(f"[runner] Row {idx} failed: {e}")

    print(f"[runner] Done. Processed={processed}.")
    print(f"[runner] Results -> {ARGS.results_jsonl}")
    print(f"[runner] ShareGPT -> {ARGS.sharegpt_jsonl}")


if __name__ == "__main__":
    main()