 # ---- debug helper ----
def _dbg(msg: str) -> None:
    try:
        print(msg, flush=True)
    except Exception:
        pass

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

# Note on lifecycle:
# - Seed and follow-up *question* agents are EPHEMERAL. We instantiate them per turn,
#   return their ChatThread, run a single completion in parallel, extract the question,
#   and discard the thread. All needed state (rephrased context + prior Q/A + label)
#   is passed explicitly in the prompt.
# - Answer threads (SimpleChatThread per chunk) are PERSISTENT across the whole episode.
#   We reuse them for every turn, committing user and assistant messages after each
#   completion so conversation context is preserved.

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

import asyncio
from typing import Tuple, Sequence

from minference.lite.models import ChatThread, ProcessedOutput, LLMClient
try:
    # Optional: some builds expose RawOutput; use if available for completeness
    from minference.lite.models import RawOutput
except Exception:  # pragma: no cover
    RawOutput = None  # type: ignore

# -----------------------
# Banned phrases loader (shared with single-turn script semantics)
# -----------------------
from pathlib import Path as _Path
import yaml as _yaml
import json as _json

_BANNED_PHRASES: List[str] = []

def _load_banned_phrases() -> List[str]:
    global _BANNED_PHRASES
    if _BANNED_PHRASES:
        return _BANNED_PHRASES
    phrases: List[str] = []
    try:
        yml_p = _Path("configs/banned_phrases.yml")
        json_p = _Path("configs/banned_phrases.json")
        if yml_p.exists():
            try:
                with open(yml_p, "r", encoding="utf-8") as f:
                    data = _yaml.safe_load(f) or {}
                phrases = list(data.get("banned_phrases") or [])
            except Exception:
                phrases = []
        elif json_p.exists():
            try:
                with open(json_p, "r", encoding="utf-8") as f:
                    data = _json.load(f) or {}
                phrases = list(data.get("banned_phrases") or [])
            except Exception:
                phrases = []
    except Exception:
        phrases = []
    # normalize + dedupe
    norm: List[str] = []
    seen = set()
    for p in phrases:
        if not isinstance(p, str):
            continue
        s = p.strip()
        if not s or s in seen:
            continue
        seen.add(s)
        norm.append(s)
    _BANNED_PHRASES = norm
    if _BANNED_PHRASES:
        _dbg(f"[banned] loaded {len(_BANNED_PHRASES)} phrases")
    return _BANNED_PHRASES


def _banned_block() -> str:
    phrases = _load_banned_phrases()
    if not phrases:
        return ""
    inline = ", ".join(phrases)
    return (
        "Strictly avoid references to specific chapters/lessons/units and do NOT use any of these phrases: "
        f"{inline}."
    )


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


#
# Pair-only sanitizer used for ShareGPT export. This keeps only clean human->gpt
# pairs and drops: system messages, orphan gpt turns, and human turns that are not
# followed by a non-empty gpt answer. It prevents half-baked pairs from leaking
# into the ShareGPT file and confusing downstream loaders.
#
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

    # Use explicit dataset_seed_present flag to determine if we should drop the first pair.
    trimmed = conversations
    drop_dataset_seed = bool(record.get("dataset_seed_present"))
    if drop_dataset_seed and isinstance(conversations, list) and len(conversations) >= 2:
        # Drop exactly the first human->gpt pair which corresponds to the dataset seed Q/A
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



# Thin wrapper around orchestrator.run_parallel_ai_completion that preserves input
# ordering, batches threads in slabs of `batch_size`, and shields the caller from
# catastrophic slab failures by returning an Exception object per failed item.
# We intentionally avoid mixing different model types in the same call list; the
# orchestrator should receive homogeneous ChatThreads for predictable latency.
async def _run_threads_in_batches(
    threads: Sequence[ChatThread],
    batch_size: int,
    orchestrator: Optional[InferenceOrchestrator] = None,
) -> List[ProcessedOutput | Exception]:
    orch = orchestrator or InferenceOrchestrator()
    results: List[ProcessedOutput | Exception] = []
    for i in range(0, len(threads), batch_size):
        slab = list(threads[i:i + batch_size])
        try:
            outs = await orch.run_parallel_ai_completion(slab)
            results.extend(outs)
        except Exception as e:
            results.extend([e] * len(slab))
    return results


# -----------------------
# Adapter callables for pipeline
#   - We prefer reusing existing conversations for seed Q/A.
#   - Follow-ups: small agent persona that proposes next question.
#   - Answers: provided by SimpleChatThread internally (ChatThread + orchestrator).
# -----------------------


# Rephraser is a no-op here: prefer precomputed rephrased_text; fall back to raw text.
async def rephraser_fn_use_existing(chunk: Dict[str, Any]) -> str:
    """Prefer chunk['rephrased_text'] else fall back to chunk['text']."""
    return chunk.get("rephrased_text") or chunk.get("text") or ""


async def seed_question_agent(chunk: Dict[str, Any], rephrased_text: str) -> str:
    """
    Returns either a dataset seed string (when present) or an EPHEMERAL agent ChatThread
    prepared to generate the seed question. Caller must run the returned ChatThread in
    parallel and capture its completion. Important: we set `chat_thread.new_message` to
    the task so the orchestrator has a user message to complete against.
    """
    conv = chunk.get("conversations") or chunk.get("seed_conversation")
    if isinstance(conv, list) and len(conv) >= 1 and conv[0].get("from") == "human":
        return conv[0]["value"]

    # Fallback: tiny agent to write part (a)
    persona = Persona(
        role="Question Writer",
        persona=(
            "You are an expert in curriculum development and exam question writing."
            "You design problem sets and structured exam questions in same language"
        ),
        objectives=["Produce one concise exam style question"],
        skills=["Question design", "Curriculum alignment"],
    )
    banned_txt = _banned_block()
    task_prompt = (
        "Context (hidden):\n" + rephrased_text +
        "\n\n##Guidelines:\n"
        "- Write a single concise question in the same language as the context.\n"
        "- Use the provided context as textbook material for question generation.\n"
        "- The question must be fully self-contained and answerable **without** additional context.\n"
        "- If the context is fiction (stories, poems, etc.), focus on **themes, ideas, or structures** rather than exact character or plot details.\n"
        "- For problem sets (maths, physics, chemistry, etc.), recreate a **complete solvable question** with necessary numbers, equations, or assumptions.\n"
        "- Do **not** generate questions that depend on diagrams or illustrations.\n"
        "- Do not use question numbers such as Q0 (a), Q0 (b) etc. and do not include the answer.\n"
        f"- {banned_txt}\n" if banned_txt else ""
        "- Directly output only the question text.\n"
    )
    agent = MarketAgent(
        name="seed-question",
        persona=persona,
        task=task_prompt,
        llm_config=LLMConfig(
            client=LLMClient.openai,
            model=os.environ.get("DATAGEN_MODEL", "gpt-4o-mini"),
            temperature=0.2,
            response_format=ResponseFormat.text,
            #reasoning_effort="minimal",
            #max_completion_tokens=4096
            max_tokens=2048
        ),
        llm_orchestrator=InferenceOrchestrator(),
        prompt_manager=PromptManager(),
        tools=[],
    )
    # Ensure the user message is set on the thread for orchestrator
    agent.chat_thread.new_message = agent.task
    # Return the thread (not agent.execute); the runner will batch it with others.
    return agent.chat_thread

# --- New: Parallel seed agent runner

#
# ID-aware stitching: we match ProcessedOutput.chat_thread_id back to the originating ChatThread.
#

# Helper to extract a thread's id in a consistent way
def _thread_id(th: ChatThread) -> str:
    """Best-effort getter for a ChatThread's stable id as a string."""
    tid = getattr(th, "chat_thread_id", None) or getattr(th, "id", None)
    return str(tid) if tid is not None else ""

# ProcessedOutput -> stable id string
def _out_id(out: ProcessedOutput) -> str:
    try:
        return str(out.chat_thread_id)
    except Exception:
        return ""

# Robustly extract text content from ProcessedOutput, with sane fallbacks
def _extract_text(out: ProcessedOutput) -> str:
    # Primary parsed content
    txt = (getattr(out, "content", None) or "").strip()
    if txt:
        return txt
    # JSON-mode outputs may carry a serialized object; try to stringify a useful field
    try:
        jo = getattr(out, "json_object", None)
        if jo:
            # common patterns: {"answer": "..."} or {"content": "..."}
            for k in ("answer", "content", "text", "output"):
                v = jo.get(k)
                if isinstance(v, str) and v.strip():
                    return v.strip()
    except Exception:
        pass
    # Fall back to raw_output fields if available
    try:
        ro = getattr(out, "raw_output", None)
        if ro is not None:
            for k in ("output_text", "content", "text"):
                v = getattr(ro, k, None)
                if isinstance(v, str) and v.strip():
                    return v.strip()
    except Exception:
        pass
    return ""

# Batch runner for seed generation. It calls the seed builder per chunk and collects
# only the ChatThreads that require model calls. Dataset-provided seeds are passed
# through untouched. We keep an index map so we can stitch model outputs back into
# the correct positions in the original chunk list.
async def run_seed_agent_parallel(
    chunks: List[Dict[str, Any]],
    batch_size: int,
) -> List[str | Exception]:
    """
    For each chunk, call seed_question_agent to obtain either:
      - a dataset-provided seed string (no model call), or
      - a ChatThread to be run via orchestrator in parallel
    Returns a seed question string per chunk in the original order.
    """
    threads: List[Optional[ChatThread]] = []
    idx_map: List[int] = []
    outs: List[str | Exception] = [None] * len(chunks)

    # Build list of ChatThreads to run
    for i, ch in enumerate(chunks):
        rephrased_text = ch.get("rephrased_text") or ch.get("text") or ""
        result = await seed_question_agent(ch, rephrased_text)
        if isinstance(result, ChatThread):
            # Ensure a user message is present on the thread. Do NOT add it to
            # history yet; we only set new_message so the orchestrator can produce
            # the assistant completion. We commit history on the persistent threads.
            result.new_message = result.new_message or getattr(result, "_pending_user_msg", None)
            threads.append(result)
            idx_map.append(i)
        else:
            # Dataset-provided seed string
            outs[i] = str(result)
            _dbg(f"[seed-agent] dataset seed idx={i} len={len(outs[i])}")

    # Run batch inference for generated seeds
    actual_threads = [t for t in threads if t is not None]
    if actual_threads:
        # Map expected ids to chunk indices
        id_to_idx: Dict[str, int] = {}
        for i2, th in zip(idx_map, actual_threads):
            tid = _thread_id(th)
            if tid:
                id_to_idx[str(tid)] = i2
        raw = await _run_threads_in_batches(actual_threads, batch_size=batch_size)
        for r in raw:
            if isinstance(r, Exception):
                continue
            rid = _out_id(r)
            idx = id_to_idx.get(rid)
            if idx is not None:
                outs[idx] = _extract_text(r)
                _dbg(f"[seed-agent] stitched rid={rid} → idx={idx} q_len={len(outs[idx])}")

    # Finalize, replacing any lingering None with an Exception
    for i, v in enumerate(outs):
        if v is None:
            outs[i] = Exception("seed missing and not generated")
    return outs


#
# Follow-up question AGENT (EPHEMERAL): builds a prompt from prior Q/A + label.
# Returns a ChatThread to be run once in a parallel batch.
#
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
    banned_txt = _banned_block()
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
        f"- {banned_txt}\n" if banned_txt else ""
        "- Directly output only the question text without question numbers such as Q0 (b), Q0 (c) etc."
    )
    agent = MarketAgent(
        name="followup-question",
        persona=persona,
        task=task,
        llm_config=LLMConfig(
            client=LLMClient.openai,
            model=os.environ.get("DATAGEN_MODEL", "gpt-4o-mini"),
            temperature=0.4,
            response_format=ResponseFormat.text,
            #reasoning_effort="minimal",
            #max_completion_tokens=4096
            max_tokens=2048
        ),
        llm_orchestrator=InferenceOrchestrator(),
        prompt_manager=PromptManager(),
        tools=[],
    )

    # Ensure the user message is set on the thread for orchestrator
    agent.chat_thread.new_message = task
    return agent.chat_thread


# Batch runner for follow-up question generation. We build N ephemeral agent threads
# (one per chunk for this turn), run a single parallel completion, and return the
# question strings. No history is persisted on these ephemeral threads.
async def run_followup_agent_parallel(
    histories: List[Tuple[List[str], List[str]]],
    labels: List[str],
    levels: List[str],
    batch_size: int,
) -> List[str | Exception]:
    threads: List[ChatThread] = []
    for (qs, ans), lab, lvl in zip(histories, labels, levels):
        th = await followup_question_agent(qs, ans, lab, lvl)
        threads.append(th)
    # Build an index map by thread id to position
    id_to_pos: Dict[str, int] = {}
    for pos, th in enumerate(threads):
        tid = _thread_id(th)
        if tid:
            id_to_pos[str(tid)] = pos

    raw = await _run_threads_in_batches(threads, batch_size=batch_size)
    outs: List[str | Exception] = [None] * len(threads)
    for r in raw:
        if isinstance(r, Exception):
            continue
        rid = _out_id(r)
        pos = id_to_pos.get(rid)
        if pos is not None:
            outs[pos] = _extract_text(r)
        # print(f"[debug] stitched id={rid} → pos={pos}")
    # Fill any gaps with Exceptions to surface issues explicitly
    for i in range(len(outs)):
        if outs[i] is None:
            outs[i] = Exception("follow-up generation failed or id missing")
    return outs



# Answer stage (PERSISTENT): we reuse one SimpleChatThread per chunk for the entire
# episode. For each turn, we set `new_message` to the next user question, run the
# parallel completion, then commit BOTH the user message and the assistant output
# to the thread history in that order.
async def answers_parallel_via_threads(
    simple_threads: List[SimpleChatThread],
    next_questions: List[str],
    batch_size: int,
) -> List[str | Exception]:
    # Prepare underlying chat threads with the next user message and record pending questions by id
    chat_threads: List[ChatThread] = []
    pending: Dict[str, str] = {}
    for th, q in zip(simple_threads, next_questions):
        tid = _thread_id(th.chat_thread)
        th.chat_thread.new_message = q
        chat_threads.append(th.chat_thread)
        pending[tid] = q
        _dbg(f"[ask] stage next user tid={tid} q_len={len(q or '')}")

    _dbg(f"[ask] running batch size={len(chat_threads)}")
    raw = await _run_threads_in_batches(chat_threads, batch_size=batch_size)

    # Map returned outputs by their chat_thread_id
    out_map: Dict[str, ProcessedOutput] = {}
    for r in raw:
        if isinstance(r, Exception):
            _dbg(f"[ask] batch item failed: {r}")
            continue
        rid = _out_id(r)
        out_map[rid] = r
        _dbg(f"[ask] got completion rid={rid} a_len={len(_extract_text(r))}")

    outs: List[str | Exception] = []
    for th in simple_threads:
        tid = _thread_id(th.chat_thread)
        r = out_map.get(tid)
        q = pending.get(tid)
        if (not r) or (q is None):
            _dbg(f"[ask] MISSING output for tid={tid}")
            outs.append(Exception("answer missing for thread"))
            continue
        ans = _extract_text(r)
        try:
            # Do NOT add_user_message here; orchestrator already added the user based on new_message
            if hasattr(th.chat_thread, "add_chat_turn_history"):
                maybe_coro = th.chat_thread.add_chat_turn_history(r)
                if asyncio.iscoroutine(maybe_coro):
                    await maybe_coro
            _dbg(f"[ask] committed assistant tid={tid} a_len={len(ans)}")
        except Exception as e:
            _dbg(f"[ask] commit failed tid={tid}: {e}")
        outs.append(ans)
    return outs



# Orchestrates one slab (batch) of rows through:
#   1) Seed question generation (mix of dataset seeds and ephemeral agent threads)
#   2) Seed answer generation via persistent SimpleChatThreads (parallel)
#   3) For each remaining turn:
#        a) Ephemeral follow-up agent threads → next question (parallel)
#        b) Persistent answer threads → answer (parallel)
#   4) Emit records with full `conversations` history
async def process_slab_multiturn(
    rows: List[Dict[str, Any]],
    num_turns: int,
    batch_size: int,
    difficulty_profile: List[str] = None,
) -> List[Dict[str, Any]]:
    difficulty_profile = difficulty_profile or ["recall", "application", "analysis"]
    chunks = [row_to_chunk(r) for r in rows]

    # Detect dataset example seed Q/A (used strictly as few-shot, not counted toward num_turns)
    dataset_seed_present: List[bool] = []
    dataset_example_q: List[str] = []
    dataset_example_a: List[str] = []
    for ch in chunks:
        conv = ch.get("conversations") or []
        q = conv[0]["value"].strip() if (isinstance(conv, list) and len(conv) >= 1 and (conv[0] or {}).get("from") == "human" and ((conv[0] or {}).get("value") or "").strip()) else ""
        a = conv[1]["value"].strip() if (isinstance(conv, list) and len(conv) >= 2 and (conv[1] or {}).get("from") == "gpt" and ((conv[1] or {}).get("value") or "").strip()) else ""
        dataset_example_q.append(q)
        dataset_example_a.append(a)
        dataset_seed_present.append(bool(q and a))

    # Step 1: generate a fresh seed question for every chunk (single cohort)
    seed_qs_raw = await run_seed_agent_parallel(chunks, batch_size=batch_size)
    seed_qs: List[str] = [sq if isinstance(sq, str) else "" for sq in seed_qs_raw]

    # Step 2: build one persistent answer thread per chunk and preload dataset example pair (few-shot) if present
    simple_threads: List[SimpleChatThread] = []
    for ch in chunks:
        ctx = ch.get("rephrased_text") or ch.get("text") or ""
        simple_threads.append(new_chatthread_fn(system_hidden_context=ctx))

    # Preload dataset example pair into thread history (no model call)
    for i, th in enumerate(simple_threads):
        tid = _thread_id(th.chat_thread)
        if dataset_example_q[i] and dataset_example_a[i]:
            _dbg(f"[preload] tid={tid} add Q0 len={len(dataset_example_q[i])}")
            th.chat_thread.new_message = dataset_example_q[i]
            try:
                th.chat_thread.add_user_message()
                _dbg(f"[preload] tid={tid} committed user(Q0)")
            except Exception as e:
                _dbg(f"[preload] tid={tid} failed add_user_message: {e}")
            # deterministically append assistant via the same path as real completions
            added_assistant = False
            try:
                if hasattr(th.chat_thread, "add_chat_turn_history"):
                    # fabricate a minimal ProcessedOutput with the correct thread id
                    po_kwargs = dict(
                        content=dataset_example_a[i],
                        json_object=None,
                        usage=None,
                        error=None,
                        time_taken=0.0,
                        llm_client=LLMClient.openai,
                        raw_output=None,
                        chat_thread_id=th.chat_thread.chat_thread_id if hasattr(th.chat_thread, "chat_thread_id") else getattr(th.chat_thread, "id", None),
                    )
                    # If RawOutput class exists and field is required, provide a minimal stub
                    if RawOutput is not None:
                        try:
                            po_kwargs["raw_output"] = RawOutput(
                                raw_result={}, completion_kwargs={}, start_time=0.0, end_time=0.0,
                                chat_thread_id=po_kwargs["chat_thread_id"], client=LLMClient.openai,
                            )
                        except Exception:
                            pass
                    po = ProcessedOutput(**po_kwargs)  # type: ignore[arg-type]

                    maybe_coro = th.chat_thread.add_chat_turn_history(po)
                    if asyncio.iscoroutine(maybe_coro):
                        await maybe_coro
                    added_assistant = True
                    _dbg(f"[preload] tid={tid} committed assistant(A0) via add_chat_turn_history")
                elif hasattr(th.chat_thread, "add_assistant_message"):
                    th.chat_thread.add_assistant_message(dataset_example_a[i])
                    added_assistant = True
                    _dbg(f"[preload] tid={tid} committed assistant(A0) via add_assistant_message")
                elif hasattr(th.chat_thread, "add_chat_turn_history_text"):
                    th.chat_thread.add_chat_turn_history_text(dataset_example_a[i])
                    added_assistant = True
                    _dbg(f"[preload] tid={tid} committed assistant(A0) via add_chat_turn_history_text")
            except Exception as e:
                _dbg(f"[preload] tid={tid} failed assistant preload: {e}")
            if not added_assistant:
                _dbg(f"[preload] tid={tid} WARNING: assistant(A0) not preloaded; proceeding without")
            # clear pending to avoid accidental reuse
            try:
                th.chat_thread.new_message = None
            except Exception:
                pass
        else:
            _dbg(f"[preload] tid={tid} no dataset example pair present")

    # Step 2 (cont.): ask seed answers in parallel for everyone
    _dbg(f"[seed] answering {len(seed_qs)} generated seeds in parallel")
    seed_answers = await answers_parallel_via_threads(
        simple_threads=simple_threads,
        next_questions=seed_qs,
        batch_size=batch_size,
    )
    _dbg("[seed] completed answers: " + ", ".join(str(len(a)) if isinstance(a, str) else "ERR" for a in seed_answers))

    # Initialize histories with the **generated** seed pair only (dataset example is few-shot)
    q_hist: List[List[str]] = [[q] for q in seed_qs]
    a_hist: List[List[str]] = [[a] if isinstance(a, str) and a.strip() else [""] for a in seed_answers]

    # Follow-up turns
    for t in range(num_turns - 1):
        label = chr(ord("b") + t)
        level = difficulty_profile[min(t, len(difficulty_profile) - 1)]
        histories = list(zip(q_hist, a_hist))
        _dbg(f"[followup t={t}] generating questions for {len(chunks)} threads")
        followups = await run_followup_agent_parallel(
            histories=histories,
            labels=[label] * len(chunks),
            levels=[level] * len(chunks),
            batch_size=batch_size,
        )
        answers = await answers_parallel_via_threads(
            simple_threads=simple_threads,
            next_questions=[fq if isinstance(fq, str) else "" for fq in followups],
            batch_size=batch_size,
        )
        _dbg(f"[followup t={t}] answered: " + ", ".join(str(len(a)) if isinstance(a, str) else "ERR" for a in answers))
        for i in range(len(chunks)):
            fq = followups[i]
            fa = answers[i]
            if isinstance(fq, str) and fq.strip():
                q_hist[i].append(fq)
            else:
                q_hist[i].append("")
            if isinstance(fa, str) and fa.strip():
                a_hist[i].append(fa)
            else:
                a_hist[i].append("")

    # Build final records for output
    # We record whether the dataset provided a seed question so the ShareGPT export
    # can drop that first pair and keep only subsequent generations.
    records: List[Dict[str, Any]] = []
    for i, ch in enumerate(chunks):
        convs: List[Dict[str, str]] = []
        for q, a in zip(q_hist[i], a_hist[i]):
            if q:
                convs.append({"from": "human", "value": q})
            if a:
                convs.append({"from": "gpt", "value": a})
        rec = {
            "id": ch.get("id"),
            "subject": ch.get("subject"),
            "grade": ch.get("grade"),
            "chapter_title": ch.get("chapter_title"),
            "source": ch.get("source"),
            "context_text": ch.get("text"),
            "rephrased_text": ch.get("rephrased_text"),
            "conversations": convs,
            "dataset_seed_present": dataset_seed_present[i],
        }
        try:
            assert len(q_hist[i]) == len(a_hist[i]) == num_turns, f"pair count mismatch id={rec['id']} q={len(q_hist[i])} a={len(a_hist[i])}"
        except AssertionError as e:
            _dbg(f"[warn] {e}")
        _dbg(f"[record] id={rec['id']} pairs={len(q_hist[i])} ds_seed={dataset_seed_present[i]}")
        records.append(rec)
    return records



# Construct a SimpleChatThread with the hidden system context set to rephrased text.
# The model name is pulled from $DATAGEN_MODEL or defaults to a reasonable baseline.
def new_chatthread_fn(system_hidden_context: str) -> SimpleChatThread:
    """Construct a ChatThread-backed wrapper that embeds the rephrased context once."""
    banned_txt = _banned_block()
    if banned_txt:
        combined = (
            f"{system_hidden_context}\n\n"
            "#Constraints:\n"
            "- Answer in the same language as the question.\n"
            "- For questions with exact answer in math and science, please reason step by step, and put your final answer within \boxed{}."
            "- Do not reference that a textbook or rephrased context exists; just answer.\n"
            f"- {banned_txt}"
        )
    else:
        combined = system_hidden_context
    return SimpleChatThread(system_hidden_context=combined, llm_model=os.environ.get("DATAGEN_MODEL", "gpt-4.1-mini"))



# Legacy modular pipeline kept for compatibility. The parallel slab driver above is
# the default execution path for this runner.
def build_pipeline(num_turns: int) -> MultiTurnQAPipeline:
    # Note: This pipeline is still available but the parallel slab driver is default in this runner.
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
        seed_question_fn=seed_question_agent,
        followup_question_fn=followup_question_agent,
        new_chatthread_fn=new_chatthread_fn,
        append_results_jsonl_fn=lambda rec: _append_jsonl_line(ARGS.results_jsonl, rec),
        append_sharegpt_jsonl_fn=lambda rec: _append_sharegpt_filtered(ARGS.sharegpt_jsonl, rec),
    )



# HF → internal chunk map. These keys are assumed downstream when building prompts
# and exporting results. Be careful when renaming or dropping fields.
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
    parser.add_argument("--batch_size", type=int, default=8, help="Parallel slab size for orchestrator.run_parallel_ai_completion")
    parser.add_argument("--results_jsonl", default=None, help="Path to results JSONL. If omitted, auto-generates under outputs/multiturn.")
    parser.add_argument("--sharegpt_jsonl", default=None, help="Path to ShareGPT JSONL. If omitted, auto-generates under outputs/multiturn.")
    return parser.parse_args()


def main() -> None:
    global ARGS
    ARGS = parse_args()

    ARGS.results_jsonl, ARGS.sharegpt_jsonl = _autogen_output_paths(ARGS.results_jsonl, ARGS.sharegpt_jsonl)
    print(f"[runner] Results will be appended to: {ARGS.results_jsonl}")
    print(f"[runner] ShareGPT will be appended to: {ARGS.sharegpt_jsonl}")

    _load_banned_phrases()

    print(f"[runner] Loading HF dataset: {ARGS.hf_path} split={ARGS.split}")
    ds = load_dataset(ARGS.hf_path, split=ARGS.split)

    end = min(len(ds), ARGS.start + ARGS.limit)
    print(f"[runner] Processing rows {ARGS.start}:{end} (num_turns={ARGS.num_turns})")

    pipeline = build_pipeline(ARGS.num_turns)

    processed = 0
    slab_size = int(ARGS.batch_size)
    idx = ARGS.start
    # Process the dataset in slabs of --batch_size. Each slab fully runs through
    # seed, follow-ups, and answers in parallel, then appends outputs to JSONLs.
    while idx < end:
        slab_rows = [ds[j] for j in range(idx, min(end, idx + slab_size))]
        try:
            recs = asyncio.run(process_slab_multiturn(
                rows=slab_rows,
                num_turns=ARGS.num_turns,
                batch_size=slab_size,
                difficulty_profile=["recall", "application", "analysis"],
            ))
            for rec in recs:
                _append_jsonl_line(ARGS.results_jsonl, rec)
                _append_sharegpt_filtered(ARGS.sharegpt_jsonl, rec)
            processed += len(recs)
            print(f"[runner] Processed {processed} items...")
        except Exception as e:
            print(f"[runner] Slab {idx}:{min(end, idx+slab_size)} failed: {e}")
        idx += slab_size

    print(f"[runner] Done. Processed={processed}.")
    print(f"[runner] Results -> {ARGS.results_jsonl}")
    print(f"[runner] ShareGPT -> {ARGS.sharegpt_jsonl}")


if __name__ == "__main__":
    main()