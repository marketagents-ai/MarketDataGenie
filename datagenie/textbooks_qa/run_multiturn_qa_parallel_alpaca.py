 # ---- debug helper ----
def _dbg(msg: str) -> None:
    try:
        print(msg, flush=True)
    except Exception:
        pass

"""
Runner script for multi-turn ShareGPT generation on HuggingFace alpaca QA dataset
It uses the modular MultiTurnQAPipeline and:
- Seeds a persistent ChatThread per chunk with the rephrased text in the system prompt
- Reuses existing single-turn Q(a)/A(a) from the dataset's `conversations` list as few-shot context (NOT counted toward num_turns)
- Generates follow-up questions via a simple agent persona (unless disabled)
- Answers all turns through ChatThread + InferenceOrchestrator
- Appends JSONL lines to results and sharegpt outputs (safe to resume)

Usage (examples):
python datagenie/textbooks_qa/run_multiturn_qa_parallel_alpaca.py \
  --hf_path your_username/your-textbook-dataset-multiturn \
  --source "your_username/your-textbook-dataset" \
  --num_turns 3 \
  --start 0 \
  --limit 100 \
  --batch_size 32 \
  --seed_question \
  --resume
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
)

# Agent bits for follow-up question generation
from market_agents.agents.base_agent.agent import Agent as MarketAgent  # thin Agent wrapper
from market_agents.agents.personas.persona import Persona
from market_agents.agents.base_agent.prompter import PromptManager

from minference.lite.models import (
    SystemPrompt,
    ChatThread,
    ChatMessage,
    MessageRole,
    ProcessedOutput,
    LLMConfig,
    LLMClient,
    ResponseFormat,
)
from minference.lite.inference import InferenceOrchestrator

# --- Local definition of SimpleChatThread (AGI-style wrapper) ---
class SimpleChatThread:
    """
    Persistent ChatThread wrapper for AGI-human conversations.
    Seeds hidden context once and appends user questions to get assistant answers via InferenceOrchestrator.

    Key changes from old implementation:
    - Uses a super-intelligent, first-principles, maximally truth-seeking system prompt.
    - Enforces <think> ... </think> reasoning blocks before every answer.
    - No legacy textbook QA framing.
    """
    def __init__(self, instructions: str, prefill_think: bool = True,  llm_model: str = "Hermes-4-405B"):
        self.instructions = instructions
        self.llm_model = llm_model
        self.prefill_think = prefill_think

        # Build updated AGI-style system prompt
        sys_content = (
        #    "You are a deep thinking AI, you may use extremely long chains of thought to deeply consider the problem and deliberate with yourself via systematic reasoning processes to help come to a correct solution prior to answering. "
        #    "You should enclose your thoughts and internal monologue inside <think> </think> tags, and then provide your solution or response to the problem.\n"
            "You are Hermes 4. Be concise and helpful "
            "You are a super-intelligent AI assistant. You reason from first principles. You are maximally truth-seeking. "
            f"#Instructions:\n{self.instructions}"
        )

        system_prompt = SystemPrompt(name="agi-thread", content=sys_content)

        # LLM configuration optimized for reasoning-style outputs
        self.llm_config = LLMConfig(
            client=LLMClient.litellm,
            model=self.llm_model,
            temperature=0.4,
            response_format=ResponseFormat.text,
            max_tokens=2048
        )

        # Underlying ChatThread
        self.chat_thread = ChatThread(
            name="agi-thread",
            system_prompt=system_prompt,
            llm_config=self.llm_config,
            tools=[],
            new_message=None,
        )

        self.orchestrator = InferenceOrchestrator()

    def append_user(self, text: str) -> None:
        user_msg = ChatMessage(role=MessageRole.user, content=text)
        self.chat_thread.history.append(user_msg)

    def append_assistant(self, text: str) -> None:
        parent = self.chat_thread.history[-1] if self.chat_thread.history else None
        assistant_msg = ChatMessage(
            role=MessageRole.assistant,
            content=text,
            parent_message_uuid=parent.id if parent else None
        )
        self.chat_thread.history.append(assistant_msg)

    async def ask(self, user_text: str) -> str:
        # Set new message and prefill <think> for reasoning-style completions
        self.chat_thread.new_message = user_text
        if self.prefill_think:
            try:
                setattr(self.chat_thread, "prefill", "<think>")
            except Exception:
                pass

        outputs = await self.orchestrator.run_parallel_ai_completion([self.chat_thread])
        if not outputs:
            return ""

        last: ProcessedOutput = outputs[-1]
        answer = (last.content or "").strip()

        # Commit this turn to ChatThread history
        self.chat_thread.new_message = user_text
        _ = self.chat_thread.add_user_message()
        await self.chat_thread.add_chat_turn_history(last)

        return answer

import asyncio
from typing import Tuple, Sequence

#
#
#
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

import re
THINK_PATTERN = re.compile(r"<think>.+?</think>", re.DOTALL)

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
    base_dir = Path("outputs/multiturn_alpaca")
    base_dir.mkdir(parents=True, exist_ok=True)
    res = results_jsonl or str(base_dir / f"multiturn_results_{ts}.jsonl")
    sgd = sharegpt_jsonl or str(base_dir / f"multiturn_sharegpt_{ts}.jsonl")
    return res, sgd


# --- Published set helpers ---

def _norm_text(s: Optional[str]) -> str:
    if not isinstance(s, str):
        return ""
    return " ".join(s.split()).strip()

# --- New helpers: first-gpt duplicate skipping ---
def _first_gpt_value(conversations: List[Dict[str, Any]]) -> str:
    """
    Return the first non-empty assistant ('gpt') message value from a ShareGPT-style conversation list.
    Normalizes whitespace. Returns '' if not found.
    """
    if not isinstance(conversations, list):
        return ""
    for m in conversations:
        try:
            if (m or {}).get("from") == "gpt":
                v = (m.get("value") or "").strip()
                if v:
                    return " ".join(v.split())
        except Exception:
            continue
    return ""

def _first_gpt_value_from_row(row: Dict[str, Any]) -> str:
    try:
        return _first_gpt_value(row.get("conversations") or [])
    except Exception:
        return ""

def _load_published_first_gpt_set(repo_id: str, split: str) -> set[str]:
    """
    Load a published HF dataset and collect a set of normalized *first gpt message* values
    from each record's `conversations`. Used for duplicate skipping when human text varies slightly.
    """
    try:
        dset = load_dataset(repo_id, split=split)
    except Exception as e:
        _dbg(f"[skip] could not load published repo {repo_id}:{split}: {e}")
        return set()
    keys: set[str] = set()
    count_rows = 0
    for row in dset:
        count_rows += 1
        v = _first_gpt_value_from_row(row)
        if v:
            keys.add(v)
    _dbg(f"[skip] loaded {len(keys)} published first-gpt keys (from {count_rows} rows) from {repo_id}:{split}")
    return keys

# Key strategy: we consider a sample already published if the (id, rephrased_text) pair exists.
# This survives minor reorderings and guards against accidental collisions.
# If 'rephrased_text' is missing in the published dataset, we fall back to id-only matching.

def _load_published_keys(repo_id: str, split: str) -> set[str]:
    try:
        dset = load_dataset(repo_id, split=split)
    except Exception as e:
        _dbg(f"[skip] could not load published repo {repo_id}:{split}: {e}")
        return set()
    keys: set[str] = set()
    for row in dset:
        rre = _norm_text(row.get("rephrased_text") or row.get("context_text") or "")
        if rre:
            keys.add(rre)
    _dbg(f"[skip] loaded {len(keys)} published rephrased_text keys from {repo_id}:{split}")
    return keys


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
# --- New: Parallel seed agent runner
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
    """
    Extract the best-available textual content from a ProcessedOutput.

    Order of precedence:
    1) Parsed `content`
    2) Parsed `json_object` fields: answer/content/text/output
    3) Provider-specific `reasoning_content` if present on the object
    4) Raw OpenAI-style payload under `raw_output.raw_result`:
       - choices[0].message.content
       - choices[0].message.reasoning_content
       If both exist, return "<think>reasoning</think>\n\ncontent".
    """
    # Primary parsed content
    txt = (getattr(out, "content", None) or "").strip()
    if txt:
        return txt

    # JSON-mode outputs may carry a serialized object; try to stringify a useful field
    try:
        jo = getattr(out, "json_object", None)
        if isinstance(jo, dict):
            for k in ("answer", "content", "text", "output"):
                v = jo.get(k)
                if isinstance(v, str) and v.strip():
                    return v.strip()
    except Exception:
        pass

    # Some providers expose reasoning separately on the ProcessedOutput
    try:
        rc = (getattr(out, "reasoning_content", None) or "").strip()
        if rc:
            return rc
    except Exception:
        pass

    # Fall back to raw OpenAI-like payload if available
    try:
        ro = getattr(out, "raw_output", None)
        raw = getattr(ro, "raw_result", None)
        if isinstance(raw, dict):
            choices = raw.get("choices") or []
            if isinstance(choices, list) and choices:
                msg = choices[0].get("message") or {}
                c = (msg.get("content") or "").strip()
                rc = (msg.get("reasoning_content") or "").strip()
                if c and rc:
                    return f"<think>\n{rc}\n</think>\n\n{c}"
                if c:
                    return c
                if rc:
                    return rc
    except Exception:
        pass

    return ""

# --- Helper: indices where most recent Q and A are both non-empty ---
from typing import List
def _active_indices_for_next_turn(q_hist: List[List[str]], a_hist: List[List[str]]) -> List[int]:
    """
    Return indices where the most recent Q and A are both non-empty.
    We only generate a follow-up and run an answer for these threads.
    """
    active: List[int] = []
    for i in range(len(q_hist)):
        try:
            q_ok = bool((q_hist[i][-1] if q_hist[i] else "").strip())
            a_ok = bool((a_hist[i][-1] if a_hist[i] else "").strip())
            if q_ok and a_ok:
                active.append(i)
        except Exception:
            continue
    return active

#
# Follow-up question AGENT (EPHEMERAL): builds a prompt from prior Q/A + label.
# Returns a ChatThread to be run once in a parallel batch.
#
async def followup_question_agent(prev_questions: List[str], prev_answers: List[str], label: str, target_level: str) -> str:
    """Generate the next self-contained human question that deepens the AGI–human conversation."""
    persona = Persona(
        role="Conversation Deepener",
        persona=(
            "You help a human probe topics with sharper, more rigorous questions. "
            "You guide the discussion toward first-principles understanding and truth-seeking."
        ),
        objectives=["Produce exactly one next user question that is self-contained and concise."],
        skills=["Socratic inquiry", "First-principles decomposition", "Clarifying assumptions"],
    )
    history_text = "\n".join([f"Q: {q}\nA: {a}" for (q, a) in zip(prev_questions, prev_answers)])
    if not history_text.strip():
        # No usable history → signal the caller to skip this item
        raise ValueError("empty history for follow-up generation")
    task = (
        f"Conversation so far:\n{history_text}\n\n"
        f"Write the next human question that:\n"
        f"- is fully self-contained and answerable without external context;\n"
        f"- recreates a **complete solvable question** with necessary numbers, equations, or assumptions for problem sets\n"
        f"- does not include any answer;\n"
        f"- stays in the same language as the prior turns\n"
        f"- may include English phrases for technical terms in brackets"
        f"Directly output the question"
    )
    agent = MarketAgent(
        name="followup-question",
        persona=persona,
        task=task,
        llm_config=LLMConfig(
            client=LLMClient.litellm,
            model=os.environ.get("DATAGEN_MODEL", "Hermes-4-70B"),
            temperature=0.4,
            response_format=ResponseFormat.text,
            max_tokens=1024,
        ),
        llm_orchestrator=InferenceOrchestrator(),
        prompt_manager=PromptManager(),
        tools=[],
    )
    agent.chat_thread.new_message = task
    return agent.chat_thread

# --- Seed question agent ---
async def seed_question_agent(chunk: Dict[str, Any], rephrased_text: str) -> ChatThread:
    """
    Build an EPHEMERAL agent ChatThread prepared to generate a *fresh* seed question.
    Caller should batch-run the returned ChatThread via the orchestrator and capture its completion.
    We set `chat_thread.new_message` to the task so the orchestrator has a user message.
    """
    # If dataset already provides a seed question and ARGS.seed_question is False,
    # the runner will use that upstream and won't call this function.
    persona = Persona(
        role="Question Writer",
        persona=(
            "You are an expert socratic inquirer who asks interesting and informative questions. "
            "You design questions for conversation with a super-intelligent AI assistant "
        ),
        objectives=["Produce one concise conversational question"],
        skills=["Question design", "Socratic Inquiry"],
    )
    banned_txt = _banned_block()
    task_prompt = (
        "Context (hidden):\n" + (rephrased_text or "") +
        "\n\n##Guidelines:\n"
        "- Write a single concise question.\n"
        "- Use the provided QA pair as context for question generation.\n"
        "- The question must be fully self-contained and answerable **without** additional context.\n"
        "- If the context is fiction (stories, poems, etc.), ask questions about **themes, ideas, or structures** rather than exact character or plot details.\n"
        "- For problem sets (maths, physics, chemistry, etc.), recreate a **complete solvable question** with necessary numbers, equations, or assumptions.\n"
        "- Do **not** generate questions that depend on diagrams or illustrations.\n"
        "- Provide complete details such as equations, markdown tables, quotes etc. required for answering questions.\n"
        "- For technical terms you may put English phrases or full forms in brackets\n"
        + (f"- {banned_txt}\n" if banned_txt else "")
        + "- Directly output only the question text.\n"
    )
    agent = MarketAgent(
        name="seed-question",
        persona=persona,
        task=task_prompt,
        llm_config=LLMConfig(
            client=LLMClient.litellm,
            model=os.environ.get("DATAGEN_MODEL", "Hermes-4-70B"),
            temperature=0.2,
            response_format=ResponseFormat.text,
            max_tokens=2048,
        ),
        llm_orchestrator=InferenceOrchestrator(),
        prompt_manager=PromptManager(),
        tools=[],
    )
    # Ensure the user message is set on the thread for orchestrator
    agent.chat_thread.new_message = agent.task
    return agent.chat_thread

# --- Batch runner for seed agent ---
async def run_seed_agent_parallel(
    chunks: List[Dict[str, Any]],
    batch_size: int,
) -> List[str | Exception]:
    """
    For each chunk, construct a seed-question agent using the dataset's first human/gpt pair
    as hidden context, then run all in parallel and return the generated seed questions.
    """
    # Build simple context from existing first pair, if available
    def _context_from_chunk(ch: Dict[str, Any]) -> str:
        conv = (ch.get("conversations") or []) if isinstance(ch, dict) else []
        q = ""
        a = ""
        if isinstance(conv, list) and len(conv) >= 1 and (conv[0] or {}).get("from") == "human":
            q = (conv[0].get("value") or "").strip()
        if isinstance(conv, list) and len(conv) >= 2 and (conv[1] or {}).get("from") == "gpt":
            a = (conv[1].get("value") or "").strip()
        ctx_parts = []
        if q:
            ctx_parts.append(f"Seed Q: {q}")
        if a:
            ctx_parts.append(f"Seed A: {a}")
        return "\n".join(ctx_parts)

    threads: List[ChatThread] = []
    id_to_pos: Dict[str, int] = {}
    for idx, ch in enumerate(chunks):
        ctx = _context_from_chunk(ch)
        th = await seed_question_agent(ch, ctx)
        threads.append(th)
        tid = _thread_id(th)
        if tid:
            id_to_pos[tid] = idx

    raw = await _run_threads_in_batches(threads, batch_size=batch_size)
    outs: List[str | Exception] = [None] * len(threads)
    for r in raw:
        if isinstance(r, Exception):
            continue
        rid = _out_id(r)
        pos = id_to_pos.get(rid)
        if pos is not None:
            outs[pos] = _extract_text(r)
    for i in range(len(outs)):
        if outs[i] is None:
            outs[i] = Exception("seed question generation failed or id missing")
    return outs


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
    valid_positions: List[int] = []
    for pos, ((qs, ans), lab, lvl) in enumerate(zip(histories, labels, levels)):
        try:
            th = await followup_question_agent(qs, ans, lab, lvl)
            threads.append(th)
            valid_positions.append(pos)
        except Exception as e:
            _dbg(f"[followup] skipping pos={pos}: {e}")
            # placeholder to preserve output alignment for caller
    # Build an index map by thread id to position
    id_to_pos: Dict[str, int] = {}
    for pos, th in enumerate(threads):
        tid = _thread_id(th)
        if tid:
            id_to_pos[str(tid)] = pos

    raw = await _run_threads_in_batches(threads, batch_size=batch_size)
    outs: List[str | Exception] = [Exception("follow-up skipped: empty history")] * len(histories)
    for r in raw:
        if isinstance(r, Exception):
            continue
        rid = _out_id(r)
        pos = id_to_pos.get(rid)
        if pos is not None:
            outs[valid_positions[pos]] = _extract_text(r)
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
        # Prefill reasoning tag to nudge the model to emit <think> first
        try:
            setattr(th.chat_thread, "prefill", "<think>\n")
        except Exception:
            pass
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



async def process_slab_multiturn(
    rows: List[Dict[str, Any]],
    num_turns: int,
    batch_size: int,
    difficulty_profile: List[str] = None,
    use_seed_agent: bool = False,
) -> List[Dict[str, Any]]:
    difficulty_profile = difficulty_profile or ["recall", "application", "analysis"]
    chunks = [row_to_chunk(r) for r in rows]

    # If requested, generate a fresh seed question via agent and answer it first.
    if use_seed_agent:
        _dbg(f"[seed-agent] generating fresh seed questions for {len(chunks)} chunks")
        # Build one persistent answer thread per chunk
        simple_threads: List[SimpleChatThread] = [new_chatthread_fn() for _ in chunks]
        # Generate seed questions
        seed_qs = await run_seed_agent_parallel(chunks=chunks, batch_size=batch_size)
        # Convert Exceptions to empty strings to keep alignment; they'll get blank answers.
        seed_qs_text = [sq if isinstance(sq, str) else "" for sq in seed_qs]
        # Answer the seed questions
        seed_as = await answers_parallel_via_threads(
            simple_threads=simple_threads,
            next_questions=seed_qs_text,
            batch_size=batch_size,
        )
        seed_as_text = [sa if isinstance(sa, str) else "" for sa in seed_as]

        # Initialize histories with the GENERATED seed turn
        q_hist: List[List[str]] = [[q] if q else [""] for q in seed_qs_text]
        a_hist: List[List[str]] = [[a] if a else [""] for a in seed_as_text]

        # Follow-up turns (num_turns already counts the seed; we generated 1, so do num_turns-1 more)
        for t in range(num_turns - 1):
            label = chr(ord("b") + t)
            level = difficulty_profile[min(t, len(difficulty_profile) - 1)]

            active = _active_indices_for_next_turn(q_hist, a_hist)
            _dbg(f"[followup t={t}] active={len(active)} / {len(chunks)}")
            if not active:
                _dbg(f"[followup t={t}] no active threads; breaking early")
                break

            # Build subset histories for active threads only
            sub_histories = [ (q_hist[i], a_hist[i]) for i in active ]
            try:
                sub_followups = await run_followup_agent_parallel(
                    histories=sub_histories,
                    labels=[label] * len(active),
                    levels=[level] * len(active),
                    batch_size=batch_size,
                )
            except Exception as e:
                _dbg(f"[followup t={t}] generation batch failed: {e}")
                sub_followups = [Exception("follow-up gen failed")] * len(active)

            # Prepare next questions only for active indices
            sub_next_questions = [fq if isinstance(fq, str) else "" for fq in sub_followups]
            sub_threads = [simple_threads[i] for i in active]

            try:
                sub_answers = await answers_parallel_via_threads(
                    simple_threads=sub_threads,
                    next_questions=sub_next_questions,
                    batch_size=batch_size,
                )
            except Exception as e:
                _dbg(f"[followup t={t}] answer batch failed: {e}")
                sub_answers = [Exception("answer failed")] * len(active)

            # Stitch back into q_hist/a_hist; append blanks for inactive to keep lengths aligned
            ans_debug = []
            for pos, i in enumerate(active):
                fq = sub_followups[pos]
                fa = sub_answers[pos]
                q_hist[i].append(fq if isinstance(fq, str) and fq.strip() else "")
                a_hist[i].append(fa if isinstance(fa, str) and fa.strip() else "")
                ans_debug.append(str(len(fa)) if isinstance(fa, str) else "ERR")
            _dbg(f"[followup t={t}] answered(active): " + ", ".join(ans_debug))

            inactive = [i for i in range(len(chunks)) if i not in active]
            for i in inactive:
                q_hist[i].append("")
                a_hist[i].append("")

        # Build final records
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
                "conversations": convs,
                # Explicitly mark this as NOT a dataset seed so exporter does not drop it
                "dataset_seed_present": False,
            }
            try:
                assert len(q_hist[i]) == len(a_hist[i]) == num_turns, f"pair count mismatch id={rec['id']} q={len(q_hist[i])} a={len(a_hist[i])}"
            except AssertionError as e:
                _dbg(f"[warn] {e}")
            _dbg(f"[record] id={rec['id']} pairs={len(q_hist[i])}")
            records.append(rec)
        return records

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

    # Filter to rows that have a dataset-provided seed Q/A; skip others
    valid_idxs = [i for i in range(len(chunks)) if dataset_example_q[i] and dataset_example_a[i]]
    dataset_example_q = [dataset_example_q[i] for i in valid_idxs]
    dataset_example_a = [dataset_example_a[i] for i in valid_idxs]
    # Step 2: build one persistent answer thread per chunk and preload dataset example pair (few-shot) if present
    simple_threads: List[SimpleChatThread] = []
    for ch in chunks:
        simple_threads.append(new_chatthread_fn())
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
    # Initialize histories with the dataset seed pair
    q_hist: List[List[str]] = [[q] for q in dataset_example_q]
    a_hist: List[List[str]] = [[a] for a in dataset_example_a]

    # Follow-up turns
    for t in range(num_turns - 1):
        label = chr(ord("b") + t)
        level = difficulty_profile[min(t, len(difficulty_profile) - 1)]

        active = _active_indices_for_next_turn(q_hist, a_hist)
        _dbg(f"[followup t={t}] active={len(active)} / {len(chunks)}")
        if not active:
            _dbg(f"[followup t={t}] no active threads; breaking early")
            break

        sub_histories = [ (q_hist[i], a_hist[i]) for i in active ]
        try:
            sub_followups = await run_followup_agent_parallel(
                histories=sub_histories,
                labels=[label] * len(active),
                levels=[level] * len(active),
                batch_size=batch_size,
            )
        except Exception as e:
            _dbg(f"[followup t={t}] generation batch failed: {e}")
            sub_followups = [Exception("follow-up gen failed")] * len(active)

        sub_next_questions = [fq if isinstance(fq, str) else "" for fq in sub_followups]
        sub_threads = [simple_threads[i] for i in active]
        try:
            sub_answers = await answers_parallel_via_threads(
                simple_threads=sub_threads,
                next_questions=sub_next_questions,
                batch_size=batch_size,
            )
        except Exception as e:
            _dbg(f"[followup t={t}] answer batch failed: {e}")
            sub_answers = [Exception("answer failed")] * len(active)

        ans_debug = []
        for pos, i in enumerate(active):
            fq = sub_followups[pos]
            fa = sub_answers[pos]
            q_hist[i].append(fq if isinstance(fq, str) and fq.strip() else "")
            a_hist[i].append(fa if isinstance(fa, str) and fa.strip() else "")
            ans_debug.append(str(len(fa)) if isinstance(fa, str) else "ERR")
        _dbg(f"[followup t={t}] answered(active): " + ", ".join(ans_debug))

        inactive = [i for i in range(len(chunks)) if i not in active]
        for i in inactive:
            q_hist[i].append("")
            a_hist[i].append("")

    # Build final records for output (minimal schema: only id and conversations)
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
            "conversations": convs,
            "dataset_seed_present": True,
        }
        try:
            assert len(q_hist[i]) == len(a_hist[i]) == num_turns, f"pair count mismatch id={rec['id']} q={len(q_hist[i])} a={len(a_hist[i])}"
        except AssertionError as e:
            _dbg(f"[warn] {e}")
        _dbg(f"[record] id={rec['id']} pairs={len(q_hist[i])}")
        records.append(rec)
    return records



# Construct a SimpleChatThread with the hidden system context set to rephrased text.
# The model name is pulled from $DATAGEN_MODEL or defaults to a reasonable baseline.
def new_chatthread_fn() -> ChatThread:
    """Construct a ChatThread-backed wrapper with an AGI system prompt and hidden context."""
    banned_txt = _banned_block()
    instructions = (
        "- Answer in the same language as the question.\n"
        "- For technical terms you may put English phrases or full forms in brackets\n"
        "- For exact-answer problems (math/science), reason step by step and put the final answer in a box: \\boxed{}.\n"
        "- Answer the question directly.\n"
    )
    if banned_txt:
        instructions += f"- {banned_txt}"
    combined = f"{instructions}"
    return SimpleChatThread(instructions=combined, llm_model=os.environ.get("DATAGEN_MODEL", "Hermes-4-405B"))


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
        rephraser_fn=None,
        seed_question_fn=None,
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
        "conversations": row.get("conversations"),  # seed Q(a)/A(a)
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run multi-turn QA generation on HF dataset.")
    parser.add_argument("--hf_path", default=None)
    parser.add_argument("--split", default="train")
    parser.add_argument("--start", type=int, default=0)
    parser.add_argument("--limit", type=int, default=None, help="Max rows to process after filtering; omit for no cap")
    parser.add_argument("--num_turns", type=int, default=3, help="Number of GENERATED user turns (seed Q/A is few-shot and NOT counted)")
    parser.add_argument("--batch_size", type=int, default=64, help="Parallel slab size for orchestrator.run_parallel_ai_completion")
    parser.add_argument("--results_jsonl", default=None, help="Path to results JSONL. If omitted, auto-generates under outputs/multiturn.")
    parser.add_argument("--sharegpt_jsonl", default=None, help="Path to ShareGPT JSONL. If omitted, auto-generates under outputs/multiturn.")
    parser.add_argument("--skip_published", action="store_true", help="Skip rows that already exist in the published multiturn HF repo")
    parser.add_argument("--published_repo_id", default="your_username/your-seed-question-dataset", help="HF dataset repo id that already contains published multiturn records")
    parser.add_argument("--published_split", default="train", help="Split to read from the published multiturn repo")
    parser.add_argument("--seed_question", action="store_true", help="Generate a fresh seed question via agent instead of using dataset's first question")
    parser.add_argument("--source", default=None, help="Filter dataset by source repository name")
    parser.add_argument("--resume", action="store_true", help="Resume generation by skipping already processed samples (based on original dataset IDs)")
    return parser.parse_args()


def _load_processed_ids(results_jsonl: str) -> set[str]:
    """Load set of already processed sample IDs from results JSONL file."""
    processed_ids = set()
    if not Path(results_jsonl).exists():
        return processed_ids
    
    try:
        with open(results_jsonl, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    record = json.loads(line)
                    sample_id = record.get("id")
                    if sample_id:
                        processed_ids.add(sample_id)
                except Exception:
                    continue
    except Exception as e:
        _dbg(f"[resume] Warning: could not load processed IDs from {results_jsonl}: {e}")
    
    _dbg(f"[resume] Loaded {len(processed_ids)} already processed sample IDs")
    return processed_ids

def _filter_processed_samples(candidates: List[Dict[str, Any]], processed_ids: set[str]) -> List[Dict[str, Any]]:
    """Filter out samples that have already been processed."""
    if not processed_ids:
        return candidates
    
    filtered = []
    skipped = 0
    for candidate in candidates:
        sample_id = candidate.get("id")
        if sample_id and sample_id in processed_ids:
            skipped += 1
            continue
        filtered.append(candidate)
    
    _dbg(f"[resume] Skipped {skipped} already processed samples, {len(filtered)} remaining")
    return filtered


def main() -> None:
    global ARGS
    ARGS = parse_args()

    ARGS.results_jsonl, ARGS.sharegpt_jsonl = _autogen_output_paths(ARGS.results_jsonl, ARGS.sharegpt_jsonl)
    print(f"[runner] Results will be appended to: {ARGS.results_jsonl}")
    print(f"[runner] ShareGPT will be appended to: {ARGS.sharegpt_jsonl}")

    _load_banned_phrases()

    print(f"[runner] Loading HF dataset: {ARGS.hf_path} split={ARGS.split}")
    ds = load_dataset(ARGS.hf_path, split=ARGS.split)

    total_len = len(ds)
    if ARGS.skip_published:
        # Scan from start to the end so we can fill the post-filter limit properly
        end = total_len
        print(f"[runner] Skip-published enabled: scanning rows {ARGS.start}:{end} (will apply --limit AFTER filtering)")
    else:
        if ARGS.limit is None:
            end = total_len
        else:
            end = min(total_len, ARGS.start + ARGS.limit)
        print(f"[runner] Processing rows {ARGS.start}:{end} (num_turns={ARGS.num_turns})")

    if ARGS.skip_published:
        print(f"[runner] Will skip rows that appear in {ARGS.published_repo_id}:{ARGS.published_split}")

    # Add source filtering
    if ARGS.source:
        print(f"[runner] Filtering by source: {ARGS.source}")
        before_source_filter = len(ds)
        ds = ds.filter(lambda x: x.get("source") == ARGS.source)
        after_source_filter = len(ds)
        print(f"[runner] Source filtering: {before_source_filter} -> {after_source_filter} rows")
        
        # Update total_len after source filtering
        total_len = len(ds)
        if ARGS.skip_published:
            end = total_len
        else:
            if ARGS.limit is None:
                end = total_len
            else:
                end = min(total_len, ARGS.start + ARGS.limit)

    pipeline = build_pipeline(ARGS.num_turns)

    processed = 0
    slab_size = int(ARGS.batch_size)

    # Build candidate rows according to start/limit
    candidates = [ds[j] for j in range(ARGS.start, end)]

    # Add resume functionality - skip already processed samples
    if ARGS.resume:
        print(f"[runner] Resume mode enabled: checking for already processed samples")
        processed_ids = _load_processed_ids(ARGS.results_jsonl)
        before_resume = len(candidates)
        candidates = _filter_processed_samples(candidates, processed_ids)
        after_resume = len(candidates)
        print(f"[runner] Resume filtering: {before_resume} -> {after_resume} candidates")

    # Optionally filter out rows that already exist in the published multiturn repo
    if ARGS.skip_published:
        # Prefer duplicate skipping by the *first assistant ('gpt') message*,
        # since the human text might vary slightly due to concatenation.
        published_gpt = _load_published_first_gpt_set(ARGS.published_repo_id, ARGS.published_split)
        before = len(candidates)

        def _row_first_gpt(r: Dict[str, Any]) -> str:
            return _first_gpt_value_from_row(r)

        # Keep rows where first gpt is either missing (we want to generate them)
        # or not present in the published set. This avoids re-generating already-done samples.
        filtered = []
        dropped = 0
        for r in candidates:
            g = _row_first_gpt(r)
            if g and g in published_gpt:
                dropped += 1
                continue
            filtered.append(r)
        candidates = filtered
        after = len(candidates)
        print(f"[runner] Skip-published enabled (first-gpt match). Filtered {dropped} rows; remaining={after}")

    # If user provided --limit, respect it AFTER filtering so we fill the requested work quota
    if ARGS.limit is not None and len(candidates) > ARGS.limit:
        _dbg(f"[runner] Applying post-filter limit: taking first {ARGS.limit} of {len(candidates)} candidates")
        candidates = candidates[:ARGS.limit]

    # Process the remaining candidates in slabs
    idx = 0
    total = len(candidates)
    while idx < total:
        slab_rows = candidates[idx: min(total, idx + slab_size)]
        try:
            recs = asyncio.run(process_slab_multiturn(
                rows=slab_rows,
                num_turns=ARGS.num_turns,
                batch_size=slab_size,
                difficulty_profile=["recall", "application", "analysis"],
                use_seed_agent=ARGS.seed_question,
            ))
            for rec in recs:
                _append_jsonl_line(ARGS.results_jsonl, rec)
                _append_sharegpt_filtered(ARGS.sharegpt_jsonl, rec)
            processed += len(recs)
            print(f"[runner] Processed {processed} items...")
        except Exception as e:
            print(f"[runner] Slab {idx}:{min(total, idx+slab_size)} failed: {e}")
        idx += slab_size

    print(f"[runner] Done. Processed={processed}.")
    print(f"[runner] Results -> {ARGS.results_jsonl}")
    print(f"[runner] ShareGPT -> {ARGS.sharegpt_jsonl}")


if __name__ == "__main__":
    main()