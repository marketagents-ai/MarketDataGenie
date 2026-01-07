# testing judge tool calling for each model served with vllm and routed via litellm

# Pydantic model for judge metrics (all fields REQUIRED)
from typing import Optional
from pydantic import Field, BaseModel
from minference.lite.models import ChatMessage, ChatThread, LLMConfig, MessageRole, ResponseFormat, StructuredTool, SystemPrompt, LLMClient
from minference.enregistry import EntityRegistry
from minference.lite.inference import InferenceOrchestrator

EntityRegistry()

# Note: avoid globally instantiating EntityRegistry() or reusing entity instances across threads.
# Some registry implementations track content by UUID; copying/updating can cause content mismatches.

import re
import math
from fractions import Fraction

THINK_OPEN = re.compile(r"<think>", re.IGNORECASE)
THINK_CLOSE = re.compile(r"</think>", re.IGNORECASE)
THINK_BLOCK = re.compile(r"<think>([\s\S]*?)</think>", re.IGNORECASE)
BOXED_ALL = re.compile(r"\\boxed\{([\s\S]*?)\}")


def extract_think_blocks(text: str) -> list[str]:
    return [m.group(1) for m in THINK_BLOCK.finditer(text or "")]

def think_is_fully_formed(text: str) -> tuple[bool, str]:
    if not text:
        return False, "empty content"
    opens = len(THINK_OPEN.findall(text))
    closes = len(THINK_CLOSE.findall(text))
    if opens == 0 and closes == 0:
        return False, "no <think> tags"
    if opens != closes:
        return False, f"unbalanced think tags: open={opens} close={closes}"
    blocks = extract_think_blocks(text)
    if not blocks:
        return False, "no matched <think>…</think> blocks"
    if any(len(b.strip()) == 0 for b in blocks):
        return False, "empty <think> block detected"
    return True, f"ok: {len(blocks)} block(s)"

def extract_all_boxed(text: str) -> list[str]:
    return [m.group(1).strip() for m in BOXED_ALL.finditer(text or "")]

def _extract_boxed(text: str) -> str | None:
    payloads = extract_all_boxed(text)
    return payloads[-1] if payloads else None

_num_pat = re.compile(r"[-+]?\d*\.?\d+(?:[eE][-+]?\d+)?")

def _find_fraction(text: str) -> Fraction | None:
    m = re.search(r"(?<![\d/])(\d+)\s*/\s*(\d+)(?![\d/])", text)
    if not m:
        return None
    try:
        return Fraction(int(m.group(1)), int(m.group(2)))
    except Exception:
        return None

def _find_first_number(text: str) -> float | None:
    m = _num_pat.search(text)
    return float(m.group(0)) if m else None

def _normalize_units(text: str) -> str:
    return text.lower()

def validate_meeting_time(answer_text: str, require_boxed: bool = True) -> tuple[bool, str]:
    """Validate train meet time for D=120km, v1=60, v2=80 toward each other.
    True target is 6/7 hours ≈ 0.8571429 h ≈ 51.4286 minutes.
    Accept any of: '6/7' (any unit), decimal hours within 2%, minutes within 2%.
    Prefer \boxed{...} if present.
    Returns (ok, detail_string).
    """
    target_h = 120/140
    target_min = target_h * 60

    payload = _extract_boxed(answer_text)
    if require_boxed and not payload:
        return False, "no \\boxed{...} answer found"
    src = payload if payload is not None else answer_text
    s = _normalize_units(src)
    used = "from boxed" if payload is not None else "from body"

    # Fraction acceptance
    frac = _find_fraction(s)
    if frac and math.isclose(float(frac), target_h, rel_tol=0.02, abs_tol=1e-4):
        return True, f"accepted fraction {frac.numerator}/{frac.denominator} h ({used})"

    # If units mention minutes, evaluate minute number
    if any(u in s for u in ["min", "minute", "minutes", "मिनेट", "मिनिट", "मि."]):
        n = _find_first_number(s)
        if n is not None and math.isclose(n, target_min, rel_tol=0.02, abs_tol=0.6):
            return True, f"accepted minutes ≈ {n:.2f} ({used})"

    # Otherwise evaluate hours
    n = _find_first_number(s)
    if n is not None and math.isclose(n, target_h, rel_tol=0.02, abs_tol=0.01):
        return True, f"accepted hours ≈ {n:.4f} ({used})"

    # Also accept hh:mm style like 0:51 or 51 min 26 sec
    m = re.search(r"(\d{1,2})\s*[:\-]\s*(\d{1,2})", s)
    if m:
        h = int(m.group(1)); mnt = int(m.group(2));
        if math.isclose(h + mnt/60, target_h, rel_tol=0.02, abs_tol=0.01):
            return True, f"accepted hh:mm {h}:{mnt:02d} ({used})"

    return False, f"no acceptable numeric form detected ({used})"

def has_think_block(text: str):
    """Return (found, snippet) where snippet is the first 120 chars of the think block if present."""
    if not text:
        return False, None
    m = re.search(r"<think>([\s\S]*?)</think>", text, re.IGNORECASE)
    if m:
        snippet = m.group(1).strip().replace("\n", " ")
        return True, (snippet[:120] + ("…" if len(snippet) > 120 else ""))
    return False, None


class MathJudgeVerdict(BaseModel):  # pyright: ignore[reportUndefinedVariable]
    """Structured verdict for math answer judging."""
    verdict: str = Field(..., pattern="^(correct|incorrect|unclear)$", description="Overall judgment")
    score: float = Field(..., ge=0.0, le=1.0, description="Confidence in the verdict, 0..1")
    reason: str = Field(..., min_length=1, max_length=160, description="One-sentence justification")

judge_tool = StructuredTool.from_pydantic(
    model=MathJudgeVerdict,
    name="judge_tool",
    description="Return a strict math-judge verdict: {verdict, score, reason}"
)

def build_judge_user_msg(question: str, answer_text: str, boxed_payload: Optional[str]) -> str:
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

def new_judge_thread(model_id: str, user_msg: str) -> ChatThread:
    # Create a BRAND-NEW LLMConfig and SystemPrompt each time to avoid entity ID reuse
    cfg = LLMConfig(
        client=LLMClient.litellm,
        model=model_id,
        temperature=0.2,
        max_tokens=8192,
        response_format=ResponseFormat.text,
    )
    # Hybrid reasoning activator for models that require it (Hermes-4 family and DeepHermes Mistral hybrid)
    sys_text = (
        "You are a deep thinking AI, you may use extremely long chains of thought to deeply consider the problem "
        "and deliberate with yourself via systematic reasoning processes to help come to a correct solution prior to answering. "
        "You should enclose your thoughts and internal monologue inside <think> </think> tags, and then provide your solution or response to the problem."
        "You should reason and answer in the same language as the question"
    )
    sys = SystemPrompt(name="math_judge", content=sys_text)
    th = ChatThread(
        system_prompt=sys,
        llm_config=cfg,
        tools=[judge_tool],
    )
    th.new_message = user_msg
    return th

def new_reasoning_thread(model_id: str, user_msg: str) -> ChatThread:
    """Create a ChatThread configured to encourage hybrid reasoning (<think> … </think>) without tools."""
    cfg = LLMConfig(
        client=LLMClient.litellm,
        model=model_id,
        temperature=0.6,
        max_tokens=768,
        response_format=ResponseFormat.text,
    )
    sys_text = (
        "You are a deep thinking AI, you may use extremely long chains of thought to deeply consider the problem "
        "and deliberate with yourself via systematic reasoning processes to help come to a correct solution prior to answering. "
        "You should enclose your thoughts and internal monologue inside <think> </think> tags, and then provide your solution or response to the problem."
    )
    sys = SystemPrompt(name="reasoning_probe", content=sys_text)
    th = ChatThread(
        system_prompt=sys,
        llm_config=cfg,
        tools=[],
    )
    th.new_message = user_msg
    th.prefill = "<think>"
    return th


import asyncio

async def test_judge_tool_integration():
    models_to_test = [
#        "NousResearch/DeepHermes-3-Llama-3-8B-Preview",
#        "Qwen/Qwen3-30B-A3B-Instruct-2507",
#        "deepseek-ai/DeepSeek-R1-Distill-Qwen-32B",
#        "mistralai/Ministral-8B-Instruct-2410",
#        "meta-llama/Llama-3.1-8B-Instruct",
#        "google/gemma-3-12b-it",
#        "Qwen/Qwen3-30B-A3B-Thinking-2507",
        "openai/gpt-oss-120b",
#        "Hermes-4-70B",
#        "NousResearch/Hermes-4-405B",
#        "Hermes-4-405B"
#        "Hermes-3-Llama-3.1-70B",
#        "DeepHermes-3-Llama-3-8B-Preview",
#        "DeepHermes-3-Mistral-24B-Preview",
#        "Hermes-3-Llama-3.1-405B",
    ]

    question = "एक आयतको क्षेत्रफल २४ वर्ग सेमी छ भने लम्बाइ ६ सेमी छ, चौडाइ कति हुन्छ?"
    assistant_answer = "हामी क्षेत्रफल = लम्बाइ × चौडाइ प्रयोग गर्छौं। \n\n२४ = ६ × चौडाइ\n⇒ \\boxed{४ \\text{ सेमी}}"
    boxed_payload = "४ \\text{ सेमी}"

    threads = []
    labels = []
    for model_id in models_to_test:
        print(f"\n=== Queueing judge tool test for model: {model_id} ===")
        msg = build_judge_user_msg(question, assistant_answer, boxed_payload)
        th = new_judge_thread(model_id, msg)
        threads.append(th)
        labels.append(model_id)

    orch = InferenceOrchestrator()
    # Submit all at once; returns a list aligned with `threads`
    results = await orch.run_parallel_ai_completion(threads)

    for model_id, result in zip(labels, results):
        try:
            # 1) First, use the direct tool output object if available
            tool = getattr(result, "tool_output", None)
            tool_out = None
            if tool and hasattr(tool, "object") and isinstance(tool.object, dict):
                tool_out = tool.object
            # 2) Next, some runners expose a json_object entity; prefer its `.object` field
            if tool_out is None:
                json_obj = getattr(result, "json_object", None)
                if json_obj is not None:
                    # If it's an entity with `.object`, use that; otherwise accept dict directly
                    if hasattr(json_obj, "object") and isinstance(json_obj.object, dict):
                        tool_out = json_obj.object
                    elif isinstance(json_obj, dict):
                        tool_out = json_obj
            # 3) Fallback: some models leak a <tool_call> blob in content; try to parse its JSON
            if tool_out is None:
                content = (getattr(result, "content", None) or "").strip()
                if "<tool_call>" in content:
                    import json, re
                    # Grab the largest JSON-looking payload inside the tool_call tag
                    blob = content.split("<tool_call>", 1)[-1]
                    blob = blob.split("</tool_call>", 1)[0]
                    # Extract the JSON part after "arguments":
                    m = re.search(r'"arguments"\s*:\s*(\{.*\}|\"\\{.*\\}\")', blob, re.S)
                    if m:
                        args_blob = m.group(1)
                        # If arguments is a quoted JSON string, unquote it
                        if args_blob.startswith('\\"') or args_blob.startswith('"'):
                            args_blob = args_blob.strip().strip('"').replace('\\\\', '\\')
                        try:
                            tool_out = json.loads(args_blob)
                        except Exception:
                            tool_out = None
            if isinstance(tool_out, dict) and {"verdict", "score", "reason"} <= set(tool_out.keys()):
                print(f"✅ {model_id} structured verdict:", tool_out)
            else:
                content = getattr(result, "content", None)
                print(f"⚠️ {model_id} unexpected format.\n  tool_output={getattr(result, 'tool_output', None)}\n  json_object={getattr(result, 'json_object', None)}\n  content={content}")
        except Exception as e:
            print(f"❌ {model_id} failed to parse result: {e}")

async def test_think_block_emission():
    models_to_test = [
        "openai/gpt-oss-120b"
#        "NousResearch/Hermes-4-405B",
#        "Hermes-4-405B",
#        "Hermes-3-Llama-3.1-70B",
#        "DeepHermes-3-Llama-3-8B-Preview",
#        "DeepHermes-3-Mistral-24B-Preview",
#        "Hermes-3-Llama-3.1-405B"
    ]

    prompt = (
    "Two trains start from stations 120 km apart and move toward each other at speeds of 60 and 80 km/h. "
    "How long will it take for them to meet? Please reason step by step, and put your final answer in \\boxed{}."
    )
    threads = [new_reasoning_thread(mid, prompt) for mid in models_to_test]
    orch = InferenceOrchestrator()
    results = await orch.run_parallel_ai_completion(threads)

    for mid, res in zip(models_to_test, results):
        content = (getattr(res, "content", None) or "").strip()
        formed, t_reason = think_is_fully_formed(content)
        blocks = extract_think_blocks(content)
        ok, why = validate_meeting_time(content, require_boxed=True)
        if formed:
            b0 = (blocks[0].strip().replace("\n", " ")[:180] + ("…" if len(blocks[0]) > 180 else "")) if blocks else ""
            think_line = f"🧠 think: formed | {t_reason} | first: {b0}"
        else:
            think_line = f"🚫 think: not formed | {t_reason}"

        print("\n--- FULL THINK BLOCKS ---")
        if blocks:
            for i, b in enumerate(blocks):
                print(f"[{i+1}] {b.strip()}\n")
        else:
            print("No <think> blocks found.")

        boxed_all = extract_all_boxed(content)
        print("--- BOXED ANSWERS ---")
        if boxed_all:
            for i, b in enumerate(boxed_all):
                print(f"[{i+1}] \\boxed{{{b}}}")
        else:
            print("No boxed answers found.")

        verdict = "✅ correct" if ok else "❌ incorrect"
        boxed_last = _extract_boxed(content)
        boxed_note = f"boxed: {'present' if boxed_last else 'missing'}"
        head = content[:140] + ("…" if len(content) > 140 else "")
        print(f"\n=== {mid} ===\n{think_line}\n{verdict} ({why}) | {boxed_note}\nAnswer head: {head}")

if __name__ == "__main__":
    async def _main():
        print("===== Starting <think> tag emission + correctness tests =====")
        await test_think_block_emission()
    asyncio.run(_main())
