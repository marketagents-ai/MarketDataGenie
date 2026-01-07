# testing judge tool calling for each model served with vllm and routed via litellm

# Pydantic model for judge metrics (all fields REQUIRED)
from typing import Optional
from pydantic import Field, BaseModel
from minference.lite.models import ChatThread, LLMConfig, ResponseFormat, StructuredTool, SystemPrompt, LLMClient
from minference.enregistry import EntityRegistry
from minference.lite.inference import InferenceOrchestrator

EntityRegistry()

# Note: avoid globally instantiating EntityRegistry() or reusing entity instances across threads.
# Some registry implementations track content by UUID; copying/updating can cause content mismatches.


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
        max_tokens=4096,
        response_format=ResponseFormat.text,
    )
    # Hybrid reasoning activator for models that require it (Hermes-4 family and DeepHermes Mistral hybrid)
    hybrid_activator = (
        "You are a deep thinking AI, you may use extremely long chains of thought to deeply consider the problem "
        "and deliberate with yourself via systematic reasoning processes to help come to a correct solution prior to answering. "
        "You should enclose your thoughts and internal monologue inside <think> </think> tags, and then provide your solution or response to the problem."
    )
    needs_hybrid = any(k in model_id for k in ["Hermes-4", "Mistral-24B-Preview"])
    sys_text = (
        (hybrid_activator + "\n\n") if needs_hybrid else ""
    ) + "Call the judge_tool. Output must be a tool call. Do not write free-form text."
    sys = SystemPrompt(name="math_judge", content=sys_text)
    th = ChatThread(
        system_prompt=sys,
        llm_config=cfg,
        tools=[judge_tool],
    )
    th.new_message = user_msg
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
#        "Hermes-4-405B",
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

if __name__ == "__main__":
    asyncio.run(test_judge_tool_integration())
