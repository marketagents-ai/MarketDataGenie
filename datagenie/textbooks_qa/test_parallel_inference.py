# test parallel inference with short output
import asyncio
import time
import math
import argparse
from typing import List
from statistics import mean
import sys

# Project imports (assumed available in repo)
from typing import Optional
from pydantic import Field, BaseModel
from minference.lite.models import ChatMessage, ChatThread, LLMConfig, MessageRole, ResponseFormat, StructuredTool, SystemPrompt, LLMClient
from minference.enregistry import EntityRegistry
from minference.lite.inference import InferenceOrchestrator

EntityRegistry()

# If your project has these types elsewhere, adjust imports accordingly.

ACTIVATOR = (
    "You are a deep thinking AI. If you deliberate, enclose thoughts inside <think> </think> and then give the final answer."
)

DEFAULT_PROMPT = (
    "Two trains start from stations 120 km apart and move toward each other at speeds of 60 and 80 km/h. "
    "How long will it take for them to meet? Please reason step by step, and put your final answer in \\boxed{}."
)


def new_reasoning_thread(model_id: str, user_msg: str, temperature: float, max_tokens: int) -> ChatThread:
    cfg = LLMConfig(
        client=LLMClient.litellm,
        model=model_id,
        temperature=temperature,
        max_tokens=max_tokens,
        response_format=ResponseFormat.text,
    )
    sys = SystemPrompt(name="bench_reasoning", content=ACTIVATOR)
    th = ChatThread(system_prompt=sys, llm_config=cfg, tools=[])
    th.new_message = user_msg
    # Nudge models that support hybrid/implicit thinking
    th.prefill = "<think>\n"
    return th


async def run_batch(threads: List[ChatThread]) -> float:
    """Run a batch in parallel and return wall-clock seconds."""
    orch = InferenceOrchestrator()
    start = time.monotonic()
    await orch.run_parallel_ai_completion(threads)
    end = time.monotonic()
    return end - start


async def benchmark(model: str, prompt: str, total_requests: int, concurrencies: List[int],
                    temperature: float, max_tokens: int, warmup: int = 2) -> None:
    print(f"Model: {model}")
    print(f"Prompt head: {prompt[:80]}{'…' if len(prompt) > 80 else ''}")
    print(f"Total requests per setting: {total_requests}")
    print()

    results = []  # collect per-concurrency rows for end-of-run summary

    # Warmup batches (small) so vLLM loads model/kv cache, etc.
    if warmup > 0:
        warm_threads = [new_reasoning_thread(model, prompt, temperature, max_tokens) for _ in range(min(warmup, 4))]
        try:
            _ = await run_batch(warm_threads)
        except Exception as e:
            print(f"[warmup] error: {e}")

    header = f"{'conc':>5}  {'batches':>7}  {'total_req':>10}  {'wall_s':>8}  {'req/s':>8}  {'avg_ms':>8}  {'bar':<20}"
    print(header)
    print("-" * len(header))

    for conc in concurrencies:
        if conc <= 0:
            continue
        batches = math.ceil(total_requests / conc)
        rem = total_requests
        wall = 0.0
        for _ in range(batches):
            this = min(rem, conc)
            rem -= this
            threads = [new_reasoning_thread(model, prompt, temperature, max_tokens) for _ in range(this)]
            # Preflight debug (first batch only for this conc)
            if _ == 0 and threads:
                try:
                    built = threads[0].vllm_messages
                    has_prefill = any(m.get('role') == 'assistant' and (m.get('content') or '').startswith('<think>') for m in built)
                    print(f"[preflight conc={conc}] assistant prefill present: {has_prefill}")
                except Exception as e:
                    print(f"[preflight conc={conc}] error building messages: {e}")
            try:
                wall += await run_batch(threads)
            except Exception as e:
                print(f"[error] batch failed at conc={conc}: {e}")
                break
        throughput = (total_requests / wall) if wall > 0 else 0.0
        avg_ms = (wall / total_requests) * 1000 if total_requests > 0 else 0.0
        # Draw a tiny bar scaled to req/s (capped at 20 chars)
        bar_len = min(20, int(throughput))
        bar = '█' * bar_len
        print(f"{conc:>5}  {batches:>7}  {total_requests:>10}  {wall:>8.2f}  {throughput:>8.2f}  {avg_ms:>8.1f}  {bar:<20}")

        results.append({
            'conc': conc,
            'batches': batches,
            'total_req': total_requests,
            'wall_s': round(wall, 3),
            'req_per_s': round(throughput, 3),
            'avg_ms': round(avg_ms, 1),
        })

    print("\nNotes:")
    print("- req/s: higher is better. avg_ms: approximate average latency per request (wall/total).")
    print("- If req/s flattens as conc increases, you're saturating the backend (CPU/GPU, KV cache, or networking).")

    if results:
        print("\n=== Summary (all concurrencies) ===")
        header2 = f"{'conc':>5}  {'batches':>7}  {'total_req':>10}  {'wall_s':>8}  {'req/s':>8}  {'avg_ms':>8}"
        print(header2)
        print("-" * len(header2))
        for r in results:
            print(f"{r['conc']:>5}  {r['batches']:>7}  {r['total_req']:>10}  {r['wall_s']:>8.2f}  {r['req_per_s']:>8.2f}  {r['avg_ms']:>8.1f}")

        # Also provide CSV for spreadsheets
        print("\nSummary CSV:")
        print("conc,batches,total_req,wall_s,req_per_s,avg_ms")
        for r in results:
            print(f"{r['conc']},{r['batches']},{r['total_req']},{r['wall_s']},{r['req_per_s']},{r['avg_ms']}")

        # Quick best throughput line
        best = max(results, key=lambda x: x['req_per_s'])
        print(f"\nBest throughput: conc={best['conc']} at {best['req_per_s']} req/s (avg_ms={best['avg_ms']})")

        # SLO evaluation for a specific concurrency target
        target_conc = args.slo_conc if 'args' in globals() else 64
        target = next((r for r in results if r['conc'] == target_conc), None)
        if target is not None:
            ok_reqps = target['req_per_s'] >= args.slo_min_reqps
            ok_avg = target['avg_ms'] <= args.slo_avg_ms
            status = "PASS" if (ok_reqps and ok_avg) else "FAIL"
            print(f"\nSLO check @ conc={target_conc}: status={status} | req/s={target['req_per_s']} (>= {args.slo_min_reqps}) & avg_ms={target['avg_ms']} (<= {args.slo_avg_ms})")
            # non-zero exit on FAIL so CI can catch it
            if status == "FAIL":
                sys.exit(2)


def parse_args():
    p = argparse.ArgumentParser(description="Parallel inference benchmark for vLLM via ChatThread")
    p.add_argument("--model", type=str, default="openai/gpt-oss-120b",
                   help="Model ID as seen by your vLLM endpoint")
    p.add_argument("--prompt", type=str, default=DEFAULT_PROMPT,
                   help="User prompt to send")
    p.add_argument("--total", type=int, default=64,
                   help="Total requests per concurrency setting")
    p.add_argument("--concurrencies", type=int, nargs="*", default=[4, 8, 16, 32, 64],
                   help="Concurrency levels to test")
    p.add_argument("--temperature", type=float, default=0.6)
    p.add_argument("--max_tokens", type=int, default=128)
    p.add_argument("--warmup", type=int, default=2)
    p.add_argument("--slo-avg-ms", type=float, default=150.0,
                   help="SLO threshold for average latency per request (ms) at target concurrency.")
    p.add_argument("--slo-min-reqps", type=float, default=8.0,
                   help="Minimum acceptable throughput (req/s) at target concurrency.")
    p.add_argument("--slo-conc", type=int, default=64,
                   help="Concurrency level to evaluate PASS/FAIL against SLOs.")
    return p.parse_args()


args = None

if __name__ == "__main__":
    args = parse_args()
    asyncio.run(benchmark(
        model=args.model,
        prompt=args.prompt,
        total_requests=args.total,
        concurrencies=args.concurrencies,
        temperature=args.temperature,
        max_tokens=args.max_tokens,
        warmup=args.warmup,
    ))