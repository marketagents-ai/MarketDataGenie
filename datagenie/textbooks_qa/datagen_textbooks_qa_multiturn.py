from __future__ import annotations

import time
import uuid
from dataclasses import dataclass, field
from typing import Callable, Dict, List, Optional, Any

# ChatThread + Orchestrator (answers come from here)
from minference.lite.models import (
    SystemPrompt,
    ChatThread,
    ProcessedOutput,
    ChatMessage,
    MessageRole,
    LLMConfig,
    LLMClient,
    ResponseFormat,
)
from minference.lite.inference import InferenceOrchestrator


# ============================================================
# Configuration
# ============================================================

@dataclass
class PipelineConfig:
    mode: str = "multiturn"  # "single" | "multiturn"
    num_turns: int = 3       # number of GENERATED user turns (seed Q/A is few-shot and not counted)
    difficulty_profile: Optional[List[str]] = field(default_factory=lambda: [
        "recall", "application", "analysis",
    ])
    include_context_in_metadata: bool = True
    include_rephrased_in_metadata: bool = True


# ============================================================
# ChatThread-backed answer runner
# ============================================================

class SimpleChatThread:
    """
    Persistent ChatThread wrapper that seeds hidden context once and
    appends user questions to get assistant answers via InferenceOrchestrator.

    NOTE:
    - We embed `rephrased_text` directly into the system prompt (string).
    - We DO NOT concatenate history into the user prompt; ChatThread manages history.
    """
    def __init__(self, system_hidden_context: str, llm_model: str = "gpt-4.1"):
        self.system_hidden_context = system_hidden_context
        self.llm_model = llm_model

        # Build system prompt with hidden context embedded once
        sys_content = (
            "You are a curriculum development and educational content creation expert. "
            "You answer structured exam questions with complete solutions.\n"
            "Answer directly and completely in same language only.\n"
            f"Context (hidden):\n{self.system_hidden_context}"
            f"\n\n#Guidelines:"
            f"\n- Use gramatically correct language and provide factually correct information."
            f"\n- Do NOT reference the existence of a textbook context or source material — write naturally as if answering an exam."
            f"\n- For theoretical concepts, provide **descriptive and insightful explanations**."
            f"\n- For problem sets, provide a **full worked-out solution**, showing steps and reasoning clearly."
            f"\n- Do NOT include texbook details such as chapter number, title, author, subject, or grade."
            f"\n- Answer the question directly."
        )
        system_prompt = SystemPrompt(name="qa-thread", content=sys_content)

        # LLM configuration for plain text answers
        self.llm_config = LLMConfig(
            client=LLMClient.openai,
            model=self.llm_model,
            temperature=0.4,
            response_format=ResponseFormat.text,
            #reasoning_effort="minimal",
            max_tokens=4096
        )

        # Create the underlying ChatThread (no tools, no initial user msg)
        self.chat_thread = ChatThread(
            name="qa-thread",
            system_prompt=system_prompt,
            llm_config=self.llm_config,
            tools=[],
            new_message=None,
        )

        # Orchestrator to run completions
        self.orchestrator = InferenceOrchestrator()

    # Properly append history to the ChatThread history, not to the user prompt string
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
        # 1) Provide the new user message (ChatThread will include history automatically)
        self.chat_thread.new_message = user_text

        # 2) Run inference
        outputs = await self.orchestrator.run_parallel_ai_completion([self.chat_thread])
        if not outputs:
            return ""

        last: ProcessedOutput = outputs[-1]
        answer = (last.content or "").strip()

        # 3) Commit this turn to ChatThread history using its API
        self.chat_thread.new_message = user_text
        _ = self.chat_thread.add_user_message()
        await self.chat_thread.add_chat_turn_history(last)

        return answer


# Thin wrapper to keep compatibility with earlier pipeline design
class ConversationRunner:
    def __init__(self, chatthread: Any):
        self.thread = chatthread

    async def ask(self, question: str) -> str:
        return await self.thread.ask(question)


# ============================================================
# Core pipeline
# ============================================================

class MultiTurnQAPipeline:
    def __init__(
        self,
        cfg: PipelineConfig,
        # Model/inference callables
        rephraser_fn: Callable[[Dict[str, Any]], Any],
        seed_question_fn: Callable[[Dict[str, Any], str], Any],
        followup_question_fn: Callable[[List[str], List[str], str, str], Any],
        new_chatthread_fn: Callable[[str], Any],
        # Persistence callables
        append_results_jsonl_fn: Callable[[Dict[str, Any]], None],
        append_sharegpt_jsonl_fn: Callable[[Dict[str, Any]], None],
    ) -> None:
        self.cfg = cfg
        self.rephraser_fn = rephraser_fn
        self.seed_question_fn = seed_question_fn
        self.followup_question_fn = followup_question_fn
        self.new_chatthread_fn = new_chatthread_fn
        self.append_results_jsonl = append_results_jsonl_fn
        self.append_sharegpt_jsonl = append_sharegpt_jsonl_fn

    async def process_chunk(self, chunk: Dict[str, Any]) -> Dict[str, Any]:
        """
        Generate one ShareGPT conversation for a textbook chunk.

        Expects fields like: id, text, subject, grade, chapter_title, source.
        If an existing single-turn is provided (chunk['conversations'] or chunk['seed_conversation']),
        we will re-use it for part (a).
        """
        t0 = time.time()

        # 1) Get rephrased text (hidden context)
        rephrased_text = await _maybe_await(self.rephraser_fn(chunk))

        # 2) Either reuse existing Q(a)/A(a) or generate from scratch
        seed_conv = chunk.get("seed_conversation") or chunk.get("conversations")
        questions: List[str] = []
        answers: List[str] = []
        generated_pairs = 0  # counts only pairs generated in this run

        # Build single persistent chat thread seeded with hidden context
        runner = ConversationRunner(self.new_chatthread_fn(rephrased_text))

        if isinstance(seed_conv, list) and len(seed_conv) >= 2:
            # Trust provided single-turn seed from HF 'conversations': [human, gpt]
            Qa = seed_conv[0]["value"]
            Aa = seed_conv[1]["value"]
            questions.append(Qa)
            answers.append(Aa)
            # Prime the thread with the seed pair so part (b) sees part (a)
            if hasattr(runner.thread, "append_user") and hasattr(runner.thread, "append_assistant"):
                runner.thread.append_user(Qa)
                runner.thread.append_assistant(Aa)
        else:
            # Generate Q(a) then A(a)
            Qa = await _maybe_await(self.seed_question_fn(chunk, rephrased_text))
            Aa = await runner.ask(Qa)
            questions.append(Qa)
            answers.append(Aa)
            generated_pairs += 1  # seed generated in this run counts toward num_turns

        # 3) Follow-ups within the same thread (multiturn only)
        if self.cfg.mode == "multiturn":
            # We must generate exactly `num_turns` user turns in this run.
            # If the dataset provided a seed Q/A, generated_pairs starts at 0.
            # If we generated the seed, generated_pairs == 1 already.
            while generated_pairs < max(0, int(self.cfg.num_turns)):
                # Compute labels based on how many questions exist:
                # len(questions) == 0 -> 'a', 1 -> 'b', etc.
                label = _label_for(len(questions))
                target_level = _target_level(self.cfg.difficulty_profile, len(questions))
                Qi = await _maybe_await(
                    self.followup_question_fn(questions, answers, label, target_level)
                )
                Ai = await runner.ask(Qi)
                questions.append(Qi)
                answers.append(Ai)
                generated_pairs += 1

        # 4) Build outputs (judge skipped)
        sharegpt_dialog: List[Dict[str, str]] = []
        for i in range(len(answers)):
            sharegpt_dialog.append({"from": "human", "value": questions[i]})
            sharegpt_dialog.append({"from": "gpt", "value": answers[i]})

        # Results JSONL record
        result_rec: Dict[str, Any] = {
            "chunk_id": chunk.get("id") or str(uuid.uuid4()),
            "qa_conversation": {
                "questions": questions,
                "answers": answers,
                "metrics": None,
                "rephrased_text": rephrased_text,
            },
            "subject": chunk.get("subject"),
            "grade": chunk.get("grade"),
            "chapter_title": chunk.get("chapter_title"),
            "source": chunk.get("source"),
            "execution_time": round(time.time() - t0, 6),
            "timestamp": time.strftime("%Y-%m-%dT%H:%M:%S"),
        }
        self.append_results_jsonl(result_rec)

        # ShareGPT JSONL record
        md: Dict[str, Any] = {
            "subject": chunk.get("subject"),
            "grade": chunk.get("grade"),
            "chapter_title": chunk.get("chapter_title"),
            "source": chunk.get("source"),
        }
        if self.cfg.include_context_in_metadata:
            md["context_text"] = chunk.get("text")
        if self.cfg.include_rephrased_in_metadata:
            md["rephrased_text"] = rephrased_text

        sharegpt_rec: Dict[str, Any] = {
            "id": result_rec["chunk_id"],
            "conversations": sharegpt_dialog,
            "metadata": md,
        }
        self.append_sharegpt_jsonl(sharegpt_rec)

        return {
            "id": result_rec["chunk_id"],
            "turns": len(answers),
            "subject": md.get("subject"),
            "grade": md.get("grade"),
        }


# ============================================================
# Helpers
# ============================================================

async def _maybe_await(x):
    if callable(getattr(x, "__await__", None)):
        return await x
    return x


def _label_for(turn_idx: int) -> str:
    # turn_idx = 1 -> 'b', 2 -> 'c', etc.
    alphabet = "abcdefghijklmnopqrstuvwxyz"
    base = 1  # part (a) is index 0, so 1 maps to 'b'
    pos = min(len(alphabet) - 1, turn_idx + base)
    return alphabet[pos]


def _target_level(profile: Optional[List[str]], turn_idx: int) -> str:
    if not profile:
        return "application"
    if turn_idx < len(profile):
        return profile[turn_idx]
    return profile[-1]
