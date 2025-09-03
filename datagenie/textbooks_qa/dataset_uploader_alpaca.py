#!/usr/bin/env python3
"""
Upload ShareGPT conversations and execution metrics JSONL files to the Hugging Face Hub
as a proper dataset repo with one split: `sharegpt`.

Usage:
  python dataset_uploader.py \
    --repo_id your-username/marketagents-textbook-qa \
    --sharegpt_jsonl outputs/textbook_qa/textbook_qa_sharegpt_20250819_095729.jsonl \
    --private false \
    --push_readme true

Auth:
  - Set an access token via env var HF_TOKEN (recommended), or pass --hf_token.
  - You can create a token at https://huggingface.co/settings/tokens

Implementation notes:
  - We validate ShareGPT lines to avoid saving broken conversations.
  - We rely on `datasets.DatasetDict.push_to_hub()` for a clean dataset repo layout.

References:
  - Upload datasets with `push_to_hub`: https://huggingface.co/docs/datasets/en/upload_dataset
  - Upload files/folders with `huggingface_hub`: https://huggingface.co/docs/huggingface_hub/en/guides/upload
  - Repo structure and splits: https://huggingface.co/docs/datasets/en/repository_structure
"""

import argparse
import hashlib
import json
import os
import re
import sys
import tempfile
import uuid
from pathlib import Path
from typing import Any, Dict, Iterable, Tuple

from datasets import Dataset, DatasetDict, load_dataset  # type: ignore
from huggingface_hub import HfApi, create_repo  # type: ignore


# --- Helpers to unwrap JSON-like question blocks and code fences ---
_CODE_FENCE_RE = re.compile(r"^\s*```[a-zA-Z0-9_-]*\s*\n|\n\s*```\s*$")
_JSON_QUESTION_RE = re.compile(r"\{[^}]*?\"question\"\s*:\s*\"(.*?)\"[^}]*?\}", re.DOTALL)

_QUOTED_QUESTION_PREFIX = re.compile(r"^\s*\"?question\"?\s*[:：]\s*", re.IGNORECASE)

# --- Helper to remove leftover quotes/escapes after prefix stripping ---
def _strip_surrounding_quotes_and_escapes(text: str) -> str:
    """Remove stray leading/trailing quotes and common escape artifacts after prefix stripping."""
    if not isinstance(text, str):
        return text
    s = text.strip()
    # Unescape common escaped quotes
    s = s.replace('\\"', '"').replace("\\'", "'")
    # Drop dangling backslashes at ends
    s = re.sub(r"^\\+", "", s)
    s = re.sub(r"\\+$", "", s)
    # Remove matching surrounding quotes
    if len(s) >= 2 and ((s[0] == '"' and s[-1] == '"') or (s[0] == "'" and s[-1] == "'")):
        s = s[1:-1].strip()
    else:
        # Remove lone leading/trailing quote if present
        s = s[1:].strip() if s.startswith(('"', "'")) else s
        s = s[:-1].strip() if s.endswith(('"', "'")) else s
    return s

def _strip_code_fences(text: str) -> str:
    if not isinstance(text, str):
        return text
    s = text
    # Remove leading/trailing triple backtick fences if present
    s = re.sub(r"^\s*```[a-zA-Z0-9_-]*\s*\n", "", s)
    s = re.sub(r"\n\s*```\s*$", "", s)
    return s.strip()

def _extract_json_question(text: str) -> str | None:
    """If text contains a JSON object with a "question" field, return its value; else None."""
    if not isinstance(text, str):
        return None
    # Direct JSON parse attempt if it looks like an object
    s = text.strip()
    if s.startswith("{") and s.endswith("}"):
        try:
            obj = json.loads(s)
            if isinstance(obj, dict) and "question" in obj and isinstance(obj["question"], str):
                return obj["question"].strip()
        except Exception:
            pass
    # Regex fallback to catch embedded objects inside prose or code blocks
    m = _JSON_QUESTION_RE.search(s)
    if m:
        return m.group(1).strip()
    return None


# --- Helper to ensure <think> opening tag if </think> is present but missing opening ---
def _ensure_opening_think(text: str) -> str:
    """If text contains a closing </think> but no opening <think>, prepend one."""
    if not isinstance(text, str) or not text:
        return text
    if "</think>" in text and "<think>" not in text:
        return "<think>\n" + text
    return text

def _strip_question_numbering(text: str) -> str:
    if not isinstance(text, str):
        return text
    s = text.lstrip()
    # Iteratively remove up to two leading markers to be safe
    for _ in range(2):
        m1 = re.match(rf"^\s*Q\s*\d+\s*(?:\({ _MARKER_INNER }\))?\s*[:\).\-]*\s*", s)
        if m1:
            s = s[m1.end():]
            continue
        m2 = re.match(rf"^\s*(?:\(\s*{ _MARKER_INNER }\s*\)|{ _MARKER_INNER }[\)\.:])\s*", s)
        # Guards: avoid stripping mixed fractions like "(1 2/3)" and decimals like "5.00 kg" or "५.०० kg"
        if m2:
            inner = m2.group(0)
            # Decimal guard: if the matched prefix ends with a '.' and the next char is a digit, it's a decimal, not a list marker
            try:
                if inner.rstrip().endswith('.') and m2.end() < len(s) and s[m2.end()].isdigit():
                    break
            except Exception:
                pass
            # Mixed-fraction guard
            if "/" in inner or re.search(r"\d+\s+\d+\/\d+", inner):
                break
            s = s[m2.end():]
            continue
        break
    return s.strip()


def _coerce_int(val):
    try:
        if val is None:
            return None
        if isinstance(val, bool):
            return None
        if isinstance(val, int):
            return val
        s = str(val).strip().replace(",", "")
        if s.lstrip("-").isdigit():
            return int(s)
    except Exception:
        pass
    return None

def _coerce_float(val):
    try:
        if val is None:
            return None
        if isinstance(val, (int, float)) and not isinstance(val, bool):
            return float(val)
        s = str(val).strip().replace(",", "")
        return float(s)
    except Exception:
        return None

def _extract_answer_tail(text: str) -> str:
    """
    Returns the portion of the text after the last </think> closing tag.
    If no </think> exists, returns the entire text.
    """
    if not isinstance(text, str):
        return ""
    try:
        tail_match = re.search(r"</think>\s*(.*)\Z", text, re.DOTALL)
        return (tail_match.group(1) if tail_match else text).strip()
    except Exception:
        return text.strip()

# --- REPETITION / BORKED GENERATION DETECTOR ---

def _max_ngram_repetition(tokens: list[str], n: int = 5) -> int:
    """Return maximum frequency of any n-gram in the token sequence."""
    try:
        if not tokens or len(tokens) < n:
            return 1
        counts = {}
        for i in range(len(tokens) - n + 1):
            key = tuple(tokens[i:i+n])
            counts[key] = counts.get(key, 0) + 1
        return max(counts.values()) if counts else 1
    except Exception:
        return 1

def _has_borked_repetition(text: str) -> bool:
    """
    Heuristics to catch looped or degenerate generations:
      - Extremely repeated n-grams (5-gram freq >= 6)
      - Same line duplicated >= 5 times
      - Character-level long repeated fragments
    """
    if not isinstance(text, str):
        return False
    s = text.strip()
    if not s:
        return False
    # Ignore inside <think>; analyze only the visible answer tail
    tail = _extract_answer_tail(s)

    # Token n-gram repetition
    tokens = re.findall(r"\S+", tail)
    if _max_ngram_repetition(tokens, n=5) >= 6:
        return True

    # Line repetition
    lines = [ln.strip() for ln in tail.splitlines() if ln.strip()]
    if lines:
        line_counts = {}
        for ln in lines:
            line_counts[ln] = line_counts.get(ln, 0) + 1
        if max(line_counts.values()) >= 5:
            return True

    # Character fragment repetition (look for a 12+ char fragment repeated 6+ times)
    # Cheap heuristic: sliding window substrings every 8 chars
    tail_compact = re.sub(r"\s+", " ", tail)
    L = len(tail_compact)
    if L >= 120:
        seen = {}
        for i in range(0, L - 12, 8):
            frag = tail_compact[i:i+24]
            if len(frag.strip()) < 12:
                continue
            seen[frag] = seen.get(frag, 0) + 1
        if seen and max(seen.values()) >= 6:
            return True

    return False

# --- Additional repetition/degeneration detectors ---
def _has_mono_token_loop(text: str) -> bool:
    """Detects degenerate outputs dominated by one token repeated.
    Triggers if the most frequent token covers ≥ 60% of tokens and occurs ≥ 30 times,
    or if any single token occurs ≥ 80 times.
    Only analyzes the answer tail after </think>.
    """
    if not isinstance(text, str):
        return False
    tail = _extract_answer_tail(text or "")
    tokens = re.findall(r"\S+", tail)
    if len(tokens) < 10:
        return False
    counts = {}
    for t in tokens:
        counts[t] = counts.get(t, 0) + 1
    if not counts:
        return False
    max_tok = max(counts.values())
    if max_tok >= 80:
        return True
    if max_tok >= 30 and (max_tok / max(1, len(tokens))) >= 0.60:
        return True
    # Low diversity heuristic: unique/share ratio very small on long tails
    if len(tokens) >= 100 and (len(counts) / len(tokens)) <= 0.15:
        return True
    return False

def _has_long_trailing_char_run(text: str) -> bool:
    """Detects a single character repeated at the end for a long run (≥ 40)."""
    if not isinstance(text, str):
        return False
    tail = _extract_answer_tail(text or "")
    s = tail.rstrip()
    if not s:
        return False
    last = s[-1]
    # Count how many times `last` repeats consecutively at the end
    i = len(s) - 1
    run = 0
    while i >= 0 and s[i] == last:
        run += 1
        i -= 1
    return run >= 40

# --- Intra-conversation deduplication helpers ---
def _norm_ws(s: str) -> str:
    if not isinstance(s, str):
        s = "" if s is None else str(s)
    return re.sub(r"\s+", " ", s.strip())

def _dedupe_conversation_pairs(convs: list[dict]) -> tuple[list[dict], int]:
    """Remove repeated human→gpt pairs and duplicate single turns.
    - Keeps the first occurrence of an identical (human_text, gpt_text) pair; drops later duplicates.
    - Also drops duplicate single turns if they appear unpaired.
    Returns: (deduped_convs, num_pairs_dropped)
    """
    out: list[dict] = []
    seen_pairs: set[tuple[str, str]] = set()
    seen_solos: set[tuple[str, str]] = set()
    dropped_pairs = 0
    i = 0
    L = len(convs)
    while i < L:
        cur = convs[i]
        frm = cur.get("from")
        val = _norm_ws(cur.get("value", ""))
        # If this is a human followed by a gpt, treat as a pair
        if frm == "human" and (i + 1) < L and isinstance(convs[i+1], dict) and convs[i+1].get("from") == "gpt":
            nxt = convs[i+1]
            gval = _norm_ws(nxt.get("value", ""))
            key = (val, gval)
            if key in seen_pairs:
                dropped_pairs += 1
                i += 2
                continue
            seen_pairs.add(key)
            out.append({"from": "human", "value": val})
            out.append({"from": "gpt", "value": gval})
            i += 2
            continue
        # Otherwise handle as a solo turn
        solo_key = (frm or "", val)
        if solo_key in seen_solos:
            i += 1
            continue
        seen_solos.add(solo_key)
        out.append({"from": frm or "human", "value": val})
        i += 1
    return out, dropped_pairs


def _is_valid_sharegpt(obj: Dict[str, Any]) -> bool:
    """
    Minimal validity for general ShareGPT-style conversations:
    - conversations is a list with at least 2 turns
    - first human and first gpt turns are non-empty strings
    """
    try:
        if not isinstance(obj, dict):
            return False
        conv = obj.get("conversations") or []
        if not isinstance(conv, list) or len(conv) < 2:
            return False
        q = (conv[0] or {}).get("value")
        a = (conv[1] or {}).get("value")
        if not (isinstance(q, str) and q.strip()):
            return False
        if not (isinstance(a, str) and a.strip()):
            return False

        return True
    except Exception:
        return False


def _filter_jsonl(in_path: Path) -> Tuple[Path, int, int]:
    """
    Filter a ShareGPT JSONL file to only valid items, write to a temp file.
    Returns: (temp_path, kept, dropped)
    """
    kept = 0
    dropped = 0
    tmp = Path(tempfile.mkstemp(prefix="sharegpt_filtered_", suffix=".jsonl")[1])
    with open(in_path, "r", encoding="utf-8") as fin, open(tmp, "w", encoding="utf-8") as fout:
        for line in fin:
            line = line.strip()
            if not line:
                continue
            try:
                obj = json.loads(line)
            except Exception:
                dropped += 1
                continue
            try:
                conv = obj.get("conversations")
                if isinstance(conv, list) and len(conv) >= 2:
                    cleaned = []
                    for idx, item in enumerate(conv):
                        if not isinstance(item, dict):
                            continue
                        frm = item.get("from")
                        val = item.get("value")
                        # Normalize role
                        if frm not in ("human", "gpt"):
                            frm = "human" if (idx % 2 == 0) else "gpt"

                        # Strip numbering/prefixes for human turns only
                        if frm == "human" and isinstance(txt, str) and txt:
                            # Remove code fences first (common when users paste JSON)
                            txt = _strip_code_fences(txt)
                            # If a JSON object with a "question" field is present, unwrap it
                            qv = _extract_json_question(txt)
                            if isinstance(qv, str) and qv:
                                txt = qv
                            txt = _strip_question_numbering(txt)
                            # Also handle a bare leading '"question":' without braces
                            txt = _QUOTED_QUESTION_PREFIX.sub("", txt).strip()
                            # Finally, remove leftover quotes/escapes caused by prefix removal
                            txt = _strip_surrounding_quotes_and_escapes(txt)
                            cleaned.append({"from": "human", "value": txt})
                            continue

                        # GPT answers: ensure <think> presence if </think> exists; drop pair if invalid
                        if frm == "gpt" and isinstance(txt, str) and txt:
                            txt = _ensure_opening_think(txt)
                            # 2) Borked repetition => drop pair
                            if _has_borked_repetition(txt):
                                if cleaned and cleaned[-1].get("from") == "human":
                                    cleaned.pop()
                                print("[uploader] dropping pair: borked repetition detected", file=sys.stderr)
                                continue
                            # 3) Mono-token loop => drop pair
                            if _has_mono_token_loop(txt):
                                if cleaned and cleaned[-1].get("from") == "human":
                                    cleaned.pop()
                                print("[uploader] dropping pair: mono-token loop detected", file=sys.stderr)
                                continue
                            # 4) Long trailing char run => drop pair
                            if _has_long_trailing_char_run(txt):
                                if cleaned and cleaned[-1].get("from") == "human":
                                    cleaned.pop()
                                print("[uploader] dropping pair: trailing char run detected", file=sys.stderr)
                                continue
                            cleaned.append({"from": "gpt", "value": txt})
                            continue

                        # Default append for any other cases
                        cleaned.append({"from": frm, "value": txt})
                    # Deduplicate repeated QA pairs within the conversation
                    cleaned, dropped_pairs = _dedupe_conversation_pairs(cleaned)
                    if dropped_pairs:
                        print(f"[uploader] dropped {dropped_pairs} duplicate QA pair(s) in-conversation", file=sys.stderr)
                    # Invalidate if only the seed pair (<=2 turns) remains after filtering
                    if len(cleaned) < 2:
                        obj = {"id": str(uuid.uuid4()), "conversations": []}
                    else:
                        obj = {"id": str(uuid.uuid4()), "conversations": cleaned}
                else:
                    obj = {"id": str(uuid.uuid4()), "conversations": []}
            except Exception:
                # Fall back to raw object if something exploded; validation will drop it
                pass
            if _is_valid_sharegpt(obj):
                fout.write(json.dumps(obj, ensure_ascii=False) + "\n")
                kept += 1
            else:
                try:
                    conv = obj.get("conversations") or []
                    qv = (conv[0] or {}).get("value") if len(conv) > 0 else ""
                    av = (conv[1] or {}).get("value") if len(conv) > 1 else ""
                    reasons = []
                    if not isinstance(qv, str) or not qv.strip():
                        reasons.append("empty_question")
                    if not isinstance(av, str) or not av.strip():
                        reasons.append("empty_answer")
                    if isinstance(conv, list) and len(conv) <= 2:
                        reasons.append("too_few_turns_after_filtering")
                    q_preview = (qv[:60] + "…") if isinstance(qv, str) and len(qv) > 60 else qv
                    a_preview = (av[:60] + "…") if isinstance(av, str) and len(av) > 60 else av
                    print(f"[uploader] dropping invalid item: reasons={','.join(reasons) or 'unknown'} | Q='{q_preview}' | A='{a_preview}'", file=sys.stderr)
                except Exception:
                    pass
                dropped += 1
    return tmp, kept, dropped


def _normalize_ws(s: str) -> str:
    if not isinstance(s, str):
        s = "" if s is None else str(s)
    return re.sub(r"\s+", " ", s.strip())

def _dedupe_key(obj: Dict[str, Any], mode: str = "auto") -> str:
    """
    Compute a stable hash key for deduplication.
    modes:
      - 'context_text': hash of normalized context_text
      - 'rephrased_text': hash of normalized rephrased_text
      - 'qa': hash of (human value + gpt value)
      - 'auto': prefer rephrased_text else qa else context_text
    """
    try:
        if mode == "context_text":
            basis = _normalize_ws(obj.get("context_text", ""))
        elif mode == "rephrased_text":
            basis = _normalize_ws(obj.get("rephrased_text", ""))
        elif mode == "qa":
            conv = obj.get("conversations") or []
            if isinstance(conv, list) and len(conv) > 0:
                vals = []
                for m in conv:
                    if isinstance(m, dict):
                        mv = _normalize_ws(m.get("value", ""))
                        if mv:
                            vals.append(mv)
                basis = " || ".join(vals)
            else:
                basis = ""
        else:
            # auto
            rt = _normalize_ws(obj.get("rephrased_text", ""))
            if rt:
                basis = rt
            else:
                conv = obj.get("conversations") or []
                q = (conv[0] or {}).get("value") if len(conv) > 0 else ""
                a = (conv[1] or {}).get("value") if len(conv) > 1 else ""
                qa = _normalize_ws(q) + " || " + _normalize_ws(a)
                if qa.strip(" |"):
                    basis = qa
                else:
                    basis = _normalize_ws(obj.get("context_text", ""))
        return hashlib.sha256(basis.encode("utf-8")).hexdigest()
    except Exception:
        # fall back to random UUID to avoid accidental collision (will be treated as new)
        return str(uuid.uuid4())

def _iter_jsonl(path: Path):
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                yield json.loads(line)
            except Exception:
                continue

def _merge_with_hub(repo_id: str, split: str, new_path: Path, key_mode: str, token: str | None) -> Path:
    """
    Pull existing dataset from Hub (split), merge with new JSONL, dedupe by key,
    and write to a new temp JSONL. Returns merged temp path.
    """
    # Build set of existing keys
    seen = set()
    existing = []
    try:
        # Use streaming to avoid loading everything into memory when possible
        ds_stream = load_dataset(repo_id, split=split, streaming=True, token=token)
        for ex in ds_stream:
            key = _dedupe_key(ex, key_mode)
            if key not in seen:
                seen.add(key)
                existing.append(ex)
    except Exception as e:
        print(f"[uploader] Warning: failed to load existing hub split '{split}' from {repo_id}: {e}")
        existing = []

    # Now iterate new_path
    new_unique = []
    for obj in _iter_jsonl(new_path):
        key = _dedupe_key(obj, key_mode)
        if key not in seen:
            seen.add(key)
            new_unique.append(obj)

    merged_path = Path(tempfile.mkstemp(prefix="sharegpt_merged_", suffix=".jsonl")[1])
    with open(merged_path, "w", encoding="utf-8") as out:
        for ex in existing:
            out.write(json.dumps(ex, ensure_ascii=False) + "\n")
        for ex in new_unique:
            out.write(json.dumps(ex, ensure_ascii=False) + "\n")

    print(f"[uploader] Merge summary: kept_existing={len(existing)}, added_new={len(new_unique)}, total={len(existing)+len(new_unique)}")
    return merged_path


def _ensure_repo(repo_id: str, token: str, private: bool) -> None:
    """
    Create the dataset repo if it does not exist.
    """
    api = HfApi()
    try:
        api.repo_info(repo_id, repo_type="dataset", token=token)
    except Exception:
        # create if missing
        create_repo(
            repo_id=repo_id,
            token=token,
            repo_type="dataset",
            private=private,
            exist_ok=True,
        )


def _build_readme(repo_id: str) -> str:
    """
    Create a minimal dataset card with useful metadata.
    """
    return f"""---
language:
- ne
pretty_name: "ShareGPT Conversations"
task_categories:
- text-generation
license: apache-2.0
tags:
- sharegpt
---

# ShareGPT Conversations

This repository contains multi-turn **human ↔ gpt** conversations.

## Splits

- `{repo_id}` provides a split named `train` by default.

## Usage

```python
from datasets import load_dataset

ds = load_dataset("{repo_id}")
train = ds["train"]
```

## Schema

Each row contains:
- `id`: unique string
- `conversations`: list of N messages (N ≥ 2), alternating `human` and `gpt` roles

Notes:
- Conversations are lightly normalized (e.g., stripping leading numbering and prefixes like "Question:").

"""


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--repo_id", required=True, help="e.g., your-username/marketagents-textbook-qa")
    p.add_argument("--sharegpt_jsonl", required=True, help="Path to validated ShareGPT JSONL")
    p.add_argument("--hf_token", default=None, help=argparse.SUPPRESS)
    p.add_argument("--private", default="false", choices=["true", "false"], help="Create repo as private")
    p.add_argument("--push_readme", default="true", choices=["true", "false"], help="Upload README.md dataset card")
    p.add_argument("--append", default="false", choices=["true", "false"], help="Append to existing Hub split by merging and deduping")
    p.add_argument("--dedupe_on", default="auto", choices=["auto", "rephrased_text", "qa", "context_text"], help="Field used to generate dedupe key")
    p.add_argument("--split", default="train", help="HF dataset split name to push to (default: train)")
    args = p.parse_args()

    # Allow token to be passed via argument, environment variable, or left as None for library default
    token = args.hf_token or os.getenv("HF_TOKEN")

    repo_id = args.repo_id
    sharegpt_path = Path(args.sharegpt_jsonl).expanduser().resolve()

    if not sharegpt_path.exists():
        print(f"ERROR: ShareGPT JSONL not found: {sharegpt_path}", file=sys.stderr)
        sys.exit(2)

    private = args.private.lower() == "true"
    push_readme = args.push_readme.lower() == "true"
    append_mode = args.append.lower() == "true"
    dedupe_on = args.dedupe_on
    split_name = args.split

    # Ensure repo exists (dataset type)
    _ensure_repo(repo_id, token, private=private)

    # Filter ShareGPT JSONL to valid rows only
    filtered_path, kept, dropped = _filter_jsonl(sharegpt_path)
    print(f"[uploader] ShareGPT validation: kept={kept}, dropped={dropped}, tmp={filtered_path.name}")

    # Optionally merge with existing Hub split and dedupe
    final_path = filtered_path
    if append_mode:
        print(f"[uploader] Append mode enabled. Merging with existing split '{split_name}' and deduping on '{dedupe_on}'")
        final_path = _merge_with_hub(repo_id, split_name, filtered_path, dedupe_on, token)

    data_files = { split_name: str(final_path) }
    dset = load_dataset("json", data_files=data_files)

    # Push to hub
    # Note: push_to_hub accepts `token` and will create/overwrite splits.
    print(f"[uploader] Pushing DatasetDict to hub: {repo_id}")
    dset.push_to_hub(repo_id, token=token, private=private)

    # Optionally upload a README.md dataset card
    if push_readme:
        readme = _build_readme(repo_id)
        readme_path = Path(tempfile.mkstemp(prefix="README_", suffix=".md")[1])
        readme_path.write_text(readme, encoding="utf-8")

        # Use huggingface_hub to upload README.md at repo root
        api = HfApi()
        api.upload_file(
            path_or_fileobj=str(readme_path),
            path_in_repo="README.md",
            repo_id=repo_id,
            repo_type="dataset",
            token=token,
            commit_message="Add dataset card",
        )
        print("[uploader] README.md uploaded")

    print("[uploader] Done.")


if __name__ == "__main__":
    main()