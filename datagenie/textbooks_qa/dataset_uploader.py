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


def _is_valid_sharegpt(obj: Dict[str, Any]) -> bool:
    """
    Minimal validity: first human question, first gpt answer, and `rephrased_text` must be non-empty.
    Judge metrics may be missing. Additionally, reject items where the first human turn, first gpt turn,
    """
    try:
        if not isinstance(obj, dict):
            return False
        conv = obj.get("conversations") or []
        if not isinstance(conv, list) or len(conv) < 2:
            return False
        q = (conv[0] or {}).get("value")
        a = (conv[1] or {}).get("value")
        r = obj.get("rephrased_text")
        if not (isinstance(q, str) and q.strip()):
            return False
        if not (isinstance(a, str) and a.strip()):
            return False
        if not (isinstance(r, str) and r.strip()):
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
            # Normalize/flatten schema while preserving new top-level fields.
            # If `metadata` exists (old schema), use it to override; else keep current top-level values.
            try:
                # Capture current top-level values first (new schema)
                cur_subject = obj.get("subject")
                cur_grade = _coerce_int(obj.get("grade"))
                cur_chapter = obj.get("chapter_title")
                cur_source = obj.get("source")
                cur_context = obj.get("context_text")
                cur_rephrased = obj.get("rephrased_text")
                cur_ljm = obj.get("llm_judge_metrics")
                cur_avg = _coerce_float(obj.get("average_score"))
                cur_ds_seed = obj.get("dataset_seed_present")  # pass-through if present

                md = obj.get("metadata") or None
                if isinstance(md, dict):
                    # Remove deprecated 'feedback'
                    md.pop("feedback", None)
                    # Normalize llm_judge_metrics from metadata
                    ljm = md.get("llm_judge_metrics")
                    if isinstance(ljm, dict):
                        ljm.pop("feedback", None)
                        ljm = {k: _coerce_float(v) for k, v in ljm.items() if k is not None}
                        if len(ljm) == 0:
                            ljm = None
                    else:
                        ljm = None if cur_ljm is None else cur_ljm

                    # Prefer metadata values when present; otherwise keep current top-level
                    subject = md.get("subject", cur_subject)
                    grade = _coerce_int(md.get("grade")) if md.get("grade") is not None else cur_grade
                    chapter = md.get("chapter_title", cur_chapter)
                    source = md.get("source", cur_source)
                    context_text = md.get("context_text", cur_context)
                    if isinstance(md.get("rephrased_text"), str):
                        rephrased_text = cur_rephrased if cur_rephrased is None else str(cur_rephrased)

                    average_score = _coerce_float(md.get("average_score")) if md.get("average_score") is not None else cur_avg

                    # Drop noisy keys
                    md.pop("timestamp", None)
                    md.pop("execution_time", None)
                else:
                    # No metadata: keep current top-level values, but trim rephrased if it's a string
                    ljm = None
                    if isinstance(cur_ljm, dict):
                        tmp = {k: _coerce_float(v) for k, v in cur_ljm.items() if k is not None}
                        ljm = tmp if len(tmp) > 0 else None
                    subject = cur_subject if cur_subject is None else str(cur_subject)
                    grade = cur_grade
                    chapter = cur_chapter if cur_chapter is None else str(cur_chapter)
                    source = cur_source if cur_source is None else str(cur_source)
                    context_text = cur_context if cur_context is None else str(cur_context)
                    rephrased_text = cur_rephrased if isinstance(cur_rephrased, str) else (cur_rephrased if cur_rephrased is None else str(cur_rephrased))
                    average_score = cur_avg

                # Write back to top-level
                obj["subject"] = subject
                obj["grade"] = grade
                obj["chapter_title"] = chapter
                obj["source"] = source
                obj["context_text"] = context_text
                obj["rephrased_text"] = rephrased_text
                obj["llm_judge_metrics"] = ljm
                obj["average_score"] = average_score
                if cur_ds_seed is not None:
                    obj["dataset_seed_present"] = bool(cur_ds_seed)

                # Remove metadata container to avoid nested variance
                if "metadata" in obj:
                    del obj["metadata"]
            except Exception:
                pass
            try:
                conv = obj.get("conversations")
                if isinstance(conv, list) and len(conv) >= 2:
                    for idx in range(len(conv)):
                        item = conv[idx]
                        if isinstance(item, dict):
                            # Normalize 'from' to known labels; when unknown, infer by index parity
                            frm = item.get("from")
                            if frm not in ("human", "gpt"):
                                item["from"] = "human" if (idx % 2 == 0) else "gpt"
                            # Normalize 'value'
                            val = item.get("value")
                            if isinstance(val, str):
                                norm = str(val)
                            item["value"] = norm
                    # Preserve ALL turns; do not truncate to two messages
                    obj["conversations"] = conv
            except Exception:
                pass
            # Normalize dataset_seed_present if present
            try:
                if "dataset_seed_present" in obj:
                    obj["dataset_seed_present"] = bool(obj["dataset_seed_present"])
            except Exception:
                pass
            if _is_valid_sharegpt(obj):
                obj["id"] = str(uuid.uuid4())
                fout.write(json.dumps(obj, ensure_ascii=False) + "\n")
                kept += 1
            else:
                # Log invalid samples with reason if possible
                try:
                    conv = obj.get("conversations") or []
                    qv = (conv[0] or {}).get("value") if len(conv) > 0 else ""
                    av = (conv[1] or {}).get("value") if len(conv) > 1 else ""
                    rv = obj.get("rephrased_text") or ""
                    reasons = []
                    if not isinstance(qv, str) or not qv.strip():
                        reasons.append("empty_question")
                    if not isinstance(av, str) or not av.strip():
                        reasons.append("empty_answer")
                    if not isinstance(rv, str) or not rv.strip():
                        reasons.append("empty_rephrased")
                    # Trim previews for logging
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
    # Tags and metadata can be tweaked as needed
    return f"""---
language:
- ne
- en
pretty_name: "Textbook QA"
task_categories:
- question-answering
- text-generation
license: apache-2.0
tags:
- sharegpt
- qa
- synthetic-data
- education
---

# Textbook Question-Answering Dataset

This repository contains **ShareGPT-style conversations** generated by the Textbook QA agentic pipeline.

## Splits

- `train`: validated conversations with non-empty question, answer, and rephrased_text.

## Usage

```python
from datasets import load_dataset

ds = load_dataset("{repo_id}")
train = ds["train"]
```

## Schema

- **train**: each row contains:
  - `id`: unique string
  - `conversations`: list of N messages (N ≥ 2) alternating `human` and `gpt`
  - `subject`
  - `grade`
  - `chapter_title`
  - `source`
  - `context_text`
  - `rephrased_text`
  - `llm_judge_metrics` (object with scores)
  - `average_score` (float)

## Notes

- Conversations are validated to include **question**, **answer**, and **rephrased_text**.
- Judge metrics may be missing by design.

### Appending and Deduplication
When `--append true` is used, the uploader pulls the existing split from the Hub, merges the new rows, and deduplicates using a stable hash. You can choose the key with `--dedupe_on`:
- `auto` (default): prefers `rephrased_text`, falls back to Q&A, then `context_text`
- `rephrased_text`, `qa`, or `context_text`
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