#!/usr/bin/env python3
"""
filter_boxed_from_hf.py

Fetch a ShareGPT-style dataset from HuggingFace, filter conversations that contain
final answers in LaTeX \boxed{...}, and export either:
  - single-turn flattened records (one record per assistant turn that has a boxed answer; we take the final \boxed{...} within that turn, but all boxed answers per turn are available), or
  - multi-turn records with a per-turn list containing at most one answer (the final \boxed{...} found in each assistant turn). No conversation-level “final” is computed. Each assistant turn may contain multiple boxed answers.

Usage examples:
  # Flatten every boxed answer to its own record
  python filter_boxed_from_hf.py \
    --repo_id your_username/textbook-qa-multiturn \
    --split train \
    --mode single \
    --out_jsonl outputs/filtered/boxed_single.jsonl

  # Keep multi-turn conversation, attach list `boxed_answers`
  python filter_boxed_from_hf.py \
    --repo_id your_username/textbook-qa-multiturn \
    --split train \
    --mode multi \
    --out_jsonl outputs/filtered/boxed_multi.jsonl

  # Push filtered records to a new HuggingFace repo
  python filter_boxed_from_hf.py \
    --repo_id your_username/textbook-qa-multiturn \
    --split train \
    --mode single \
    --out_jsonl outputs/filtered/boxed_single.jsonl \
    --push_repo_id your-username/filtered-textbook-qa \
    --push_split train \
    --push_private
"""

import re
import json
import argparse
import os
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

from datasets import load_dataset
from datasets import Dataset

# Labels that often precede the final answer
FINAL_LABEL_RE = re.compile(r"(?:अन्तिम\s*उत्तर|final\s*answer)[:：]?", re.IGNORECASE)
# Token to find \\boxed (also tolerates optional slashes and whitespace); include known typo "oxed/"
BOXED_TOKEN_RE = re.compile(r"(?:(?:\\|/){0,2}\s*boxed\b|oxed/)\s*", re.IGNORECASE)

# Robust patterns to catch variants like \boxed{}, /boxed{}, //boxed{}, plain boxed{}, and known typos like oxed/ { }
BOX_PATTERNS = [
    re.compile(r"(?:\\|/){1,2}\s*boxed\s*\{([^}]*)\}", re.IGNORECASE),  # \boxed{} or /boxed{} or //boxed{} (optionally spaced)
    re.compile(r"\\boxed\s*\{([^}]*)\}", re.IGNORECASE),                # canonical fallback
    re.compile(r"/boxed\s*\{([^}]*)\}", re.IGNORECASE),                 # single slash
    re.compile(r"\bboxed\s*\{([^}]*)\}", re.IGNORECASE),                # missing slash (word boundary)
    re.compile(r"oxed/\s*\{([^}]*)\}", re.IGNORECASE),                  # typo variant seen in data
]

TFRAC_RE = re.compile(r"\\(?:tfrac|frac)\s*\{([^}]*)\}\s*\{([^}]*)\}")

def _extract_boxed_balanced_with_spans(text: str) -> List[Tuple[int, str]]:
    """Return list of (start_index, payload) for each boxed occurrence, using balanced brace parsing.
    Handles nested braces and multi-line TeX, e.g., \boxed{\begin{aligned} ... \end{aligned}}.
    """
    spans: List[Tuple[int, str]] = []
    if not text:
        return spans
    s = text
    n = len(s)
    for m in BOXED_TOKEN_RE.finditer(s):
        i = m.end()
        # skip whitespace to find the opening '{'
        while i < n and s[i].isspace():
            i += 1
        if i >= n or s[i] != '{':
            continue
        start_payload = i + 1
        depth = 1
        j = start_payload
        while j < n:
            ch = s[j]
            if ch == '\\':  # skip escaped char
                j += 2
                continue
            if ch == '{':
                depth += 1
            elif ch == '}':
                depth -= 1
                if depth == 0:
                    payload = s[start_payload:j]
                    spans.append((m.start(), payload))
                    break
            j += 1
        # if we exit without depth==0, ignore this malformed occurrence
    return spans

def _canonicalize_payload(p: str) -> str:
    """Light canonicalization: normalize TeX fractions to a/b, keep everything else intact."""
    return _normalize_tex_fraction(p.strip())

def _prioritized_final(spans: List[Tuple[int, str]], text: str) -> Optional[str]:
    """Choose the final boxed payload, prioritizing those that appear *after* the last 'अन्तिम उत्तर' label.
    If none appear after the label, return the last payload in the text.
    """
    if not spans:
        return None
    last_label = None
    m = None
    for m in FINAL_LABEL_RE.finditer(text or ""):
        last_label = m.end()
    # Select spans after last label if present
    if last_label is not None:
        after = [p for (pos, p) in spans if pos >= last_label]
        if after:
            return after[-1]
    # Otherwise the last overall
    return spans[-1][1]

def _last_boxed_in_conversation(conv: List[Dict[str, Any]]) -> Optional[Tuple[int, str, str, Optional[str]]]:
    """
    Returns the last occurrence of a \boxed{...} in the conversation.
    Output tuple: (assistant_turn_index, assistant_text, boxed_payload, preceding_human_text).
    If no boxed is found, returns None.
    """
    last: Optional[Tuple[int, str, str, Optional[str]]] = None
    prev_h: Optional[str] = None
    for i, m in enumerate(conv or []):
        role = m.get("from")
        val = (m.get("value") or "").strip()
        if not val:
            continue
        if role == "human":
            prev_h = val
            continue
        if role != "gpt":
            continue
        # scan all boxed occurrences in this assistant turn and remember the last one
        local_prev_h = prev_h
        for pat in BOX_PATTERNS:
            for mm in pat.finditer(val):
                payload = (mm.group(1) or "").strip()
                last = (i, val, payload, local_prev_h)
    return last

def _normalize_tex_fraction(s: str) -> str:
    """Convert \\tfrac{a}{b} or \\frac{a}{b} to 'a/b'. Leaves other strings unchanged."""
    s = s.strip()
    m = TFRAC_RE.fullmatch(s)
    if m:
        num, den = m.group(1).strip(), m.group(2).strip()
        return f"{num}/{den}"
    # Also handle inline occurrences inside longer strings
    def _repl(mm):
        return f"{mm.group(1).strip()}/{mm.group(2).strip()}"
    return TFRAC_RE.sub(_repl, s)

def _choose_canonical_boxed(vals: List[str]) -> str:
    """Pick a single representative from a list of boxed payloads.
    Preference order:
      1) A clean fraction form (contains '/' or TeX frac) after normalization
      2) Otherwise the first value
    """
    if not vals:
        return ""
    normed = [(_normalize_tex_fraction(v), v) for v in vals]
    # prefer entries that clearly look like a/b after normalization
    for nv, orig in normed:
        if "/" in nv and len(nv) >= 3:
            return nv
    return normed[0][0]

def _ensure_parent(path: str) -> None:
    Path(path).parent.mkdir(parents=True, exist_ok=True)

def _append_jsonl(path: str, rec: Dict[str, Any]) -> None:
    _ensure_parent(path)
    with open(path, "a", encoding="utf-8") as f:
        f.write(json.dumps(rec, ensure_ascii=False) + "\n")

def _first_human(conversations: List[Dict[str, Any]]) -> Optional[str]:
    for m in conversations or []:
        if m.get("from") == "human":
            v = (m.get("value") or "").strip()
            if v:
                return v
    return None

def _last_assistant(conversations: List[Dict[str, Any]]) -> Optional[str]:
    for m in reversed(conversations or []):
        if m.get("from") == "gpt":
            v = (m.get("value") or "").strip()
            if v:
                return v
    return None

def _extract_all_boxed(text: str) -> List[str]:
    """Return all boxed payloads using balanced parsing, preserving full content including nested braces."""
    spans = _extract_boxed_balanced_with_spans(text)
    return [p for _, p in spans]

def _last_boxed_in_text(text: str) -> Optional[str]:
    """Return the prioritized final boxed payload in this text (see _prioritized_final)."""
    spans = _extract_boxed_balanced_with_spans(text or "")
    return _prioritized_final(spans, text or "")

def _iter_assistant_turns_with_questions(conv: List[Dict[str, Any]]) -> List[Tuple[int, str, Optional[str]]]:
    """
    Yields tuples (assistant_index, assistant_text, preceding_human_text) for each assistant turn.
    Note: a single assistant turn may contain multiple \boxed{...} occurrences; downstream logic
    should decide whether to keep all or only the final one.
    """
    prev_h = None
    out = []
    for i, m in enumerate(conv or []):
        role = m.get("from")
        val = (m.get("value") or "").strip()
        if not val:
            continue
        if role == "human":
            prev_h = val
        elif role == "gpt":
            out.append((i, val, prev_h))
    return out

def filter_and_export(repo_id: str, split: str, mode: str, out_jsonl: str) -> Tuple[Dict[str, int], List[Dict[str, Any]]]:
    ds = load_dataset(repo_id, split=split)
    kept, scanned, boxed_msgs, written = 0, 0, 0, 0
    out_records: List[Dict[str, Any]] = []

    for row in ds:
        scanned += 1
        conv = row.get("conversations") or []
        if mode == "multi":
            boxed_all: List[Dict[str, Any]] = []
            for idx, msg in enumerate(conv):
                if msg.get("from") == "gpt":
                    a_text = msg.get("value") or ""
                    spans = _extract_boxed_balanced_with_spans(a_text)
                    if spans:
                        payloads_raw = [p for _, p in spans]
                        payloads_canon = [_canonicalize_payload(p) for p in payloads_raw]
                        final_raw = _prioritized_final(spans, a_text)
                        final_canon = _canonicalize_payload(final_raw) if final_raw is not None else None
                        boxed_msgs += len(payloads_raw)
                        boxed_all.append({
                            "turn_index": idx,
                            "boxed_list": payloads_canon,
                            "boxed_raw_list": payloads_raw,
                            "final_boxed": final_canon,
                            "final_boxed_raw": final_raw,
                        })
            if boxed_all:
                kept += 1
                rec = dict(row)
                rec["boxed_answers"] = boxed_all  # list per turn
                out_records.append(rec)
                _append_jsonl(out_jsonl, rec)
                written += 1

        elif mode == "single":
            # Flatten: one record per assistant turn that has a final \boxed{...} in that turn.
            for idx, a_text, prev_q in _iter_assistant_turns_with_questions(conv):
                spans = _extract_boxed_balanced_with_spans(a_text)
                if not spans:
                    continue
                payloads_raw = [p for _, p in spans]
                payloads_canon = [_canonicalize_payload(p) for p in payloads_raw]
                final_raw = _prioritized_final(spans, a_text)
                final_canon = _canonicalize_payload(final_raw) if final_raw is not None else None

                boxed_msgs += len(payloads_raw)
                kept += 1
                rec = {
                    "id": row.get("id"),
                    "subject": row.get("subject"),
                    "grade": row.get("grade"),
                    "chapter_title": row.get("chapter_title"),
                    "source": row.get("source"),
                    "context_text": row.get("context_text"),
                    "rephrased_text": row.get("rephrased_text"),
                    "turn_index": idx,
                    "problem": prev_q or _first_human(conv) or "",
                    "generated_solution": a_text,
                    # Canonicalized answers (list) for sub-questions in a single turn
                    "extracted_answers": payloads_canon,
                    # Convenience: prioritized final (typically the last, or last after 'अन्तिम उत्तर:')
                    "final_answer": final_canon,
                }
                out_records.append(rec)
                _append_jsonl(out_jsonl, rec)
                written += 1
        else:
            raise ValueError("mode must be 'single' or 'multi'")

    return {
        "scanned": scanned,
        "kept_conversations": kept,
        "assistant_turns_with_boxed": boxed_msgs,
        "written": written,
    }, out_records

def push_to_hub_if_requested(records: List[Dict[str, Any]], repo_id: Optional[str], split: str, private: bool) -> None:
    if not repo_id:
        return
    if not records:
        print(f"[push] no records to upload to {repo_id}:{split}")
        return
    try:
        dset = Dataset.from_list(records)
        # Respect HF token from env if set: HF_TOKEN or HUGGINGFACE_TOKEN
        token = os.environ.get("HF_TOKEN") or os.environ.get("HUGGINGFACE_TOKEN")
        print(f"[push] uploading {len(records)} records to {repo_id}:{split} private={private}")
        dset.push_to_hub(repo_id, split=split, private=private, token=token)
        print(f"[push] done: {repo_id}:{split}")
    except Exception as e:
        print(f"[push] failed: {e}")

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Filter HF ShareGPT dataset for \\boxed{...} answers and extract them.")
    p.add_argument("--repo_id", required=True, help="HuggingFace dataset repo id")
    p.add_argument("--split", default="train", help="Dataset split")
    p.add_argument("--mode", choices=["single","multi"], default="single",
                   help="single: flatten; one record per assistant turn using the final \\boxed{...} within that turn. multi: keep conversation and attach a per-turn list with at most one (final-in-turn) answer; no convo-level final.")
    p.add_argument("--out_jsonl", required=True, help="Output JSONL path")
    p.add_argument("--push_repo_id", default=None, help="If set, also push filtered records to this HF dataset repo")
    p.add_argument("--push_split", default="train", help="Split name for push_to_hub")
    p.add_argument("--push_private", action="store_true", help="Create/update the repo as private")
    return p.parse_args()

def main():
    args = parse_args()
    stats, records = filter_and_export(args.repo_id, args.split, args.mode, args.out_jsonl)
    print(f"[stats] scanned={stats['scanned']} kept_conversations={stats['kept_conversations']} "
          f"assistant_turns_with_boxed={stats['assistant_turns_with_boxed']} written={stats['written']}")
    push_to_hub_if_requested(records, args.push_repo_id, args.push_split, args.push_private)

if __name__ == "__main__":
    main()