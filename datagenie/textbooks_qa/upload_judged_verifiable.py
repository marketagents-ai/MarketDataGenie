import argparse
import json
from typing import List, Tuple, Optional, Dict, Any
import re
from datasets import load_dataset, Dataset, DatasetDict

# -------------------------
# Balanced boxed extraction
# -------------------------
FINAL_LABEL_RE = re.compile(r"(?:final\s*answer)[:：]?", re.IGNORECASE)
BOXED_TOKEN_RE = re.compile(r"(?:(?:\\|/){0,2}\s*boxed\b|oxed/)\s*", re.IGNORECASE)

# Minimal fraction normalizer used elsewhere in the project
_TFRAC = re.compile(r"\\t?frac\s*\{([^}]*)\}\s*\{([^}]*)\}")

def _normalize_tex_fraction(s: str) -> str:
    def _r(m: re.Match) -> str:
        return f"{m.group(1).strip()}/{m.group(2).strip()}"
    return _TFRAC.sub(_r, s)

def _extract_boxed_balanced_with_spans(text: str) -> List[Tuple[int, str]]:
    spans: List[Tuple[int, str]] = []
    if not text:
        return spans
    s = text
    n = len(s)
    for m in BOXED_TOKEN_RE.finditer(s):
        i = m.end()
        while i < n and s[i].isspace():
            i += 1
        if i >= n or s[i] != '{':
            continue
        start_payload = i + 1
        depth = 1
        j = start_payload
        while j < n:
            ch = s[j]
            if ch == '\\':
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
    return spans

def _canonicalize_payload(p: str) -> str:
    return _normalize_tex_fraction(p.strip())

def _prioritized_final(spans: List[Tuple[int, str]], text: str) -> Optional[str]:
    if not spans:
        return None
    last_label = None
    for m in FINAL_LABEL_RE.finditer(text or ""):
        last_label = m.end()
    if last_label is not None:
        after = [payload for (pos, payload) in spans if pos >= last_label]
        if after:
            return after[-1]
    return spans[-1][1]

# -------------------------
# CLI + main
# -------------------------

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Prepare and upload verified dataset with extracted_answers and final_answer. Input can be HF repo or a JSONL.")
    src = p.add_mutually_exclusive_group(required=True)
    src.add_argument("--in_repo", help="Source HF dataset repo id (e.g., your_username/textbook-qa-judged)")
    p.add_argument("--in_split", default="train", help="Source split (for --in_repo)")
    src.add_argument("--in_jsonl", help="Path to local JSONL of judged results")
    p.add_argument("--out_repo", required=True, help="Destination HF dataset repo id")
    p.add_argument("--out_split", default="train", help="Destination split")
    p.add_argument("--private", action="store_true", help="Push destination as private")
    p.add_argument("--max_items", type=int, default=None, help="Optional cap for a smoke test")
    return p.parse_args()

def _load_jsonl(path: str) -> List[Dict[str, Any]]:
    rows = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            rows.append(json.loads(line))
    return rows

def process_record(rec: Dict[str, Any]) -> Dict[str, Any]:
    # Prefer assistant_text; some datasets used generated_solution
    atext = (rec.get("assistant_text") or rec.get("generated_solution") or "")
    spans = _extract_boxed_balanced_with_spans(atext)
    payloads_raw = [p for _, p in spans]
    payloads_canon = [_canonicalize_payload(p) for p in payloads_raw]
    final_raw = _prioritized_final(spans, atext)
    final_canon = _canonicalize_payload(final_raw) if final_raw is not None else None

    # Normalize verdict, score, tally, judges from either top-level or `verification`
    verdict = rec.get("verdict")
    score = rec.get("score")
    tally = rec.get("tally")
    judges = rec.get("llm_judges") or rec.get("judges")
    if verdict is None or score is None:
        ver = rec.get("verification") or {}
        agg = ver.get("aggregate") or {}
        verdict = verdict or agg.get("verdict")
        score = score or agg.get("score")
        tally = tally or agg.get("tally")
        judges = judges or ver.get("judges")

    # Build a clean, ordered record
    out: Dict[str, Any] = {}
    out["id"] = rec.get("id")
    out["problem"] = rec.get("question") or rec.get("problem")
    out["generated_solution"] = atext
    out["extracted_answers"] = payloads_canon
    out["final_answer"] = final_canon
    out["verdict"] = verdict
    out["score"] = score
    out["tally"] = tally
    out["llm_judges"] = judges
    out["rephrased_text"] = rec.get("rephrased_text")
    out["context_text"] = rec.get("context_text")

    # Append any remaining useful metadata except deprecated/noisy fields
    excluded = {
        "assistant_text", "generated_solution", "question", "verification",
        "extracted_answer", "boxed_raw", "boxed_raw_all", "old_extracted_answer",
        "extracted_answers", "final_answer", "verdict", "score", "tally",
        "llm_judges", "rephrased_text", "context_text", "problem", "id"
    }
    for k, v in rec.items():
        if k in excluded:
            continue
        # Preserve common meta: subject/grade/source/chapter_title/etc.
        out.setdefault(k, v)

    return out

def main():
    args = parse_args()

    # Load source rows
    if args.in_repo:
        print(f"[io] loading {args.in_repo}:{args.in_split} ...")
        ds = load_dataset(args.in_repo, split=args.in_split)
        if args.max_items:
            ds = ds.select(range(min(args.max_items, len(ds))))
        rows = [dict(r) for r in ds]
    else:
        print(f"[io] loading jsonl from {args.in_jsonl} ...")
        rows = _load_jsonl(args.in_jsonl)
        if args.max_items:
            rows = rows[: args.max_items]
    print(f"[io] rows: {len(rows)}")

    # Process
    processed = [process_record(r) for r in rows]

    # Reorder columns explicitly
    ordered_rows: List[Dict[str, Any]] = []
    front_keys = [
        "id", "problem", "generated_solution",
        "extracted_answers", "final_answer",
        "verdict", "score", "tally", "llm_judges",
        "rephrased_text", "context_text",
    ]
    for r in processed:
        o = {k: r.get(k) for k in front_keys}
        # add any remaining keys in stable order
        for k, v in r.items():
            if k not in o:
                o[k] = v
        ordered_rows.append(o)

    # Create dataset and push
    ds_out = Dataset.from_list(ordered_rows)
    dsd = DatasetDict({args.out_split: ds_out})
    print(f"[upload] pushing to {args.out_repo}:{args.out_split} ...")
    dsd.push_to_hub(repo_id=args.out_repo, private=args.private)
    print("[done] push complete.")

if __name__ == "__main__":
    main()
