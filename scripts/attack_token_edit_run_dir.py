#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Token-level random edits on CSV files in a directory.
Similar接口 to attack_char_edit_run_dir.py:
  - reads all CSVs in --in_dir (filtered by --glob_regex)
  - writes attacked CSVs to --out_dir with suffix _attacked.csv
  - adds columns: attacked_full_text, token_edit_ops, token_edit_replace/delete/insert
  - if --overwrite_full_text: replace full_text/text with attacked version, keep orig_full_text

Edits are applied on tokens decoded with the model tokenizer from run_meta["model"].
"""

import argparse
import csv
import os
import random
import re
from typing import Dict, List, Tuple, Optional

from transformers import AutoTokenizer


def stable_seed(base: int, s: str) -> int:
    import hashlib

    h = hashlib.blake2b(digest_size=8)
    h.update(str(base).encode("utf-8"))
    h.update(b"|")
    h.update((s or "").encode("utf-8", errors="ignore"))
    return int.from_bytes(h.digest(), "little") & 0x7FFFFFFF


def detect_prompt_full(row: Dict[str, str]) -> Tuple[str, str, str]:
    prompt = row.get("prompt_text") or row.get("prompt") or ""
    completion = row.get("completion_text") or row.get("completion") or ""
    full_text = row.get("full_text") or row.get("text") or ""
    if not full_text:
        full_text = (prompt or "") + (completion or "")
    return str(prompt), str(completion), str(full_text)


def infer_completion(prompt: str, full_text: str) -> Optional[str]:
    if prompt and full_text.startswith(prompt):
        return full_text[len(prompt) :]
    return None


def token_edit(
    tok: AutoTokenizer,
    text: str,
    *,
    rng: random.Random,
    edit_ratio: float,
    ops: List[str],
) -> Tuple[str, int, Dict[str, int]]:
    """
    Token-level edits: replace/delete/insert on token ids, then decode.
    """
    ops = [o.strip().lower() for o in ops if o.strip()]
    if not ops:
        ops = ["replace", "delete", "insert"]

    enc = tok(text, add_special_tokens=False, return_tensors=None)
    ids: List[int] = list(enc["input_ids"])
    if not ids:
        return text, 0, {op: 0 for op in ops}

    k = int(round(len(ids) * float(edit_ratio)))
    if k <= 0:
        return text, 0, {op: 0 for op in ops}

    # candidate ids for replacement/insertion (non-special)
    special = set(tok.all_special_ids or [])
    cand_ids = [i for i in range(tok.vocab_size) if i not in special]
    if not cand_ids:
        cand_ids = [i for i in range(tok.vocab_size)]

    cnt = {op: 0 for op in ops}
    for _ in range(k):
        op = ops[rng.randrange(len(ops))]
        if op == "replace":
            if not ids:
                continue
            i = rng.randrange(len(ids))
            ids[i] = rng.choice(cand_ids)
            cnt["replace"] += 1
        elif op == "delete":
            if not ids:
                continue
            i = rng.randrange(len(ids))
            ids.pop(i)
            cnt["delete"] += 1
        elif op == "insert":
            i = rng.randrange(len(ids) + 1)
            ids.insert(i, rng.choice(cand_ids))
            cnt["insert"] += 1
        else:
            # treat unknown op as replace
            if not ids:
                continue
            i = rng.randrange(len(ids))
            ids[i] = rng.choice(cand_ids)
            cnt["replace"] += 1

    if not ids:
        # avoid empty decode
        return text, sum(cnt.values()), cnt

    attacked = tok.decode(ids, skip_special_tokens=True)
    return attacked, sum(cnt.values()), cnt


def process_csv(
    tok: AutoTokenizer,
    in_path: str,
    out_path: str,
    *,
    edit_ratio: float,
    ops: List[str],
    seed: int,
    attack_generated_only: bool,
    overwrite_full_text: bool,
) -> None:
    with open(in_path, "r", encoding="utf-8") as f:
        rows = list(csv.DictReader(f))
    if not rows:
        raise ValueError(f"Empty CSV: {in_path}")

    fieldnames = list(rows[0].keys())
    extra_cols = [
        "attacked_full_text",
        "token_edit_ops",
        "token_edit_replace",
        "token_edit_delete",
        "token_edit_insert",
    ]
    for c in extra_cols:
        if c not in fieldnames:
            fieldnames.append(c)
    if overwrite_full_text and "full_text" in fieldnames and "orig_full_text" not in fieldnames:
        fieldnames.append("orig_full_text")

    out_rows = []
    for idx, r in enumerate(rows):
        prompt, completion, full = detect_prompt_full(r)
        # decide target to edit
        if attack_generated_only:
            target = completion if completion else (infer_completion(prompt, full) or full)
            rng = random.Random(stable_seed(seed, f"{in_path}|{idx}|{prompt}"))
            adv_target, n_ops, cnt = token_edit(tok, target, rng=rng, edit_ratio=edit_ratio, ops=ops)
            if completion:
                adv_full = (prompt or "") + adv_target
            else:
                if prompt and full.startswith(prompt):
                    adv_full = prompt + adv_target
                else:
                    adv_full = adv_target
        else:
            rng = random.Random(stable_seed(seed, f"{in_path}|{idx}|{prompt}"))
            adv_full, n_ops, cnt = token_edit(tok, full, rng=rng, edit_ratio=edit_ratio, ops=ops)

        rr = dict(r)
        rr["attacked_full_text"] = adv_full
        rr["token_edit_ops"] = str(n_ops)
        rr["token_edit_replace"] = str(cnt.get("replace", 0))
        rr["token_edit_delete"] = str(cnt.get("delete", 0))
        rr["token_edit_insert"] = str(cnt.get("insert", 0))

        if overwrite_full_text and "full_text" in rr:
            rr["orig_full_text"] = rr["full_text"]
            rr["full_text"] = adv_full
        elif overwrite_full_text and "text" in rr:
            rr["orig_full_text"] = rr["text"]
            rr["text"] = adv_full

        out_rows.append(rr)

    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    with open(out_path, "w", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames)
        w.writeheader()
        w.writerows(out_rows)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--run_meta", required=True, help="JSON with model path (run_metadata.json)")
    ap.add_argument("--in_dir", required=True)
    ap.add_argument("--out_dir", required=True)
    ap.add_argument("--edit_ratio", type=float, default=0.02)
    ap.add_argument("--ops", default="replace,delete,insert")
    ap.add_argument("--seed", type=int, default=1234)
    ap.add_argument("--attack_generated_only", action="store_true")
    ap.add_argument(
        "--overwrite_full_text",
        action="store_true",
        help="If set, overwrite full_text/text column with attacked text (keep orig_full_text).",
    )
    ap.add_argument(
        "--glob_regex", default=r".*\.csv$", help="Regex to select csv files inside in_dir (default: all .csv)."
    )
    args = ap.parse_args()

    import json

    with open(args.run_meta, "r", encoding="utf-8") as f:
        run_meta = json.load(f)
    model_path = run_meta["model"]
    tok = AutoTokenizer.from_pretrained(model_path)

    ops = [x.strip().lower() for x in args.ops.split(",") if x.strip()]
    rx = re.compile(args.glob_regex)

    os.makedirs(args.out_dir, exist_ok=True)
    files = []
    for fn in sorted(os.listdir(args.in_dir)):
        if rx.match(fn):
            files.append(fn)
    if not files:
        raise FileNotFoundError(f"No csv matched in {args.in_dir} by regex {args.glob_regex}")

    print("=" * 100)
    print(f"[IN ] {args.in_dir}")
    print(f"[OUT] {args.out_dir}")
    print(f"[ATTACK token] edit_ratio={args.edit_ratio} ops={ops} generated_only={args.attack_generated_only}")
    print(f"[WRITE] overwrite_full_text={args.overwrite_full_text}")
    print("=" * 100)

    for fn in files:
        in_path = os.path.join(args.in_dir, fn)
        out_path = os.path.join(args.out_dir, fn.replace(".csv", "_attacked.csv"))
        print(f"[DO] {fn} -> {os.path.basename(out_path)}")
        process_csv(
            tok,
            in_path,
            out_path,
            edit_ratio=args.edit_ratio,
            ops=ops,
            seed=args.seed,
            attack_generated_only=args.attack_generated_only,
            overwrite_full_text=args.overwrite_full_text,
        )
    print("[OK] done.")


if __name__ == "__main__":
    main()
