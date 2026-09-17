#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import argparse
import csv
import hashlib
import os
import random
import re
from typing import Dict, List, Tuple, Optional

ASCII_POOL = (
    "abcdefghijklmnopqrstuvwxyz"
    "ABCDEFGHIJKLMNOPQRSTUVWXYZ"
    "0123456789"
    " .,;:!?'-\"()[]{}"
)

ZWSP = "\u200b"  # zero-width space
ZWJ  = "\u200d"  # zero-width joiner

# small homoglyph map (ASCII -> visually similar Unicode)
HOMO = {
    "a": "а",  # Cyrillic a
    "c": "с",
    "e": "е",
    "i": "і",
    "o": "о",
    "p": "р",
    "x": "х",
    "y": "у",
    "A": "Α",  # Greek Alpha
    "B": "Β",
    "E": "Ε",
    "H": "Η",
    "I": "Ι",
    "K": "Κ",
    "M": "Μ",
    "N": "Ν",
    "O": "Ο",
    "P": "Ρ",
    "T": "Τ",
    "X": "Χ",
}


def stable_seed(base: int, s: str) -> int:
    h = hashlib.blake2b(digest_size=8)
    h.update(str(base).encode("utf-8"))
    h.update(b"|")
    h.update((s or "").encode("utf-8", errors="ignore"))
    return int.from_bytes(h.digest(), "little") & 0x7FFFFFFF


def detect_prompt_full(row: Dict[str, str]) -> Tuple[str, str, str]:
    """
    Return (prompt, completion, full_text) from common column names.
    """
    prompt = row.get("prompt_text") or row.get("prompt") or ""
    completion = row.get("completion_text") or row.get("completion") or ""
    full_text = row.get("full_text") or row.get("text") or ""

    if not full_text:
        full_text = (prompt or "") + (completion or "")
    return str(prompt), str(completion), str(full_text)


def infer_completion(prompt: str, full_text: str) -> Optional[str]:
    if prompt and full_text.startswith(prompt):
        return full_text[len(prompt):]
    return None


def char_edit(
    s: str,
    *,
    rng: random.Random,
    edit_ratio: float,
    ops: List[str],
    mode: str,
) -> Tuple[str, int, Dict[str, int]]:
    """
    True char-level edits on raw string.
    mode:
      - ascii: replace/insert uses ASCII_POOL
      - zw:    insert uses ZWSP/ZWJ
      - homo:  replace prefers homoglyph substitution when possible
      - mixed: mixture of ascii + zw + homo behavior
    """
    if not s:
        return s, 0, {op: 0 for op in ops}

    ops = [o.strip().lower() for o in ops if o.strip()]
    if not ops:
        ops = ["replace", "delete", "insert"]

    k = int(round(len(s) * float(edit_ratio)))
    if k <= 0:
        return s, 0, {op: 0 for op in ops}

    arr = list(s)
    cnt = {op: 0 for op in ops}

    def rand_ascii():
        return ASCII_POOL[rng.randrange(len(ASCII_POOL))]

    def rand_zw():
        return ZWSP if rng.random() < 0.5 else ZWJ

    def do_replace(i: int):
        ch = arr[i]
        if mode == "homo" or (mode == "mixed" and rng.random() < 0.7):
            if ch in HOMO:
                arr[i] = HOMO[ch]
                return
        # fallback ascii replace
        arr[i] = rand_ascii()

    def do_insert(i: int):
        if mode == "zw" or (mode == "mixed" and rng.random() < 0.6):
            arr.insert(i, rand_zw())
        else:
            arr.insert(i, rand_ascii())

    for _ in range(k):
        op = ops[rng.randrange(len(ops))]

        if op == "replace":
            if not arr:
                continue
            i = rng.randrange(len(arr))
            do_replace(i)
            cnt["replace"] += 1

        elif op == "delete":
            if not arr:
                continue
            i = rng.randrange(len(arr))
            arr.pop(i)
            cnt["delete"] += 1

        elif op == "insert":
            i = rng.randrange(len(arr) + 1)
            do_insert(i)
            cnt["insert"] += 1

        else:
            # unknown op -> treat as replace
            if not arr:
                continue
            i = rng.randrange(len(arr))
            do_replace(i)
            cnt["replace"] += 1

    return "".join(arr), sum(cnt.values()), cnt


def process_csv(
    in_path: str,
    out_path: str,
    *,
    edit_ratio: float,
    ops: List[str],
    mode: str,
    seed: int,
    attack_generated_only: bool,
    overwrite_full_text: bool,
) -> None:
    with open(in_path, "r", encoding="utf-8") as f:
        rows = list(csv.DictReader(f))

    if not rows:
        raise ValueError(f"Empty CSV: {in_path}")

    fieldnames = list(rows[0].keys())
    # add output columns
    extra_cols = ["attacked_full_text", "char_edit_ops", "char_edit_replace", "char_edit_delete", "char_edit_insert"]
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
            adv_target, n_ops, cnt = char_edit(target, rng=rng, edit_ratio=edit_ratio, ops=ops, mode=mode)
            if completion:
                adv_full = (prompt or "") + adv_target
            else:
                # inferred completion path
                if prompt and full.startswith(prompt):
                    adv_full = prompt + adv_target
                else:
                    adv_full = adv_target
        else:
            rng = random.Random(stable_seed(seed, f"{in_path}|{idx}|{prompt}"))
            adv_full, n_ops, cnt = char_edit(full, rng=rng, edit_ratio=edit_ratio, ops=ops, mode=mode)

        rr = dict(r)
        rr["attacked_full_text"] = adv_full
        rr["char_edit_ops"] = str(n_ops)
        rr["char_edit_replace"] = str(cnt.get("replace", 0))
        rr["char_edit_delete"] = str(cnt.get("delete", 0))
        rr["char_edit_insert"] = str(cnt.get("insert", 0))

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


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--in_dir", required=True, help="e.g. outputs/c4_samples_head_200")
    ap.add_argument("--out_dir", required=True, help="e.g. outputs/attack_char_head_er002")
    ap.add_argument("--edit_ratio", type=float, default=0.02)
    ap.add_argument("--ops", default="replace,delete,insert")
    ap.add_argument("--mode", default="ascii", choices=["ascii", "zw", "homo", "mixed"])
    ap.add_argument("--seed", type=int, default=1234)
    ap.add_argument("--attack_generated_only", action="store_true")
    ap.add_argument("--overwrite_full_text", action="store_true",
                    help="If set, overwrite full_text/text column with attacked text (keep orig_full_text).")
    ap.add_argument("--glob_regex", default=r".*\.csv$",
                    help="Regex to select csv files inside in_dir (default: all .csv).")
    args = ap.parse_args()

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
    print(f"[ATTACK] edit_ratio={args.edit_ratio} ops={ops} mode={args.mode} generated_only={args.attack_generated_only}")
    print(f"[WRITE] overwrite_full_text={args.overwrite_full_text}")
    print("=" * 100)

    for fn in files:
        in_path = os.path.join(args.in_dir, fn)
        out_path = os.path.join(args.out_dir, fn.replace(".csv", "_attacked.csv"))
        print(f"[DO] {fn} -> {os.path.basename(out_path)}")
        process_csv(
            in_path, out_path,
            edit_ratio=args.edit_ratio,
            ops=ops,
            mode=args.mode,
            seed=args.seed,
            attack_generated_only=args.attack_generated_only,
            overwrite_full_text=args.overwrite_full_text,
        )

    print("[OK] done.")


if __name__ == "__main__":
    main()
