#!/usr/bin/env python3
# -*- coding: utf-8 -*-
from __future__ import annotations

import argparse
import json
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import torch
from transformers import AutoTokenizer
from tqdm import tqdm


# ----------------------------
# import helpers (robust to different package roots)
# ----------------------------
ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

def _try_import():
    # Try MarkLLM first (your analyze_v6_kgw_attack.py uses this) :contentReference[oaicite:9]{index=9}
    try:
        from MarkLLM.watermark.bytekgwV6.token_bytes import TokenByteVocabV6  # type: ignore
        from MarkLLM.watermark.bytekgwV6.prf import RobustPartitioner  # type: ignore
        return TokenByteVocabV6, RobustPartitioner
    except ModuleNotFoundError:
        pass

    # Fallback: if your repo uses "watermark" as top-level package
    try:
        from watermark.bytekgwV6.token_bytes import TokenByteVocabV6  # type: ignore
        from watermark.bytekgwV6.prf import RobustPartitioner  # type: ignore
        return TokenByteVocabV6, RobustPartitioner
    except ModuleNotFoundError as e:
        raise ModuleNotFoundError(
            "Cannot import TokenByteVocabV6 / RobustPartitioner. "
            "Tried MarkLLM.watermark... and watermark.... "
            "Fix: ensure repo root is on PYTHONPATH, and adjust import paths to match your tree."
        ) from e

TokenByteVocabV6, RobustPartitioner = _try_import()


# ----------------------------
# utils
# ----------------------------
def load_json(path: str) -> Dict[str, Any]:
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)

def _to_bytes(key) -> bytes:
    if isinstance(key, bytes):
        return key
    if isinstance(key, int):
        return int(key).to_bytes(8, "little", signed=False)
    if isinstance(key, str):
        s = key.strip()
        # allow "0x..." hex
        if s.startswith("0x") or all(c in "0123456789abcdefABCDEF" for c in s):
            try:
                s2 = s[2:] if s.startswith("0x") else s
                return bytes.fromhex(s2)
            except Exception:
                pass
        return s.encode("utf-8")
    raise TypeError(f"Unsupported key type: {type(key)}")

def popcount_bytes(x: bytes) -> int:
    # x is short (m_bytes=32), pure python popcount is fine
    return sum(int(b).bit_count() for b in x)

@dataclass
class TokInfo:
    ids: torch.Tensor               # [T]
    offsets: Optional[List[Tuple[int,int]]]  # len=T, char offsets in original string
    tokens: List[str]               # len=T, token strings (convert_ids_to_tokens)

def tokenize_with_offsets(tok, text: str, add_special_tokens: bool) -> TokInfo:
    """
    Prefer fast tokenizer offsets. If offsets are unavailable, offsets=None.
    """
    enc = tok(
        text,
        return_tensors="pt",
        add_special_tokens=add_special_tokens,
        return_offsets_mapping=True,
    )
    ids = enc["input_ids"][0]
    offsets = None
    if "offset_mapping" in enc:
        # fast tokenizers return offsets; shape [1,T,2]
        try:
            offsets = [tuple(map(int, x)) for x in enc["offset_mapping"][0].tolist()]
        except Exception:
            offsets = None
    toks = tok.convert_ids_to_tokens(ids.tolist())
    return TokInfo(ids=ids, offsets=offsets, tokens=toks)

def apply_char_attack_replace_X(full_text: str, prompt: str, attack_ratio: float, seed: int) -> Tuple[str, List[int]]:
    """
    Replace characters in continuation with 'X'.
    Returns attacked_text and absolute attacked character indices (w.r.t full_text).
    """
    if attack_ratio <= 0:
        return full_text, []

    rng = np.random.RandomState(seed)
    start = len(prompt)
    cont = full_text[start:]
    n = len(cont)
    if n <= 0:
        return full_text, []

    k = int(round(n * attack_ratio))
    k = max(1, k) if attack_ratio > 0 else 0
    k = min(k, n)

    idxs = rng.choice(n, size=k, replace=False)
    idxs = sorted(int(i) for i in idxs)

    cont_list = list(cont)
    for i in idxs:
        cont_list[i] = "X"

    attacked = full_text[:start] + "".join(cont_list)
    abs_idxs = [start + i for i in idxs]
    return attacked, abs_idxs


def tail_str(s: str, n: int) -> str:
    if n <= 0:
        return ""
    return s[-n:]

def build_v6_partitioner(v6_cfg: Dict[str, Any]) -> RobustPartitioner:
    partitioner = RobustPartitioner(
        master_key=_to_bytes(v6_cfg.get("hash_key", 15485863)),
        m_bits=int(v6_cfg.get("m_bits", 256)),
        target_anchors=int(v6_cfg.get("target_anchors", 96)),
        k_choices=tuple(v6_cfg.get("k_choices", [4, 5, 6])),
        normalize_whitespace=bool(v6_cfg.get("normalize_whitespace", True)),
    )
    return partitioner


# ----------------------------
# main trace
# ----------------------------
def parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser()
    ap.add_argument("--run_meta", required=True)
    ap.add_argument("--v6_config", required=True)
    ap.add_argument("--input_csv", required=True)
    ap.add_argument("--attack_ratio", type=float, default=0.02)
    ap.add_argument("--attack_seed", type=int, default=0)
    ap.add_argument("--indices", default="0,1,2", help="comma-separated row indices in csv")
    ap.add_argument("--max_tokens", type=int, default=200)
    ap.add_argument("--stride", type=int, default=1, help="trace every stride-th token (after prompt)")
    ap.add_argument("--device", default="cpu")
    ap.add_argument("--out_dir", default="traces_v6_process")
    return ap.parse_args()

def main() -> None:
    args = parse_args()
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    run_meta = load_json(args.run_meta)
    v6_cfg = load_json(args.v6_config)

    model_path = run_meta["model"]
    tok = AutoTokenizer.from_pretrained(model_path)

    add_special_tokens = bool(v6_cfg.get("add_special_tokens", True))
    seed_window_chars = int(v6_cfg.get("seed_window_chars", 18))
    n_bytes = int(v6_cfg.get("n_bytes", 3))

    # Build partitioner + vocab->firstn mapping
    partitioner = build_v6_partitioner(v6_cfg)
    vocab = TokenByteVocabV6.from_tokenizer(tok, skip_markers=True).to(args.device)
    firstn_ids = vocab.first_n_id(torch.device(args.device), n_bytes)  # [V]

    df = pd.read_csv(args.input_csv)
    wanted = [int(x) for x in args.indices.split(",") if x.strip()]

    for row_idx in wanted:
        row = df.iloc[row_idx]
        full: str = row["full_text"]
        prompt: str = row.get("prompt_text", "")

        attacked, abs_attacked_chars = apply_char_attack_replace_X(
            full, prompt, args.attack_ratio, seed=args.attack_seed + row_idx
        )

        # tokenize clean/attack with offsets
        clean_t = tokenize_with_offsets(tok, full, add_special_tokens=add_special_tokens)
        att_t = tokenize_with_offsets(tok, attacked, add_special_tokens=add_special_tokens)

        if clean_t.offsets is None or att_t.offsets is None:
            raise RuntimeError(
                "Tokenizer offsets are unavailable (slow tokenizer?). "
                "Use a fast tokenizer so return_offsets_mapping works."
            )

        # map clean token start offset -> index (for same-char-start alignment)
        clean_start2i: Dict[int, int] = {}
        for i, (s, e) in enumerate(clean_t.offsets):
            clean_start2i[s] = i

        # prompt boundary in chars (we only care continuation)
        prompt_len_chars = len(prompt)

        # TSV header
        header = [
            # attacked side
            "pos_a","char_s_a","char_e_a","tok_id_a","uid_a","tok_str_a",
            "tail_a","fp_a_hex","hd_tok_fp_a","is_green_a",
            "n_attacked_in_tailwin","attacked_pos_in_tailwin",
            # clean-aligned (same char start)
            "pos_c","char_s_c","char_e_c","tok_id_c","uid_c","tok_str_c",
            "tail_c","fp_c_hex","hd_tok_fp_c","is_green_c",
            # comparisons
            "same_uid","token_id_changed_at_same_char_start","no_clean_token_at_same_char_start",
            "tail_equal","fp_hd","green_changed_for_uid",
            "is_green_uid_under_clean_aligned_fp","green_flip_cleanfp_vs_attackfp",
        ]

        lines: List[str] = []
        lines.append("\t".join(header))

        # stats
        flip_cnt = 0
        flip_den = 0
        tail_eq_cnt = 0
        tail_den = 0
        extra_tok_cnt = 0
        traced = 0

        # iterate attacked tokens (by token index), but align by char start
        for pos_a, (s_a, e_a) in enumerate(att_t.offsets):
            if traced >= args.max_tokens:
                break
            if s_a < prompt_len_chars:
                continue
            if (pos_a % max(1, args.stride)) != 0:
                continue

            traced += 1
            tok_id_a = int(att_t.ids[pos_a].item())
            uid_a = int(firstn_ids[tok_id_a].item())
            tok_str_a = att_t.tokens[pos_a]

            # prefix ends at token start
            prefix_end = s_a
            prefix_a = attacked[:prefix_end]
            prefix_c = full[:min(prefix_end, len(full))]  # substitution attack keeps length; safe clamp

            # fingerprint + tail
            fp_a = partitioner.fingerprint(prefix_a, seed_window_chars)
            fp_c = partitioner.fingerprint(prefix_c, seed_window_chars)
            tail_a = tail_str(prefix_a, seed_window_chars)
            tail_c = tail_str(prefix_c, seed_window_chars)

            # hd(token_vector(uid) xor fp)
            w = partitioner.token_vector(uid_a)
            hd_tok_fp_a = popcount_bytes(bytes(x ^ y for x, y in zip(w, fp_a)))
            hd_tok_fp_c = popcount_bytes(bytes(x ^ y for x, y in zip(w, fp_c)))

            # green decisions for THIS uid
            is_green_a = int(partitioner.is_green(uid_a, fp_a))
            is_green_uid_under_clean_fp = int(partitioner.is_green(uid_a, fp_c))
            green_flip_cleanfp_vs_attackfp = int(is_green_a != is_green_uid_under_clean_fp)

            # tail-window attacked char positions
            win_l = max(0, prefix_end - seed_window_chars)
            in_win = [p for p in abs_attacked_chars if win_l <= p < prefix_end]
            n_in_win = len(in_win)

            tail_equal = int(tail_a == tail_c)
            fp_hd = popcount_bytes(bytes(x ^ y for x, y in zip(fp_a, fp_c)))
            green_changed_for_uid = int(is_green_a != is_green_uid_under_clean_fp)

            tail_den += 1
            tail_eq_cnt += tail_equal

            flip_den += 1
            flip_cnt += green_flip_cleanfp_vs_attackfp

            # align clean token at same char start
            pos_c = clean_start2i.get(s_a, -1)
            if pos_c == -1:
                # extra token (no clean token starts here)
                extra_tok_cnt += 1
                char_s_c = char_e_c = tok_id_c = uid_c = -1
                tok_str_c = ""
                is_green_c = -1
                hd_tok_fp_c_cleanuid = -1
            else:
                char_s_c, char_e_c = clean_t.offsets[pos_c]
                tok_id_c = int(clean_t.ids[pos_c].item())
                uid_c = int(firstn_ids[tok_id_c].item())
                tok_str_c = clean_t.tokens[pos_c]

                # clean's own fp at that start (for display)
                prefix_c2 = full[:char_s_c]
                fp_c2 = partitioner.fingerprint(prefix_c2, seed_window_chars)
                tail_c2 = tail_str(prefix_c2, seed_window_chars)
                # but for comparability, we keep fp_c_hex as fp_c (aligned by s_a)
                # and tail_c as tail_c (aligned by s_a)

                # green for clean's uid under clean fp at its own start
                is_green_c = int(partitioner.is_green(uid_c, fp_c2))
                hd_tok_fp_c_cleanuid = popcount_bytes(
                    bytes(x ^ y for x, y in zip(partitioner.token_vector(uid_c), fp_c2))
                )

            same_uid = int((pos_c != -1) and (uid_c == uid_a))
            token_id_changed = int((pos_c != -1) and (tok_id_c != tok_id_a))
            no_clean = int(pos_c == -1)

            row_out = [
                str(pos_a), str(s_a), str(e_a), str(tok_id_a), str(uid_a), tok_str_a,
                tail_a.replace("\t","\\t").replace("\n","\\n"),
                fp_a.hex(), str(hd_tok_fp_a), str(is_green_a),
                str(n_in_win), ",".join(map(str, in_win)),
                str(pos_c), str(char_s_c), str(char_e_c), str(tok_id_c), str(uid_c), tok_str_c,
                tail_c.replace("\t","\\t").replace("\n","\\n"),
                fp_c.hex(), str(hd_tok_fp_c_cleanuid if pos_c!=-1 else -1), str(is_green_c),
                str(same_uid), str(token_id_changed), str(no_clean),
                str(tail_equal), str(fp_hd), str(green_changed_for_uid),
                str(is_green_uid_under_clean_fp), str(green_flip_cleanfp_vs_attackfp),
            ]
            lines.append("\t".join(row_out))

        # write files
        tsv_path = out_dir / f"trace_row{row_idx}_r{args.attack_ratio:.3f}.tsv"
        tsv_path.write_text("\n".join(lines), encoding="utf-8")

        summary = {
            "row_idx": row_idx,
            "attack_ratio": args.attack_ratio,
            "seed_window_chars": seed_window_chars,
            "n_bytes": n_bytes,
            "clean_tokens": int(clean_t.ids.numel()),
            "attacked_tokens": int(att_t.ids.numel()),
            "extra_token_count_in_traced": int(extra_tok_cnt),
            "traced_tokens": int(traced),
            "tail_match_rate_over_traced": float(tail_eq_cnt / max(1, tail_den)),
            "green_flip_rate_uid_cleanfp_vs_attackfp": float(flip_cnt / max(1, flip_den)),
        }
        (out_dir / f"summary_row{row_idx}_r{args.attack_ratio:.3f}.json").write_text(
            json.dumps(summary, ensure_ascii=False, indent=2),
            encoding="utf-8"
        )

        print(f"\n[row {row_idx}] wrote: {tsv_path}")
        print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
