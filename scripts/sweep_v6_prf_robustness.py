#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from __future__ import annotations
import argparse
import importlib
import json
import random
import sys
from dataclasses import dataclass
from itertools import product
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import pandas as pd
import torch
from transformers import AutoTokenizer

# -------- robust imports (MarkLLM / watermark) --------
ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

def _try_import_v6():
    candidates = [
        ("MarkLLM.watermark.bytekgwV6.token_bytes", "TokenByteVocabV6",
         "MarkLLM.watermark.bytekgwV6.prf", "RobustPartitioner"),
        ("watermark.bytekgwV6.token_bytes", "TokenByteVocabV6",
         "watermark.bytekgwV6.prf", "RobustPartitioner"),
    ]
    last = None
    for m1, c1, m2, c2 in candidates:
        try:
            mod1 = importlib.import_module(m1)
            mod2 = importlib.import_module(m2)
            return getattr(mod1, c1), getattr(mod2, c2)
        except Exception as e:
            last = e
    raise ModuleNotFoundError(f"Cannot import V6 modules. Last error: {last}")

TokenByteVocabV6, RobustPartitioner = _try_import_v6()

# -------- helpers --------
def load_json(path: str) -> Dict[str, Any]:
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)

def apply_char_attack_replace_X(full_text: str, prompt: str, ratio: float, seed: int) -> Tuple[str, List[int]]:
    """
    Replace k chars in continuation with 'X' (length unchanged).
    Return attacked text and absolute attacked char indices.
    """
    if ratio <= 0:
        return full_text, []
    rng = random.Random(seed)

    start = len(prompt) if full_text.startswith(prompt) else 0
    cont = full_text[start:]
    n = len(cont)
    if n <= 0:
        return full_text, []

    k = max(1, int(round(n * ratio)))
    k = min(k, n)
    idxs = rng.sample(range(n), k)
    idxs.sort()

    cont_list = list(cont)
    for i in idxs:
        cont_list[i] = "X"
    attacked = full_text[:start] + "".join(cont_list)
    abs_idxs = [start + i for i in idxs]
    return attacked, abs_idxs

def tokenize_with_offsets(tok, text: str, add_special_tokens: bool) -> Tuple[List[int], List[Tuple[int,int]]]:
    enc = tok(text, add_special_tokens=add_special_tokens, return_offsets_mapping=True)
    if "offset_mapping" not in enc or enc["offset_mapping"] is None:
        raise RuntimeError("offset_mapping unavailable; please use a fast tokenizer.")
    ids = enc["input_ids"]
    offsets = [(int(a), int(b)) for (a, b) in enc["offset_mapping"]]
    return ids, offsets

def _to_bytes(key) -> bytes:
    if isinstance(key, bytes):
        return key
    if isinstance(key, int):
        return int(key).to_bytes(16, "little", signed=False)
    if isinstance(key, str):
        s = key.strip()
        if s.startswith("0x"):
            try:
                return int(s, 16).to_bytes(16, "little", signed=False)
            except Exception:
                pass
        # try hex without 0x
        try:
            return bytes.fromhex(s)
        except Exception:
            return s.encode("utf-8")
    # fallback
    return str(key).encode("utf-8")

def build_partitioner(cfg: Dict[str, Any]) -> Any:
    """
    Match your v6 partitioner signature.
    """
    # most likely signature in your repo
    tries = [
        lambda: RobustPartitioner(
            master_key=_to_bytes(cfg.get("hash_key", 15485863)),
            m_bits=int(cfg.get("m_bits", 256)),
            target_anchors=int(cfg.get("target_anchors", 96)),
            k_choices=tuple(cfg.get("k_choices", [4,5,6])),
            normalize_whitespace=bool(cfg.get("normalize_whitespace", True)),
        ),
        # alt naming
        lambda: RobustPartitioner(
            hash_key=_to_bytes(cfg.get("hash_key", 15485863)),
            m_bits=int(cfg.get("m_bits", 256)),
            target_anchors=int(cfg.get("target_anchors", 96)),
            k_choices=tuple(cfg.get("k_choices", [4,5,6])),
            normalize_whitespace=bool(cfg.get("normalize_whitespace", True)),
        ),
    ]
    last = None
    for fn in tries:
        try:
            return fn()
        except Exception as e:
            last = e
    raise RuntimeError(f"Cannot construct RobustPartitioner. Last error={last}")

@dataclass
class CachedSample:
    ridx: int
    clean_text: str
    attack_text: str
    prompt: str
    clean_ids: List[int]
    clean_off: List[Tuple[int,int]]
    attack_ids: List[int]
    attack_off: List[Tuple[int,int]]
    clean_start2i: Dict[int,int]
    prompt_chars: int

def cache_samples(
    tok,
    df: pd.DataFrame,
    indices: List[int],
    attack_ratio: float,
    attack_seed: int,
    add_special_tokens: bool
) -> List[CachedSample]:
    out: List[CachedSample] = []
    for ridx in indices:
        row = df.iloc[ridx]
        full = str(row["full_text"])
        prompt = str(row.get("prompt_text", ""))
        attacked, _ = apply_char_attack_replace_X(full, prompt, attack_ratio, seed=attack_seed + ridx)

        clean_ids, clean_off = tokenize_with_offsets(tok, full, add_special_tokens=add_special_tokens)
        att_ids, att_off = tokenize_with_offsets(tok, attacked, add_special_tokens=add_special_tokens)

        m: Dict[int,int] = {}
        for i, (s, e) in enumerate(clean_off):
            if s not in m:
                m[s] = i

        prompt_chars = len(prompt) if full.startswith(prompt) else 0
        out.append(
            CachedSample(
                ridx=ridx,
                clean_text=full,
                attack_text=attacked,
                prompt=prompt,
                clean_ids=clean_ids,
                clean_off=clean_off,
                attack_ids=att_ids,
                attack_off=att_off,
                clean_start2i=m,
                prompt_chars=prompt_chars,
            )
        )
    return out

def parse_int_list(s: str) -> List[int]:
    return [int(x.strip()) for x in s.split(",") if x.strip()]

def parse_kchoices_list(s: str) -> List[List[int]]:
    """
    Input like: "4|5|6,3|4|5" -> [[4,5,6],[3,4,5]]
    """
    items = []
    for part in s.split(","):
        part = part.strip()
        if not part:
            continue
        items.append([int(x) for x in part.split("|") if x.strip()])
    return items

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run_meta", required=True)
    ap.add_argument("--base_v6_config", required=True, help="e.g. config/ByteKGWv6.json")
    ap.add_argument("--input_csv", required=True)
    ap.add_argument("--indices", default="0,1,2")
    ap.add_argument("--attack_ratio", type=float, default=0.02)
    ap.add_argument("--attack_seed", type=int, default=0)
    ap.add_argument("--max_tokens", type=int, default=200)
    ap.add_argument("--stride", type=int, default=1)
    ap.add_argument("--device", default="cpu")

    # sweep grid knobs
    ap.add_argument("--seed_windows", default="8,10,12,18")
    ap.add_argument("--m_bits_list", default="256,512")
    ap.add_argument("--target_anchors_list", default="64,96,128")
    ap.add_argument("--n_bytes_list", default="2,3,4")
    ap.add_argument("--k_choices_list", default="4|5|6")
    ap.add_argument("--normalize_whitespace_list", default="true")

    # optional “区分度/随机性”粗检
    ap.add_argument("--corr_pairs", type=int, default=0, help=">0 to estimate green agreement across unrelated contexts")
    ap.add_argument("--corr_uids", type=int, default=512, help="how many random uids used for correlation estimate")

    ap.add_argument("--out_dir", required=True)
    args = ap.parse_args()

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    run_meta = load_json(args.run_meta)
    base_cfg = load_json(args.base_v6_config)

    tok = AutoTokenizer.from_pretrained(run_meta["model"])
    add_special_tokens = bool(base_cfg.get("add_special_tokens", True))

    df = pd.read_csv(args.input_csv)
    indices = parse_int_list(args.indices)

    # cache tokenization once (PRF sweep won't change tokenization)
    cached = cache_samples(tok, df, indices, args.attack_ratio, args.attack_seed, add_special_tokens)

    # prepare sweep grid
    seed_windows = parse_int_list(args.seed_windows)
    m_bits_list = parse_int_list(args.m_bits_list)
    target_anchors_list = parse_int_list(args.target_anchors_list)
    n_bytes_list = parse_int_list(args.n_bytes_list)
    k_choices_list = parse_kchoices_list(args.k_choices_list)

    norm_list = [x.strip().lower() for x in args.normalize_whitespace_list.split(",") if x.strip()]
    norm_bool_list = [True if x in ("1","true","yes","y") else False for x in norm_list]

    results: List[Dict[str, Any]] = []

    # build vocab once
    vocab = TokenByteVocabV6.from_tokenizer(tok, skip_markers=True).to(args.device)

    for (seed_w, m_bits, targ, n_bytes, k_choices, norm_ws) in product(
        seed_windows, m_bits_list, target_anchors_list, n_bytes_list, k_choices_list, norm_bool_list
    ):
        cfg = dict(base_cfg)
        cfg["seed_window_chars"] = int(seed_w)
        cfg["m_bits"] = int(m_bits)
        cfg["target_anchors"] = int(targ)
        cfg["n_bytes"] = int(n_bytes)
        cfg["k_choices"] = list(k_choices)
        cfg["normalize_whitespace"] = bool(norm_ws)

        partitioner = build_partitioner(cfg)
        firstn_ids = vocab.first_n_id(torch.device(args.device), n_bytes).to("cpu")

        # counters
        total = 0
        extra = 0
        aligned = 0
        uid_change = 0
        same_uid = 0
        prf_flip = 0
        green_a_cnt = 0  # attack green count (for p_attack)
        green_c_cnt = 0  # clean green count (for p_clean)

        # main measurement
        for s in cached:
            # map clean start -> idx already in cache
            for pos_a, (cs, ce) in enumerate(s.attack_off):
                if total >= args.max_tokens:
                    break
                if cs < s.prompt_chars:
                    continue
                if (pos_a % max(1, args.stride)) != 0:
                    continue

                total += 1
                tok_a = int(s.attack_ids[pos_a])

                pos_c = s.clean_start2i.get(cs, -1)
                if pos_c == -1:
                    extra += 1
                    # still count green on attack side (it enters z)
                    uid_a = int(firstn_ids[tok_a].item())
                    prefix_a = s.attack_text[:cs]
                    fp_a = partitioner.fingerprint(prefix_a, seed_w)
                    green_a_cnt += int(partitioner.is_green(uid_a, fp_a))
                    continue

                aligned += 1
                tok_c = int(s.clean_ids[pos_c])

                uid_a = int(firstn_ids[tok_a].item())
                uid_c = int(firstn_ids[tok_c].item())

                prefix_a = s.attack_text[:cs]
                prefix_c = s.clean_text[:cs]
                fp_a = partitioner.fingerprint(prefix_a, seed_w)
                fp_c = partitioner.fingerprint(prefix_c, seed_w)

                ga = int(partitioner.is_green(uid_a, fp_a))
                gc = int(partitioner.is_green(uid_c, fp_c))
                green_a_cnt += ga
                green_c_cnt += gc

                if uid_a != uid_c:
                    uid_change += 1
                    continue

                same_uid += 1
                # compare same uid under clean vs attack prefix (flip)
                g_under_clean = int(partitioner.is_green(uid_a, fp_c))
                if ga != g_under_clean:
                    prf_flip += 1

        # compute rates
        extra_rate = extra / max(total, 1)
        uid_change_rate_aligned = uid_change / max(aligned, 1)
        prf_flip_rate_same_uid = prf_flip / max(same_uid, 1)

        p_attack = green_a_cnt / max(total, 1)
        p_clean = green_c_cnt / max(aligned, 1) if aligned > 0 else 0.0

        row = {
            "seed_window_chars": seed_w,
            "m_bits": m_bits,
            "target_anchors": targ,
            "n_bytes": n_bytes,
            "k_choices": "|".join(map(str, k_choices)),
            "normalize_whitespace": norm_ws,

            "total_traced": total,
            "extra_tokens": extra,
            "extra_rate": extra_rate,
            "aligned_tokens": aligned,
            "uid_change_aligned": uid_change,
            "uid_change_rate_aligned": uid_change_rate_aligned,
            "same_uid_tokens": same_uid,
            "prf_flip_same_uid": prf_flip,
            "prf_flip_rate_same_uid": prf_flip_rate_same_uid,

            "p_attack_green": p_attack,
            "p_clean_green_aligned": p_clean,
        }

        # optional: “区分度/随机性”粗检：不同上下文下同 uid 颜色一致率（理想≈0.5）
        if args.corr_pairs > 0:
            # sample contexts (prefix strings) from cached attacked texts
            contexts: List[str] = []
            for s in cached:
                # pick a few random token starts
                for _ in range(2):
                    j = random.randrange(0, min(len(s.attack_off), 200))
                    cs = s.attack_off[j][0]
                    if cs > 0:
                        contexts.append(s.attack_text[:cs])
            if len(contexts) < 2:
                corr = float("nan")
            else:
                # sample uids from uniq firstn ids
                uids = torch.unique(firstn_ids).tolist()
                rnd = random.Random(0)
                pick_uids = [int(uids[rnd.randrange(len(uids))]) for _ in range(args.corr_uids)]

                same = 0
                tot = 0
                for _ in range(args.corr_pairs):
                    a = contexts[rnd.randrange(len(contexts))]
                    b = contexts[rnd.randrange(len(contexts))]
                    fp1 = partitioner.fingerprint(a, seed_w)
                    fp2 = partitioner.fingerprint(b, seed_w)
                    for uid in pick_uids:
                        g1 = int(partitioner.is_green(uid, fp1))
                        g2 = int(partitioner.is_green(uid, fp2))
                        same += int(g1 == g2)
                        tot += 1
                corr = same / max(tot, 1)

            row["context_green_agreement_rate"] = corr

        results.append(row)
        print(f"[done] seed_w={seed_w} m={m_bits} anchors={targ} n_bytes={n_bytes} "
              f"flip={prf_flip_rate_same_uid:.4f} uidchg={uid_change_rate_aligned:.4f} extra={extra_rate:.4f}")

    out_csv = out_dir / "v6_prf_sweep_results.tsv"
    pd.DataFrame(results).sort_values(
        ["prf_flip_rate_same_uid","uid_change_rate_aligned","extra_rate"]
    ).to_csv(out_csv, sep="\t", index=False)
    print(f"\nWrote: {out_csv}")

if __name__ == "__main__":
    main()
