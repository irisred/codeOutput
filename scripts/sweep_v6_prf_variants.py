#!/usr/bin/env python3
# scripts/sweep_v6_prf_variants.py
from __future__ import annotations

import argparse, json, random, sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Tuple

import numpy as np
import pandas as pd
import torch
from transformers import AutoTokenizer

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

# ---- import your v6 pieces (same pattern you used before) ----
import importlib
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
    raise ModuleNotFoundError(f"Cannot import V6 modules. last={last}")

TokenByteVocabV6, RobustPartitioner = _try_import_v6()


def load_json(p: str) -> Dict[str, Any]:
    with open(p, "r", encoding="utf-8") as f:
        return json.load(f)

def _to_bytes(x) -> bytes:
    if isinstance(x, bytes):
        return x
    if isinstance(x, int):
        return int(x).to_bytes(16, "little", signed=False)
    s = str(x).strip()
    if s.startswith("0x"):
        try:
            return int(s, 16).to_bytes(16, "little", signed=False)
        except Exception:
            pass
    try:
        return bytes.fromhex(s)
    except Exception:
        return s.encode("utf-8")

def apply_attack_replace_X(full: str, prompt: str, ratio: float, seed: int) -> str:
    if ratio <= 0:
        return full
    rng = random.Random(seed)
    start = len(prompt) if full.startswith(prompt) else 0
    cont = full[start:]
    n = len(cont)
    if n <= 0:
        return full
    k = max(1, int(round(n * ratio)))
    k = min(k, n)
    idxs = rng.sample(range(n), k)
    cont_list = list(cont)
    for i in idxs:
        cont_list[i] = "X"
    return full[:start] + "".join(cont_list)

def tokenize_with_offsets(tok, text: str, add_special_tokens: bool):
    enc = tok(text, add_special_tokens=add_special_tokens, return_offsets_mapping=True)
    if "offset_mapping" not in enc or enc["offset_mapping"] is None:
        raise RuntimeError("Need a fast tokenizer with offset_mapping.")
    ids = enc["input_ids"]
    offs = [(int(a), int(b)) for a, b in enc["offset_mapping"]]
    return ids, offs

def hd_bits(a: bytes, b: bytes) -> int:
    return sum((x ^ y).bit_count() for x, y in zip(a, b))


# ---------------- PRF variants (fingerprint on tail string) ----------------
def blake64(key: bytes, data: bytes) -> int:
    import hashlib
    h = hashlib.blake2b(data, key=key, digest_size=8).digest()
    return int.from_bytes(h, "little", signed=False)

def blake_bytes(key: bytes, data: bytes, out_bytes: int) -> bytes:
    import hashlib
    return hashlib.blake2b(data, key=key, digest_size=out_bytes).digest()

def fp_ngram_simhash(
    *,
    tail: str,
    key: bytes,
    m_bits: int,
    ngrams: Tuple[int, ...] = (3, 4),
    topk: int = 0,         # 0 => use all
) -> bytes:
    """
    Build simhash fingerprint from character n-grams of tail.
    - robust to small edits because only O(n) grams affected
    """
    m_bytes = m_bits // 8
    tail_bytes = tail.encode("utf-8", errors="ignore")

    feats: List[int] = []
    L = len(tail)
    for n in ngrams:
        if L < n:
            continue
        for i in range(0, L - n + 1):
            gram = tail[i:i+n].encode("utf-8", errors="ignore")
            feats.append(blake64(key, b"g" + n.to_bytes(1,"little") + gram))
    if not feats:
        # fallback: hash whole tail
        return blake_bytes(key, b"empty" + tail_bytes, m_bytes)

    if topk and topk > 0 and len(feats) > topk:
        feats = sorted(feats)[:topk]   # smallest hashes = stable-ish selection

    # simhash accumulator
    acc = np.zeros(m_bits, dtype=np.int32)
    for h in feats:
        v = blake_bytes(key, b"v" + h.to_bytes(8,"little"), m_bytes)
        for bi in range(m_bits):
            bit = (v[bi >> 3] >> (bi & 7)) & 1
            acc[bi] += 1 if bit else -1

    # produce bits
    out = bytearray(m_bytes)
    # deterministic tie-break
    tie = blake_bytes(key, b"tie" + tail_bytes, m_bytes)
    for bi in range(m_bits):
        if acc[bi] > 0:
            bit = 1
        elif acc[bi] < 0:
            bit = 0
        else:
            bit = (tie[bi >> 3] >> (bi & 7)) & 1
        if bit:
            out[bi >> 3] |= (1 << (bi & 7))
    return bytes(out)

def fp_minimizer_simhash(
    *,
    tail: str,
    key: bytes,
    m_bits: int,
    n: int = 4,
    win: int = 4,
    topk: int = 0,
) -> bytes:
    """
    Winnowing/minimizer on n-gram hashes, then simhash.
    Better for insert/delete (shift) than pure positional anchors.
    """
    m_bytes = m_bits // 8
    L = len(tail)
    if L < n:
        return fp_ngram_simhash(tail=tail, key=key, m_bits=m_bits, ngrams=(1,2,3), topk=0)

    hs = []
    for i in range(0, L - n + 1):
        gram = tail[i:i+n].encode("utf-8", errors="ignore")
        hs.append((blake64(key, b"m" + gram), i))

    # minimizers per window
    mins = []
    w = max(1, win)
    for i in range(0, len(hs)):
        j = min(i + w, len(hs))
        hmin, pos = min(hs[i:j], key=lambda x: x[0])
        mins.append(hmin)
    # dedup
    mins = sorted(set(mins))

    if topk and topk > 0 and len(mins) > topk:
        mins = mins[:topk]

    if not mins:
        return blake_bytes(key, b"mmempty" + tail.encode("utf-8", errors="ignore"), m_bytes)

    acc = np.zeros(m_bits, dtype=np.int32)
    tie = blake_bytes(key, b"tie" + tail.encode("utf-8", errors="ignore"), m_bytes)
    for h in mins:
        v = blake_bytes(key, b"v" + h.to_bytes(8,"little"), m_bytes)
        for bi in range(m_bits):
            bit = (v[bi >> 3] >> (bi & 7)) & 1
            acc[bi] += 1 if bit else -1

    out = bytearray(m_bytes)
    for bi in range(m_bits):
        if acc[bi] > 0:
            bit = 1
        elif acc[bi] < 0:
            bit = 0
        else:
            bit = (tie[bi >> 3] >> (bi & 7)) & 1
        if bit:
            out[bi >> 3] |= (1 << (bi & 7))
    return bytes(out)


# ---------------- core sweep ----------------
@dataclass
class Cached:
    ridx: int
    clean: str
    attack: str
    prompt: str
    clean_ids: List[int]
    clean_off: List[Tuple[int,int]]
    attack_ids: List[int]
    attack_off: List[Tuple[int,int]]
    start2i: Dict[int,int]
    prompt_chars: int

def build_partitioner(cfg: Dict[str, Any]) -> Any:
    return RobustPartitioner(
        master_key=_to_bytes(cfg.get("hash_key", 15485863)),
        m_bits=int(cfg.get("m_bits", 256)),
        target_anchors=int(cfg.get("target_anchors", 96)),
        k_choices=tuple(cfg.get("k_choices", [4,5,6])),
        normalize_whitespace=bool(cfg.get("normalize_whitespace", True)),
    )

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run_meta", required=True)
    ap.add_argument("--v6_config", required=True)
    ap.add_argument("--input_csv", required=True)
    ap.add_argument("--indices", default="0,1,2")
    ap.add_argument("--attack_ratio", type=float, default=0.02)
    ap.add_argument("--attack_seed", type=int, default=0)
    ap.add_argument("--max_tokens", type=int, default=200)
    ap.add_argument("--stride", type=int, default=1)
    ap.add_argument("--device", default="cpu")
    ap.add_argument("--out_tsv", default="v6_prf_variant_sweep.tsv")

    # variants knobs
    ap.add_argument("--ngram_topk", default="0,16,32,64")
    ap.add_argument("--min_topk", default="0,16,32,64")
    ap.add_argument("--min_win", default="3,4,5")
    args = ap.parse_args()

    run_meta = load_json(args.run_meta)
    v6_cfg = load_json(args.v6_config)

    tok = AutoTokenizer.from_pretrained(run_meta["model"])
    add_special = bool(v6_cfg.get("add_special_tokens", True))
    seed_w = int(v6_cfg.get("seed_window_chars", 18))  # keep fixed at 18
    m_bits = int(v6_cfg.get("m_bits", 256))
    m_bytes = m_bits // 8
    key = _to_bytes(v6_cfg.get("hash_key", 15485863))

    partitioner = build_partitioner(v6_cfg)

    vocab = TokenByteVocabV6.from_tokenizer(tok, skip_markers=True).to(args.device)
    firstn = vocab.first_n_id(torch.device(args.device), int(v6_cfg.get("n_bytes", 3))).to("cpu")

    df = pd.read_csv(args.input_csv)
    indices = [int(x.strip()) for x in args.indices.split(",") if x.strip()]

    cached: List[Cached] = []
    for ridx in indices:
        row = df.iloc[ridx]
        clean = str(row["full_text"])
        prompt = str(row.get("prompt_text", ""))
        attack = apply_attack_replace_X(clean, prompt, args.attack_ratio, args.attack_seed + ridx)

        c_ids, c_off = tokenize_with_offsets(tok, clean, add_special)
        a_ids, a_off = tokenize_with_offsets(tok, attack, add_special)

        s2i: Dict[int,int] = {}
        for i,(s,e) in enumerate(c_off):
            if s not in s2i:
                s2i[s] = i
        pchars = len(prompt) if clean.startswith(prompt) else 0

        cached.append(Cached(ridx, clean, attack, prompt, c_ids, c_off, a_ids, a_off, s2i, pchars))

    def is_green(uid: int, fp: bytes) -> int:
        # prefer partitioner.is_green if available
        if hasattr(partitioner, "is_green"):
            return int(partitioner.is_green(uid, fp))
        # else compute via token_vector + hd
        w = partitioner.token_vector(uid)
        hdv = hd_bits(w, fp)
        half = m_bits // 2
        if hdv < half:
            return 1
        if hdv > half:
            return 0
        # tie break deterministic
        t = blake64(key, b"tie_uid" + uid.to_bytes(8,"little"))
        return 1 if (t & 1) == 0 else 0

    # collect contexts for discriminability check
    ctxs: List[str] = []
    for s in cached:
        for j in range(min(50, len(s.attack_off))):
            cs = s.attack_off[j][0]
            if cs > s.prompt_chars:
                ctxs.append(s.attack[:cs][-seed_w:])

    uniq_uids = torch.unique(firstn).tolist()
    rng = random.Random(0)

    def context_agreement(fp_fn, trials: int = 200, uids: int = 256) -> float:
        if len(ctxs) < 2 or len(uniq_uids) < 1:
            return float("nan")
        same = 0
        tot = 0
        pick_uids = [int(uniq_uids[rng.randrange(len(uniq_uids))]) for _ in range(uids)]
        for _ in range(trials):
            a = ctxs[rng.randrange(len(ctxs))]
            b = ctxs[rng.randrange(len(ctxs))]
            fpa = fp_fn(a)
            fpb = fp_fn(b)
            for uid in pick_uids:
                same += int(is_green(uid, fpa) == is_green(uid, fpb))
                tot += 1
        return same / max(tot, 1)

    # define variant list
    variants: List[Tuple[str, Any]] = []

    # baseline = current partitioner fingerprint
    def fp_baseline(tail: str) -> bytes:
        return partitioner.fingerprint(tail, seed_w)

    variants.append(("baseline_v6", fp_baseline))

    # ngram simhash variants
    for topk in [int(x) for x in args.ngram_topk.split(",") if x.strip()]:
        def make_fp(topk_val: int):
            return lambda tail: fp_ngram_simhash(
                tail=tail, key=key, m_bits=m_bits, ngrams=(3,4), topk=topk_val
            )
        variants.append((f"ngram_simhash_34_topk{topk}", make_fp(topk)))

    # minimizer simhash variants
    for win in [int(x) for x in args.min_win.split(",") if x.strip()]:
        for topk in [int(x) for x in args.min_topk.split(",") if x.strip()]:
            def make_fp2(win_val: int, topk_val: int):
                return lambda tail: fp_minimizer_simhash(
                    tail=tail, key=key, m_bits=m_bits, n=4, win=win_val, topk=topk_val
                )
            variants.append((f"minimizer_n4_win{win}_topk{topk}", make_fp2(win, topk)))

    rows = []
    for name, fp_fn in variants:
        total = 0
        extra = 0
        aligned = 0
        uidchg = 0
        same_uid = 0
        flip = 0
        hd_sum = 0

        for s in cached:
            # walk attacked tokens
            for pos_a, (cs, ce) in enumerate(s.attack_off):
                if total >= args.max_tokens:
                    break
                if cs < s.prompt_chars:
                    continue
                if (pos_a % max(1, args.stride)) != 0:
                    continue

                total += 1
                tok_a = int(s.attack_ids[pos_a])
                uid_a = int(firstn[tok_a].item())

                pos_c = s.start2i.get(cs, -1)
                if pos_c == -1:
                    extra += 1
                    continue

                aligned += 1
                tok_c = int(s.clean_ids[pos_c])
                uid_c = int(firstn[tok_c].item())
                if uid_a != uid_c:
                    uidchg += 1
                    continue

                same_uid += 1
                tail_a = s.attack[:cs][-seed_w:]
                tail_c = s.clean[:cs][-seed_w:]
                fp_a = fp_fn(tail_a)
                fp_c = fp_fn(tail_c)
                hd_sum += hd_bits(fp_a, fp_c)

                ga = is_green(uid_a, fp_a)
                g_under_clean = is_green(uid_a, fp_c)
                if ga != g_under_clean:
                    flip += 1

        rows.append({
            "variant": name,
            "seed_window_chars": seed_w,
            "total": total,
            "extra_rate": extra / max(total,1),
            "uid_change_rate_aligned": uidchg / max(aligned,1),
            "prf_flip_rate_same_uid": flip / max(same_uid,1),
            "fp_hd_bits_mean_same_uid": hd_sum / max(same_uid,1),
            "context_green_agreement": context_agreement(fp_fn, trials=80, uids=128),
        })
        print(f"[{name}] flip={rows[-1]['prf_flip_rate_same_uid']:.4f} "
              f"uidchg={rows[-1]['uid_change_rate_aligned']:.4f} "
              f"extra={rows[-1]['extra_rate']:.4f} "
              f"agree={rows[-1]['context_green_agreement']:.4f}")

    out = pd.DataFrame(rows).sort_values(["prf_flip_rate_same_uid","uid_change_rate_aligned","extra_rate"])
    out.to_csv(args.out_tsv, sep="\t", index=False)
    print(f"\n[ok] wrote {args.out_tsv}")

if __name__ == "__main__":
    main()
