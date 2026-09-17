#!/usr/bin/env python3
# scripts/sweep_v6_prf_variants2.py
from __future__ import annotations

import argparse, json, random, sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Tuple, Callable, Optional

import numpy as np
import pandas as pd
import torch
from transformers import AutoTokenizer
import importlib
from tqdm.auto import tqdm

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

# ---- keyed hash helpers ----
def blake_bytes(key: bytes, data: bytes, out_bytes: int) -> bytes:
    import hashlib
    return hashlib.blake2b(data, key=key, digest_size=out_bytes).digest()

def blake64(key: bytes, data: bytes) -> int:
    b = blake_bytes(key, data, 8)
    return int.from_bytes(b, "little", signed=False)

# ---- simhash from char ngrams (supports skip-bigrams) ----
def fp_ngram_simhash(
    *,
    tail: str,
    key: bytes,
    m_bits: int,
    ngrams: Tuple[int, ...] = (3, 4),
    use_skip_bi: bool = False,
) -> bytes:
    m_bytes = m_bits // 8
    L = len(tail)
    feats: List[int] = []

    # normal n-grams
    for n in ngrams:
        if L < n:
            continue
        for i in range(0, L - n + 1):
            gram = tail[i:i+n].encode("utf-8", errors="ignore")
            feats.append(blake64(key, b"g" + n.to_bytes(1,"little") + gram))

    # skip-bigrams: (i, i+2) helps insertion/deletion a bit
    if use_skip_bi and L >= 3:
        for i in range(0, L - 2):
            gram = (tail[i] + "\x1f" + tail[i+2]).encode("utf-8", errors="ignore")
            feats.append(blake64(key, b"s2" + gram))

    if not feats:
        return blake_bytes(key, b"empty" + tail.encode("utf-8", errors="ignore"), m_bytes)

    acc = np.zeros(m_bits, dtype=np.int32)
    tie = blake_bytes(key, b"tie" + tail.encode("utf-8", errors="ignore"), m_bytes)
    for h in feats:
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

def fp_bit_majority(fps: List[bytes], key: bytes, tie_tag: bytes = b"fpmaj_tie") -> bytes:
    assert fps, "empty fps"
    m_bytes = len(fps[0])
    for f in fps:
        assert len(f) == m_bytes
    out = bytearray(m_bytes)
    # tie bits (deterministic)
    tie = blake_bytes(key, tie_tag + b"".join(fps), m_bytes)
    for bi in range(m_bytes * 8):
        ones = 0
        for f in fps:
            ones += (f[bi >> 3] >> (bi & 7)) & 1
        if ones * 2 > len(fps):
            bit = 1
        elif ones * 2 < len(fps):
            bit = 0
        else:
            bit = (tie[bi >> 3] >> (bi & 7)) & 1
        if bit:
            out[bi >> 3] |= (1 << (bi & 7))
    return bytes(out)

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
    ap.add_argument("--out_tsv", default="v6_prf_variant_sweep2.tsv")

    # discriminability check size
    ap.add_argument("--agree_trials", type=int, default=120)
    ap.add_argument("--agree_uids", type=int, default=256)
    ap.add_argument("--num_workers", type=int, default=1, help="parallelize variants (threads)")
    ap.add_argument("--progress", action="store_true", help="show per-variant progress bar (disabled when num_workers>1)")
    ap.add_argument("--heartbeat", type=int, default=0, help="log every N processed tokens inside a variant (0=off)")

    args = ap.parse_args()

    run_meta = load_json(args.run_meta)
    cfg = load_json(args.v6_config)

    tok = AutoTokenizer.from_pretrained(run_meta["model"])
    add_special = bool(cfg.get("add_special_tokens", True))
    seed_w = int(cfg.get("seed_window_chars", 18))  # fixed
    m_bits = int(cfg.get("m_bits", 256))
    key = _to_bytes(cfg.get("hash_key", 15485863))

    partitioner = build_partitioner(cfg)
    vocab = TokenByteVocabV6.from_tokenizer(tok, skip_markers=True).to(args.device)
    firstn = vocab.first_n_id(torch.device(args.device), int(cfg.get("n_bytes", 3))).to("cpu")

    # is_green helper (use repo if possible)
    def is_green(uid: int, fp: bytes) -> int:
        return int(partitioner.is_green(uid, fp))

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

    # contexts for agreement test
    ctxs: List[str] = []
    for s in cached:
        for j in range(min(120, len(s.attack_off))):
            cs = s.attack_off[j][0]
            if cs > s.prompt_chars:
                ctxs.append(s.attack[:cs][-seed_w:])

    uniq_uids = torch.unique(firstn).tolist()

    def _log(msg: str):
        tqdm.write(msg)
        sys.stdout.flush()

    def agreement(green_fn: Callable[[int,str], int]) -> float:
        """
        Fresh RNG per call to keep results deterministic even when threaded.
        """
        if len(ctxs) < 2 or len(uniq_uids) < 1:
            return float("nan")
        rng = random.Random(0)
        pick_uids = [int(uniq_uids[rng.randrange(len(uniq_uids))]) for _ in range(args.agree_uids)]
        same = 0
        tot = 0
        for _ in range(args.agree_trials):
            a = ctxs[rng.randrange(len(ctxs))]
            b = ctxs[rng.randrange(len(ctxs))]
            for uid in pick_uids:
                same += int(green_fn(uid, a) == green_fn(uid, b))
                tot += 1
        return same / max(tot, 1)

    # ---- define variants as green_fn(uid, tail)->0/1 with optional fp_fn(tail) for hd ----
    Variant = Tuple[str, Callable[[int,str], int], Optional[Callable[[str], bytes]]]
    variants: List[Variant] = []

    # baseline
    def fp_base(tail: str) -> bytes:
        return partitioner.fingerprint(tail, seed_w)
    def g_base(uid: int, tail: str) -> int:
        return is_green(uid, fp_base(tail))
    variants.append(("baseline_v6", g_base, fp_base))

    # single-view ngram simhash families
    for name, ngrams, skip in [
        ("simhash_34", (3,4), False),
        ("simhash_234", (2,3,4), False),
        ("simhash_1234", (1,2,3,4), False),
        ("simhash_34_skip", (3,4), True),
        ("simhash_234_skip", (2,3,4), True),
    ]:
        def make_fp(ngrams_val, skip_val):
            return lambda tail: fp_ngram_simhash(tail=tail, key=key, m_bits=m_bits, ngrams=ngrams_val, use_skip_bi=skip_val)
        fp = make_fp(ngrams, skip)
        def make_g(fp_fn):
            return lambda uid, tail: is_green(uid, fp_fn(tail))
        variants.append((name, make_g(fp), fp))

    # shift-bagging: fp-majority of 3 shifted tails
    for base_name, ngrams, skip in [
        ("shiftmaj_34", (3,4), False),
        ("shiftmaj_234", (2,3,4), False),
        ("shiftmaj_34_skip", (3,4), True),
    ]:
        fp_single = lambda t, ng=ngrams, sk=skip: fp_ngram_simhash(tail=t, key=key, m_bits=m_bits, ngrams=ng, use_skip_bi=sk)
        def make_fpmaj(fp_s):
            def _fp(tail: str) -> bytes:
                a = tail
                b = tail[:-1] if len(tail) >= 1 else tail
                c = tail[1:] if len(tail) >= 1 else tail
                return fp_bit_majority([fp_s(a), fp_s(b), fp_s(c)], key=key, tie_tag=b"maj3")
            return _fp
        fpmaj = make_fpmaj(fp_single)
        gmaj = lambda uid, tail, fp_fn=fpmaj: is_green(uid, fp_fn(tail))
        variants.append((base_name, gmaj, fpmaj))

    # green-vote: vote on green bits across 3 shifted tails (stronger anti-flip)
    for base_name, ngrams, skip in [
        ("greenvote3_34", (3,4), False),
        ("greenvote3_234", (2,3,4), False),
    ]:
        fp_s = lambda t, ng=ngrams, sk=skip: fp_ngram_simhash(tail=t, key=key, m_bits=m_bits, ngrams=ng, use_skip_bi=sk)
        def make_gvote(fp_single):
            def _g(uid: int, tail: str) -> int:
                a = tail
                b = tail[:-1] if len(tail) >= 1 else tail
                c = tail[1:] if len(tail) >= 1 else tail
                gs = [
                    is_green(uid, fp_single(a)),
                    is_green(uid, fp_single(b)),
                    is_green(uid, fp_single(c)),
                ]
                return 1 if sum(gs) >= 2 else 0
            return _g
        gvote = make_gvote(fp_s)
        variants.append((base_name, gvote, None))  # fp_hd not applicable

    def eval_variant(name: str, green_fn, fp_fn, *, show_prog: bool = False, position: Optional[int] = None):
        total = extra = aligned = uidchg = same_uid = flip = 0
        hd_sum = 0
        pbar = None
        if show_prog:
            est_total = 0
            for s in cached:
                est_total += min(len(s.attack_off), args.max_tokens)
            pbar = tqdm(total=est_total, desc=name, leave=True, position=position, dynamic_ncols=True)
            pbar.update(0)  # force display
        _log(f"[start] {name}")

        for s in cached:
            for pos_a, (cs, ce) in enumerate(s.attack_off):
                if total >= args.max_tokens:
                    break
                if cs < s.prompt_chars:
                    continue
                if (pos_a % max(1, args.stride)) != 0:
                    continue

                total += 1
                if pbar is not None:
                    pbar.update(1)
                if args.heartbeat > 0 and (total % args.heartbeat == 0):
                    _log(f"[heartbeat] {name} total={total}")
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

                ga = green_fn(uid_a, tail_a)
                g_under_clean = green_fn(uid_a, tail_c)
                if ga != g_under_clean:
                    flip += 1

                if fp_fn is not None:
                    fpa = fp_fn(tail_a)
                    fpc = fp_fn(tail_c)
                    hd_sum += hd_bits(fpa, fpc)

        if pbar is not None:
            pbar.close()

        row = {
            "variant": name,
            "seed_window_chars": seed_w,
            "total": total,
            "extra_rate": extra / max(total, 1),
            "uid_change_rate_aligned": uidchg / max(aligned, 1),
            "prf_flip_rate_same_uid": flip / max(same_uid, 1),
            "context_green_agreement": agreement(green_fn),
        }
        if fp_fn is not None:
            row["fp_hd_bits_mean_same_uid"] = hd_sum / max(same_uid, 1)
        else:
            row["fp_hd_bits_mean_same_uid"] = float("nan")
        _log(
            f"[done] {name} total={total} flip={row['prf_flip_rate_same_uid']:.4f} "
            f"agree={row['context_green_agreement']:.4f}"
        )
        return row

    rows = []
    progress_enabled = args.progress
    _log(f"Running {len(variants)} variants sequentially (per-variant workers={args.num_workers}) ...")
    for idx, (name, green_fn, fp_fn) in enumerate(variants):
        rows.append(
            eval_variant(
                name,
                green_fn,
                fp_fn,
                show_prog=progress_enabled,
                position=idx if progress_enabled else None,
            )
        )

    out = pd.DataFrame(rows).sort_values(["prf_flip_rate_same_uid", "context_green_agreement"])
    out.to_csv(args.out_tsv, sep="\t", index=False)
    _log(f"[ok] wrote {args.out_tsv}")

if __name__ == "__main__":
    main()
