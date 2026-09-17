#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from __future__ import annotations
import argparse
import importlib
import importlib.util
import inspect
import json
import os
import random
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import pandas as pd
import torch
from transformers import AutoTokenizer


# ---------------- CLI ----------------
def parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser()
    ap.add_argument("--run_meta", required=True)
    ap.add_argument("--kgw_config", required=True)
    ap.add_argument("--input_csv", required=True)
    ap.add_argument("--attack_ratio", type=float, default=0.02)
    ap.add_argument("--attack_seed", type=int, default=0)
    ap.add_argument("--indices", default="0,1,2")
    ap.add_argument("--max_tokens", type=int, default=200)
    ap.add_argument("--stride", type=int, default=1)
    ap.add_argument("--device", default="cpu")
    ap.add_argument("--out_dir", required=True)
    ap.add_argument("--add_special_tokens", action="store_true", help="force add_special_tokens=True")
    return ap.parse_args()


def load_json(path: str) -> Dict[str, Any]:
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


# ---------------- attack ----------------
def apply_char_attack_replace_X(full_text: str, prompt: str, ratio: float, seed: int) -> Tuple[str, List[int]]:
    """
    Replace k chars in continuation with 'X'. Length unchanged.
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


# ---------------- tokenizer offsets ----------------
def tokenize_with_offsets(tok, text: str, add_special_tokens: bool, device: str) -> Tuple[torch.Tensor, List[Tuple[int, int]]]:
    enc = tok(
        text,
        add_special_tokens=add_special_tokens,
        return_offsets_mapping=True,
    )
    ids = torch.tensor(enc["input_ids"], device=device)
    offsets = enc.get("offset_mapping", None)
    if offsets is None:
        raise RuntimeError("offset_mapping unavailable. Please use a fast tokenizer.")
    offsets = [(int(a), int(b)) for (a, b) in offsets]
    return ids, offsets


# ---------------- KGW utils loader ----------------
class KGWGreenlistProvider:
    def __init__(self, obj: Any):
        self.obj = obj

    def get_greenlist_ids(self, prefix_ids: torch.Tensor) -> List[int]:
        out = self.obj.get_greenlist_ids(prefix_ids)
        # normalize to python list[int]
        if isinstance(out, torch.Tensor):
            return [int(x) for x in out.detach().cpu().tolist()]
        if isinstance(out, (list, tuple)):
            return [int(x) for x in out]
        # some impl returns set/np array
        try:
            return [int(x) for x in list(out)]
        except Exception as e:
            raise TypeError(f"Unexpected greenlist type: {type(out)}") from e


def _try_import_module_names() -> Optional[Any]:
    candidates = [
        # common
        "watermark.kgw.kgw_utils",
        "watermark.kgw.utils",
        "kgw.kgw_utils",
        "kgw.utils",
        # if your code ever used MarkLLM
        "MarkLLM.watermark.kgw.kgw_utils",
        "MarkLLM.watermark.kgw.utils",
    ]
    for name in candidates:
        try:
            return importlib.import_module(name)
        except Exception:
            continue
    return None


def _find_python_file_with_symbol(root: Path, symbol: str) -> Optional[Path]:
    for dirpath, _, filenames in os.walk(root):
        for fn in filenames:
            if not fn.endswith(".py"):
                continue
            p = Path(dirpath) / fn
            try:
                txt = p.read_text(encoding="utf-8", errors="ignore")
            except Exception:
                continue
            if symbol in txt and "get_greenlist_ids" in txt:
                return p
    return None


def _load_module_from_file(py_path: Path) -> Any:
    spec = importlib.util.spec_from_file_location(py_path.stem, py_path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Cannot load module from {py_path}")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)  # type: ignore
    return mod


def build_kgw_utils(root: Path, tokenizer, kgw_cfg: Dict[str, Any], device: str) -> KGWGreenlistProvider:
    """
    Try to build the exact kgw_utils object your repo already uses.
    Expectation: it has a method get_greenlist_ids(prefix_ids: Tensor)->(list/tensor of ids)
    """
    mod = _try_import_module_names()
    if mod is None:
        hit = _find_python_file_with_symbol(root, "get_greenlist_ids")
        if hit is None:
            raise ModuleNotFoundError(
                "Cannot find KGW implementation with get_greenlist_ids.\n"
                "Hint: locate the file used by scripts/analyze_kgw_prf_overlap.py and make sure it's under repo root."
            )
        mod = _load_module_from_file(hit)

    # if module itself exposes an already-built object (rare)
    for attr in ["kgw_utils", "KGW_UTILS", "utils"]:
        if hasattr(mod, attr):
            obj = getattr(mod, attr)
            if hasattr(obj, "get_greenlist_ids"):
                return KGWGreenlistProvider(obj)

    # try typical class names
    class_candidates = ["KGWUtils", "KGW", "KGWWatermark", "KGWPartitioner", "KGWHelper"]
    for cname in class_candidates:
        if hasattr(mod, cname):
            cls = getattr(mod, cname)
            if inspect.isclass(cls):
                # try a few constructor patterns
                for ctor in [
                    lambda: cls(tokenizer, kgw_cfg),
                    lambda: cls(tokenizer=tokenizer, config=kgw_cfg),
                    lambda: cls(tokenizer=tokenizer, **kgw_cfg),
                    lambda: cls(config=kgw_cfg, tokenizer=tokenizer),
                    lambda: cls(kgw_cfg, tokenizer),
                ]:
                    try:
                        obj = ctor()
                        if hasattr(obj, "to"):
                            try:
                                obj = obj.to(device)
                            except Exception:
                                pass
                        if hasattr(obj, "get_greenlist_ids"):
                            return KGWGreenlistProvider(obj)
                    except Exception:
                        continue

    # try a plain function get_greenlist_ids(prefix_ids, ...)
    if hasattr(mod, "get_greenlist_ids") and callable(getattr(mod, "get_greenlist_ids")):
        fn = getattr(mod, "get_greenlist_ids")

        class _Wrap:
            def __init__(self, fn, tokenizer, cfg):
                self.fn = fn
                self.tokenizer = tokenizer
                self.cfg = cfg

            def get_greenlist_ids(self, prefix_ids: torch.Tensor):
                # try signature variants
                for call in [
                    lambda: self.fn(prefix_ids),
                    lambda: self.fn(prefix_ids, self.tokenizer),
                    lambda: self.fn(prefix_ids, self.cfg),
                    lambda: self.fn(prefix_ids, self.tokenizer, self.cfg),
                ]:
                    try:
                        return call()
                    except Exception:
                        continue
                raise RuntimeError("Found get_greenlist_ids but cannot call it with known signatures.")

        return KGWGreenlistProvider(_Wrap(fn, tokenizer, kgw_cfg))

    # Fallback: directly build KGWUtils from MarkLLM.watermark.kgw.kgw if present
    try:
        from MarkLLM.watermark.kgw.kgw import KGWUtils  # type: ignore

        class SimpleKGWConfig:
            def __init__(self, cfg: Dict[str, Any], vocab_size: int, device: str) -> None:
                self.gamma = float(cfg.get("gamma", 0.5))
                self.delta = float(cfg.get("delta", 2.0))
                self.hash_key = int(cfg.get("hash_key", 15485863))
                self.z_threshold = float(cfg.get("z_threshold", 4.0))
                self.prefix_length = int(cfg.get("prefix_length", 4))
                self.f_scheme = cfg.get("f_scheme", "additive")
                self.window_scheme = cfg.get("window_scheme", "left")
                self.vocab_size = int(vocab_size)
                self.device = device
                self.gen_kwargs = {}
                self.generation_model = None
                self.generation_tokenizer = None

        cfg_obj = SimpleKGWConfig(kgw_cfg, vocab_size=len(tokenizer), device=device)
        obj = KGWUtils(cfg_obj)
        # ensure rng/device align with prefixes
        try:
            obj.rng = torch.Generator(device=device)
            obj.config.device = device
        except Exception:
            pass
        return KGWGreenlistProvider(obj)
    except Exception:
        pass

    raise ModuleNotFoundError(
        "Found a candidate KGW module, but couldn't build a kgw_utils object with get_greenlist_ids."
    )


# ---------------- trace core ----------------
def is_green_token(kgw: KGWGreenlistProvider, prefix_ids: torch.Tensor, token_id: int) -> int:
    gl = kgw.get_greenlist_ids(prefix_ids)
    return 1 if token_id in set(gl) else 0


def main() -> None:
    args = parse_args()
    root = Path(__file__).resolve().parents[1]
    if str(root) not in sys.path:
        sys.path.insert(0, str(root))

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    run_meta = load_json(args.run_meta)
    kgw_cfg = load_json(args.kgw_config)

    tok = AutoTokenizer.from_pretrained(run_meta["model"])
    add_special_tokens = bool(args.add_special_tokens) or bool(kgw_cfg.get("add_special_tokens", False))

    kgw = build_kgw_utils(root, tok, kgw_cfg, args.device)

    df = pd.read_csv(args.input_csv)
    indices = [int(x.strip()) for x in args.indices.split(",") if x.strip()]

    for ridx in indices:
        row = df.iloc[ridx]
        full = str(row["full_text"])
        prompt = str(row.get("prompt_text", ""))

        attacked, attacked_positions = apply_char_attack_replace_X(
            full, prompt, args.attack_ratio, seed=args.attack_seed + ridx
        )

        ids_c, off_c = tokenize_with_offsets(tok, full, add_special_tokens=add_special_tokens, device=args.device)
        ids_a, off_a = tokenize_with_offsets(tok, attacked, add_special_tokens=add_special_tokens, device=args.device)

        # map clean char_start -> clean token index
        clean_start2i: Dict[int, int] = {}
        for i, (s, e) in enumerate(off_c):
            if s not in clean_start2i:
                clean_start2i[s] = i

        prompt_chars = len(prompt) if full.startswith(prompt) else 0

        rows: List[Dict[str, Any]] = []
        traced = 0

        # summary counters
        total = 0
        extra = 0
        aligned = 0
        token_changed = 0
        same_token = 0

        # PRF flip: two useful口径
        # (1) same-token flip: token_id same at same char_start, but is_green differs (clean prefix vs attack prefix)
        prf_flip_same_token = 0
        # (2) attacked-token flip: for the attacked token itself, compare is_green under clean-prefix vs attack-prefix
        prf_flip_attacked_token = 0

        # “clean绿->attack红”损失
        loss_total = 0
        loss_due_to_token_change = 0
        loss_due_to_prf_flip_same_token = 0
        loss_due_to_other = 0

        for pos_a, (s_a, e_a) in enumerate(off_a):
            if traced >= args.max_tokens:
                break
            if s_a < prompt_chars:
                continue
            if (pos_a % max(1, args.stride)) != 0:
                continue

            traced += 1
            total += 1

            tok_id_a = int(ids_a[pos_a].item())
            prefix_ids_a = ids_a[:pos_a]

            # align clean token by same char_start
            pos_c = clean_start2i.get(s_a, -1)
            if pos_c == -1:
                extra += 1
                rows.append(
                    {
                        "pos_a": pos_a,
                        "char_start": s_a,
                        "char_end": e_a,
                        "token_id_a": tok_id_a,
                        "token_str_a": tok.convert_ids_to_tokens([tok_id_a])[0],
                        "pos_c": -1,
                        "token_id_c": -1,
                        "token_str_c": "",
                        "same_token": 0,
                        "is_green_a": is_green_token(kgw, prefix_ids_a, tok_id_a),
                        "is_green_c": -1,
                        "prf_flip_same_token": 0,
                        "prf_flip_attacked_token": 0,  # no aligned clean prefix
                        "reason": "no_clean_token_at_same_char_start",
                    }
                )
                continue

            aligned += 1
            tok_id_c = int(ids_c[pos_c].item())
            prefix_ids_c = ids_c[:pos_c]

            g_a = is_green_token(kgw, prefix_ids_a, tok_id_a)
            g_c = is_green_token(kgw, prefix_ids_c, tok_id_c)

            st = int(tok_id_a == tok_id_c)
            if st:
                same_token += 1
            else:
                token_changed += 1

            # (2) attacked-token flip under clean-prefix vs attack-prefix (更像 v6 的 same_uid flip 概念)
            g_a_under_cleanprefix = is_green_token(kgw, prefix_ids_c, tok_id_a)
            flip_att = int(g_a_under_cleanprefix != g_a)
            prf_flip_attacked_token += flip_att

            # (1) same-token flip (严格口径)
            flip_same = 0
            if st:
                flip_same = int(g_a != g_c)
                prf_flip_same_token += flip_same

            # loss: clean green but attack red (对齐位置)
            if g_c == 1 and g_a == 0:
                loss_total += 1
                if not st:
                    loss_due_to_token_change += 1
                elif flip_same:
                    loss_due_to_prf_flip_same_token += 1
                else:
                    loss_due_to_other += 1

            rows.append(
                {
                    "pos_a": pos_a,
                    "char_start": s_a,
                    "char_end": e_a,
                    "token_id_a": tok_id_a,
                    "token_str_a": tok.convert_ids_to_tokens([tok_id_a])[0],
                    "pos_c": pos_c,
                    "token_id_c": tok_id_c,
                    "token_str_c": tok.convert_ids_to_tokens([tok_id_c])[0],
                    "same_token": st,
                    "is_green_a": g_a,
                    "is_green_c": g_c,
                    "is_green_a_under_cleanprefix": g_a_under_cleanprefix,
                    "prf_flip_same_token": flip_same,
                    "prf_flip_attacked_token": flip_att,
                    "reason": ("token_id_changed_at_same_char_start" if not st else "aligned_same_token"),
                }
            )

        out_join = out_dir / f"row_{ridx}_kgw_join_trace.tsv"
        pd.DataFrame(rows).to_csv(out_join, sep="\t", index=False)

        # rates
        extra_rate = extra / max(total, 1)
        token_changed_rate_aligned = token_changed / max(aligned, 1)
        prf_flip_rate_attacked_token = prf_flip_attacked_token / max(aligned, 1)
        prf_flip_rate_same_token = prf_flip_same_token / max(same_token, 1)

        summary = {
            "row_index": ridx,
            "attack_ratio": args.attack_ratio,
            "total_traced": total,
            "extra_tokens": extra,
            "extra_rate": extra_rate,
            "aligned_tokens": aligned,
            "token_changed_at_same_char_start": token_changed,
            "token_changed_rate_aligned": token_changed_rate_aligned,
            "same_token_aligned": same_token,
            "prf_flip_attacked_token_count": prf_flip_attacked_token,
            "prf_flip_rate_attacked_token_aligned": prf_flip_rate_attacked_token,
            "prf_flip_same_token_count": prf_flip_same_token,
            "prf_flip_rate_same_token": prf_flip_rate_same_token,
            "loss_clean_green_to_attack_red": loss_total,
            "loss_due_to_token_change": loss_due_to_token_change,
            "loss_due_to_prf_flip_same_token": loss_due_to_prf_flip_same_token,
            "loss_due_to_other": loss_due_to_other,
            "attacked_char_positions_head": attacked_positions[:40],
        }

        out_sum = out_dir / f"row_{ridx}_kgw_summary.json"
        out_sum.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")

        print(f"\n[row {ridx}] wrote:\n  {out_join}\n  {out_sum}")
        print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
