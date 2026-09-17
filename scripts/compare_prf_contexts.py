#!/usr/bin/env python3
"""Compare generator vs detector PRF contexts for the first N steps."""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import List

import sys

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.append(str(REPO_ROOT))

from MarkLLM.charm_v2.charm_kgw import CharmKGW
from MarkLLM.charm_v2.prf_trace import PRFTracer
from MarkLLM.utils.transformers_config import TransformersConfig


def load_prompt(prompt_arg: str | None, prompt_file: str | None, prompt_id: int, trim: int) -> str:
    if prompt_arg:
        return prompt_arg
    if not prompt_file:
        raise SystemExit("Either --prompt or --prompt-file must be specified.")
    lines: List[str] = []
    with Path(prompt_file).open("r", encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if not line:
                continue
            lines.append(line[:trim] if trim > 0 else line)
    if not lines:
        raise SystemExit(f"No prompts in {prompt_file}")
    idx = min(max(prompt_id, 0), len(lines) - 1)
    return lines[idx]


def build_runtime(model_path: str, device: str, **gen_kwargs) -> TransformersConfig:
    tokenizer = AutoTokenizer.from_pretrained(model_path)
    if tokenizer.pad_token_id is None and tokenizer.eos_token_id is not None:
        tokenizer.pad_token = tokenizer.eos_token
    torch_dtype = torch.float16 if device.startswith("cuda") else None
    model = AutoModelForCausalLM.from_pretrained(model_path, torch_dtype=torch_dtype).to(device)
    model.eval()
    tf_cfg = TransformersConfig(
        model=model,
        tokenizer=tokenizer,
        device=device,
        **gen_kwargs,
    )
    return tf_cfg


def pretty_record(rec) -> str:
    extra = rec.extra
    extra_str = ""
    if extra:
        extra_str = " " + ", ".join(f"{k}={v}" for k, v in extra.items())
    return (
        f"idx={rec.token_index} tail={rec.token_tail} byte_pos={rec.byte_pos} "
        f"prefix={rec.prefix_bytes_hex} green_len={len(rec.greenlist)}{extra_str}"
    )


def compare_records(gen_records, det_records, *, h: int, start_idx: int, limit: int) -> List[str]:
    """
    Align by token_index and compare only the last h tokens + prefix_bytes/byte_pos.
    start_idx: typically prompt_len so we start from first generated token.
    """
    lines: List[str] = []
    if not gen_records or not det_records:
        lines.append("No records to compare.")
        return lines
    gmap = {r.token_index: r for r in gen_records}
    dmap = {r.token_index: r for r in det_records}
    count = 0
    idx = start_idx
    while count < limit:
        g = gmap.get(idx)
        d = dmap.get(idx)
        if g is None or d is None:
            lines.append(f"[{idx}] missing gen/det")
            break
        g_tail = g.token_tail[-h:]
        d_tail = d.token_tail[-h:]
        tail_match = g_tail == d_tail
        prefix_match = (g.prefix_bytes_hex == d.prefix_bytes_hex) and (g.byte_pos == d.byte_pos)
        status = "MATCH" if (tail_match and prefix_match) else "MISMATCH"
        lines.append(f"[{idx}] {status} tail_match={tail_match} prefix_match={prefix_match}")
        lines.append(f"  gen: tail={g_tail} byte_pos={g.byte_pos} prefix={g.prefix_bytes_hex}")
        lines.append(f"  det: tail={d_tail} byte_pos={d.byte_pos} prefix={d.prefix_bytes_hex}")
        count += 1
        if status != "MATCH":
            break
        idx += 1
    return lines


def main() -> None:
    ap = argparse.ArgumentParser(description="Compare PRF contexts between generator and detector.")
    ap.add_argument("--config", default="MarkLLM/config/CharmKGW.json")
    ap.add_argument("--model", default="../Meta-Llama-3-8B")
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--prompt", default=None, help="Direct prompt text.")
    ap.add_argument("--prompt-file", default="data/prompts_c4.txt")
    ap.add_argument("--prompt-id", type=int, default=0)
    ap.add_argument("--prompt-trim", type=int, default=128)
    ap.add_argument("--max-new-tokens", type=int, default=64)
    ap.add_argument("--temperature", type=float, default=1.0)
    ap.add_argument("--top-p", type=float, default=0.95)
    ap.add_argument("--limit", type=int, default=32, help="Number of contexts to log per phase.")
    ap.add_argument(
        "--token-tail", type=int, default=32, help="How many trailing token IDs to record per context."
    )
    ap.add_argument(
        "--compare-byte-pos",
        type=int,
        default=0,
        help="Only keep detector records with this byte_pos (default: 0 = first byte).",
    )
    args = ap.parse_args()

    prompt = load_prompt(args.prompt, args.prompt_file, args.prompt_id, args.prompt_trim)
    # prompt token count (no specials) for alignment
    tokenizer = None
    tf_cfg = build_runtime(
        args.model,
        args.device,
        max_new_tokens=args.max_new_tokens,
        do_sample=True,
        temperature=args.temperature,
        top_p=args.top_p,
    )
    tokenizer = tf_cfg.tokenizer
    prompt_ids = tokenizer(prompt, return_tensors="pt", add_special_tokens=False)["input_ids"][0]
    prompt_len = len(prompt_ids)

    charm = CharmKGW(args.config, tf_cfg)
    # 需要记录：所有 prompt token 的上下文 + 额外 args.limit 步，以便对齐比较
    tracer_limit = None  # 不限制记录条数，调试对齐更直观
    tracer = PRFTracer(limit=tracer_limit, token_tail=args.token_tail)
    charm.runtime.logits_processor.set_debug_tracer(
        tracer,
        phase="gen",
        byte_pos_filter=args.compare_byte_pos,
    )
    charm.detector.set_debug_tracer(tracer, byte_pos_filter=args.compare_byte_pos)
    prf_h = charm.detector.prefix_length

    wm_text = charm.generate_watermarked_text(prompt, max_new_tokens=args.max_new_tokens)
    print("=== Prompt ===")
    print(prompt)
    print("=== Watermarked Text ===")
    print(wm_text)

    det_res = charm.detector.detect(wm_text, return_dict=True)
    print("=== Detection Score ===")
    print(det_res.get("score"))

    gen_records = tracer.phase_records("gen")
    det_records = tracer.phase_records("det")
    if args.compare_byte_pos is not None:
        det_records = [rec for rec in det_records if rec.byte_pos == args.compare_byte_pos]

    print(f"--- Counts: gen={len(gen_records)} det={len(det_records)} (after filter/cut) ---")

    # 仅展示前 args.limit 条，避免过长
    print("=== Generator PRF contexts ===")
    for rec in gen_records[: args.limit]:
        print("  ", pretty_record(rec))

    print("=== Detector PRF contexts (filtered) ===")
    for rec in det_records[: args.limit]:
        print("  ", pretty_record(rec))

    compare_lines = compare_records(
        gen_records,
        det_records,
        h=charm.detector.prefix_length,
        start_idx=prompt_len,
        limit=args.limit,
    )
    print("=== Pairwise comparison ===")
    for line in compare_lines:
        print(line)


if __name__ == "__main__":
    main()
