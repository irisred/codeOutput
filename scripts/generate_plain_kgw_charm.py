#!/usr/bin/env python3
"""
Generate plain / KGW / Charm samples into CSV files.

Requirements from user:
  - Use two GPUs (configurable via --devices).
  - Each CSV uses prompts truncated to 256 chars; completion max_new_tokens=256.
  - Plain: 1000 samples (no watermark).
  - KGW: 3 deltas (1,2,3) * 1000 samples each = 3000 rows split into 3 CSVs.
  - Charm: same as KGW (delta grid 1,2,3).
"""

from __future__ import annotations

import argparse
import csv
import math
import os
import random
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

from tqdm import tqdm

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from MarkLLM.charm_v2.charm_kgw import CharmKGW
from MarkLLM.utils.transformers_config import TransformersConfig
from MarkLLM.watermark.kgw.kgw import KGW

CSV_COLUMNS = [
    "method",
    "label",
    "delta",
    "prompt_id",
    "seed",
    "device",
    "text",
    "cond_ppl",
    "z_first_byte",
    "z_other_byte",
    "z_k121",
    "z_score",
]


@dataclass
class DeviceRuntime:
    device: str
    tokenizer: AutoTokenizer
    model: AutoModelForCausalLM
    tf_config: TransformersConfig


def parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser(description="Generate plain/KGW/Charm CSV datasets.")
    ap.add_argument("--model", default="../Meta-Llama-3-8B", help="HF model path/id")
    ap.add_argument("--kgw-config", default="MarkLLM/config/KGW.json")
    ap.add_argument("--charm-config", default="MarkLLM/config/CharmKGW.json")
    ap.add_argument("--prompts", default="data/prompts_c4_head32.txt", help="Prompt file (one per line)")
    ap.add_argument("--prompt-trim", type=int, default=0, help="Trim prompt to this many characters (0=disabled)")
    ap.add_argument("--max-new-tokens", type=int, default=128)
    ap.add_argument("--temperature", type=float, default=1.0)
    ap.add_argument("--top-p", type=float, default=0.95)
    ap.add_argument("--devices", default="cuda:0,cuda:1", help="Comma-separated devices to use")
    ap.add_argument("--plain-samples", type=int, default=1000)
    ap.add_argument("--wm-samples", type=int, default=1000, help="Samples per watermark delta")
    ap.add_argument("--deltas", default="1,2,3,4,5", help="Comma-separated deltas for KGW/Charm")
    ap.add_argument("--output-dir", default="saved_data", help="Directory for CSV outputs")
    ap.add_argument("--output-prefix", default="bulk", help="CSV prefix")
    ap.add_argument("--seed", type=int, default=20250101)
    ap.add_argument("--charm-first-byte-only", action="store_true", help="Force Charm generation to use first-byte-only bias.")
    return ap.parse_args()


def prepare_runtime(model_path: str, device: str, args: argparse.Namespace) -> DeviceRuntime:
    tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code="qwen" in model_path.lower())
    if tokenizer.pad_token_id is None and tokenizer.eos_token_id is not None:
        tokenizer.pad_token = tokenizer.eos_token
    torch_dtype = torch.float16 if device.startswith("cuda") else None
    model = AutoModelForCausalLM.from_pretrained(
        model_path,
        torch_dtype=torch_dtype,
        trust_remote_code="qwen" in model_path.lower(),
    ).to(device)
    model.eval()
    gen_kwargs = dict(
        max_new_tokens=args.max_new_tokens,
        do_sample=True,
        temperature=args.temperature,
        top_p=args.top_p,
    )
    tf_cfg = TransformersConfig(
        model=model,
        tokenizer=tokenizer,
        device=device,
        **gen_kwargs,
    )
    return DeviceRuntime(device=device, tokenizer=tokenizer, model=model, tf_config=tf_cfg)


def load_prompts(path: str | Path, limit: Optional[int], trim: int) -> List[str]:
    prompts: List[str] = []
    with Path(path).open("r", encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if not line:
                continue
            prompt = line[:trim] if (trim and trim > 0) else line
            prompts.append(prompt)
            if limit and len(prompts) >= limit:
                break
    if limit and len(prompts) < limit:
        raise SystemExit(f"Prompt file {path} only has {len(prompts)} usable lines (need {limit}).")
    return prompts


def set_seed(seed: int, device: str) -> None:
    random.seed(seed)
    torch.manual_seed(seed)
    if device.startswith("cuda"):
        torch.cuda.manual_seed_all(seed)


def generate_plain(runtime: DeviceRuntime, prompt: str, seed: int) -> str:
    set_seed(seed, runtime.device)
    encoded = runtime.tokenizer(
        prompt,
        return_tensors="pt",
        add_special_tokens=True,
    ).to(runtime.device)
    gen_ids = runtime.model.generate(**encoded, **runtime.tf_config.gen_kwargs)
    return runtime.tokenizer.batch_decode(gen_ids, skip_special_tokens=True)[0]


class KGWManager:
    def __init__(self, cfg_path: str, runtimes: Dict[str, DeviceRuntime]) -> None:
        self.cfg_path = cfg_path
        self.runtimes = runtimes
        self.cache: Dict[Tuple[str, float], KGW] = {}

    def get(self, device: str, delta: float) -> KGW:
        key = (device, float(delta))
        if key not in self.cache:
            tf_cfg = self.runtimes[device].tf_config
            inst = KGW(self.cfg_path, tf_cfg)
            inst.config.delta = float(delta)
            inst.logits_processor.config.delta = float(delta)
            self.cache[key] = inst
        return self.cache[key]


class CharmManager:
    def __init__(self, cfg_path: str, runtimes: Dict[str, DeviceRuntime], first_byte_only: bool) -> None:
        self.cfg_path = cfg_path
        self.runtimes = runtimes
        self.cache: Dict[Tuple[str, float], CharmKGW] = {}
        self.first_byte_only = bool(first_byte_only)

    def _apply_first_byte_flag(self, inst: CharmKGW) -> None:
        inst.config.first_byte_only_bias = self.first_byte_only

    def get(self, device: str, delta: float) -> CharmKGW:
        key = (device, float(delta))
        if key not in self.cache:
            tf_cfg = self.runtimes[device].tf_config
            inst = CharmKGW(self.cfg_path, tf_cfg)
            self._apply_first_byte_flag(inst)
            inst.runtime.logits_processor.delta = float(delta)
            self.cache[key] = inst
        else:
            inst = self.cache[key]
            self._apply_first_byte_flag(inst)
            inst.runtime.logits_processor.delta = float(delta)
        return self.cache[key]


def run_task(
    method: str,
    device: str,
    runtime: DeviceRuntime,
    kgw_mgr: KGWManager,
    charm_mgr: CharmManager,
    prompt: str,
    prompt_id: int,
    delta: Optional[float],
    seed: int,
    max_new_tokens: int,
) -> Dict[str, object]:
    if method == "plain":
        text = generate_plain(runtime, prompt, seed)
        label = 0
    elif method == "kgw":
        set_seed(seed, runtime.device)
        kgw = kgw_mgr.get(device, delta or 0.0)
        text = kgw.generate_watermarked_text(prompt)
        label = 1
    elif method == "charm":
        set_seed(seed, runtime.device)
        charm = charm_mgr.get(device, delta or 0.0)
        text = charm.generate_watermarked_text(prompt, max_new_tokens=max_new_tokens)
        label = 1
    else:
        raise ValueError(f"Unknown method {method}")

    cond_ppl = compute_conditional_ppl(runtime, prompt, text)
    charm_metrics = {"z_first_byte": "", "z_other_byte": "", "z_k121": ""}
    kgw_z = ""
    if method in ("charm", "plain"):
        charm = charm_mgr.get(device, delta or 0.0)
        charm_metrics = compute_charm_metrics(charm, text)
    if method in ("kgw", "plain"):
        kgw = kgw_mgr.get(device, delta or 0.0)
        kgw_z = compute_kgw_z(kgw, text)

    return {
        "method": method,
        "label": label,
        "delta": float(delta) if delta is not None else "",
        "prompt_id": prompt_id,
        "seed": seed,
        "device": device,
        "text": text,
        "cond_ppl": cond_ppl,
        "z_first_byte": charm_metrics["z_first_byte"],
        "z_other_byte": charm_metrics["z_other_byte"],
        "z_k121": charm_metrics["z_k121"],
        "z_score": kgw_z,
    }


def compute_conditional_ppl(runtime: DeviceRuntime, prompt: str, text: str) -> str:
    if not text:
        return ""
    tokenizer = runtime.tokenizer
    device = runtime.device
    try:
        full = tokenizer(text, return_tensors="pt", add_special_tokens=True)
        full = {k: v.to(device) for k, v in full.items()}
        prompt_ids = tokenizer(prompt, return_tensors="pt", add_special_tokens=True)["input_ids"].to(device)
    except Exception:
        return ""
    labels = full["input_ids"].clone()
    prompt_len = min(prompt_ids.shape[1], labels.shape[1])
    labels[:, :prompt_len] = -100
    try:
        with torch.no_grad():
            outputs = runtime.model(input_ids=full["input_ids"], attention_mask=full.get("attention_mask"), labels=labels)
    except Exception:
        return ""
    loss = outputs.loss
    if loss is None:
        return ""
    value = math.exp(float(loss.item()))
    return f"{value:.6f}"


def _format_float(value: object) -> str:
    try:
        val = float(value)
    except Exception:
        return ""
    if not math.isfinite(val):
        return ""
    return f"{val:.6f}"


def _bucket_z_from_stats(stats: Optional[Dict[str, float]], gamma: float) -> str:
    if not stats:
        return ""
    try:
        count = float(stats.get("count", 0.0))
        hits = float(stats.get("hits", 0.0))
    except Exception:
        return ""
    if count <= 0.0 or not (0.0 < gamma < 1.0):
        return ""
    denom = math.sqrt(count * gamma * (1.0 - gamma))
    if denom <= 0.0:
        return ""
    z = (hits - count * gamma) / denom
    return _format_float(z)


def compute_charm_metrics(charm: CharmKGW, text: str) -> Dict[str, str]:
    result = {"z_first_byte": "", "z_other_byte": "", "z_k121": ""}
    if not text:
        return result
    try:
        detection = charm.detector.detect(text, return_dict=True)
    except Exception:
        return result
    bucket_stats = detection.get("bucket_stats") or {}
    seg = (bucket_stats.get("segmented_start") or {}).get("z")
    nonseg = (bucket_stats.get("nonsegmented_start") or {}).get("z")
    result["z_first_byte"] = _format_float(seg)
    result["z_other_byte"] = _format_float(nonseg)
    k_stats = detection.get("k_stats") or {}
    stat121 = k_stats.get(121) or k_stats.get("121")
    result["z_k121"] = _bucket_z_from_stats(stat121, charm.detector.gamma)
    return result


def compute_kgw_z(kgw: KGW, text: str) -> str:
    if not text:
        return ""
    try:
        res = kgw.detect_watermark(text, return_dict=True)
    except Exception:
        return ""
    return _format_float(res.get("score"))


def generate_csv_for_method(
    method: str,
    deltas: Sequence[float],
    prompts: List[str],
    args: argparse.Namespace,
    runtimes: Dict[str, DeviceRuntime],
    kgw_mgr: KGWManager,
    charm_mgr: CharmManager,
    devices: List[str],
    output_dir: Path,
) -> None:
    samples = args.plain_samples if method == "plain" else args.wm_samples
    prompt_subset = prompts[:samples]
    delta_list = [None] if method == "plain" else [float(d) for d in deltas]
    base_seed = args.seed

    for delta_idx, delta in enumerate(delta_list):
        rows: List[Dict[str, object]] = []
        with ThreadPoolExecutor(max_workers=len(devices)) as executor:
            futures = []
            for idx, prompt in enumerate(prompt_subset):
                device = devices[idx % len(devices)]
                runtime = runtimes[device]
                seed = base_seed + delta_idx * 100000 + idx
                futures.append(
                    executor.submit(
                        run_task,
                        method,
                        device,
                        runtime,
                        kgw_mgr,
                        charm_mgr,
                        prompt,
                        idx,
                        delta,
                        seed,
                        args.max_new_tokens,
                    )
                )
            for fut in tqdm(as_completed(futures), total=len(futures), desc=f"{method.upper()} delta={delta}"):
                rows.append(fut.result())

        rows.sort(key=lambda r: r["prompt_id"])
        if method == "plain":
            csv_name = f"{args.output_prefix}_plain_s{samples}.csv"
        else:
            csv_name = f"{args.output_prefix}_{method}_d{int(delta)}_s{samples}.csv"
        csv_path = output_dir / csv_name
        with csv_path.open("w", newline="", encoding="utf-8") as handle:
            writer = csv.DictWriter(handle, fieldnames=CSV_COLUMNS)
            writer.writeheader()
            for row in rows:
                writer.writerow(row)
        print(f"[info] Wrote {len(rows)} rows to {csv_path}")


def main() -> None:
    args = parse_args()
    devices = [d.strip() for d in args.devices.split(",") if d.strip()]
    if not devices:
        raise SystemExit("No devices specified.")

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    prompts_all = load_prompts(args.prompts, max(args.plain_samples, args.wm_samples), args.prompt_trim)

    runtimes = {dev: prepare_runtime(args.model, dev, args) for dev in devices}
    kgw_mgr = KGWManager(args.kgw_config, runtimes)
    charm_mgr = CharmManager(args.charm_config, runtimes, args.charm_first_byte_only)

    deltas = [float(x) for x in args.deltas.split(",") if x.strip()]

    generate_csv_for_method("plain", deltas, prompts_all, args, runtimes, kgw_mgr, charm_mgr, devices, output_dir)
    generate_csv_for_method("kgw", deltas, prompts_all, args, runtimes, kgw_mgr, charm_mgr, devices, output_dir)
    generate_csv_for_method("charm", deltas, prompts_all, args, runtimes, kgw_mgr, charm_mgr, devices, output_dir)

    print("[done] Generation complete.")


if __name__ == "__main__":
    main()
