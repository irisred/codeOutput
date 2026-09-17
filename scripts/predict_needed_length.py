#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import argparse
import math
import pandas as pd

def z_from_p(p: float, n: int) -> float:
    # z = (hits - 0.5n)/sqrt(n*0.25) = (p-0.5)*2*sqrt(n)
    return (p - 0.5) * 2.0 * math.sqrt(max(n, 1))

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--sweep_tsv", required=True)
    ap.add_argument("--lengths", default="64,128,256,384,512")
    ap.add_argument("--topk", type=int, default=10)
    args = ap.parse_args()

    df = pd.read_csv(args.sweep_tsv, sep="\t")
    df = df.sort_values(["prf_flip_rate_same_uid","uid_change_rate_aligned","extra_rate"]).head(args.topk)

    lengths = [int(x.strip()) for x in args.lengths.split(",") if x.strip()]
    rows = []
    for _, r in df.iterrows():
        p = float(r["p_attack_green"])
        out = {k: r[k] for k in ["seed_window_chars","m_bits","target_anchors","n_bytes","k_choices",
                                 "prf_flip_rate_same_uid","uid_change_rate_aligned","extra_rate","p_attack_green"]}
        for n in lengths:
            out[f"z_est_N{n}"] = z_from_p(p, n)
        rows.append(out)

    out_df = pd.DataFrame(rows)
    print(out_df.to_string(index=False))

if __name__ == "__main__":
    main()
