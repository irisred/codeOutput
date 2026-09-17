import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl

# ====== 1) Paths: your 4 attack_ratio files ======
paths = [
    "tpr_attack_by_bucket_0.01.csv",  # attack_ratio=0.01
    "tpr_attack_by_bucket_0.02.csv",  # attack_ratio=0.02
    "tpr_attack_by_bucket_0.05.csv",  # attack_ratio=0.05
    "tpr_attack_by_bucket_0.10.csv",  # attack_ratio=0.10
]

df = pd.concat([pd.read_csv(p) for p in paths], ignore_index=True)

# ====== 2) Keep only PPL<12 buckets (consistent with your main tables) ======
keep_buckets = {"ppl_0_3", "ppl_3_5", "ppl_5_8", "ppl_8_12"}
df = df[df["bucket"].isin(keep_buckets)].copy()

# ====== 3) Rename algorithms for plot legend ======
algo_map = {"bytekgw_v6": "Ours", "kgw": "KGW", "dip": "DIP", "unbiased": "UniBased"}
df["algo_plot"] = df["algo"].map(algo_map).fillna(df["algo"])

# ====== 4) Retention (%) clipped at 100 ======
# Note: tpr_clean/tpr_attack are in [0,1] in your CSVs
df["retention"] = (df["tpr_attack"] / df["tpr_clean"]).clip(upper=1.0) * 100.0

# ====== 5) Aggregate to "overall over buckets" using n-weighted average ======
# For each (atk_style, fpr, attack_ratio, algo), compute weighted mean retention across buckets
g = df.groupby(["atk_style", "fpr", "attack_ratio", "algo_plot"], as_index=False)
agg = g.apply(lambda x: pd.Series({
    "retention": np.average(x["retention"], weights=x["n"]),
    "n_total": x["n"].sum()
})).reset_index(drop=True)

# Optional: add epsilon=0 baseline point (definition: no attack => retention = 100)
base = agg[["atk_style", "fpr", "algo_plot"]].drop_duplicates().copy()
base["attack_ratio"] = 0.0
base["retention"] = 100.0
base["n_total"] = np.nan
agg = pd.concat([agg, base], ignore_index=True).sort_values(
    ["atk_style", "fpr", "algo_plot", "attack_ratio"]
)

# ====== 6) Plot style (modern) ======
mpl.rcParams.update({
    "font.family": "sans-serif",
    "font.sans-serif": ["Inter", "Segoe UI", "Helvetica", "Arial", "DejaVu Sans"],
    "mathtext.fontset": "dejavusans",
    "font.size": 10,
    "axes.titlesize": 10,
    "axes.labelsize": 10,
    "xtick.labelsize": 9,
    "ytick.labelsize": 9,
    "legend.fontsize": 9,
    "axes.linewidth": 0.9,
})

palette = {"Ours":"#3B82F6","KGW":"#10B981","DIP":"#F59E0B","UniBased":"#8B5CF6"}
order_methods = ["Ours","KGW","DIP","UniBased"]
fprs = [0.01, 0.05, 0.10, 0.20]
styles = ["token", "char"]

# y-limits shared
ymin = np.floor((agg["retention"].min() - 2) / 5) * 5
ymax = 101.0

fig, axes = plt.subplots(2, 4, figsize=(14.5, 6.0), sharey=True)
fig.subplots_adjust(top=0.82, left=0.06, right=0.995, bottom=0.12, wspace=0.12, hspace=0.20)

# x ticks as percents
xt = [0.0, 0.01, 0.02, 0.05, 0.10]
xt_lbl = [f"{int(x*100)}%" for x in xt]

for r, sty in enumerate(styles):
    for c, fpr in enumerate(fprs):
        ax = axes[r, c]
        ax.set_axisbelow(True)
        ax.grid(axis="y", linestyle="-", linewidth=0.7, alpha=0.15)
        ax.axhline(100.0, linestyle="--", linewidth=1.0, alpha=0.55)

        for m in order_methods:
            sub = agg[(agg["atk_style"] == sty) & (agg["fpr"] == fpr) & (agg["algo_plot"] == m)]
            sub = sub.sort_values("attack_ratio")

            ax.plot(
                sub["attack_ratio"].values,
                sub["retention"].values,
                marker="o", markersize=4,
                linewidth=2.0, alpha=0.95,
                color=palette.get(m, None),
                label=m if (r == 0 and c == 0) else None
            )

        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
        ax.set_ylim(ymin, ymax)
        ax.set_xticks(xt)
        ax.set_xticklabels(xt_lbl)

        if c == 0:
            ax.set_ylabel("TPR retention (%)")

        # Put FPR information at the bottom row x-label (matches your earlier preference)
        if r == 1:
            ax.set_xlabel(f"Edit rate ε  (FPR={int(fpr*100)}%)")
        else:
            ax.set_xlabel("")
            ax.set_title(f"FPR={int(fpr*100)}%")

# Row labels
fig.text(0.012, 0.67, "Token edit", rotation=90, va="center", ha="left", fontsize=11, alpha=0.85)
fig.text(0.012, 0.25, "Char edit", rotation=90, va="center", ha="left", fontsize=11, alpha=0.85)

# Global legend (slightly lowered)
handles, labels = axes[0, 0].get_legend_handles_labels()
fig.legend(handles, labels, loc="upper center", ncol=4, frameon=False, bbox_to_anchor=(0.5, 0.945))

fig.suptitle("Robustness vs edit rate", y=0.995, fontsize=12)

plt.savefig("robustness_vs_editrate_retention.pdf", bbox_inches="tight")
plt.savefig("robustness_vs_editrate_retention.png", dpi=300, bbox_inches="tight")
plt.show()
