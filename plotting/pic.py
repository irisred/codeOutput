import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl

# ====== 1) Paths ======
paths = [
    "tpr_attack_by_bucket_0.01.csv",
    "tpr_attack_by_bucket_0.02.csv",
    "tpr_attack_by_bucket_0.05.csv",
    "tpr_attack_by_bucket_0.10.csv",
]
df = pd.concat([pd.read_csv(p) for p in paths], ignore_index=True)

# ====== 2) Keep only PPL<12 buckets and map to RelPPL ======
keep_buckets = {"ppl_0_3", "ppl_3_5", "ppl_5_8", "ppl_8_12"}
bucket2rel = {"ppl_0_3": 1.0, "ppl_3_5": 1.5, "ppl_5_8": 2.0, "ppl_8_12": 3.0}

df = df[df["bucket"].isin(keep_buckets)].copy()
df["relppl"] = df["bucket"].map(bucket2rel)

# ====== 3) Rename algorithms for legend ======
algo_map = {"bytekgw_v6": "Ours", "kgw": "KGW", "dip": "DIP", "unbiased": "UniBased"}
df["algo_plot"] = df["algo"].map(algo_map).fillna(df["algo"])

# ====== 4) Retention (%) clipped at 100 ======
df["retention"] = (df["tpr_attack"] / df["tpr_clean"]).clip(upper=1.0) * 100.0

# ====== 5) Choose ONE edit rate for this RelPPL-x-axis plot ======
attack_ratio_to_plot = 0.01   # <- 改这里：0.02 / 0.05 / 0.10 也行
dfp = df[df["attack_ratio"] == attack_ratio_to_plot].copy()

# 若有重复行（一般没有），用 n 做加权平均汇总
g = dfp.groupby(["atk_style", "fpr", "algo_plot", "relppl"], as_index=False)
dfp = g.apply(lambda x: pd.Series({
    "retention": np.average(x["retention"], weights=x["n"]),
})).reset_index(drop=True)

# ====== 6) Plot style (保持你原风格) ======
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
ymin = np.floor((dfp["retention"].min() - 2) / 5) * 5
ymax = 101.0

fig, axes = plt.subplots(2, 4, figsize=(14.5, 6.0), sharey=True)
fig.subplots_adjust(top=0.82, left=0.06, right=0.995, bottom=0.12, wspace=0.12, hspace=0.20)

# x ticks: RelPPL
xt = [1.0, 1.5, 2.0, 3.0]
xt_lbl = ["1.0", "1.5", "2.0", "3.0"]
x = np.arange(len(xt))
group_width = 0.78
bar_w = group_width / len(order_methods)

for r, sty in enumerate(styles):
    for c, fpr in enumerate(fprs):
        ax = axes[r, c]
        ax.set_axisbelow(True)
        ax.grid(axis="y", linestyle="-", linewidth=0.7, alpha=0.15)
        ax.axhline(100.0, linestyle="--", linewidth=1.0, alpha=0.55)

        sub = dfp[(dfp["atk_style"] == sty) & (dfp["fpr"] == fpr)].copy()

        for i, m in enumerate(order_methods):
            sm = sub[sub["algo_plot"] == m].set_index("relppl").reindex(xt)
            ys = sm["retention"].values

            offset = (i - (len(order_methods) - 1) / 2) * bar_w
            ax.bar(
                x + offset, ys, width=bar_w,
                color=palette[m], alpha=0.90,
                edgecolor="white", linewidth=0.6,
                label=m if (r == 0 and c == 0) else None
            )

        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
        ax.set_ylim(ymin, ymax)

        ax.set_xticks(x)
        if r == 1:
            ax.set_xticklabels(xt_lbl)
            ax.set_xlabel(f"RelPPL (FPR={int(fpr*100)}%)")
        else:
            ax.set_xticklabels([])

        if c == 0:
            ax.set_ylabel("TPR retention (%)")

# Row labels
fig.text(0.012, 0.67, "Token edit", rotation=90, va="center", ha="left", fontsize=11, alpha=0.85)
fig.text(0.012, 0.25, "Char edit", rotation=90, va="center", ha="left", fontsize=11, alpha=0.85)

# Global legend
handles, labels = axes[0, 0].get_legend_handles_labels()
fig.legend(handles, labels, loc="upper center", ncol=4, frameon=False, bbox_to_anchor=(0.5, 0.945))

fig.suptitle(f"TPR retention vs RelPPL (edit rate ε={attack_ratio_to_plot:.2f})", y=0.995, fontsize=12)

plt.savefig("retention_vs_relppl.pdf", bbox_inches="tight")
plt.savefig("retention_vs_relppl.png", dpi=300, bbox_inches="tight")
plt.show()
