import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl
from io import StringIO

# ====== 1) Data (replace with your own loading if needed) ======
csv_text = """n,mean_entropy_prefix,mean_entropy_full,mean_ratio
1,0.13921693528268678,0.897237977499608,0.1551616614252684
2,0.17835159324660424,0.897237977499608,0.1987784709510718
3,0.7837943283056661,0.897237977499608,0.8735634780974355
4,0.8603095475560445,0.897237977499608,0.9588421011263094
5,0.8812512762129123,0.897237977499608,0.9821823176374602
6,0.8886146824201997,0.897237977499608,0.9903890658937116
7,0.891812581767553,0.897237977499608,0.9939532254896585
"""
df = pd.read_csv(StringIO(csv_text))

# Optionally recompute ratio from definition:
# df["mean_ratio"] = df["mean_entropy_prefix"] / df["mean_entropy_full"]

# ====== 2) Plot style (match your modern style) ======
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

# Single-line palette (pick one close to yours)
line_color = "#3B82F6"  # same blue as your palette["Ours"]

# ====== 3) Plot ======
x = df["n"].to_numpy()
y = df["mean_ratio"].to_numpy()

# y-limits similar logic to your script (slight padding, cap at 1.01)
ymin = np.floor((y.min() - 0.02) / 0.05) * 0.05
ymax = 1.01

fig, ax = plt.subplots(figsize=(6.2, 3.8))
fig.subplots_adjust(left=0.12, right=0.98, bottom=0.16, top=0.92)

ax.set_axisbelow(True)
ax.grid(axis="y", linestyle="-", linewidth=0.7, alpha=0.15)   # light horizontal grid
ax.grid(axis="x", linestyle="-", linewidth=0.6, alpha=0.06)   # very light vertical grid (optional)

# Reference line at 1.0 (like your 100% dashed line)
ax.axhline(1.0, linestyle="--", linewidth=1.0, alpha=0.55)

ax.plot(
    x, y,
    marker="o", markersize=4,
    linewidth=2.0, alpha=0.95,
    color=line_color,
)

# Clean spines (match your style)
ax.spines["top"].set_visible(False)
ax.spines["right"].set_visible(False)

ax.set_xlabel("Prefix length n (bytes)")
ax.set_ylabel(r"Entropy retention $\rho(n)$")

ax.set_xticks(x)
ax.set_ylim(ymin, ymax)

# Optional: title
# ax.set_title("Entropy retention vs prefix length")

# Save
plt.savefig("entropy_retention_modern.pdf", bbox_inches="tight")
plt.savefig("entropy_retention_modern.png", dpi=300, bbox_inches="tight")
plt.show()
