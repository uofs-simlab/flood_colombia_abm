"""
Builds the two Experiment A figures:
  - figure_experimentA_main_sweep.png: dose-response of mortality, peak
    stranding, and household wealth across the five community-organization
    levels (n=10/level).
  - figure_experimentA_channel_ablation.png: boxplot comparing the
    channel-ablation configurations at full organization (org=1.0), showing
    that the depth-attenuation channel alone reproduces almost the entire
    effect while the social-capital channel alone does not.
"""
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os

OUT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "data_collection"))
FIG_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), "figures"))

pa = pd.read_csv(os.path.join(OUT, "experiment_A_main_sweep_summary.csv"))
g = pa.groupby("community_organization_level")

fig, axes = plt.subplots(1, 3, figsize=(14, 4.3))

metrics = [
    ("Death_final", "Deaths (fraction of population)", "#d7191c"),
    ("Stranded_peak", "Peak stranded (fraction of population)", "#2c7bb6"),
    ("Wealth_People_final", "Household wealth vs. pre-flood\nbaseline", "#1a9641"),
]
x = sorted(pa["community_organization_level"].unique())
for ax, (col, label, color) in zip(axes, metrics):
    med = g[col].median().reindex(x)
    q1 = g[col].quantile(0.25).reindex(x)
    q3 = g[col].quantile(0.75).reindex(x)
    yerr = np.vstack([med - q1, q3 - med])
    ax.errorbar(x, med, yerr=yerr, marker="o", color=color, capsize=4, linewidth=2)
    ax.set_xlabel("Community organization level")
    ax.set_ylabel(label, fontsize=9)
    ax.grid(alpha=0.3)

axes[0].set_title("(a) Mortality\n(n=10/level)", fontsize=9.5)
axes[1].set_title("(b) Peak stranded", fontsize=9.5)
axes[2].set_title("(c) Household wealth", fontsize=9.5)

fig.suptitle(
    "Experiment A -- community organization dose-response (median, IQR whiskers, n=10/level)",
    fontsize=12,
)
plt.tight_layout(rect=[0, 0, 1, 0.90])
plt.savefig(os.path.join(FIG_DIR, "figure_experimentA_main_sweep.png"), dpi=200)
print("saved figure_experimentA_main_sweep.png")

# ---- channel ablation ----------------------------------------------------
abl = pd.read_csv(os.path.join(OUT, "experiment_A_channel_ablation_summary.csv"))
both_default = pa.loc[pa.community_organization_level == 1.0, "Death_final"].values
configs = {
    "org=0.0\n(no organization)": pa.loc[pa.community_organization_level == 0.0, "Death_final"].values,
    "both channels\n(default 0.5/0.3)": both_default,
    "depth-attenuation\nchannel ONLY": abl.loc[abl.ablation_config == "depth_only", "Death_final"].values,
    "social-capital\nchannel ONLY": abl.loc[abl.ablation_config == "sct_only", "Death_final"].values,
    "both channels,\nweak coef. (0.1)": abl.loc[abl.ablation_config == "coef_0.1", "Death_final"].values,
}

fig2, ax2 = plt.subplots(figsize=(8.5, 5.2))
names = list(configs.keys())
data = [configs[n] * 100 for n in names]
bp = ax2.boxplot(data, tick_labels=names, patch_artist=True, showmeans=True)
colors = ["#999999", "#1a9641", "#66c2a5", "#fc8d62", "#8da0cb"]
for patch, c in zip(bp["boxes"], colors):
    patch.set_facecolor(c)
    patch.set_alpha(0.75)
for i, d in enumerate(data):
    ax2.scatter(np.random.normal(i + 1, 0.04, size=len(d)), d, color="black", s=14, zorder=5, alpha=0.6)
ax2.set_ylabel("Deaths (% of population)")
ax2.set_title(
    "Channel ablation + coefficient sweep at full organization (org=1.0)\n"
    "Isolating the depth-attenuation channel vs. the social-capital channel",
    fontsize=10,
)
ax2.tick_params(axis="x", labelsize=8.5)
ax2.grid(alpha=0.3, axis="y")
plt.tight_layout()
plt.savefig(os.path.join(FIG_DIR, "figure_experimentA_channel_ablation.png"), dpi=200)
print("saved figure_experimentA_channel_ablation.png")
