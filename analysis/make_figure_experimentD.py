"""
Builds figure_experimentD_capacity_distance_fleet.png: a two-panel figure
showing (a) the shelter capacity x distance factorial (does distance matter
more when the shelter is small, medium, or large?) and (b) rescue-fleet-size
sensitivity at the two most extreme sites (does adding boats matter more
than moving the shelter closer?).
"""
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os
import sys

sys.path.append(os.path.dirname(__file__))
from summarize import per_run_summary  # noqa: E402

OUT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "data_collection"))
FIG_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), "figures"))

DIST_KM = {"nearby_crossing": 2.0, "current_hypothetical": 38.4,
           "intermediate": 58.0, "barranquilla": 99.9}
SITE_ORDER = ["nearby_crossing", "current_hypothetical", "intermediate", "barranquilla"]

d = pd.read_csv(os.path.join(OUT, "experiment_D_results.csv"))
pdd = per_run_summary(d, ["shelter_site_name", "shelter_cap_limit", "rescue_boats_available"])

fig, axes = plt.subplots(1, 2, figsize=(13, 5))

# ---- panel (a): capacity x distance factorial ---------------------------
ax = axes[0]
colors = {1: "#fee08b", 10: "#fc8d59", 30: "#d73027"}
for cap in [1, 10, 30]:
    meds, q1s, q3s = [], [], []
    for site in SITE_ORDER:
        sub = pdd[(pdd.shelter_site_name == site) & (pdd.shelter_cap_limit == cap) &
                  (pdd.rescue_boats_available == 2)]["Death_final"] * 100
        meds.append(sub.median())
        q1s.append(sub.quantile(0.25))
        q3s.append(sub.quantile(0.75))
    x = [DIST_KM[s] for s in SITE_ORDER]
    yerr = np.vstack([np.array(meds) - np.array(q1s), np.array(q3s) - np.array(meds)])
    ax.errorbar(x, meds, yerr=yerr, marker="o", label=f"{cap}% shelter capacity",
                color=colors[cap], linewidth=2, capsize=4)
ax.set_xlabel("Real routed distance to shelter (km)")
ax.set_ylabel("Deaths (% of population)")
ax.set_title("(a) Capacity x distance factorial (n=8/cell)", fontsize=10)
ax.legend(fontsize=8.5, loc="upper left")
ax.grid(alpha=0.3)

# ---- panel (b): fleet-size sensitivity -----------------------------------
ax = axes[1]
for site, color, marker in [("nearby_crossing", "#1a9641", "s"), ("barranquilla", "#762a83", "o")]:
    boats_x = [1, 2, 5]
    meds, q1s, q3s = [], [], []
    for b in boats_x:
        if b == 2:
            sub = pdd[(pdd.shelter_site_name == site) & (pdd.shelter_cap_limit == 10) &
                      (pdd.rescue_boats_available == 2)]["Death_final"] * 100
        else:
            sub = pdd[(pdd.shelter_site_name == f"{site}_fleet{b}") &
                      (pdd.rescue_boats_available == b)]["Death_final"] * 100
        meds.append(sub.median())
        q1s.append(sub.quantile(0.25))
        q3s.append(sub.quantile(0.75))
    yerr = np.vstack([np.array(meds) - np.array(q1s), np.array(q3s) - np.array(meds)])
    label = f"{site.replace('_', ' ')} ({DIST_KM[site]} km)"
    ax.errorbar(boats_x, meds, yerr=yerr, marker=marker, label=label, color=color,
                linewidth=2, capsize=4, markersize=8)
ax.set_xlabel("Rescue-boat fleet size")
ax.set_xticks([1, 2, 5])
ax.set_ylabel("Deaths (% of population)")
ax.set_title("(b) Fleet-size sensitivity @ 10% capacity (n=6-8/cell)", fontsize=10)
ax.legend(fontsize=8.5)
ax.grid(alpha=0.3)

fig.suptitle("Experiment D -- capacity, distance, and rescue-fleet size (median, IQR whiskers)", fontsize=12.5)
plt.tight_layout(rect=[0, 0, 1, 0.92])
plt.savefig(os.path.join(FIG_DIR, "figure_experimentD_capacity_distance_fleet.png"), dpi=200)
print("saved figure_experimentD_capacity_distance_fleet.png")
