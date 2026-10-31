"""
Statistical analysis for Experiment A (community organization) and
Experiment D (shelter capacity, distance, and rescue-fleet size).

Computes, for every comparison: a Kruskal-Wallis omnibus test (non-
parametric, since outcome fractions are bounded and often skewed) with
epsilon-squared effect size (Tomczak & Tomczak, 2014), and, where the
omnibus test is significant, pairwise Mann-Whitney U tests with
Holm-Bonferroni correction (Holm, 1979) and Cliff's delta effect size
(Romano et al., 2006).

Writes a full report to data_collection/statistics_report.txt (and prints
it to stdout), plus the per-run summary CSVs used by the figure scripts.
"""
import os
import numpy as np
import pandas as pd
from scipy import stats

import sys
sys.path.append(os.path.dirname(__file__))
from summarize import per_run_summary  # noqa: E402

OUT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "data_collection"))
REPORT_LINES = []


def log(*args):
    s = " ".join(str(a) for a in args)
    print(s)
    REPORT_LINES.append(s)


# ---------------------------------------------------------------- helpers --
def epsilon_squared(H, k, n):
    """Effect size for Kruskal-Wallis (Tomczak & Tomczak 2014):
    eps^2 = (H - k + 1) / (n - k). Conventional small/medium/large
    thresholds are approximately 0.01/0.06/0.14."""
    if n - k <= 0:
        return float("nan")
    return (H - k + 1) / (n - k)


def cliffs_delta(a, b):
    """Cliff's delta effect size for two independent samples. Ranges -1..1;
    |d|<0.147 negligible, <0.33 small, <0.474 medium, else large (Romano
    et al. 2006 thresholds)."""
    a = np.asarray(a)
    b = np.asarray(b)
    n_gt = sum(1 for x in a for y in b if x > y)
    n_lt = sum(1 for x in a for y in b if x < y)
    return (n_gt - n_lt) / (len(a) * len(b))


def cliffs_label(d):
    ad = abs(d)
    if ad < 0.147:
        return "negligible"
    if ad < 0.33:
        return "small"
    if ad < 0.474:
        return "medium"
    return "large"


def holm_bonferroni(pvals):
    """Returns adjusted p-values (Holm 1979 step-down procedure)."""
    idx = np.argsort(pvals)
    m = len(pvals)
    adj = np.empty(m)
    running_max = 0.0
    for rank, i in enumerate(idx):
        val = (m - rank) * pvals[i]
        running_max = max(running_max, val)
        adj[i] = min(running_max, 1.0)
    return adj


def median_iqr(x):
    x = np.asarray(x)
    med = np.median(x)
    q1, q3 = np.percentile(x, [25, 75])
    return med, q1, q3


def kw_report(groups_dict, label):
    """groups_dict: name -> array of values. Runs Kruskal-Wallis + eps^2,
    prints median/IQR per group."""
    names = list(groups_dict.keys())
    arrays = [np.asarray(groups_dict[n]) for n in names]
    n_total = sum(len(a) for a in arrays)
    k = len(arrays)
    H, p = stats.kruskal(*arrays)
    eps2 = epsilon_squared(H, k, n_total)
    log(f"\n-- Kruskal-Wallis: {label} --")
    for name, arr in zip(names, arrays):
        med, q1, q3 = median_iqr(arr)
        log(f"   {name}: n={len(arr)}, median={med:.3f}, IQR=[{q1:.3f}, {q3:.3f}], "
            f"mean={np.mean(arr):.3f}, sd={np.std(arr, ddof=1):.3f}")
    log(f"   H({k-1}, n={n_total}) = {H:.3f}, p = {p:.4f}, epsilon^2 = {eps2:.3f}")
    return dict(H=H, p=p, eps2=eps2, k=k, n=n_total)


def pairwise_mw_holm(groups_dict, label):
    """All pairwise Mann-Whitney U tests within groups_dict, Holm-Bonferroni
    corrected, with Cliff's delta effect sizes."""
    names = list(groups_dict.keys())
    pairs = [(a, b) for i, a in enumerate(names) for b in names[i + 1:]]
    raw_p, deltas = [], []
    for a, b in pairs:
        u, p = stats.mannwhitneyu(groups_dict[a], groups_dict[b], alternative="two-sided")
        raw_p.append(p)
        deltas.append(cliffs_delta(groups_dict[a], groups_dict[b]))
    adj_p = holm_bonferroni(np.array(raw_p))
    log(f"\n-- Pairwise Mann-Whitney (Holm-Bonferroni corrected): {label} --")
    for (a, b), p_raw, p_adj, d in zip(pairs, raw_p, adj_p, deltas):
        sig_raw = "*" if p_raw < 0.05 else " "
        sig_adj = "*" if p_adj < 0.05 else " "
        log(f"   {a:>22s} vs {b:<22s}  p_raw={p_raw:.4f}{sig_raw}  "
            f"p_holm={p_adj:.4f}{sig_adj}  Cliff's delta={d:+.3f} ({cliffs_label(d)})")
    return pairs, raw_p, adj_p, deltas


# =============================================================== EXPERIMENT A
log("=" * 78)
log("EXPERIMENT A -- community organization, main sweep (n=10/level)")
log("=" * 78)
a = pd.read_csv(os.path.join(OUT, "experiment_A_main_sweep_results.csv"))
pa = per_run_summary(a, ["community_organization_level"])
pa.to_csv(os.path.join(OUT, "experiment_A_main_sweep_summary.csv"), index=False)

groups_A_death = {
    f"org={lvl}": pa.loc[pa.community_organization_level == lvl, "Death_final"].values
    for lvl in sorted(pa.community_organization_level.unique())
}
kw_report(groups_A_death, "Death_final across community_organization_level")
pairwise_mw_holm(groups_A_death, "Death_final, community_organization_level")

# --- channel ablation + coefficient sweep ---
log("\n" + "-" * 78)
log("EXPERIMENT A -- channel ablation + coefficient sweep (org=1.0 fixed)")
log("-" * 78)
abl = pd.read_csv(os.path.join(OUT, "experiment_A_channel_ablation_results.csv"))
p_abl = per_run_summary(abl, ["ablation_config"])
p_abl.to_csv(os.path.join(OUT, "experiment_A_channel_ablation_summary.csv"), index=False)

both_default = pa.loc[pa.community_organization_level == 1.0, "Death_final"].values  # org=1.0, both channels, default coefs, from the main sweep
groups_ablation = {
    "both_default_0.5_0.3": both_default,
    "depth_only": p_abl.loc[p_abl.ablation_config == "depth_only", "Death_final"].values,
    "sct_only": p_abl.loc[p_abl.ablation_config == "sct_only", "Death_final"].values,
    "coef_0.1_both": p_abl.loc[p_abl.ablation_config == "coef_0.1", "Death_final"].values,
}
kw_report(groups_ablation, "Death_final across channel-ablation configs (org=1.0)")
pairwise_mw_holm(groups_ablation, "channel-ablation configs (org=1.0)")

baseline0 = pa.loc[pa.community_organization_level == 0.0, "Death_final"].values
log("\n-- Each channel/coefficient config vs. org=0.0 baseline (no organization) --")
for name, arr in groups_ablation.items():
    u, p = stats.mannwhitneyu(baseline0, arr, alternative="two-sided")
    d = cliffs_delta(baseline0, arr)
    log(f"   org=0.0 vs {name}: p={p:.4f}, Cliff's delta={d:+.3f} ({cliffs_label(d)}), "
        f"median_diff={np.median(arr) - np.median(baseline0):+.2f}")

# =============================================================== EXPERIMENT D
log("\n" + "=" * 78)
log("EXPERIMENT D -- shelter capacity, distance, and rescue-fleet size")
log("=" * 78)

DIST_KM = {"nearby_crossing": 2.0, "current_hypothetical": 38.4,
           "intermediate": 58.0, "barranquilla": 99.9}
CANONICAL_SITES = ["nearby_crossing", "current_hypothetical", "intermediate", "barranquilla"]

d = pd.read_csv(os.path.join(OUT, "experiment_D_results.csv"))
pd_ = per_run_summary(d, ["shelter_site_name", "shelter_cap_limit", "rescue_boats_available"])
pd_["distance_km"] = pd_["shelter_site_name"].map(lambda s: DIST_KM.get(s.split("_fleet")[0], np.nan))
pd_.to_csv(os.path.join(OUT, "experiment_D_summary.csv"), index=False)

# --- capacity-stratified distance test: does distance matter, and does it
# depend on how large the shelter is? ---
for cap in [1, 10, 30]:
    sub = pd_[(pd_.shelter_cap_limit == cap) & (pd_.rescue_boats_available == 2) &
              (pd_.shelter_site_name.isin(CANONICAL_SITES))]
    groups = {s: sub.loc[sub.shelter_site_name == s, "Death_final"].values for s in CANONICAL_SITES}
    kw_report(groups, f"Death_final across sites @ shelter_cap_limit={cap}%")

log("\n-- Pairwise Mann-Whitney (Holm-Bonferroni), cap=10% --")
sub10 = pd_[(pd_.shelter_cap_limit == 10) & (pd_.rescue_boats_available == 2) &
            (pd_.shelter_site_name.isin(CANONICAL_SITES))]
groups10 = {s: sub10.loc[sub10.shelter_site_name == s, "Death_final"].values for s in CANONICAL_SITES}
pairwise_mw_holm(groups10, "Death_final across sites @ 10% capacity")

log("\n-- Stranded_peak across sites @ 10% capacity (mechanism-verification check) --")
groups10_stranded = {s: sub10.loc[sub10.shelter_site_name == s, "Stranded_peak"].values for s in CANONICAL_SITES}
kw_report(groups10_stranded, "Stranded_peak across sites @ 10% capacity")

log("\n-- Levene's test for equal variances (Death_final), cap=10% across sites --")
lev_stat, lev_p = stats.levene(*[groups10[s] for s in CANONICAL_SITES])
log(f"   Levene W={lev_stat:.3f}, p={lev_p:.4f}")

# --- fleet-size sensitivity vs. distance: direct magnitude comparison ---
log("\n" + "-" * 78)
log("Fleet-size sensitivity: boats in {1, 2 (reference), 5} @ 10% capacity")
log("-" * 78)
for site in ["nearby_crossing", "barranquilla"]:
    ref = pd_[(pd_.shelter_site_name == site) & (pd_.shelter_cap_limit == 10) &
              (pd_.rescue_boats_available == 2)]["Death_final"].values
    b1 = pd_[(pd_.shelter_site_name == f"{site}_fleet1") & (pd_.rescue_boats_available == 1)]["Death_final"].values
    b5 = pd_[(pd_.shelter_site_name == f"{site}_fleet5") & (pd_.rescue_boats_available == 5)]["Death_final"].values
    groups_fleet = {"1_boat": b1, "2_boats_ref": ref, "5_boats": b5}
    kw_report(groups_fleet, f"Death_final across fleet size @ {site} (dist={DIST_KM[site]}km)")
    pairwise_mw_holm(groups_fleet, f"fleet size @ {site}")

log("\n-- Direct magnitude comparison: fleet-size effect vs. distance effect --")
near = groups10["nearby_crossing"]
far = groups10["barranquilla"]
d_distance = cliffs_delta(near, far)
log(f"   Distance effect (10%% cap, 2 boats): nearby_crossing (2km, median={np.median(near):.3f}) vs "
    f"barranquilla (99.9km, median={np.median(far):.3f}): Cliff's delta={d_distance:+.3f} ({cliffs_label(d_distance)}), "
    f"median diff={np.median(far)-np.median(near):+.4f}")
for site in ["nearby_crossing", "barranquilla"]:
    ref = pd_[(pd_.shelter_site_name == site) & (pd_.shelter_cap_limit == 10) &
              (pd_.rescue_boats_available == 2)]["Death_final"].values
    b1 = pd_[(pd_.shelter_site_name == f"{site}_fleet1") & (pd_.rescue_boats_available == 1)]["Death_final"].values
    d_fleet = cliffs_delta(b1, ref)
    log(f"   Fleet effect @ {site}: 1 boat (median={np.median(b1):.3f}) vs 2 boats (median={np.median(ref):.3f}): "
        f"Cliff's delta={d_fleet:+.3f} ({cliffs_label(d_fleet)}), median diff={np.median(ref)-np.median(b1):+.4f}")

# --- Spearman correlation: honest treatment of the pseudo-replication issue
# (32 runs at cap=10%, but only 4 distinct distance values) ---
log("\n" + "-" * 78)
log("Spearman correlation: Death_final vs. distance_km @ 10% capacity")
log("-" * 78)
naive_rho, naive_p = stats.spearmanr(sub10["distance_km"], sub10["Death_final"])
log(f"   Naive (n=32, treats each run as an independent x-observation -- "
    f"incorrect, only 4 distinct distance values exist): rho={naive_rho:.3f}, p={naive_p:.4f}")
site_means = sub10.groupby("shelter_site_name").agg(distance_km=("distance_km", "first"),
                                                       Death_final=("Death_final", "mean")).reset_index()
site_means = site_means.sort_values("distance_km")
correct_rho, correct_p = stats.spearmanr(site_means["distance_km"], site_means["Death_final"])
log(f"   Corrected (n=4 site means, the actual number of independent x-values): "
    f"rho={correct_rho:.3f}, p={correct_p:.4f} (with only 4 points, p can never reach "
    f"conventional significance regardless of effect size -- report descriptively, not inferentially)")

with open(os.path.join(OUT, "statistics_report.txt"), "w") as f:
    f.write("\n".join(REPORT_LINES))

print("\n[statistics report complete -> data_collection/statistics_report.txt]")
