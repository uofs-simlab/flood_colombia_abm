"""
Summarizes raw per-step simulation output into per-run scalar outcomes
(one row per simulation run), then aggregates (mean +/- std across
iterations) per policy level. Used by both the figure scripts and
run_statistics.py.

Metric convention (established by inspecting a full time series, see
project notes): Death is cumulative/monotonic non-decreasing over the run,
so its END-OF-RUN value is the right summary. Injured/Stranded/Sheltered/
Hospitalized/Evacuated are instantaneous point-in-time fractions (agents
recover/return), so their PEAK value over the run is the right summary of
severity. Houses_Flooded is an exogenous exposure check (should not vary
by policy lever -- used as a sanity check that all runs saw comparable
flood exposure). Wealth_People is END-OF-RUN (cumulative economic effect,
relative deviation from pre-flood per-capita GDP baseline).
"""
import pandas as pd
import os

OUT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "data_collection"))

PEAK_COLS = ["Injured", "Stranded", "Sheltered", "Hospitalized", "Evacuated", "Houses_Flooded"]
FINAL_COLS = ["Death", "Wealth_People", "Wealth_Businesses"]


def per_run_summary(df, group_cols):
    rows = []
    for keys, g in df.groupby(group_cols + ["iteration"]):
        g = g.sort_values("Step")
        row = dict(zip(group_cols + ["iteration"], keys if isinstance(keys, tuple) else (keys,)))
        for c in PEAK_COLS:
            row[c + "_peak"] = g[c].max()
        for c in FINAL_COLS:
            row[c + "_final"] = g[c].iloc[-1]
        # also grab SES-2 (vulnerable class) dead/injured peak/final for equity lens
        row["dead_SES2_final"] = g["dead_total_pop_SES_2_0_0.3"].iloc[-1] + g["dead_total_pop_SES_2_0.7_1"].iloc[-1]
        row["dead_SES1_final"] = g["dead_total_pop_SES_1_0_0.3"].iloc[-1] + g["dead_total_pop_SES_1_0.7_1"].iloc[-1]
        row["evac_SES2_peak"] = g[["evacuated_total_pop_SES_2_0_0.3", "evacuated_total_pop_SES_2_0.7_1"]].sum(axis=1).max()
        row["evac_SES1_peak"] = g[["evacuated_total_pop_SES_1_0_0.3", "evacuated_total_pop_SES_1_0.7_1"]].sum(axis=1).max()
        rows.append(row)
    return pd.DataFrame(rows)


def agg(df, group_cols):
    metric_cols = [c for c in df.columns if c not in group_cols + ["iteration"]]
    return df.groupby(group_cols)[metric_cols].agg(["mean", "std"]).round(4)


if __name__ == "__main__":
    # Quick standalone demo: regenerate the per-run summary CSVs for both
    # experiments shipped in this repository. For the full statistical
    # analysis (Kruskal-Wallis, effect sizes, pairwise tests), run
    # run_statistics.py instead -- it calls per_run_summary() itself.
    pd.set_option("display.width", 200)
    pd.set_option("display.max_columns", 30)

    print("################ EXPERIMENT A: community organization ################")
    a = pd.read_csv(os.path.join(OUT, "experiment_A_main_sweep_results.csv"))
    pa = per_run_summary(a, ["community_organization_level"])
    pa.to_csv(os.path.join(OUT, "experiment_A_main_sweep_summary.csv"), index=False)
    print(agg(pa, ["community_organization_level"]))

    print("\n################ EXPERIMENT D: shelter capacity, distance, rescue fleet ################")
    d = pd.read_csv(os.path.join(OUT, "experiment_D_results.csv"))
    pdd = per_run_summary(d, ["shelter_site_name", "shelter_cap_limit", "rescue_boats_available"])
    pdd.to_csv(os.path.join(OUT, "experiment_D_summary.csv"), index=False)
    print(agg(pdd, ["shelter_site_name", "shelter_cap_limit", "rescue_boats_available"]))
