"""Validate the archived runs and generate manuscript tables and figures.

This script does not run the ABM or modify its raw data, checkpoints, or
legacy summaries. The simulation environment is not needed for re-analysis.
"""

import hashlib
import itertools
import json
import platform
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import scipy
from scipy import stats

from summarize import per_run_summary

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "data_collection"
OUT = ROOT / "analysis" / "archive_generated"
SITES = ["nearby_crossing", "current_hypothetical", "intermediate", "barranquilla"]
LABELS = {
    "nearby_crossing": "Nearby crossing",
    "current_hypothetical": "Hypothetical site",
    "intermediate": "Intermediate",
    "barranquilla": "Barranquilla",
    "depth_only": "Depth only",
    "sct_only": "Social capital only",
    "coef_0.1": "Both, weak attenuation",
}
DISTANCES = dict(zip(SITES, [2.0, 38.4, 58.0, 99.9]))
FRACTIONS = ["Death", "Stranded", "Injured", "Sheltered", "Hospitalized",
             "Evacuated", "Houses_Flooded"]


def require(condition, message):
    if not condition:
        raise ValueError(message)


def holm_adjust(pvalues):
    pvalues = np.asarray(pvalues, dtype=float)
    require(pvalues.ndim == 1 and np.isfinite(pvalues).all()
            and ((pvalues >= 0) & (pvalues <= 1)).all(),
            "Holm adjustment requires finite probabilities.")
    order = np.argsort(pvalues)
    adjusted = np.empty(len(pvalues))
    adjusted[order] = np.minimum(
        1.0, np.maximum.accumulate(
            pvalues[order] * np.arange(len(pvalues), 0, -1)))
    return adjusted


def cliffs_delta(a, b):
    a, b = np.asarray(a), np.asarray(b)
    require(a.size > 0 and b.size > 0, "Cliff's delta requires two samples.")
    return float(np.sign(a[:, None] - b[None, :]).mean())


def validate_raw(df, keys, expected_runs):
    require(not df.duplicated(keys + ["iteration", "Step"]).any(),
            f"Duplicate run/step rows: {keys}")
    values = df[FRACTIONS].to_numpy(dtype=float)
    require(np.isfinite(values).all() and
            ((values >= 0) & (values <= 1)).all(),
            "Missing or out-of-range fraction outcome.")
    for column, expected in [("N_persons", 972), ("pre_flood_days", 7),
                             ("flood_days", 5), ("post_flood_days", 5),
                             ("healthcare_cap_limit", 5)]:
        require(df[column].eq(expected).all(), f"Unexpected {column}.")
    groups = df.groupby(keys + ["iteration"], dropna=False)
    require(len(groups) == expected_runs, "Unexpected number of runs.")
    for key, group in groups:
        group = group.sort_values("Step")
        require(group["Step"].tolist() == list(range(0, 408, 24)),
                f"Incomplete daily time series: {key}")
        require((np.diff(group["Death"]) >= -1e-12).all(),
                f"Non-monotone cumulative deaths: {key}")


def verify_summary(df, keys, stored_path):
    summary = per_run_summary(df, keys)
    independent = df.sort_values("Step").groupby(
        keys + ["iteration"], dropna=False).agg(
            Death_final=("Death", "last"),
            Stranded_peak=("Stranded", "max"),
            Injured_peak=("Injured", "max"))
    actual = summary.set_index(keys + ["iteration"]).sort_index()
    for column in independent:
        np.testing.assert_allclose(actual[column],
                                   independent.sort_index()[column],
                                   rtol=1e-12, atol=1e-12)
    stored = pd.read_csv(stored_path).set_index(
        keys + ["iteration"]).sort_index()
    require(actual.index.equals(stored.index),
            f"Archived summary run keys differ: {stored_path.name}")
    for column in actual:
        np.testing.assert_allclose(actual[column], stored[column],
                                   rtol=1e-10, atol=1e-12,
                                   err_msg=f"Summary mismatch: {column}")
    return summary


def pairwise(groups, family):
    rows = []
    for (name_a, a), (name_b, b) in itertools.combinations(groups.items(), 2):
        result = stats.mannwhitneyu(a, b, alternative="two-sided",
                                   method="asymptotic", use_continuity=True)
        rows.append({"family": family, "a": name_a, "b": name_b,
                     "n_a": len(a), "n_b": len(b), "U": float(result.statistic),
                     "p_raw": float(result.pvalue),
                     "cliffs_delta_a_minus_b": cliffs_delta(a, b),
                     "median_difference_pp_b_minus_a":
                         float(100 * (np.median(b) - np.median(a)))})
    adjusted = holm_adjust([row["p_raw"] for row in rows])
    for row, pvalue in zip(rows, adjusted):
        row["p_holm"] = float(pvalue)
    return rows


def omnibus(groups, family):
    samples = list(groups.values())
    h, pvalue = stats.kruskal(*samples)
    n, k = sum(map(len, samples)), len(samples)
    return {"family": family, "H": float(h), "df": k - 1,
            "n": n, "p_raw": float(pvalue),
            "rank_effect_raw": float((h - k + 1) / (n - k)),
            "rank_effect_nonnegative": max(0.0, float((h - k + 1) / (n - k)))}


def ptext(value):
    return r"$<0.0001$" if value < 0.0001 else f"{value:.4f}"


def interval(values):
    values = np.asarray(values) * 100
    q1, med, q3 = np.quantile(values, [0.25, 0.5, 0.75])
    return f"{med:.2f} [{q1:.2f}, {q3:.2f}]"


def write_table(filename, headers, rows, alignment):
    lines = [r"\begin{tabular}{" + alignment + "}", r"\toprule",
             " & ".join(headers) + r" \\", r"\midrule"]
    lines.extend(" & ".join(map(str, row)) + r" \\" for row in rows)
    lines += [r"\bottomrule", r"\end{tabular}", ""]
    (OUT / filename).write_text("\n".join(lines), encoding="ascii")


def plot_points(ax, groups, ylabel):
    rng = np.random.default_rng(20261026)
    for index, (name, values) in enumerate(groups.items()):
        values = np.asarray(values) * 100
        q1, med, q3 = np.quantile(values, [0.25, 0.5, 0.75])
        ax.scatter(rng.normal(index, 0.045, len(values)), values,
                   color="#2b6e8c", s=20, alpha=0.65)
        ax.errorbar(index, med, yerr=[[med - q1], [q3 - med]],
                    color="black", fmt="D", capsize=5, markersize=4)
    ax.set_xticks(range(len(groups)), list(groups))
    ax.set_ylabel(ylabel)
    ax.grid(axis="y", alpha=0.2)
    ax.spines[["top", "right"]].set_visible(False)


def main():
    datasets = [
        ("experiment_A_main_sweep", ["community_organization_level"], 50),
        ("experiment_A_channel_ablation", ["ablation_config"], 18),
        ("experiment_D", ["shelter_site_name", "shelter_cap_limit",
                          "rescue_boats_available"], 120),
    ]
    summaries, integrity, inputs = [], [], []
    for stem, keys, count in datasets:
        path = DATA / f"{stem}_results.csv"
        raw = pd.read_csv(path)
        validate_raw(raw, keys, count)
        summary = verify_summary(raw, keys, DATA / f"{stem}_summary.csv")
        summaries.append(summary)
        integrity.append({"dataset": stem, "runs": count, "rows": len(raw),
                          "columns": len(raw.columns), "last_saved_hour": 384})
        inputs.append({"path": str(path.relative_to(ROOT)),
                       "sha256": hashlib.sha256(path.read_bytes()).hexdigest()})

    a, ablation, d = summaries
    require(set(a.community_organization_level) == {0, .25, .5, .75, 1},
            "Unexpected organization settings.")
    require(a.groupby("community_organization_level").size().eq(10).all(),
            "Main-sweep replication counts differ.")
    require(set(ablation.ablation_config) == {"depth_only", "sct_only", "coef_0.1"}
            and ablation.groupby("ablation_config").size().eq(6).all(),
            "Unexpected ablation configurations or counts.")
    for site in SITES:
        for cap in [1, 10, 30]:
            cell = d[(d.shelter_site_name == site) &
                     (d.shelter_cap_limit == cap) &
                     (d.rescue_boats_available == 2)]
            require(set(cell.iteration) == set(range(8)),
                    f"Incomplete factorial cell: {site}, {cap}")
    for boats in [1, 5]:
        for site in [SITES[0], SITES[-1]]:
            cell = d[(d.shelter_site_name == f"{site}_fleet{boats}") &
                     (d.shelter_cap_limit == 10) &
                     (d.rescue_boats_available == boats)]
            require(set(cell.iteration) == set(range(6)),
                    f"Incomplete fleet cell: {site}, {boats}")

    checkpoints = list(DATA.glob("*.checkpoint.json"))
    checkpoint_count = sum(len(json.loads(p.read_text())) for p in checkpoints)
    require(len(checkpoints) == 20 and checkpoint_count == 188,
            "Checkpoint counts disagree with archived experiment design.")
    for path in checkpoints:
        keys = json.loads(path.read_text())
        require(len(keys) == len(set(keys)), f"Duplicate checkpoint keys: {path.name}")

    org = {f"{level:g}": group.Death_final.to_numpy()
           for level, group in a.groupby("community_organization_level")}
    channels = {"Both, default": org["1"]}
    channels.update({LABELS[name]: ablation.loc[
        ablation.ablation_config == name, "Death_final"].to_numpy()
        for name in ["depth_only", "sct_only", "coef_0.1"]})
    tests, pairs = [], []
    for name, groups in [("A_main", org), ("A_ablation", channels)]:
        tests.append(omnibus(groups, name))
        pairs.extend(pairwise(groups, name))

    baseline_rows = []
    for name, values in channels.items():
        baseline_rows.extend(pairwise({"No organization": org["0"], name: values},
                                      "A_baseline_comparisons"))
    for row, adjusted in zip(baseline_rows, holm_adjust(
            [row["p_raw"] for row in baseline_rows])):
        row["p_holm"] = float(adjusted)
    pairs.extend(baseline_rows)

    distance_groups = {}
    for cap in [1, 10, 30]:
        distance_groups[cap] = {
            LABELS[site]: d.loc[(d.shelter_site_name == site) &
                               (d.shelter_cap_limit == cap) &
                               (d.rescue_boats_available == 2),
                               "Death_final"].to_numpy() for site in SITES}
        family = f"D_distance_cap{cap}"
        tests.append(omnibus(distance_groups[cap], family))
        pairs.extend(pairwise(distance_groups[cap], family))
    fleets = {}
    for site in [SITES[0], SITES[-1]]:
        fleets[site] = {f"{boats}": d.loc[
            (d.shelter_site_name == (site if boats == 2 else f"{site}_fleet{boats}")) &
            (d.shelter_cap_limit == 10) & (d.rescue_boats_available == boats),
            "Death_final"].to_numpy() for boats in [1, 2, 5]}
        tests.append(omnibus(fleets[site], f"D_fleet_{site}"))
        pairs.extend(pairwise(fleets[site], f"D_fleet_{site}"))

    require(len(tests) == 7, "Unexpected omnibus family size.")
    for row, adjusted in zip(tests, holm_adjust([r["p_raw"] for r in tests])):
        row["p_holm_seven_omnibus"] = float(adjusted)

    OUT.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(tests).to_csv(OUT / "omnibus_tests.csv", index=False)
    pd.DataFrame(pairs).to_csv(OUT / "pairwise_tests.csv", index=False)
    for summary, (stem, _, _) in zip(summaries, datasets):
        summary.to_csv(OUT / f"{stem}_verified_summary.csv", index=False)
    write_table("organization_table.tex",
                ["Organization", "$n$", "Mortality (\\%)",
                 "Sampled peak stranded (\\%)"],
                [[f"{level:g}", len(g), interval(g.Death_final),
                  interval(g.Stranded_peak)]
                 for level, g in a.groupby("community_organization_level")],
                "rrrr")
    write_table("ablation_table.tex", ["Configuration", "$n$", "Mortality (\\%)"],
                [["No organization", 10, interval(org["0"])]] +
                [[name, len(values), interval(values)]
                 for name, values in channels.items()], "lrr")
    write_table("distance_table.tex",
                ["Site", "Distance (km)", "1\\% capacity",
                 "10\\% capacity", "30\\% capacity"],
                [[LABELS[site], f"{DISTANCES[site]:.1f}"] +
                 [interval(distance_groups[cap][LABELS[site]])
                  for cap in [1, 10, 30]] for site in SITES], "lrrrr")
    write_table("fleet_table.tex",
                ["Site", "1 boat ($n=6$)", "2 boats ($n=8$)", "5 boats ($n=6$)"],
                [[LABELS[site]] + [interval(fleets[site][str(b)])
                                  for b in [1, 2, 5]] for site in fleets], "lrrr")
    test_names = ["Organization", "Channel ablation", "Distance, 1\\%",
                  "Distance, 10\\%", "Distance, 30\\%",
                  "Fleet, nearby crossing", "Fleet, Barranquilla"]
    write_table("tests_table.tex",
                ["Comparison", "$H$", "df", "$p$", "$p_{\\mathrm{Holm}}$",
                 "$\\eta_H^2$"],
                [[name, f"{test['H']:.3f}", test["df"], ptext(test["p_raw"]),
                  ptext(test["p_holm_seven_omnibus"]),
                  f"{test['rank_effect_nonnegative']:.3f}"]
                 for name, test in zip(test_names, tests)], "lrrrrr")
    baseline_frame = pd.DataFrame(baseline_rows)
    write_table("baseline_table.tex",
                ["Configuration vs. no organization", "$p$", "$p_{\\mathrm{Holm}}$",
                 "Cliff's $\\delta$"],
                [[row["b"], ptext(row["p_raw"]), ptext(row["p_holm"]),
                  f"{row['cliffs_delta_a_minus_b']:+.3f}"]
                 for row in baseline_rows], "lrrr")

    plt.rcParams.update({"font.size": 10, "pdf.fonttype": 42})
    fig, axes = plt.subplots(1, 2, figsize=(10, 3.9))
    plot_points(axes[0], org, "Mortality at last saved hour (%)")
    stranded = {f"{level:g}": group.Stranded_peak.to_numpy()
                for level, group in a.groupby("community_organization_level")}
    plot_points(axes[1], stranded, "Sampled peak stranded (%)")
    for ax, label in zip(axes, ["(a) Mortality", "(b) Stranding"]):
        ax.set_xlabel("Community organization level")
        ax.set_title(label, loc="left")
    fig.tight_layout()
    fig.savefig(OUT / "organization.pdf", bbox_inches="tight")
    plt.close(fig)
    fig, ax = plt.subplots(figsize=(9, 4))
    groups = {"No\norganization": org["0"]}
    groups.update({name.replace(", ", ",\n"): values for name, values in channels.items()})
    plot_points(ax, groups, "Mortality at last saved hour (%)")
    fig.tight_layout()
    fig.savefig(OUT / "ablation.pdf", bbox_inches="tight")
    plt.close(fig)
    fig, axes = plt.subplots(1, 2, figsize=(10.5, 4))
    for cap in [1, 10, 30]:
        medians = [100 * np.median(v) for v in distance_groups[cap].values()]
        axes[0].plot(list(DISTANCES.values()), medians, "o-",
                     label=f"{cap}% capacity")
    axes[0].set_xlabel("Scenario route distance (km)")
    axes[0].set_ylabel("Median mortality (%)")
    axes[0].set_title("(a) Capacity and distance", loc="left")
    for site, groups in fleets.items():
        medians = [100 * np.median(v) for v in groups.values()]
        q1 = [100 * np.quantile(v, .25) for v in groups.values()]
        q3 = [100 * np.quantile(v, .75) for v in groups.values()]
        axes[1].errorbar([1, 2, 5], medians,
                         yerr=[np.array(medians) - q1, np.array(q3) - medians],
                         fmt="o-", capsize=4, label=LABELS[site])
    axes[1].set_xlabel("Number of rescue boats")
    axes[1].set_ylabel("Median mortality (%)")
    axes[1].set_xticks([1, 2, 5])
    axes[1].set_title("(b) Fleet size, 10% capacity", loc="left")
    for ax in axes:
        ax.legend(fontsize=8)
        ax.grid(alpha=.2)
        ax.spines[["top", "right"]].set_visible(False)
    fig.tight_layout()
    fig.savefig(OUT / "logistics.pdf", bbox_inches="tight")
    plt.close(fig)

    report = {
        "status": "Archived-data consistency checks passed; not model validation.",
        "datasets": integrity,
        "checkpoint_files": len(checkpoints), "checkpointed_runs": checkpoint_count,
        "inputs": inputs,
        "environment": {"python": platform.python_version(),
                        "numpy": np.__version__, "pandas": pd.__version__,
                        "scipy": scipy.__version__, "matplotlib": matplotlib.__version__},
        "last_saved_hour": 384, "simulated_hours": 408,
        "sampling_warning": "Peaks are maxima over saved daily observations, not hourly maxima.",
        "summary_checks": "All archived summary columns agree within 1e-10 relative tolerance.",
        "inference": "Two-sided asymptotic Mann-Whitney with continuity correction; Holm within each family; seven mortality omnibus tests also Holm-adjusted.",
        "omnibus": tests,
        "baseline_vs_weak_attenuation": baseline_frame.loc[
            baseline_frame.b == LABELS["coef_0.1"]].to_dict("records"),
    }
    (OUT / "validation.json").write_text(
        json.dumps(report, indent=2, allow_nan=False) + "\n", encoding="ascii")
    print(json.dumps({"runs_verified": sum(r["runs"] for r in integrity),
                      "rows_verified": sum(r["rows"] for r in integrity),
                      "omnibus": tests,
                      "baseline_comparisons": baseline_rows}, indent=2))
    print(f"Manuscript assets written to {OUT}")


if __name__ == "__main__":
    main()
