"""Verify the complete standard-model cohort and regenerate publication results."""

import argparse
from collections import defaultdict
import itertools
import json
import math
from pathlib import Path
import shutil
import subprocess
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.patches import Patch, Rectangle
from matplotlib.ticker import FuncFormatter, MaxNLocator
import geopandas as gpd
import numpy as np
import pandas as pd
from scipy import stats

ROOT = Path(__file__).resolve().parents[1]
PAPER = ROOT.with_name("flood_columbia_paper")
PUBLICATION_OUTPUT = ROOT / "analysis" / "paper_generated"
sys.path.insert(0, str(ROOT / "run"))
from run_study import study_manifest
from hpc.transfer import execution_fingerprints, run_fingerprint
from study_design import SCENARIOS, SITES
from verify_model import complete, sha256, write_json
from analyze_archive import holm_adjust, require
from agents import colombia_calibration as CAL

POPULATION = 972
MORTALITY = "Mortality_pct"
SHORT_SITES = ["Near", "Hypothetical", "Intermediate", "Far"]
FAMILY_LABELS = {
    "organization": "Organization levels", "channels": "Organization channels",
    "distance_cap1": "Distance, 1\\%", "distance_cap10": "Distance, 10\\%",
    "distance_cap30": "Distance, 30\\%", "fleet_near": "Fleet, near",
    "fleet_far": "Fleet, far",
}
CONTRASTS = [
    ("No organization to both", "organization", "A_org_0", "A_org_1"),
    ("No organization to depth only", "channels", "A_org_0", "A_depth_only"),
    ("No organization to social only", "channels", "A_org_0", "A_social_only"),
    ("No organization to weak depth", "channels", "A_org_0", "A_weak_depth"),
    ("Both to depth only", "channels", "A_org_1", "A_depth_only"),
    ("Near to far, 30\\%", "distance_cap30", "D_cap30_nearby_crossing", "D_cap30_barranquilla"),
    ("One to five boats, far", "fleet_far", "D_fleet1_barranquilla", "D_fleet5_barranquilla"),
]
BLUE, ORANGE, GREEN = "#0072B2", "#D55E00", "#009E73"
FIGURE_STYLE = {
    "font.size": 9, "axes.labelsize": 9, "axes.titlesize": 10,
    "axes.spines.top": False, "axes.spines.right": False,
    "xtick.labelsize": 8, "ytick.labelsize": 8, "legend.fontsize": 8,
    "pdf.fonttype": 42, "ps.fonttype": 42, "savefig.facecolor": "white",
}
PUBLICATION_ASSETS = (
    "organization_table.tex", "ablation_table.tex", "distance_table.tex",
    "fleet_table.tex", "process_table.tex", "tests_table.tex", "contrasts_table.tex",
    "organization.pdf", "ablation.pdf", "logistics.pdf", "rescue_process.pdf",
    "study_area.pdf", "paired_effects.pdf",
    "parameters_table.tex", "design_table.tex", "verification_table.tex",
    "study_abstract.tex", "study_results_text.tex", "study_process_text.tex",
    "study_organization_text.tex", "study_logistics_text.tex", "study_inference_text.tex",
    "study_discussion_organization.tex", "study_discussion_logistics.tex",
    "study_conclusion.tex", "study_design_numbers.tex",
    "run_summary.csv", "omnibus_tests.csv", "pairwise_tests.csv",
    "monte_carlo_precision.csv", "experiment_A_main_sweep_summary.csv",
    "experiment_A_channel_ablation_summary.csv", "experiment_D_summary.csv",
)


def mean_interval(values):
    a = np.asarray(values, dtype=float)
    require(a.ndim == 1 and len(a) >= 2 and np.isfinite(a).all(),
            "Monte Carlo intervals require at least two finite seed outcomes.")
    mean = float(a.mean())
    half = float(stats.t.ppf(.975, len(a)-1) * a.std(ddof=1) / math.sqrt(len(a)))
    return mean, mean - half, mean + half


def paired_sign_flip(differences):
    """Exact two-sided sign-flip tail probability for integer death differences."""
    values = np.asarray(differences, dtype=float)
    require(values.ndim == 1 and len(values) >= 2 and np.isfinite(values).all()
            and np.allclose(values, np.rint(values), rtol=0, atol=1e-9),
            "Exact sign-flip tests require finite integer seed-paired death differences.")
    values = np.rint(values).astype(int)
    observed = abs(int(values.sum()))
    nonzero = [abs(int(v)) for v in values if v]
    distribution = {0: 1}
    for magnitude in nonzero:
        updated = defaultdict(int)
        for total, count in distribution.items():
            updated[total + magnitude] += count
            updated[total - magnitude] += count
        distribution = dict(updated)
    tail = sum(count for total, count in distribution.items() if abs(total) >= observed)
    return float(tail / 2**len(nonzero))


def families():
    return {
        "organization": [f"A_org_{level:g}" for level in [0, .25, .5, .75, 1]],
        "channels": ["A_org_0", "A_org_1", "A_depth_only", "A_social_only", "A_weak_depth"],
        **{f"distance_cap{cap}": [f"D_cap{cap}_{site}" for site in SITES]
           for cap in [1, 10, 30]},
        "fleet_near": ["D_fleet1_nearby_crossing", "D_cap10_nearby_crossing",
                      "D_fleet5_nearby_crossing"],
        "fleet_far": ["D_fleet1_barranquilla", "D_cap10_barranquilla", "D_fleet5_barranquilla"],
    }


def analyze_pairs(frame, seed_order):
    require(len(seed_order) >= 2 and len(set(seed_order)) == len(seed_order)
            and set(frame.seed) == set(seed_order)
            and set(frame.scenario) == set(SCENARIOS)
            and not frame.duplicated(["seed", "scenario"]).any(),
            "Expected one outcome for every prespecified setting and seed.")
    deaths = frame.Deaths.to_numpy(dtype=float)
    require(np.isfinite(deaths).all() and (deaths >= 0).all()
            and (deaths <= POPULATION).all() and (deaths == np.rint(deaths)).all(),
            "Mortality comparisons require integer population death counts.")
    wide = frame.pivot(index="seed", columns="scenario", values="Deaths").reindex(seed_order)
    require(not wide.isna().any().any(), "Incomplete seed pairing.")
    tests, pairs = [], []
    for family, names in families().items():
        matrix = wide[names].to_numpy(dtype=float)
        equal_within_every_seed = bool(np.all(matrix == matrix[:, :1]))
        if equal_within_every_seed:
            statistic, probability = 0.0, 1.0
        else:
            result = stats.friedmanchisquare(*matrix.T)
            statistic, probability = float(result.statistic), float(result.pvalue)
        require(math.isfinite(statistic) and math.isfinite(probability),
                f"Nonfinite Friedman result: {family}")
        tests.append({
            "family": family, "n_seeds": len(seed_order), "k_settings": len(names),
            "Friedman_statistic": statistic, "df": len(names)-1, "p_raw": probability,
            "Kendall_W": statistic / (len(seed_order) * (len(names)-1)),
            "all_within_seed_differences_zero": equal_within_every_seed,
        })
        local = []
        for first, second in itertools.combinations(names, 2):
            differences = (wide[second] - wide[first]).to_numpy(dtype=float)
            mean, low, high = mean_interval(differences * 100 / POPULATION)
            local.append({
                "family": family, "a": first, "b": second, "n_seeds": len(seed_order),
                "mean_b_minus_a_pp": mean,
                "median_seedwise_b_minus_a_pp": float(np.median(differences) * 100 / POPULATION),
                "mc95_low_pp": low, "mc95_high_pp": high,
                "nonzero_pairs": int(np.count_nonzero(differences)),
                "p_raw": paired_sign_flip(differences),
            })
        for row, adjusted in zip(local, holm_adjust([p["p_raw"] for p in local])):
            row["p_holm_family"] = float(adjusted)
        pairs.extend(local)
    for row, adjusted in zip(tests, holm_adjust([t["p_raw"] for t in tests])):
        row["p_holm_seven_families"] = float(adjusted)
    return pd.DataFrame(tests), pd.DataFrame(pairs)


def load_cohort(batch, *, expected_manifest=None, scenario_names=None):
    metadata = json.loads((batch / "manifest.json").read_text(encoding="utf-8"))
    expected_manifest = (study_manifest(metadata["replications"], metadata["seed_base"])
                         if expected_manifest is None else expected_manifest)
    scenario_names = list(SCENARIOS) if scenario_names is None else list(scenario_names)
    require(len(scenario_names) == len(set(scenario_names))
            and set(scenario_names) <= set(metadata["scenarios"]), "Invalid cohort scenario selection.")
    require(metadata == expected_manifest,
            "The executed cohort differs from current source, inputs, environment or design.")
    expected_names = {f"{scenario}__seed{seed}" for scenario in scenario_names for seed in metadata["seeds"]}
    observed = {p.parent.name for p in batch.glob("*/completion.json")}
    require(observed == expected_names,
            f"Incomplete or unexpected cohort: {len(observed)}/{len(expected_names)} completion markers; "
            f"missing examples: {sorted(expected_names-observed)[:3]}")
    fingerprint = sha256(batch / "manifest.json")
    executions = execution_fingerprints(batch)
    rows, trajectories = [], defaultdict(list)
    fractions = ["Death", "Stranded", "Injured", "Sheltered", "Hospitalized", "Evacuated"]
    for seed in metadata["seeds"]:
        for scenario in scenario_names:
            directory = batch / f"{scenario}__seed{seed}"
            executed_fingerprint = run_fingerprint(directory, fingerprint, executions)
            require(complete(directory, executed_fingerprint), f"Run is incomplete: {directory}")
            task = json.loads((directory / "task.json").read_text(encoding="utf-8"))
            summary = json.loads((directory / "summary.json").read_text(encoding="utf-8"))
            for record in [task, summary]:
                require(record["scenario"] == scenario and record["seed"] == seed
                        and record["manifest_sha256"] == executed_fingerprint,
                        f"Run identity mismatch: {directory}")
            states = pd.read_csv(directory / "hourly_states.csv")
            events = pd.read_csv(directory / "rescue_intervals.csv")
            admissions = pd.read_csv(directory / "rescue_admissions.csv")
            require(states.Simulation_Hour.tolist() == list(range(409))
                    and events.Interval_Start_Hour.tolist() == list(range(408))
                    and events.Interval_End_Hour.tolist() == list(range(1, 409)),
                    f"Incomplete hourly observations: {directory}")
            values = states[fractions].to_numpy()
            require(np.isfinite(values).all() and ((values >= 0) & (values <= 1)).all()
                    and (np.diff(states.Death) >= -1e-12).all(),
                    f"Invalid population fractions: {directory}")
            people = values * POPULATION
            require(np.allclose(people, np.rint(people), rtol=0, atol=1e-8),
                    f"Population fractions do not correspond to integer people: {directory}")
            deaths = states.Death.to_numpy() * POPULATION
            counts = events.Rescue_Admissions.to_numpy()
            slots = events.Rescue_Slots.to_numpy()
            require(np.isfinite(counts).all() and (counts >= 0).all()
                    and (counts == np.floor(counts)).all()
                    and np.isfinite(slots).all() and (counts <= slots).all(),
                    f"Invalid or excessive rescue admissions: {directory}")
            rate = events.Nominal_Rescue_Rate.to_numpy()
            parameters = metadata["scenarios"][scenario]
            distance = parameters.get("shelter_route_distance_km", 0)
            water = distance if parameters.get("shelter_route_is_water_crossing", False) else min(
                distance, CAL.FLOOD_EXIT_DISTANCE_KM)
            travel = water / CAL.BOAT_SPEED_KMH + (distance-water) / CAL.DRY_ROAD_SPEED_KMH
            boats = parameters.get("rescue_boats_available")
            seats = parameters.get("boat_capacity_per_trip")
            expected_rate = (
                (CAL.RESCUE_BOATS_AVAILABLE if boats is None else boats)
                * (CAL.BOAT_CAPACITY_PER_TRIP if seats is None else seats)
                / max(2 * travel, .2))
            require(np.isfinite(rate).all() and (rate == rate[0]).all() and rate[0] >= 0
                    and math.isclose(rate[0], expected_rate, rel_tol=0, abs_tol=1e-12),
                    f"Invalid nominal rescue throughput: {directory}")
            allowance = np.floor(np.arange(1, 409) * rate[0] + 1e-10)
            require(np.array_equal(slots, np.diff(np.r_[0, allowance]))
                    and np.array_equal(events.Admissions_Cumulative, np.cumsum(counts))
                    and np.array_equal(states.Rescue_Admissions, np.r_[0, np.cumsum(counts)]),
                    f"Rescue-credit or cumulative-count mismatch: {directory}")
            records = admissions[["hour", "person_index", "stranding_hours"]].to_numpy(dtype=float)
            require(np.isfinite(records).all() and (records == np.rint(records)).all()
                    and admissions.hour.between(0, 407).all()
                    and admissions.stranding_hours.le(admissions.hour + 1).all()
                    and len(admissions) == admissions.person_index.nunique()
                    == summary["Rescue_Admissions"] == summary["Distinct_Rescued"]
                    == int(counts.sum()) and admissions.person_index.between(0, POPULATION-1).all()
                    and admissions.stranding_hours.ge(2).all(),
                    f"Invalid rescue identities/durations: {directory}")
            interval_counts = admissions.groupby("hour").size().reindex(range(408), fill_value=0)
            require(np.array_equal(interval_counts, counts)
                    and np.array_equal(states.Distinct_Rescued, states.Rescue_Admissions)
                    and np.array_equal(events.Distinct_Rescued, states.Distinct_Rescued.iloc[1:]),
                    f"Rescue-event/state alignment mismatch: {directory}")
            for state, event, capacity in [
                ("Sheltered", "Shelter_Occupancy_End", "shelter_cap_limit"),
                ("Hospitalized", "Hospital_Occupancy_End", "healthcare_cap_limit"),
            ]:
                occupancy = events[event].to_numpy(dtype=float)
                require(np.isfinite(occupancy).all() and (occupancy >= 0).all()
                        and (occupancy <= math.ceil(parameters[capacity] * POPULATION / 100)).all()
                        and np.allclose(occupancy, states[state].iloc[1:] * POPULATION,
                                        rtol=0, atol=1e-8),
                        f"Facility occupancy/state mismatch: {directory}")
            h = summary["households"]
            require(h["people"] == h["residential_entries"] == POPULATION
                    and h["houses"] == 286 and h["three_person_houses"] == 172
                    and h["four_person_houses"] == 114 and h["houses_without_adult"] == 0
                    and h["unique_membership"] is True,
                    f"Household invariant mismatch: {directory}")
            for key, value in {
                "Death_hour384": states.Death.iloc[384],
                "Death_hour408": states.Death.iloc[-1],
                "Stranded_peak_hourly": states.Stranded.max(),
                "Shelter_Peak_People": round(states.Sheltered.max() * POPULATION),
                "Hospital_Peak_People": round(states.Hospitalized.max() * POPULATION),
                "hourly_checks": 408,
            }.items():
                require(math.isclose(summary[key], value, rel_tol=0, abs_tol=1e-14),
                        f"Summary mismatch for {key}: {directory}")
            mean_wait = float(admissions.stranding_hours.mean()) if len(admissions) else None
            recorded_wait = summary["Stranding_Hours_At_Admission_Mean"]
            require((mean_wait is None and recorded_wait is None)
                    or (mean_wait is not None and recorded_wait is not None
                        and math.isclose(mean_wait, recorded_wait, abs_tol=1e-12)),
                    f"Rescue duration summary mismatch: {directory}")
            rows.append({
                "scenario": scenario, "seed": seed, "Deaths": int(round(deaths[-1])),
                MORTALITY: float(states.Death.iloc[-1] * 100),
                "Stranded_peak_pct": float(states.Stranded.max() * 100),
                "Injured_peak_pct": float(states.Injured.max() * 100),
                "Shelter_peak_people": int(round(states.Sheltered.max() * POPULATION)),
                "Hospital_peak_people": int(round(states.Hospitalized.max() * POPULATION)),
                "Distinct_rescued": len(admissions), "Rescue_admissions": len(admissions),
                "Stranding_hours_at_admission_mean": mean_wait,
                "Stranding_hours_at_admission_max": (
                    int(admissions.stranding_hours.max()) if len(admissions) else None),
                "Hours_shelter_full": int((events.Shelter_Occupancy_End == math.ceil(
                    metadata["scenarios"][scenario]["shelter_cap_limit"] * POPULATION / 100)).sum()),
                "Nominal_rescue_rate": float(rate[0]),
                "Elapsed_seconds": summary["elapsed_seconds"],
            })
            if scenario in {"D_cap30_nearby_crossing", "D_cap30_barranquilla"}:
                trajectories[scenario].append(states[
                    ["Rescue_Admissions", "Sheltered", "Hospitalized", "Death"]].to_numpy())
    return pd.DataFrame(rows), trajectories, metadata


def ptext(value):
    return r"$<0.0001$" if value < .0001 else f"{value:.4f}"


def interval(values, decimals=2):
    a = np.asarray(values, dtype=float)
    require(a.size > 0 and np.isfinite(a).all(), "A table interval requires finite outcomes.")
    low, middle, high = np.quantile(a, [.25, .5, .75])
    return f"{middle:.{decimals}f} [{low:.{decimals}f}, {high:.{decimals}f}]"


def write_table(out, name, headers, rows, alignment):
    content = [r"\begin{tabular}{" + alignment + "}", r"\toprule",
               " & ".join(headers) + r" \\", r"\midrule"]
    content.extend(" & ".join(str(value) for value in row) + r" \\" for row in rows)
    content.extend([r"\bottomrule", r"\end{tabular}", ""])
    (out / name).write_text("\n".join(content), encoding="ascii")


def plot_groups(ax, groups, labels, ylabel):
    aligned = pd.concat(groups, axis=1).sort_index()
    require(len(aligned) >= 2 and not aligned.isna().any().any(),
            "Paired plots require the same complete seed blocks in every setting.")
    connections = ax.plot(range(len(groups)), aligned.to_numpy().T,
                          color="#888888", alpha=.28, linewidth=.65, zorder=1)
    rng = np.random.default_rng(20261026)
    for i, values in enumerate(aligned.T.to_numpy()):
        a = np.asarray(values)
        low, middle, high = np.quantile(a, [.25, .5, .75])
        points = ax.scatter(rng.uniform(i-.07, i+.07, len(a)), a, alpha=.65, s=17,
                            color=BLUE, edgecolors="white", linewidths=.3, zorder=2)
        summary = ax.errorbar(i, middle, yerr=[[middle-low], [high-middle]], fmt="D",
                             color="black", capsize=4, markersize=4)
    ax.set_xticks(range(len(labels)), labels)
    ax.set_ylabel(ylabel)
    ax.grid(axis="y", alpha=.2)
    ax.set_ylim(bottom=0)
    ax.spines[["right", "top"]].set_visible(False)
    return [points, connections[0], summary]


def selected_contrasts(pairs):
    rows = []
    for label, family, first, second in CONTRASTS:
        match = pairs.loc[(pairs.family == family) & (pairs.a == first) & (pairs.b == second)]
        require(len(match) == 1, f"Missing prespecified contrast: {label}")
        rows.append({"label": label, **match.iloc[0].to_dict()})
    return pd.DataFrame(rows)


def spatial_layers(metadata):
    parameters = metadata["scenarios"]["A_org_0"]
    paths = {name: parameters[f"{name}_file"] for name in
             ["houses", "businesses", "schools", "healthcare", "government"]}
    paths["flood"] = parameters["flood_file_1"]
    paths.update({site: metadata["scenarios"][f"D_cap1_{site}"]["shelter_file"] for site in SITES})
    layers = {name: gpd.read_file(path).to_crs("EPSG:32618") for name, path in paths.items()}
    for name, layer in layers.items():
        require(not layer.empty and layer.geometry.notna().all()
                and layer.geometry.is_valid.all(), f"Invalid study-map geometry: {name}")
    return layers


def generate_methods_assets(metadata, out):
    layers = spatial_layers(metadata)
    parameters = metadata["scenarios"]["A_org_0"]
    capacity = lambda value: math.ceil(value * parameters["N_persons"] / 100)
    alignment = r"@{}p{.24\linewidth}p{.32\linewidth}p{.36\linewidth}@{}"
    write_table(out, "parameters_table.tex", ["Component", "Specification", "Evidence and interpretation"], [
        ["Spatial sample", f"{len(layers['houses'])} homes; {len(layers['businesses'])} businesses; "
         f"{len(layers['schools'])} schools",
         "Supplied vector layers; mapped sample, not the complete municipality"],
        ["Population", "972 people; 172 three-person and 114 four-person homes",
         "3.4-person departmental household-size proxy; adult-anchored synthetic allocation"],
        ["Age composition", "268 aged 0--14; 626 aged 15--64; 78 aged 65--100",
         "Integer allocation from municipal 2026 projection shares; not 2010 event demography"],
        ["Education proxy", f"{100*CAL.PERC_EDUCATION_PEOPLE:.1f}\\% full adult education score",
         "Departmental 2005 literacy proxy; not measured flood-risk comprehension"],
        ["Identity shares", f"{100*CAL.PERC_INDIGENA:.1f}\\% Indigenous; "
         f"{100*CAL.PERC_AFROCOLOMBIANO:.1f}\\% Afro-Colombian; remainder other/no declared affiliation",
         "National 2018 proxies with inherited categorical response rules"],
        ["Flood and time", "408 activations; active common footprint at hours 169--268",
         "Three identical layer unions; fresh $\\operatorname{Unif}(10,55)$ depth per exposed query"],
        ["Organization level $g_{\\mathrm{org}}$",
         "$0\\leq g_{\\mathrm{org}}\\leq1$; $\\alpha_{\\mathrm{depth}}=0.5$; $\\alpha_{\\mathrm{social}}=0.3$",
         "Depth attenuation and social-score increase, respectively; "
         "weak attenuation uses $\\alpha_{\\mathrm{depth}}=0.1$"],
        ["Shelter / healthcare", f"10, 98 or 292 shelter places; {capacity(parameters['healthcare_cap_limit'])} beds",
         "Ceiling of nominal population percentages; exclusive protected states"],
        ["Rescue transport", "Two boats; six people/trip; 5 km/h water, 60 km/h road",
         "Average-throughput assumptions; no explicit boat or queue trajectories"],
        ["Protection and return", "Two-hour minimum stranding; 12-hour minimum shelter stay",
         "Return requires a footprint-clear home; unsafe recovered patients retain a protected place"],
    ], alignment)
    write_table(out, "design_table.tex", ["Experiment", "Varied settings", "Held fixed / reference"], [
        ["Organization sweep (5)", "$g_{\\mathrm{org}}=0,0.25,0.5,0.75,1$",
         "Both channels; 1\\% shelter capacity; default zero-distance rescue reference"],
        ["Channel comparison (3 additional)", "Depth only; social only; weak attenuation",
         "$g_{\\mathrm{org}}=1$; reuse no-organization and both-channel sweep references"],
        ["Capacity--site (12)", "1\\%, 10\\%, 30\\% capacity $\\times$ four sites",
         "$g_{\\mathrm{org}}=0$; two boats; routes 2.0, 38.4, 58.0 and 99.9 km"],
        ["Fleet (4 additional)", "One / five boats at near / far sites",
         "$g_{\\mathrm{org}}=0$; 10\\% capacity; reuse two-boat capacity--site references"],
        ["Replication", f"{metadata['replications']} common seed blocks per setting",
         f"{len(SCENARIOS)*metadata['replications']} unique primary runs; shared references are not additional runs"],
    ], alignment)
    write_table(out, "verification_table.tex", ["Property", "Recorded verification", "What it does not establish"], [
        ["Population", "Unique residential membership; balanced household sizes; an adult in every home",
         "Empirical kinship or household wealth/size distributions"],
        ["Protected states", "Hourly exclusive membership, facility capacity and spatial-placement checks",
         "Real-world shelter safety or clinical outcomes"],
        ["Clinical progression", "One injury-clock increment per hour; common death deadline; matched care-state and recovery-boundary tests",
         "Calibrated recovery eligibility, survival thresholds or off-site access"],
        ["Rescue accounting", "Hourly fractional-credit conservation; admission identities; no readmissions",
         "Observed boat trajectories or queue waits"],
        ["Outcome measurement", "409 states and 408 intervals per run; integer-person outcomes; endpoint readback",
         "Sub-hourly extrema or accuracy of assumed mortality thresholds"],
        ["Repeatability", "Independent full-horizon repeats; four exact reference settings before cross-platform import",
         "External validation or universal cross-platform identity"],
        ["Data integrity", "Source/input/environment manifests, output hashes and retained execution provenance",
         "Input licenses or historical provenance not yet supplied"],
    ], alignment)

    fig, axes = plt.subplots(1, 2, figsize=(8.5, 5.2), constrained_layout=True,
                             gridspec_kw={"width_ratios": [1.35, 1]})
    colors = [BLUE, "#CC79A7", GREEN, ORANGE]
    for ax in axes:
        layers["flood"].plot(ax=ax, facecolor="#DCEAF2", edgecolor="#7196AD", linewidth=.7)
        ax.set_aspect("equal")
        ax.xaxis.set_major_formatter(FuncFormatter(lambda value, _: f"{value/1000:g}"))
        ax.yaxis.set_major_formatter(FuncFormatter(lambda value, _: f"{value/1000:g}"))
        ax.xaxis.set_major_locator(MaxNLocator(4))
        ax.yaxis.set_major_locator(MaxNLocator(5))
        ax.set_xlabel("Easting (km)")
        ax.set_ylabel("Northing (km)")
        ax.tick_params(labelsize=8)
    layers["houses"].centroid.plot(ax=axes[0], color="#444444", markersize=5, zorder=3)
    for name, marker, color in [("businesses", "s", ORANGE), ("schools", "^", "#AA8800"),
                                 ("healthcare", "P", GREEN), ("government", "D", "#8855AA")]:
        layers[name].centroid.plot(ax=axes[0], marker=marker, color=color,
                                   markersize=35, edgecolor="white", linewidth=.4, zorder=4)
    for site, color in zip(SITES, colors):
        point = layers[site].geometry.centroid.iloc[0]
        for ax in axes:
            ax.scatter(point.x, point.y, marker="*", color=color, s=90,
                       edgecolor="black", linewidth=.3, zorder=5)
    local = [501800, 507000, 1138400, 1145000]
    axes[0].set_xlim(local[:2])
    axes[0].set_ylim(local[2:])
    axes[0].plot([502100, 503100], [1138800, 1138800], color="black", linewidth=2)
    axes[0].text(502600, 1139000, "1 km", ha="center", fontsize=8)
    axes[0].annotate("N", xy=(.93, .93), xytext=(.93, .79),
                     xycoords="axes fraction", ha="center",
                     arrowprops={"arrowstyle": "-|>", "color": "black"})
    axes[0].set_title("a  Residential sample and local exposure", loc="left")
    axes[1].set_xlim(489000, 529000)
    axes[1].set_ylim(1135000, 1220000)
    axes[1].add_patch(Rectangle((local[0], local[2]), local[1]-local[0], local[3]-local[2],
                                fill=False, linestyle="--", linewidth=.8, edgecolor="#444444"))
    label_positions = [(508000, 1144500), (490000, 1157000),
                       (505000, 1180000), (500000, 1213500)]
    for site, label, position in zip(SITES, ["Near: 2.0 km", "Hypothetical: 38.4 km",
                                           "Intermediate: 58.0 km", "Far: 99.9 km"], label_positions):
        point = layers[site].geometry.centroid.iloc[0]
        axes[1].annotate(label, xy=(point.x, point.y), xytext=position, fontsize=7.5,
                         arrowprops={"arrowstyle": "-", "color": "#888888", "linewidth": .6})
    axes[1].text(506000, 1136000, "Santa Lucia", fontsize=8, ha="center")
    axes[1].set_title("b  Candidate shelter locations", loc="left")
    handles = [
        Patch(facecolor="#DCEAF2", edgecolor="#7196AD", label="Supplied flood footprint"),
        Line2D([], [], marker=".", color="#444444", linestyle="None", label="Residential sample"),
        Line2D([], [], marker="s", color=ORANGE, linestyle="None", label="Businesses"),
        Line2D([], [], marker="^", color="#AA8800", linestyle="None", label="Schools"),
        Line2D([], [], marker="P", color=GREEN, linestyle="None", label="Healthcare"),
        Line2D([], [], marker="D", color="#8855AA", linestyle="None", label="Government"),
        Line2D([], [], marker="*", color="black", linestyle="None", label="Candidate shelter"),
    ]
    fig.legend(handles=handles, loc="outside lower center", ncols=3, frameon=False, fontsize=8)
    fig.savefig(out / "study_area.pdf")
    plt.close(fig)


def generate_assets(frame, trajectories, tests, pairs, out):
    out.mkdir(parents=True, exist_ok=True)
    by = {name: group.set_index("seed") for name, group in frame.groupby("scenario")}
    main = families()["organization"]
    channels = families()["channels"]
    channel_labels = ["None", "Both", "Depth only", "Social only", "Weak depth"]
    write_table(out, "organization_table.tex",
                ["Organization", "Mortality (\\%)", "Stranding peak (\\%)", "Distinct rescued"],
                [[f"{level:g}", interval(by[name][MORTALITY]),
                  interval(by[name].Stranded_peak_pct), interval(by[name].Distinct_rescued, 0)]
                 for level, name in zip([0, .25, .5, .75, 1], main)], "rrrr")
    write_table(out, "ablation_table.tex",
                ["Channels", "Mortality (\\%)", "Stranding peak (\\%)"],
                [[label, interval(by[name][MORTALITY]), interval(by[name].Stranded_peak_pct)]
                 for label, name in zip(channel_labels, channels)], "lrr")
    write_table(out, "distance_table.tex", ["Capacity", *SHORT_SITES],
                [[f"{capacity}\\%", *[interval(by[f"D_cap{capacity}_{site}"][MORTALITY])
                                      for site in SITES]] for capacity in [1, 10, 30]], "lrrrr")
    write_table(out, "fleet_table.tex", ["Site", "One boat", "Two boats", "Five boats"],
                [[label, *[interval(by[name][MORTALITY]) for name in families()[family]]]
                 for label, family in [("Near", "fleet_near"), ("Far", "fleet_far")]], "lrrr")
    process_names = ["D_cap30_nearby_crossing", "D_cap30_barranquilla",
                     "D_fleet1_barranquilla", "D_fleet5_barranquilla"]
    labels = ["Near, 30\\%, two boats", "Far, 30\\%, two boats",
              "Far, 10\\%, one boat", "Far, 10\\%, five boats"]
    process_rows = []
    for label, name in zip(labels, process_names):
        wait = by[name].Stranding_hours_at_admission_mean.dropna()
        process_rows.append([
            label, interval(by[name].Distinct_rescued, 0),
            interval(by[name].Shelter_peak_people, 0),
            interval(by[name].Hours_shelter_full, 1),
            interval(wait, 1) if len(wait) else "Not observed",
        ])
    write_table(out, "process_table.tex",
                ["Setting", "Distinct rescued", "Peak occupants", "Hours full", "Stranding (h)"],
                process_rows, "lrrrr")
    write_table(out, "tests_table.tex",
                ["Family", "$Q_{\\mathrm{Fr}}$", "df", "$p_{\\mathrm{raw}}$",
                 "$p_{\\mathrm{Holm}}$", "$A_{\\mathrm{rank}}$"],
                [[FAMILY_LABELS[row.family], f"{row.Friedman_statistic:.3f}", int(row.df),
                  ptext(row.p_raw), ptext(row.p_holm_seven_families), f"{row.Kendall_W:.3f}"]
                 for row in tests.itertuples()], "lrrrrr")
    contrasts = selected_contrasts(pairs)
    contrast_rows = []
    for row in contrasts.itertuples():
        contrast_rows.append([
            row.label, f"{row.mean_b_minus_a_pp:+.2f}",
            f"[{row.mc95_low_pp:+.2f}, {row.mc95_high_pp:+.2f}]", ptext(row.p_holm_family),
        ])
    write_table(out, "contrasts_table.tex",
                ["Contrast", "Change (pp)", "95\\% MC interval", "$p_{\\mathrm{Holm}}$"],
                contrast_rows, "lrrr")

    fig, axes = plt.subplots(1, 2, figsize=(8.5, 3.9), constrained_layout=True)
    handles = plot_groups(axes[0], [by[name][MORTALITY] for name in main],
                          ["0", ".25", ".5", ".75", "1"], "Mortality at hour 408 (%)")
    plot_groups(axes[1], [by[name].Stranded_peak_pct for name in main],
                ["0", ".25", ".5", ".75", "1"], "Hourly-state stranding peak (%)")
    for ax in axes:
        ax.set_xlabel("Organization level")
    axes[0].set_title("a  Endpoint mortality", loc="left")
    axes[1].set_title("b  Exposure response", loc="left")
    fig.legend(handles, ["Individual seed outcome", "Same-seed connection", "Median and IQR"],
               loc="outside lower center", ncols=3, frameon=False)
    fig.savefig(out / "organization.pdf")
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(7, 3.9), constrained_layout=True)
    handles = plot_groups(ax, [by[name][MORTALITY] for name in channels],
                          channel_labels, "Mortality at hour 408 (%)")
    ax.set_xlabel("Organization-channel setting")
    fig.legend(handles, ["Individual seed outcome", "Same-seed connection", "Median and IQR"],
               loc="outside lower center", ncols=3, frameon=False)
    fig.savefig(out / "ablation.pdf")
    plt.close(fig)

    fig, axes = plt.subplots(1, 2, figsize=(9, 4.8), sharey=True, constrained_layout=True)
    jitter = np.random.default_rng(20261026)
    handles = []
    for capacity, color in zip([1, 10, 30], ["#777777", BLUE, ORANGE]):
        values = [by[f"D_cap{capacity}_{site}"][MORTALITY] for site in SITES]
        med = np.array([a.median() for a in values])
        low = med - np.array([a.quantile(.25) for a in values])
        high = np.array([a.quantile(.75) for a in values]) - med
        handles.append(axes[0].errorbar(
            range(4), med, yerr=[low, high], marker="o", capsize=3, color=color,
            label=f"a: {capacity}% shelter ({math.ceil(capacity*972/100)} places)"))
        for i, outcomes in enumerate(values):
            axes[0].scatter(jitter.uniform(i-.06, i+.06, len(outcomes)), outcomes,
                            color=color, s=9, alpha=.22, zorder=1)
    axes[0].set_xticks(range(4), ["Near\n2.0 km", "Hypothetical\n38.4 km",
                                 "Intermediate\n58.0 km", "Far\n99.9 km"])
    axes[0].set_xlabel("Candidate site and assumed route length")
    for family, label, color in [("fleet_near", "Near (2.0 km)", BLUE),
                                  ("fleet_far", "Far (99.9 km)", ORANGE)]:
        values = [by[name][MORTALITY] for name in families()[family]]
        med = np.array([a.median() for a in values])
        handles.append(axes[1].errorbar(
            [1, 2, 5], med,
            yerr=[med - np.array([a.quantile(.25) for a in values]),
                  np.array([a.quantile(.75) for a in values])-med],
            marker="o", capsize=3, label=f"b: {label}", color=color))
        for boats, outcomes in zip([1, 2, 5], values):
            axes[1].scatter(jitter.uniform(boats-.07, boats+.07, len(outcomes)), outcomes,
                            color=color, s=12, alpha=.25, zorder=1)
    axes[1].set_xlabel("Boats, 10% shelter capacity")
    axes[1].set_xticks([1, 2, 5])
    axes[0].set_title("a  Capacity and candidate site (two boats)", loc="left")
    axes[1].set_title("b  Fleet size (10% shelter capacity)", loc="left")
    for ax in axes:
        ax.set_ylabel("Mortality at hour 408 (%)")
        ax.grid(axis="y", alpha=.2)
        ax.set_ylim(bottom=0)
    handles.append(Line2D([], [], marker="o", linestyle="None", color="#777777",
                          alpha=.35, markersize=3, label="Individual seed outcome"))
    fig.legend(handles=handles, loc="outside lower center", ncols=2, frameon=False,
               title="Large markers and whiskers: median and IQR")
    fig.savefig(out / "logistics.pdf")
    plt.close(fig)

    fig, axes = plt.subplots(2, 2, figsize=(9, 6.7), constrained_layout=True)
    labels = ["a  Cumulative rescue admissions", "b  Shelter occupancy",
              "c  Hospital occupancy", "d  Cumulative mortality"]
    for name, arrays in trajectories.items():
        values = np.stack(arrays)
        label = "Near" if "nearby" in name else "Far"
        for i, ax in enumerate(axes.flat):
            scale = 1 if i == 0 else POPULATION if i in [1, 2] else 100
            low, middle, high = np.quantile(values[:, :, i] * scale, [.25, .5, .75], axis=0)
            color = BLUE if label == "Near" else ORANGE
            ax.plot(range(409), middle, label=label, color=color, linewidth=1.6)
            ax.fill_between(range(409), low, high, alpha=.17, color=color, linewidth=0)
            ax.set_xlabel("Simulation hour")
            ax.set_ylabel("Mortality (%)" if i == 3 else "People")
            ax.set_title(labels[i], loc="left")
            ax.grid(alpha=.2)
    for i, ax in enumerate(axes.flat):
        ax.axvspan(169, 269, color="#777777", alpha=.09, zorder=0)
        ax.set_xlim(156, 408)
        ax.set_ylim(bottom=0)
        if i in [1, 2]:
            limit = 292 if i == 1 else 49
            ax.axhline(limit, color="#555555", linestyle="--", linewidth=.9)
            ax.set_ylim(0, max(ax.get_ylim()[1], limit*1.12))
            ax.text(403, limit, f" {limit} places", va="bottom", ha="right", fontsize=8)
    handles, site_labels = axes[0, 0].get_legend_handles_labels()
    fig.legend(handles=[
        *handles,
        Patch(facecolor=BLUE, alpha=.17, label="IQR band (site colour)"),
        Patch(facecolor="#777777", alpha=.09, label="Active flood footprint"),
        Line2D([], [], color="#555555", linestyle="--", linewidth=.9,
               label="Facility capacity"),
    ], labels=[*[f"{label}: median" for label in site_labels], "IQR band (site colour)",
               "Active flood footprint", "Facility capacity"],
        loc="outside lower center", ncols=3, frameon=False)
    fig.savefig(out / "rescue_process.pdf")
    plt.close(fig)

    fig, axes = plt.subplots(1, 2, figsize=(9.2, 4.1), constrained_layout=True,
                             gridspec_kw={"width_ratios": [1.35, 1]})
    for ax, rows, color, title in [
        (axes[0], contrasts.iloc[:5], BLUE, "a  Organization channels"),
        (axes[1], contrasts.iloc[5:], ORANGE, "b  Rescue logistics"),
    ]:
        means = rows.mean_b_minus_a_pp.to_numpy()
        ax.errorbar(means, range(len(rows)),
                    xerr=[means-rows.mc95_low_pp, rows.mc95_high_pp-means],
                    fmt="o", color=color, capsize=4, markersize=5)
        labels = [label.replace("No organization to ", "None to ").replace("\\%", "%")
                  for label in rows.label]
        ax.set_yticks(range(len(rows)), labels)
        ax.invert_yaxis()
        ax.set_ylim(len(rows)-.6, -.6)
        ax.axvline(0, color="#555555", linestyle="--", linewidth=.9)
        ax.set_xlabel("Mean mortality change (percentage points)")
        ax.set_title(title, loc="left")
        ax.grid(axis="x", alpha=.2)
    fig.legend(handles=[
        Line2D([], [], marker="o", color="#444444", linestyle="None",
               label="Mean paired difference"),
        Line2D([], [], color="#444444", label="95% Monte Carlo interval"),
        Line2D([], [], color="#555555", linestyle="--", linewidth=.9,
               label="No difference (zero)"),
    ], loc="outside lower center", ncols=3, frameon=False)
    fig.savefig(out / "paired_effects.pdf")
    plt.close(fig)


def generate_text(frame, tests, pairs, precision, out, replications):
    by = {name: group for name, group in frame.groupby("scenario")}
    med = lambda name: float(by[name][MORTALITY].median())
    significant_families = int((tests.p_holm_seven_families < .05).sum())
    flagged = int(precision.mc95_half_width_pp.gt(.5).sum())
    org0, org1 = med("A_org_0"), med("A_org_1")
    depth, social = med("A_depth_only"), med("A_social_only")
    near, far = med("D_cap30_nearby_crossing"), med("D_cap30_barranquilla")
    boat1, boat5 = med("D_fleet1_barranquilla"), med("D_fleet5_barranquilla")
    count = len(frame)
    contrasts = selected_contrasts(pairs).set_index("label")
    organization_effect = contrasts.loc["No organization to both"]
    distance_effect = contrasts.loc["Near to far, 30\\%"]
    fleet_effect = contrasts.loc["One to five boats, far"]
    social_effect = contrasts.loc["No organization to social only"]
    effect_text = lambda row: (
        f"{row.mean_b_minus_a_pp:+.2f} percentage points "
        f"(95\\% Monte Carlo interval {row.mc95_low_pp:+.2f} to {row.mc95_high_pp:+.2f})")
    abstract = (
        f"We simulate a population of 972 people over 408 hours, "
        f"using {count} runs across 24 settings and {replications} random seeds per setting. "
        "Community organization is represented by two separate assumptions: reduced "
        "flood depth and increased social-capital scores. Rescue experiments vary "
        "shelter capacity, location and boat fleet size. "
        f"Median mortality is {org0:.2f}\\% without organization and "
        f"{org1:.2f}\\% with full organization, compared with {depth:.2f}\\% when only "
        f"depth is reduced and {social:.2f}\\% when only social-capital scores increase. "
        f"The mean paired change from no to full organization is {organization_effect.mean_b_minus_a_pp:.2f} "
        f"percentage points (95\\% Monte Carlo interval "
        f"{organization_effect.mc95_low_pp:.2f} to {organization_effect.mc95_high_pp:.2f}). "
        f"With shelter places for 30\\% of the population, median mortality is "
        f"{near:.2f}\\% at the nearer site and {far:.2f}\\% at the farther site. "
        f"At the farther site with 10\\% shelter capacity, the medians are {boat1:.2f}\\% "
        f"with one boat and {boat5:.2f}\\% with five boats.\n"
    )
    (out / "study_abstract.tex").write_text(abstract, encoding="ascii")
    narrative = (
        f"All {count} prescribed runs completed 408 hourly activations, with 409 "
        f"recorded states per run and {replications} common seeds per setting. "
        "All runs passed the population, hourly-state and rescue-accounting checks "
        "summarized in Appendix~\\ref{app:verification}. Shared reference settings "
        "are counted once. The figures show variation among seeds, not only "
        "summary values for each setting.\n"
    )
    (out / "study_results_text.tex").write_text(narrative, encoding="ascii")
    organization = (
        f"At organization levels 0, 0.25, 0.5, 0.75 and 1, median mortality at hour 408 is "
        + ", ".join(f"{med(name):.2f}\\%" for name in families()["organization"])
        + ", respectively (Table~\\ref{tab:organization}). "
        "Figure~\\ref{fig:organization} connects results sharing the same seed and "
        "shows mortality alongside the largest hourly proportion stranded. "
        f"The mean seed-paired change from no to full organization is "
        f"{effect_text(organization_effect)}.\n\n"
        f"The depth-only and social-only medians are {depth:.2f}\\% and {social:.2f}\\%, "
        f"respectively. With weaker depth attenuation, the median is {med('A_weak_depth'):.2f}\\%. "
        "The channel-separated results (Table~\\ref{tab:ablation}; "
        "Figure~\\ref{fig:ablation}) separate the effect of reducing sampled flood "
        "depths from the effect of increasing social-capital scores in the decision "
        "rules. Figure~\\ref{fig:effects} displays the mean paired differences and "
        "their Monte Carlo intervals. An interval spanning zero is compatible with "
        "differences in either direction; it does not prove that the settings are equivalent.\n"
    )
    (out / "study_organization_text.tex").write_text(organization, encoding="ascii")
    logistics = (
        "Figure~\\ref{fig:logistics} and Tables~\\ref{tab:distance}--\\ref{tab:fleet} "
        "summarize the capacity--site and fleet designs. "
        f"With shelter places for 30\\% of the population, median mortality is "
        f"{near:.2f}\\% at the near site and {far:.2f}\\% at the far site. "
        f"The mean paired change from the near to the far site is {effect_text(distance_effect)}. "
        "Distance comparisons are tested separately at each shelter capacity. "
        "A significant result at one capacity and a nonsignificant result at another "
        "do not, by themselves, show that capacity changes the distance effect.\n\n"
        f"At the far site with 10\\% shelter capacity, median mortality is "
        f"{boat1:.2f}\\% with one boat and {boat5:.2f}\\% with five boats. "
        f"The mean paired one-to-five-boat "
        f"change is {effect_text(fleet_effect)}, with within-family Holm-adjusted "
        "$p_{\\mathrm{Holm}}="
        f"{fleet_effect.p_holm_family:.6f}$. Its absolute mean change is "
        + ("smaller than" if abs(fleet_effect.mean_b_minus_a_pp)
           < abs(organization_effect.mean_b_minus_a_pp) else "at least as large as")
        + " the full-organization difference. However, the experiments use different "
        "resources and reference conditions; without comparable costs they cannot "
        "establish which option gives better value for money.\n"
    )
    (out / "study_logistics_text.tex").write_text(logistics, encoding="ascii")
    inference = (
        f"Of the seven overall tests, {significant_families} have Holm-adjusted "
        "$p_{\\mathrm{Holm}}$ values below 0.05 (Appendix~\\ref{app:inference}). That appendix also "
        "reports the differences in Figure~\\ref{fig:effects}, including comparisons "
        "without statistically detected differences. All 44 pairwise comparisons are retained "
        "in the machine-readable results.\n\n"
        f"The largest setting-specific 95\\% Monte Carlo half-width is "
        f"{precision.mc95_half_width_pp.max():.3f} percentage points; {flagged} of 24 "
        "settings exceed the prespecified 0.5-percentage-point flag. This precision "
        "assessment measures variability among random seeds under fixed assumptions. "
        "It does not measure uncertainty in the assumptions or real-world outcomes. "
        "Observing zero deaths in every sampled run would not establish zero risk.\n"
    )
    (out / "study_inference_text.tex").write_text(inference, encoding="ascii")
    organization_discussion = (
        f"The mean paired mortality change from no to full organization is {effect_text(organization_effect)}. "
        f"Depth-only and social-only mortality medians are {depth:.2f}\\% and "
        f"{social:.2f}\\%, compared with {org1:.2f}\\% under both channels. "
        + ("The depth-only median is closer to the combined-channel median than "
           "the social-only median. This suggests that the assumed depth reduction "
           "drives most of the modeled improvement, but similar medians do not prove "
           "that depth-only and combined protection are equivalent. "
           if abs(depth-org1) < abs(social-org1) else
           "Their different responses should be assessed through the reported "
           "paired channel contrasts, not a presumed mechanism ranking. ")
        + ("The social-only paired interval spans zero: the sampled runs do not "
           "clearly resolve whether this change lowers or raises mortality, and "
           "they do not prove that its effect is absent. "
           if social_effect.mc95_low_pp <= 0 <= social_effect.mc95_high_pp else
           "The social-only pointwise interval excludes zero; adjusted "
           "within-family probabilities are reported separately and the result "
           "does not establish a universal social-capital effect. ")
        + "The depth reduction is imposed by the researcher rather than produced "
        "by simulated community action. These comparisons separate its consequences "
        "from those of the social-capital scores; they do not rank the real-world "
        "value of community interventions.\n"
    )
    (out / "study_discussion_organization.tex").write_text(organization_discussion, encoding="ascii")
    logistic_discussion = (
        f"At 30\\% shelter capacity, moving from the near to the far site changes mean mortality by "
        f"{effect_text(distance_effect)}. At the far site, increasing the fleet from one to five boats "
        f"at 10\\% shelter capacity changes mean mortality by {effect_text(fleet_effect)}. "
        + (f"This fleet comparison does not meet the 0.05 threshold after "
           "Holm adjustment within its family ($p_{\\mathrm{Holm}}="
           f"{fleet_effect.p_holm_family:.4f}$), "
           "although its unadjusted Monte Carlo interval excludes zero. The test "
           "adjusts for multiple comparisons, whereas the interval describes this "
           "comparison alone; they therefore need not give the same conclusion. "
           if fleet_effect.p_holm_family >= .05
           and not fleet_effect.mc95_low_pp <= 0 <= fleet_effect.mc95_high_pp else
           f"The primary fleet contrast has within-family Holm-adjusted "
           "$p_{\\mathrm{Holm}}="
           f"{fleet_effect.p_holm_family:.4f}$; its Monte Carlo interval and "
           "adjusted test address different inferential summaries. ")
        + "The overall tests compare sites within each shelter capacity and fleets "
        "within each site. Comparing their adjusted probabilities does not test whether one "
        "resource changes the effect of another. The results depend on the assumed "
        "routes, facility protection, injury progression and rescue-throughput rules; "
        "they do not show that expanding capacity or fleets always improves outcomes.\n"
    )
    (out / "study_discussion_logistics.tex").write_text(logistic_discussion, encoding="ascii")
    process = []
    for label, name in [("near", "D_cap30_nearby_crossing"), ("far", "D_cap30_barranquilla")]:
        group = by[name]
        wait = group.Stranding_hours_at_admission_mean.dropna()
        wait_text = f"{wait.median():.2f} h" if len(wait) else "not observed"
        process.append(
            f"At the {label} site with 30\\% capacity, the median number of distinct "
            f"rescued people across runs is {group.Distinct_rescued.median():.1f}, and "
            f"median peak shelter occupancy is {group.Shelter_peak_people.median():.1f} "
            f"of 292 places. We calculate each run's mean stranding duration among "
            f"people admitted to shelter; the median of those means is {wait_text}. "
            f"The shelter is full at a median of {group.Hours_shelter_full.median():.1f} "
            f"hourly observation points per run. "
        )
    process.append(
        "Stranding durations describe admitted people only; they exclude people who "
        "were never rescued and do not separately measure queue waits or travel times. "
        "Figure~\\ref{fig:process} shows the hourly rescue, occupancy and mortality "
        "series for both sites. Slow transport can delay admissions early in the "
        "event, while lack of shelter space can prevent admissions later. The total "
        "number admitted cannot show which constraint limits rescue at each hour.\n"
    )
    (out / "study_process_text.tex").write_text("".join(process), encoding="ascii")
    conclusion = (
        f"The {count}-run primary experiment separates three mechanisms often aggregated "
        "under flood-response policy: exposure reduction, behavioural response and "
        "access to safe shelter and care. Running the organization channels separately "
        "distinguishes the assumed reduction in flood depth from changes to "
        "social-capital scores. The weaker-depth-reduction scenario shows how "
        "strongly the response depends on that physical assumption. The rescue "
        "experiments compare shelter locations and fleet sizes at specified shelter "
        "capacities. Their hourly records show why the number rescued is not enough "
        "to assess response: admission timing, length of stay and stranding duration "
        "also matter. These are findings about the study model, not proven "
        "real-world intervention effects. Independent hazard and response data, "
        "together with tests of alternative model structures, are needed before "
        "the results can guide operational decisions.\n"
    )
    (out / "study_conclusion.tex").write_text(conclusion, encoding="ascii")
    (out / "study_design_numbers.tex").write_text(
        f"\\newcommand{{\\StudyRuns}}{{{count}}}\n"
        f"\\newcommand{{\\StudySeeds}}{{{replications}}}\n"
        f"\\newcommand{{\\StudySeedFirst}}{{{int(frame.seed.min())}}}\n"
        f"\\newcommand{{\\StudySeedLast}}{{{int(frame.seed.max())}}}\n", encoding="ascii")


def publication_snapshot(out, names):
    """Locate previous generated text before regenerating its analysis files."""
    text = (PAPER / "flood_columbia.tex").read_text(encoding="utf-8")
    positions = {}
    for name in names:
        if not name.endswith(".tex"):
            continue
        require((out / name).is_file(),
                f"Previous publication text is required to refresh the manuscript: {name}")
        content = (out / name).read_text(encoding="utf-8").rstrip("\n")
        require(content.strip(), f"Previous publication text is empty: {name}")
        require(text.count(content) == 1,
                f"Expected exactly one occurrence of previous publication text for {name}. "
                "Reconcile the manuscript with its previous analysis output before refreshing.")
        start = text.index(content)
        positions[name] = (start, start + len(content))
    spans = sorted(positions.values())
    require(all(first[1] <= second[0] for first, second in zip(spans, spans[1:])),
            "Previous publication text fragments overlap in the manuscript.")
    return text, positions


def publish_assets(out, names, snapshot):
    """Refresh plain manuscript text and separate graphs without adding markers."""
    manuscript = PAPER / "flood_columbia.tex"
    text, positions = snapshot
    require(manuscript.read_text(encoding="utf-8") == text,
            "The manuscript changed during analysis; publication was not applied.")
    for name in names:
        require((out / name).is_file(), f"Missing generated publication asset: {name}")
    replacements = []
    for name in names:
        if not name.endswith(".tex"):
            continue
        require(name in positions, f"Missing manuscript publication snapshot for {name}")
        content = (out / name).read_text(encoding="utf-8").rstrip("\n")
        require(content.strip(), f"Generated manuscript content is empty: {name}")
        start, end = positions[name]
        replacements.append((start, end, content))
    for start, end, content in sorted(replacements, reverse=True):
        text = text[:start] + content + text[end:]
    figures = PAPER / "figures"
    figures.mkdir(exist_ok=True)
    for name in names:
        if name.endswith(".pdf"):
            shutil.copyfile(out / name, figures / name)
    manuscript.write_text(text, encoding="utf-8")


def compile_paper():
    for pass_index in range(5):
        result = subprocess.run(
            ["pdflatex", "--disable-installer", "-synctex=1", "-interaction=nonstopmode",
             "-halt-on-error", "-file-line-error", "flood_columbia.tex"],
            cwd=PAPER, text=True, capture_output=True)
        if result.returncode:
            raise RuntimeError("Manuscript compilation failed:\n" + result.stdout[-8000:])
        log = (PAPER / "flood_columbia.log").read_text(encoding="utf-8", errors="replace")
        if pass_index >= 1 and "Rerun to get cross-references right" not in log:
            break
    lines = log.splitlines()
    warnings = ["\n".join(lines[i:i+3]) for i, line in enumerate(lines)
                if "Overfull" in line or "undefined references" in line or "LaTeX Warning" in line]
    require(not warnings, "Manuscript build has layout/reference warnings:\n" + "\n".join(warnings))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--batch", type=Path, default=ROOT / "data_collection" / "study")
    parser.add_argument("--out", type=Path, default=PUBLICATION_OUTPUT)
    parser.add_argument("--compile", action="store_true")
    args = parser.parse_args()
    batch = args.batch.resolve()
    out = args.out.resolve()
    require(not out.is_relative_to(PAPER.resolve()),
            "Analysis outputs must stay outside the manuscript working copy.")
    require(not args.compile or out == PUBLICATION_OUTPUT.resolve(),
            "Manuscript compilation requires the standard publication-asset destination.")
    snapshot = publication_snapshot(out, PUBLICATION_ASSETS) if out == PUBLICATION_OUTPUT.resolve() else None
    frame, trajectories, metadata = load_cohort(batch)
    tests, pairs = analyze_pairs(frame, metadata["seeds"])
    precision = []
    for name in SCENARIOS:
        group = frame.loc[frame.scenario == name, MORTALITY]
        mean, low, high = mean_interval(group)
        precision.append({"scenario": name, "n_seeds": len(group),
                          "mean_mortality_pct": mean, "mc95_low_pct": low,
                          "mc95_high_pct": high, "mc95_half_width_pp": (high-low)/2,
                          "all_observed_deaths_zero": bool(group.eq(0).all())})
    precision = pd.DataFrame(precision)
    out.mkdir(parents=True, exist_ok=True)
    ready = out / "study_ready.tex"
    if ready.exists():
        ready.unlink()
    frame.to_csv(out / "run_summary.csv", index=False)
    tests.to_csv(out / "omnibus_tests.csv", index=False)
    pairs.to_csv(out / "pairwise_tests.csv", index=False)
    precision.to_csv(out / "monte_carlo_precision.csv", index=False)
    for family, names in [("experiment_A_main_sweep", families()["organization"]),
                          ("experiment_A_channel_ablation", families()["channels"]),
                          ("experiment_D", [n for n in SCENARIOS if n.startswith("D_")])]:
        frame.loc[frame.scenario.isin(names)].to_csv(out / f"{family}_summary.csv", index=False)
    with plt.rc_context(FIGURE_STYLE):
        generate_assets(frame, trajectories, tests, pairs, out)
        generate_methods_assets(metadata, out)
    generate_text(frame, tests, pairs, precision, out, metadata["replications"])
    validation = {
        "runs": len(frame), "settings": len(SCENARIOS), "replications": metadata["replications"],
        "seed_order": metadata["seeds"], "states_per_run": 409, "intervals_per_run": 408,
        "study_manifest_sha256": sha256(batch / "manifest.json"),
        "analysis_source_sha256": {
            str(p.relative_to(ROOT)): sha256(p) for p in [Path(__file__), ROOT / "analysis" / "analyze_archive.py"]
        },
        "source_completion_hashes": {p.parent.name: sha256(p)
                                     for p in sorted(batch.glob("*/completion.json"))},
        "execution_evidence_sha256": {
            str(p.relative_to(batch)): sha256(p)
            for p in sorted((batch / "executions").rglob("*")) if p.is_file()
        },
        "precision_flag_threshold_pp": .5,
        "precision_flags": precision.loc[precision.mc95_half_width_pp > .5, "scenario"].tolist(),
        "analysis_plan": metadata["analysis_plan"],
        "generated_sha256": {name: sha256(out / name) for name in PUBLICATION_ASSETS},
    }
    write_json(out / "validation.json", validation)
    ready.write_text("% Complete standard-model cohort verified; generated assets ready.\n", encoding="ascii")
    if out == PUBLICATION_OUTPUT.resolve():
        publish_assets(out, PUBLICATION_ASSETS, snapshot)
    if args.compile:
        compile_paper()
    print(f"Verified {len(frame)} runs across {len(SCENARIOS)} settings.")
    print(f"Generated assets: {out}")
    print(f"Monte Carlo precision flags (>0.5 pp): {len(validation['precision_flags'])}")


if __name__ == "__main__":
    main()
