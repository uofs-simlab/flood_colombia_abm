"""Verify and report targeted robustness without altering primary-study outcomes."""

import argparse
import json
import math
from pathlib import Path
import sys

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

import paper_results as primary

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "run"))
from sensitivity import design
from hpc.transfer import current_manifest
from verify_model import sha256, write_json

ASSETS = (
    "sensitivity_design_table.tex", "sensitivity_summary_table.tex",
    "sensitivity_effects_table.tex", "sensitivity_robustness.pdf",
    "sensitivity_methods_text.tex", "sensitivity_results_text.tex",
    "sensitivity_discussion_text.tex", "sensitivity_abstract.tex",
    "sensitivity_conclusion.tex", "sensitivity_design_numbers.tex",
    "sensitivity_run_summary.csv", "sensitivity_effects.csv",
)
FACTOR_NAMES = {
    "warning": "Warning", "hospital": "Hospital", "boat_capacity": "Boat capacity",
    "attenuation": "Depth coefficient", "social_boost": "Social coefficient",
}
COLORS = dict(zip(design.FACTORS, [primary.BLUE, primary.ORANGE, primary.GREEN, "#8855AA", "#CC79A7"]))
SHORT_CONTRASTS = {"organization": "Organization", "social_only": "Social only",
                   "site": "Site", "fleet": "Fleet"}


def effects(reference, sensitivity, seed_order):
    require = primary.require
    require(len(seed_order) >= 2 and len(set(seed_order)) == len(seed_order),
            "Sensitivity analysis requires unique common seed labels.")
    reference = reference.loc[reference.scenario.isin(design.ANCHORS)
                              & reference.seed.isin(seed_order)]
    expected = {(name, seed) for name in design.variants() for seed in seed_order}
    for frame, identities in [
        (reference, {(name, seed) for name in design.ANCHORS for seed in seed_order}),
        (sensitivity, expected),
    ]:
        observed = list(zip(frame.scenario, frame.seed))
        require(len(observed) == len(set(observed)) and set(observed) == identities,
                "Incomplete, duplicated or unexpected sensitivity seed pairing.")
        values = frame.Deaths.to_numpy(dtype=float)
        require(np.isfinite(values).all() and (values == np.rint(values)).all()
                and (values >= 0).all() and (values <= primary.POPULATION).all(),
                "Sensitivity contrasts require integer population death counts.")
    base = reference.pivot(index="seed", columns="scenario", values="Deaths").reindex(seed_order)
    changed = sensitivity.pivot(index="seed", columns="scenario", values="Deaths").reindex(seed_order)
    rows = []
    for contrast, label, first, second in design.CONTRASTS:
        original = (base[second]-base[first]).to_numpy(dtype=float)
        for factor, value in [("baseline", None), *[
            (factor, value) for factor, info in design.FACTORS.items()
            if first in info["anchors"] or second in info["anchors"]
            for value in info["levels"]
        ]]:
            if factor == "baseline":
                a, b = base[first], base[second]
            else:
                info = design.FACTORS[factor]
                a = changed[design.variant_name(factor, value, first)] if first in info["anchors"] else base[first]
                b = changed[design.variant_name(factor, value, second)] if second in info["anchors"] else base[second]
            difference = (b-a).to_numpy(dtype=float)
            modification = difference-original
            mean, low, high = primary.mean_interval(difference*100/primary.POPULATION)
            shift, shift_low, shift_high = primary.mean_interval(modification*100/primary.POPULATION)
            rows.append({
                "contrast": contrast, "label": label, "factor": factor, "value": value,
                "n_seeds": len(seed_order), "mean_effect_pp": mean,
                "first_mean_mortality_pct": float(a.mean()*100/primary.POPULATION),
                "second_mean_mortality_pct": float(b.mean()*100/primary.POPULATION),
                "mc95_low_pp": low, "mc95_high_pp": high,
                "change_from_reference_pp": shift,
                "change_mc95_low_pp": shift_low, "change_mc95_high_pp": shift_high,
                "p_change_raw": (primary.paired_sign_flip(modification) if factor != "baseline" else None),
            })
    result = pd.DataFrame(rows)
    applicable = result.factor.ne("baseline")
    require(int(applicable.sum()) == 30, "The 30 prescribed effect-modification comparisons differ.")
    result.loc[applicable, "p_change_holm_30"] = primary.holm_adjust(
        result.loc[applicable, "p_change_raw"].tolist())
    return result


def factor_label(factor, value, latex=False):
    if factor == "baseline":
        return "Primary reference"
    suffix = {"warning": " d", "hospital": r"\%" if latex else "%",
              "boat_capacity": " people"}.get(factor, "")
    return f"{FACTOR_NAMES[factor]}: {value:g}{suffix}"


def effect_interval(row, change=False):
    if change:
        return (f"{row.change_from_reference_pp:+.2f} "
                f"[{row.change_mc95_low_pp:+.2f}, {row.change_mc95_high_pp:+.2f}]")
    return f"{row.mean_effect_pp:+.2f} [{row.mc95_low_pp:+.2f}, {row.mc95_high_pp:+.2f}]"


def generate_assets(result, metadata, out):
    settings = []
    for factor, info in design.FACTORS.items():
        values = [f"{value:g}" + (f" ({math.ceil(value*primary.POPULATION/100)})"
                  if factor == "hospital" else "")
                  for value in [info["baseline"], *info["levels"]]]
        unit = info["unit"].replace("%", r"\%")
        label = info["label"] + (f" ({unit})" if unit else "")
        settings.append([label, values[0], " and ".join(values[1:]),
                         str(len(info["anchors"])*len(info["levels"]))])
    primary.write_table(out, "sensitivity_design_table.tex",
                        ["Assumption", "Reference", "Test levels", "New settings"],
                        settings, "lrrr")
    summary = []
    for contrast, group in result.groupby("contrast", sort=False):
        base = group.loc[group.factor.eq("baseline")].iloc[0]
        changed = group.loc[group.factor.ne("baseline")]
        summary.append([
            SHORT_CONTRASTS[contrast], effect_interval(base),
            f"[{changed.mean_effect_pp.min():+.2f}, {changed.mean_effect_pp.max():+.2f}]",
            f"{changed.change_from_reference_pp.abs().max():.2f}",
        ])
    primary.write_table(out, "sensitivity_summary_table.tex",
                        ["Contrast", "Reference effect [95\\% MC]", "Tested-effect range", "Max. shift"],
                        summary, "lrrr")
    primary.write_table(out, "sensitivity_effects_table.tex",
                        ["Contrast", "Assumption", "Effect [95\\% MC]", "Change [95\\% MC]", "$p_{\\mathrm{Holm}}$"], [
        [SHORT_CONTRASTS[row.contrast], factor_label(row.factor, row.value, latex=True),
         effect_interval(row), effect_interval(row, change=True), primary.ptext(row.p_change_holm_30)]
        for row in result.itertuples() if row.factor != "baseline"
    ], "llrrr")

    fig, axes = plt.subplots(2, 2, figsize=(10.2, 9.8), constrained_layout=True)
    titles = ["a  No organization to full", "b  No organization to social only",
              "c  Near to far, 30% capacity", "d  One to five boats, far"]
    for ax, (contrast, _, _, _), title in zip(axes.flat, design.CONTRASTS, titles):
        group = result.loc[result.contrast.eq(contrast)]
        for y, row in enumerate(group.itertuples()):
            ax.errorbar(row.mean_effect_pp, y,
                        xerr=[[row.mean_effect_pp-row.mc95_low_pp],
                              [row.mc95_high_pp-row.mean_effect_pp]],
                        fmt="D" if row.factor == "baseline" else "o",
                        color="black" if row.factor == "baseline" else COLORS[row.factor],
                        markersize=4, capsize=3)
        ax.set_yticks(range(len(group)), [
            factor_label(row.factor, row.value) for row in group.itertuples()], fontsize=8)
        ax.set_ylim(len(group)-.4, -.6)
        ax.axvline(0, color="#555555", linestyle="--", linewidth=.8)
        ax.axvline(group.iloc[0].mean_effect_pp, color="#AAAAAA", linestyle=":", linewidth=.8)
        ax.set_title(title, loc="left")
        ax.set_xlabel("Mean mortality change (percentage points)")
        ax.grid(axis="x", alpha=.2)
    fig.legend(handles=[
        primary.Line2D([], [], marker="D", color="black", linestyle="None",
                       label="Primary reference"),
        *[primary.Line2D([], [], marker="o", color=COLORS[factor], linestyle="None",
                         label=info["label"]) for factor, info in design.FACTORS.items()],
        primary.Line2D([], [], color="#444444", label="95% Monte Carlo interval"),
        primary.Line2D([], [], color="#AAAAAA", linestyle=":", linewidth=.8,
                       label="Primary mean"),
        primary.Line2D([], [], color="#555555", linestyle="--", linewidth=.8,
                       label="No difference (zero)"),
    ], loc="outside lower center", ncols=3, frameon=False, fontsize=9,
        title="Points show mean paired differences", title_fontsize=9)
    fig.savefig(out / "sensitivity_robustness.pdf")
    plt.close(fig)

    count = len(metadata["execution_scenarios"])*metadata["replications"]
    (out / "sensitivity_design_numbers.tex").write_text(
        f"\\newcommand{{\\SensitivityRuns}}{{{count}}}\n"
        f"\\newcommand{{\\SensitivitySeeds}}{{{metadata['replications']}}}\n", encoding="ascii")
    methods = (
        f"We conduct {count} additional runs covering 48 distinct settings, each "
        f"using the same {metadata['replications']} seed labels as the primary study. "
        "Table~\\ref{tab:sensitivity-design} gives the reference values and the "
        "alternative test values. Seven primary settings provide the references: "
        "no organization, full organization, social-only organization, the near and "
        "far sites at 30\\% shelter capacity, and one and five boats at the far "
        "site with 10\\% shelter capacity. Warning lead time, hospital capacity and "
        "people per boat trip are varied across all seven. Depth reduction is varied "
        "only at full organization; the social-capital increase is varied only at "
        "full and social-only organization. We omit parameter changes that do not "
        "apply to a setting. Settings unaffected by a parameter change reuse the "
        "verified primary runs with the same seeds and do not count as new runs.\n\n"
        "We examine four changes: from no to full organization, from no to "
        "social-only organization, from the near to the far shelter at 30\\% capacity, "
        "and from one to five boats at the far site with 10\\% capacity. For each tested "
        "parameter value, we calculate the mortality difference within each seed "
        "and average across seeds. We then subtract that seed's primary mortality "
        "difference before averaging again. This second quantity, a paired "
        "difference of differences, measures how much the parameter change alters "
        "the comparison. Reusing the same seeds preserves the dependence created "
        "by shared reference runs.\n\n"
        "Exact sign-flip tests use the integer death-count changes and the same "
        "sign-exchangeability assumption as the primary tests. Holm correction "
        "covers all 30 applicable changes in comparisons. Each approximate "
        "95\\% Monte Carlo interval based on Student's t distribution concerns "
        "one estimate, not all estimates "
        "jointly. The design was fixed after primary analysis and before the "
        "sensitivity runs. It tests selected parameter values individually; it "
        "does not measure each parameter's share of total outcome variation or "
        "the consequences of changing several parameters together.\n"
    )
    (out / "sensitivity_methods_text.tex").write_text(methods, encoding="ascii")
    paragraphs = [
        f"All {count} additional runs completed the same 408-hour horizon and passed "
        "the full hourly, household, institutional and rescue-accounting checks. "
        "Four reference repetitions reproduced primary-study numerical outputs "
        "exactly before distributed execution. Figure~\\ref{fig:sensitivity} and "
        "Table~\\ref{tab:sensitivity-summary} show how the four mortality comparisons "
        "change when individual assumptions are varied. Each effect is the "
        "second setting minus the first, in percentage points; negative values "
        "mean lower mortality in the second setting. Brackets give 95\\% Monte "
        "Carlo intervals unless identified as ranges across settings.\n"
    ]
    for contrast, label, _, _ in design.CONTRASTS:
        label = label.lower().replace("%", r"\%")
        group = result.loc[result.contrast.eq(contrast)]
        base = group.loc[group.factor.eq("baseline")].iloc[0]
        changed = group.loc[group.factor.ne("baseline")]
        widest = changed.loc[changed.change_from_reference_pp.abs().idxmax()]
        paragraphs.append(
            f"For the comparison from {label}, the primary mean mortality "
            f"difference is {effect_interval(base)} percentage points. Across the "
            f"{len(changed)} applicable sensitivity settings, mean differences range from "
            f"{changed.mean_effect_pp.min():+.2f} to {changed.mean_effect_pp.max():+.2f}. "
            f"The largest change from the primary difference occurs in the "
            f"``{factor_label(widest.factor, widest.value, latex=True)}'' setting. "
            f"The mean shift is {effect_interval(widest, change=True)} percentage points.\n"
        )
    significant = int(result.p_change_holm_30.lt(.05).sum())
    organization_reference = result.loc[
        result.contrast.eq("organization") & result.factor.eq("baseline")].iloc[0]
    hospital_low = result.loc[
        result.contrast.eq("organization") & result.factor.eq("hospital") & result.value.eq(1)].iloc[0]
    hospital_high = result.loc[
        result.contrast.eq("organization") & result.factor.eq("hospital") & result.value.eq(10)].iloc[0]
    clinical_reversal = bool(hospital_low.first_mean_mortality_pct
                             < organization_reference.first_mean_mortality_pct)
    if clinical_reversal:
        paragraphs.append(
            "The hospital-capacity response is counterintuitive in absolute outcomes, "
            "not merely a change of contrasts. Without organization, reducing nominal "
            "hospital capacity from 5\\% to 1\\% lowers mean mortality from "
            f"{organization_reference.first_mean_mortality_pct:.2f}\\% to "
            f"{hospital_low.first_mean_mortality_pct:.2f}\\%. Under full organization "
            f"the corresponding means are {organization_reference.second_mean_mortality_pct:.2f}\\% "
            f"and {hospital_low.second_mean_mortality_pct:.2f}\\%. This is a response of "
            "the implemented clinical and behavioral structure, not evidence that fewer beds improve "
            "real health outcomes.\n"
        )
    else:
        paragraphs.append(
            "The mortality levels in each scenario help explain the hospital-capacity "
            "results, not only the differences between scenarios. Without organization, mean mortality "
            f"is {hospital_low.first_mean_mortality_pct:.2f}\\%, "
            f"{organization_reference.first_mean_mortality_pct:.2f}\\% and "
            f"{hospital_high.first_mean_mortality_pct:.2f}\\% at hospital capacities "
            "of 1\\%, 5\\% and 10\\%, respectively. Under full organization, "
            f"the corresponding means are {hospital_low.second_mean_mortality_pct:.2f}\\%, "
            f"{organization_reference.second_mean_mortality_pct:.2f}\\% and "
            f"{hospital_high.second_mean_mortality_pct:.2f}\\%. "
            "These values describe the model under its assumed care rules; they "
            "are not measured clinical treatment effects.\n"
        )
    site_reference = result.loc[
        result.contrast.eq("site") & result.factor.eq("baseline")].iloc[0]
    site_low = result.loc[
        result.contrast.eq("site") & result.factor.eq("hospital") & result.value.eq(1)].iloc[0]
    paragraphs.append(
        "At 30\\% shelter capacity and 5\\% hospital capacity, mean mortality is "
        f"{site_reference.first_mean_mortality_pct:.2f}\\% at the near site and "
        f"{site_reference.second_mean_mortality_pct:.2f}\\% at the far site. "
        f"With 1\\% hospital capacity, these means are "
        f"{site_low.first_mean_mortality_pct:.2f}\\% and "
        f"{site_low.second_mean_mortality_pct:.2f}\\%, respectively. "
        "Hospital capacity therefore affects both mortality levels as well as "
        "the difference between locations.\n"
    )
    paragraphs.append(
        f"After Holm adjustment across all 30 tests, {significant} have "
        "$p_{\\mathrm{Holm}}$ values "
        "below 0.05 (Appendix~\\ref{app:sensitivity}). These tests ask whether a "
        "parameter change alters a mortality difference, not whether that difference "
        "is itself distinguishable from zero at the tested value. A nonsignificant "
        "change does not prove that the comparison is unaffected. The range of "
        "estimated differences across chosen settings is a descriptive summary, "
        "not a probability interval for real-world uncertainty.\n"
    )
    (out / "sensitivity_results_text.tex").write_text("\n".join(paragraphs), encoding="ascii")
    organization = result.loc[result.contrast.eq("organization") & result.factor.ne("baseline")]
    lower_at_all = bool(organization.mc95_high_pp.lt(0).all())
    direction = (
        "Full organization produces lower mean mortality than no organization "
        "at every tested parameter value. Each comparison's Monte Carlo interval "
        "also lies below zero. "
        if lower_at_all else
        "The full-organization mortality difference is not clearly resolved "
        "at every tested parameter value by its Monte Carlo interval. "
    )
    site = result.loc[result.contrast.eq("site") & result.factor.ne("baseline")]
    site_direction = (
        "At 30\\% shelter capacity, mean mortality is higher at the far site "
        "than at the near site at every tested parameter value. Each interval "
        "for the far-minus-near difference lies above zero, although the size "
        "of the difference changes with people per trip and hospital capacity. "
        if site.mc95_low_pp.gt(0).all() else
        "The far-minus-near mortality difference is not clearly resolved "
        "at every tested parameter value. "
    )
    fleet_large = result.loc[
        result.contrast.eq("fleet") & result.factor.eq("boat_capacity") & result.value.eq(9)].iloc[0]
    fleet_small = result.loc[
        result.contrast.eq("fleet") & result.factor.eq("boat_capacity") & result.value.eq(3)].iloc[0]
    fleet_probability = (
        "$p_{\\mathrm{Holm}}<0.0001$" if fleet_small.p_change_holm_30 < .0001
        else "$p_{\\mathrm{Holm}}=" + f"{fleet_small.p_change_holm_30:.4f}$"
    )
    fleet_text = (
        f"At three people per boat trip, increasing the far-site fleet from one "
        f"to five boats changes mean mortality by "
        f"{effect_interval(fleet_small)} percentage points; its change from the "
        f"primary difference has Holm-adjusted {fleet_probability}. "
        f"At nine people per trip, the mortality change is "
        f"{effect_interval(fleet_large)} percentage points. "
        + ("Its interval spans zero, so this setting does not clearly establish "
           "a mortality benefit from the additional boats. Nor does it show that "
           "additional boats cause harm. "
           if fleet_large.mc95_low_pp <= 0 <= fleet_large.mc95_high_pp else
           "This comparison remains conditional on per-trip capacity. ")
        + ("None of the tested parameter changes produces a statistically "
           "detected change in the fleet comparison after adjustment. "
        if not result.loc[result.contrast.eq("fleet"), "p_change_holm_30"].lt(.05).any() else
        "Some parameter changes alter the fleet comparison after adjustment; "
        "the primary result is therefore not a general estimate of the benefit "
        "of adding boats. ")
    )
    clinical_text = (
        "Hospital-capacity results depend on the common hourly "
        "injury-survival deadline and the fixed recovery-eligibility rule. "
        "An injury continues to progress outside hospital, including while "
        "waiting in shelter and after the flood. Evacuees are assumed to receive "
        "off-site care under the same recovery rule without using local beds; "
        "this access assumption is not empirically calibrated. Boundary tests "
        "verify identical survival deadlines for nonrecovering people across "
        "care states. Applying the same rule to every injured person does not "
        "guarantee that adding beds lowers total mortality in every random run: "
        "admissions, discharge and later exposure can also change. "
        + ("The nonmonotone aggregate response must not support a recommendation "
           "to restrict care and needs process-level interpretation. "
           if clinical_reversal else "")
        + "Recovery assumptions, survival deadlines and off-site access need "
        "independent evidence before the results can guide clinical planning.\n\n"
    )
    weak_depth = result.loc[
        result.contrast.eq("organization") & result.factor.eq("attenuation")
        & result.value.eq(.25)].iloc[0]
    (out / "sensitivity_discussion_text.tex").write_text(
        direction + "With a depth-reduction coefficient of 0.25, the mean "
        "mortality change from no to full organization is "
        f"{effect_interval(weak_depth)} percentage points, changing by "
        f"{effect_interval(weak_depth, change=True)} percentage points from the "
        "primary difference. "
        + ("This positive shift means a smaller modeled mortality reduction. "
           if weak_depth.change_from_reference_pp > 0 else
           "This negative shift means a larger modeled mortality reduction. "
           if weak_depth.change_from_reference_pp < 0 else
           "The estimated mortality reduction is unchanged. ")
        + "The findings for full organization do not establish "
        "a consistent benefit from social-capital increases alone, or prove that "
        "social capital has no effect.\n\n"
        + site_direction + fleet_text + "\n\n" + clinical_text
        + "The study tests selected comparisons under selected assumptions; "
        "it does not fit the parameters or establish that the findings hold under "
        "every plausible model. A change in the difference between settings "
        "shows sensitivity to a parameter, not which intermediate process caused "
        "the change. Mortality can rise similarly in both settings while their "
        "difference changes little. Hospital-capacity tests retain the same "
        "injury and discharge rules. Boat-capacity tests change people per trip, "
        "not loading time, travel speed or routing. Alternative depth dynamics, "
        "survival rules, household construction and return or discharge processes "
        "are not tested here. Changing several assumptions together may also "
        "produce responses not captured by these individual tests.\n",
        encoding="ascii")
    (out / "sensitivity_abstract.tex").write_text(
        f"A further {count} runs vary five assumptions one at a time. "
        + ("Full organization continues to lower modeled mortality relative to "
           "no organization throughout the tested ranges. " if lower_at_all else
           "The mortality difference between full and no organization is not "
           "clearly resolved at every tested parameter value. ")
        + ("The nearer shelter also continues to have lower mortality at "
           "30\\% shelter capacity." if site.mc95_low_pp.gt(0).all() else
           "The mortality difference between the near and far shelters is not "
           "clearly resolved at every tested parameter value.")
        + (" Lower hospital capacity produces an unexpected mortality response "
           "that requires further investigation." if clinical_reversal else "")
        + "\n", encoding="ascii")
    (out / "sensitivity_conclusion.tex").write_text(
        f"The {count}-run follow-up examines how these comparisons change with "
        "warning lead time, hospital beds, people per boat trip and the two "
        "organization coefficients. It tests only the listed values, one "
        "parameter at a time. Combined parameter changes and alternative model "
        "structures remain to be examined. "
        + ("The nonmonotone hospital response requires process-level interpretation "
           "before policy-facing use.\n"
           if clinical_reversal else
           "The injury, recovery and off-site care rules also require independent "
           "validation.\n"), encoding="ascii")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--batch", type=Path, default=ROOT / "data_collection" / "sensitivity")
    parser.add_argument("--primary", type=Path, default=ROOT / "data_collection" / "study")
    parser.add_argument("--out", type=Path, default=primary.PUBLICATION_OUTPUT)
    parser.add_argument("--compile", action="store_true")
    args = parser.parse_args()
    out, batch, baseline = args.out.resolve(), args.batch.resolve(), args.primary.resolve()
    primary.require(not out.is_relative_to(primary.PAPER.resolve()),
                    "Analysis outputs must stay outside the manuscript working copy.")
    primary.require(not args.compile or out == primary.PUBLICATION_OUTPUT.resolve(),
                    "Manuscript compilation requires the standard publication-asset destination.")
    snapshot = primary.publication_snapshot(out, ASSETS) if out == primary.PUBLICATION_OUTPUT.resolve() else None
    metadata = json.loads((batch / "manifest.json").read_text(encoding="utf-8"))
    design.validate_primary(metadata, baseline)
    reference, _, _ = primary.load_cohort(baseline)
    sensitivity, _, _ = primary.load_cohort(
        batch, expected_manifest=current_manifest(metadata),
        scenario_names=metadata["execution_scenarios"])
    result = effects(reference, sensitivity, metadata["seeds"])
    out.mkdir(parents=True, exist_ok=True)
    ready = out / "sensitivity_ready.tex"
    if ready.exists():
        ready.unlink()
    sensitivity.to_csv(out / "sensitivity_run_summary.csv", index=False)
    result.to_csv(out / "sensitivity_effects.csv", index=False)
    with plt.rc_context(primary.FIGURE_STYLE):
        generate_assets(result, metadata, out)
    validation = {
        "new_runs": len(sensitivity), "new_settings": len(metadata["execution_scenarios"]),
        "replications": metadata["replications"], "seeds": metadata["seeds"],
        "baseline_manifest_sha256": sha256(baseline / "manifest.json"),
        "sensitivity_manifest_sha256": sha256(batch / "manifest.json"),
        "source_completion_hashes": {path.parent.name: sha256(path)
                                     for path in sorted(batch.glob("*/completion.json"))},
        "baseline_completion_hashes": {
            f"{name}__seed{seed}": sha256(baseline / f"{name}__seed{seed}" / "completion.json")
            for name in design.ANCHORS for seed in metadata["seeds"]},
        "execution_evidence_sha256": {
            str(path.relative_to(batch)): sha256(path)
            for path in sorted((batch / "executions").rglob("*")) if path.is_file()},
        "analysis_source_sha256": {str(path.relative_to(ROOT)): sha256(path)
                                   for path in [Path(__file__), Path(primary.__file__)]},
        "analysis_plan": metadata["analysis_plan"],
        "generated_sha256": {name: sha256(out / name) for name in ASSETS},
    }
    write_json(out / "sensitivity_validation.json", validation)
    ready.write_text("% Complete targeted sensitivity cohort verified.\n", encoding="ascii")
    if out == primary.PUBLICATION_OUTPUT.resolve():
        primary.publish_assets(out, ASSETS, snapshot)
    if args.compile:
        primary.compile_paper()
    print(f"Verified {len(sensitivity)} new sensitivity runs; reused "
          f"{len(design.ANCHORS)*metadata['replications']} primary outcomes.")
    print(f"Reported {len(result)-4} applicable effect modifications; outputs: {out}")


if __name__ == "__main__":
    main()
