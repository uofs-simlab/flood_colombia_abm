"""Verify a complete implementation pilot and export descriptive review evidence."""

import argparse
import json
from pathlib import Path
import sys
import zipfile

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "run"))
import paper_results as primary
from hpc.transfer import RUNTIME_PACKAGES, execution_fingerprints, portable_contract, run_fingerprint
from structural import pilot
from verify_model import complete, sha256, write_json


def require(condition, message):
    if not condition:
        raise ValueError(message)


def read_json(path):
    return json.loads(path.read_text(encoding="utf-8"))


def normalized_hashes(values):
    return {name.replace("\\", "/"): value for name, value in values.items()}


def ordered(frame, columns):
    return frame.sort_values(columns).reset_index(drop=True)


def describe(changes):
    rows = []
    for (variant, contrast), group in changes.groupby(["variant", "contrast"], sort=True):
        require(group.seed.tolist() == list(pilot.SEEDS), "Unexpected descriptive seed pairing.")
        reference = group.reference_difference_deaths
        alternative = group.variant_difference_deaths
        rows.append({
            "variant": variant, "contrast": contrast, "pilot_seeds": len(group),
            "reference_mean_difference_pp": float(reference.mean() * 100 / 972),
            "alternative_mean_difference_pp": float(alternative.mean() * 100 / 972),
            "mean_change_in_difference_pp": float(group.change_in_difference_deaths.mean() * 100 / 972),
            "largest_absolute_seed_change_pp": float(group.change_in_difference_deaths.abs().max() * 100 / 972),
            "sign_reversals_in_pilot_seeds": int(((reference * alternative) < 0).sum()),
            "alternative_zero_in_pilot_seeds": int((alternative == 0).sum()),
            "interpretation": "Descriptive two-seed diagnostic, not robustness or policy inference.",
        })
    require(len(rows) == 32, "Expected all eight variants and four contrasts.")
    return pd.DataFrame(rows)


def review(batch, reference, output, expected_fingerprint):
    require(len(expected_fingerprint) == 64
            and all(c in "0123456789abcdef" for c in expected_fingerprint),
            "An approved SHA256 execution fingerprint is required.")
    require(sha256(batch / "manifest.json") == expected_fingerprint,
            "Execution manifest differs from the approved fingerprint.")
    metadata = read_json(batch / "manifest.json")
    require(metadata.get("study_kind") == "structural_pilot"
            and metadata["steps"] == 408 and metadata["seeds"] == list(pilot.SEEDS)
            and metadata["anchors"] == list(pilot.ANCHORS)
            and metadata["variants"] == pilot.VARIANTS,
            "Pilot configuration differs from the approved protocol.")
    current = pilot.manifest(reference)
    require(metadata["python"].split()[0].split(".")[:2]
            == current["python"].split()[0].split(".")[:2] == ["3", "12"],
            "Pilot review requires the pinned Python 3.12 runtime.")
    packages = [normalized_hashes({name.lower().replace("_", "-"): version
                                  for name, version in item["packages"].items()})
                for item in (metadata, current)]
    for name in RUNTIME_PACKAGES:
        require(name in packages[0] and packages[0][name] == packages[1].get(name),
                f"Scientific review runtime differs: {name}")
    require(portable_contract(metadata)["scenarios"] == portable_contract(current)["scenarios"],
            "Pilot scenario semantics differ from the approved protocol.")
    for field in ("source_sha256", "spatial_sha256"):
        require(normalized_hashes(metadata[field]) == normalized_hashes(current[field]),
                f"Pilot {field} no longer matches the retained implementation.")
    expected_names = {
        f"{name}__seed{seed}" for name in metadata["scenarios"] for seed in pilot.SEEDS}
    require(len(expected_names) == 126, "Unexpected pilot run budget.")
    validation = read_json(batch / "validation.json")
    require(validation["manifest_sha256"] == expected_fingerprint
            and validation["runs"] == 126 and validation["hours_per_run"] == 408
            and validation["historical_reference_matches"] == 14
            and validation["variant_runs"] == 112
            and validation["independent_hourly_output_readback"] is True
            and validation["policy_effect_inference"] is False
            and validation["production_runs_authorized"] is False,
            "Final pilot validation is missing or inconsistent.")
    require(set(validation["completion_sha256"]) == expected_names,
            "Final validation does not cover every planned completion record.")
    for name in expected_names:
        require(complete(batch / name, expected_fingerprint)
                and sha256(batch / name / "completion.json") == validation["completion_sha256"][name],
                f"Pilot completion changed: {name}")
    expected_outputs = {"pilot_run_summary.csv", "pilot_contrast_changes.csv"}
    require(set(validation["descriptive_outputs_sha256"]) == expected_outputs,
            "Unexpected descriptive pilot outputs.")
    for name, expected in validation["descriptive_outputs_sha256"].items():
        require(sha256(batch / name) == expected, f"Saved pilot description changed: {name}")
    gate = read_json(batch / "reference_gate.json")
    require(gate["manifest_sha256"] == expected_fingerprint
            and gate["exact_reference_matches"] == 14
            and gate["historical_manifest_sha256"] == metadata["reference_manifest_sha256"],
            "The historical-reference gate is inconsistent.")
    original_hash = sha256(reference / "manifest.json")
    executions = execution_fingerprints(reference)
    require(metadata["reference_manifest_sha256"] in {original_hash, *executions},
            "The canonical references do not retain the approved historical execution.")
    for seed in pilot.SEEDS:
        for anchor in pilot.ANCHORS:
            historical = reference / f"{anchor}__seed{seed}"
            pilot.compare_reference(
                batch / f"reference__{anchor}__seed{seed}", historical,
                run_fingerprint(historical, original_hash, executions))
    frame, _, _ = primary.load_cohort(
        batch, expected_manifest=metadata, scenario_names=metadata["scenarios"])
    changes = pilot.contrast_changes(frame)
    pd.testing.assert_frame_equal(
        ordered(pd.read_csv(batch / "pilot_run_summary.csv", float_precision="round_trip"),
                ["scenario", "seed"]),
        ordered(frame, ["scenario", "seed"]), check_exact=True)
    pd.testing.assert_frame_equal(
        ordered(pd.read_csv(batch / "pilot_contrast_changes.csv", float_precision="round_trip"),
                ["seed", "contrast", "variant"]),
        ordered(changes, ["seed", "contrast", "variant"]), check_exact=True)
    overview = describe(changes)
    require(not output.exists() or not any(output.iterdir()),
            "Review output already contains evidence; choose a fresh destination.")
    require(not output.is_relative_to(batch) and not batch.is_relative_to(output)
            and not output.is_relative_to(reference) and not reference.is_relative_to(output),
            "Review output must be separate from simulation and reference data.")
    output.mkdir(parents=True, exist_ok=True)
    overview.to_csv(output / "descriptive_overview.csv", index=False)
    report = {
        "execution_manifest_sha256": expected_fingerprint,
        "review_source_sha256": sha256(Path(__file__)),
        "verified_runs": len(frame),
        "hourly_activations_per_run": 408,
        "independent_hourly_output_readback": True,
        "exact_canonical_reference_matches": 14,
        "seedwise_descriptive_modifications": len(changes),
        "variant_contrast_descriptions": len(overview),
        "policy_effect_or_robustness_inference": False,
        "production_runs_authorized": False,
        "reported_publication_assets_modified": False,
        "overview_sha256": sha256(output / "descriptive_overview.csv"),
        "diagnostics": overview.to_dict(orient="records"),
        "next_design_for_author_decision": {
            "retain_all_variants": True,
            "reason": "Do not retain or drop variants according to two-seed effect sizes or signs.",
            "candidate_common_seeds": list(range(20261026, 20261046)),
            "candidate_configurations": 9,
            "candidate_reference_settings": 7,
            "candidate_total_runs": 1260,
            "candidate_effect_modification_comparisons": 32,
            "candidate_inference": (
                "Paired integer-death differences in contrasts; exact sign-flip tests "
                "with Holm correction across all 32 modifications; unadjusted 95% "
                "Monte Carlo intervals quantify finite-seed precision, not empirical uncertainty."
            ),
            "reuse_of_pilot_runs": "Requires a separately verified execution bridge; never relabel pilot output.",
            "authorization": "Proposal only; requires author approval and a frozen production contract.",
            "limits": "One-at-a-time structural tests do not establish clinical calibration or joint/global robustness.",
        },
    }
    write_json(output / "review.json", report)
    members = {}
    for path in batch.rglob("*"):
        if path.is_file():
            require(not path.is_symlink() and path.suffix in (".json", ".csv", ".log"),
                    f"Unexpected file in completed pilot: {path}")
            members[path.relative_to(batch).as_posix()] = path
    for field in ("source_sha256", "spatial_sha256"):
        for name in metadata[field]:
            normalized = name.replace("\\", "/")
            members["source/" + normalized] = ROOT / normalized
    members["source/analysis/structural_pilot_review.py"] = Path(__file__)
    members["review/review.json"] = output / "review.json"
    members["review/descriptive_overview.csv"] = output / "descriptive_overview.csv"
    inventory = {name: sha256(path) for name, path in sorted(members.items())}
    archive = output / "verified_pilot_results.zip"
    with zipfile.ZipFile(archive, "x", zipfile.ZIP_DEFLATED) as handle:
        for name, path in sorted(members.items()):
            require(not path.is_symlink(), f"Cannot export a symbolic link: {path}")
            handle.write(path, name)
        handle.writestr("retrieval_inventory.json", json.dumps(inventory, indent=2, sort_keys=True) + "\n")
    import hashlib
    with zipfile.ZipFile(archive) as handle:
        for name, expected in inventory.items():
            require(hashlib.sha256(handle.read(name)).hexdigest() == expected,
                    f"Export readback failed: {name}")
    write_json(output / "export.json", {
        "archive": archive.name, "archive_sha256": sha256(archive),
        "execution_manifest_sha256": expected_fingerprint,
        "archived_files": len(inventory),
        "review_sha256": sha256(output / "review.json"),
        "all_archive_member_hashes_verified": True,
        "public_release": False,
    })
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--batch", type=Path, required=True)
    parser.add_argument("--reference-batch", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--expected-manifest-sha256", required=True)
    args = parser.parse_args()
    report = review(args.batch.resolve(), args.reference_batch.resolve(),
                    args.out.resolve(), args.expected_manifest_sha256)
    print(f"PASS: verified all {report['verified_runs']} pilot runs, 14 exact references "
          "and all descriptive outputs; checksum-verified export ready. No production authorization.")


if __name__ == "__main__":
    main()
