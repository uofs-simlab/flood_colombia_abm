"""Verify saved structural outcomes and report the complete fixed replicated design."""

import argparse
import hashlib
import json
from pathlib import Path
import sys
import zipfile

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "run"))
import paper_results as primary
from structural import production
from hpc.transfer import RUNTIME_PACKAGES, portable_contract
from verify_model import complete, sha256, write_json
from structural_pilot_review import normalized_hashes, read_json, require

OUTPUTS = ("production_run_summary.csv", "production_contrast_changes.csv", "production_effects.csv")


def effects(frame):
    expected = {(name, seed) for name in production.pilot.scenarios() for seed in production.SEEDS}
    identities = list(zip(frame.scenario, frame.seed))
    require(len(identities) == len(set(identities)) and set(identities) == expected,
            "Incomplete, duplicated or unexpected structural seed pairing.")
    values = frame.Deaths.to_numpy(dtype=float)
    require(np.isfinite(values).all() and (values == np.rint(values)).all()
            and (values >= 0).all() and (values <= primary.POPULATION).all(),
            "Structural contrasts require finite integer population death counts.")
    wide = frame.pivot(index="seed", columns="scenario", values="Deaths").reindex(production.SEEDS)
    rows, changes = [], []
    for contrast, label, first, second in production.pilot.CONTRASTS:
        original = (wide[f"reference__{second}"] - wide[f"reference__{first}"]).to_numpy(dtype=int)
        for variant in production.pilot.VARIANTS:
            a, b = wide[f"{variant}__{first}"], wide[f"{variant}__{second}"]
            difference = (b - a).to_numpy(dtype=int)
            modification = difference - original
            mean, low, high = primary.mean_interval(difference * 100 / primary.POPULATION)
            shift, shift_low, shift_high = primary.mean_interval(modification * 100 / primary.POPULATION)
            rows.append({
                "variant": variant, "contrast": contrast, "label": label, "n_seeds": 20,
                "first_mean_mortality_pct": float(a.mean() * 100 / primary.POPULATION),
                "second_mean_mortality_pct": float(b.mean() * 100 / primary.POPULATION),
                "mean_effect_pp": mean, "mc95_low_pp": low, "mc95_high_pp": high,
                "change_from_reference_pp": shift,
                "change_mc95_low_pp": shift_low, "change_mc95_high_pp": shift_high,
                "p_change_raw": (None if variant == "reference"
                                 else primary.paired_sign_flip(modification)),
                "positive_difference_seeds": int((difference > 0).sum()),
                "negative_difference_seeds": int((difference < 0).sum()),
                "zero_difference_seeds": int((difference == 0).sum()),
                "effect_mc_half_width_pp": (high - low) / 2,
                "change_mc_half_width_pp": (shift_high - shift_low) / 2,
                "precision_flag_effect": (high - low) / 2 > .5,
                "precision_flag_change": (shift_high - shift_low) / 2 > .5,
            })
            if variant != "reference":
                changes.extend({
                    "variant": variant, "contrast": contrast, "seed": seed,
                    "reference_difference_deaths": int(base),
                    "variant_difference_deaths": int(changed),
                    "change_in_difference_deaths": int(delta),
                    "change_in_difference_pp": float(delta * 100 / primary.POPULATION),
                } for seed, base, changed, delta in zip(
                    production.SEEDS, original, difference, modification))
    result = pd.DataFrame(rows)
    applicable = result.variant.ne("reference")
    require(len(result) == 36 and int(applicable.sum()) == 32 and len(changes) == 640,
            "The fixed 32 structural comparisons or 640 paired changes differ.")
    result.loc[applicable, "p_change_holm_32"] = primary.holm_adjust(
        result.loc[applicable, "p_change_raw"].tolist())
    return result, pd.DataFrame(changes)


def review(batch, reference, approval, output, expected_fingerprint):
    require(len(expected_fingerprint) == 64
            and all(c in "0123456789abcdef" for c in expected_fingerprint),
            "An approved SHA256 execution fingerprint is required.")
    require(sha256(batch / "manifest.json") == expected_fingerprint,
            "Production manifest differs from the approved fingerprint.")
    metadata = read_json(batch / "manifest.json")
    current = production.manifest(reference, approval)
    require(metadata["study_kind"] == "structural_production"
            and metadata["authorization"] == production.APPROVAL
            and metadata["seeds"] == list(production.SEEDS)
            and metadata["steps"] == 408 and metadata["approval_sha256"] == sha256(approval),
            "Production configuration or authorization differs.")
    require(metadata["python"].split()[0].split(".")[:2]
            == current["python"].split()[0].split(".")[:2] == ["3", "12"],
            "Structural review requires pinned Python 3.12.")
    packages = [{name.lower().replace("_", "-"): version
                 for name, version in record["packages"].items()} for record in (metadata, current)]
    for name in RUNTIME_PACKAGES:
        require(name in packages[0] and packages[0][name] == packages[1].get(name),
                f"Scientific runtime differs: {name}")
    require(portable_contract(metadata)["scenarios"] == portable_contract(current)["scenarios"],
            "Production scenario semantics changed.")
    for field in ("source_sha256", "spatial_sha256"):
        require(normalized_hashes(metadata[field]) == normalized_hashes(current[field]),
                f"Retained production {field} changed.")
    names = {f"{name}__seed{seed}" for name in metadata["scenarios"] for seed in production.SEEDS}
    validation = read_json(batch / "validation.json")
    require(validation["manifest_sha256"] == expected_fingerprint
            and validation["runs"] == 1260 and validation["hours_per_run"] == 408
            and validation["historical_reference_matches"] == 140
            and validation["variant_runs"] == 1120
            and validation["fresh_runs_only"] is True and validation["pilot_runs_reused"] == 0
            and set(validation["completion_sha256"]) == names,
            "Final structural validation does not cover all 1,260 fresh runs.")
    for name in names:
        require(complete(batch / name, expected_fingerprint)
                and sha256(batch / name / "completion.json") == validation["completion_sha256"][name],
                f"Structural completion changed: {name}")
    require(sha256(batch / "reference_gate.json") == validation["reference_gate_sha256"],
            "Reference gate changed.")
    require(read_json(batch / "reference_gate.json")
            == production.reference_gate(batch, reference, expected_fingerprint),
            "All 140 canonical historical comparisons must still pass.")
    frame, _, _ = primary.load_cohort(batch, expected_manifest=metadata,
                                    scenario_names=metadata["scenarios"])
    result, changes = effects(frame)
    require(not output.exists() or not any(output.iterdir()), "Review evidence already exists.")
    require(not any(a.is_relative_to(b) or b.is_relative_to(a)
                    for a, b in [(output, batch), (output, reference)]),
            "Review output must be separate from outcomes and references.")
    output.mkdir(parents=True, exist_ok=True)
    for name, data in zip(OUTPUTS, (frame, changes, result)):
        data.to_csv(output / name, index=False, lineterminator="\n")
    report = {
        "execution_manifest_sha256": expected_fingerprint,
        "review_source_sha256": sha256(Path(__file__)),
        "verified_runs": 1260, "hours_per_run": 408,
        "exact_canonical_reference_matches": 140,
        "independent_hourly_output_readback": True,
        "paired_seedwise_contrast_changes": 640,
        "effect_modification_comparisons": 32, "holm_family_size": 32,
        "sign_flip_assumption": current["analysis_plan"]["sign_flip_assumption"],
        "precision_flag_threshold_pp": .5,
        "mc_intervals": "Unadjusted 95% t intervals describe finite-seed precision, not empirical uncertainty.",
        "production_runs_authorized": True, "pilot_runs_reused": 0,
        "reported_primary_sensitivity_and_pilot_outputs_modified": False,
        "clinical_or_empirical_validation": False,
        "global_or_joint_robustness": False,
        "analysis_outputs_sha256": {name: sha256(output / name) for name in OUTPUTS},
        "all_variants_retained": True,
        "precision_flags_effect": int(result.precision_flag_effect.sum()),
        "precision_flags_change": int(result.precision_flag_change.sum()),
    }
    write_json(output / "review.json", report)
    members = {}
    for path in batch.rglob("*"):
        if path.is_file():
            require(not path.is_symlink() and path.suffix in (".json", ".csv", ".log"),
                    f"Unexpected production outcome file: {path}")
            members[path.relative_to(batch).as_posix()] = path
    for field in ("source_sha256", "spatial_sha256"):
        for name in metadata[field]:
            normalized = name.replace("\\", "/")
            members["source/" + normalized] = ROOT / normalized
    members["authorization/approved_design.json"] = approval
    for name in (*OUTPUTS, "review.json"):
        members["review/" + name] = output / name
    inventory = {name: sha256(path) for name, path in sorted(members.items())}
    archive = output / "verified_structural_results.zip"
    with zipfile.ZipFile(archive, "x", zipfile.ZIP_DEFLATED) as handle:
        for name, path in sorted(members.items()):
            require(not path.is_symlink(), f"Cannot export symbolic link: {path}")
            handle.write(path, name)
        handle.writestr("retrieval_inventory.json", json.dumps(inventory, indent=2, sort_keys=True) + "\n")
    with zipfile.ZipFile(archive) as handle:
        require(set(handle.namelist()) == set(inventory) | {"retrieval_inventory.json"},
                "Export members differ from inventory.")
        for name, digest in inventory.items():
            require(hashlib.sha256(handle.read(name)).hexdigest() == digest,
                    f"Export member readback failed: {name}")
    write_json(output / "export.json", {
        "archive": archive.name, "archive_sha256": sha256(archive),
        "execution_manifest_sha256": expected_fingerprint,
        "archived_files": len(inventory), "review_sha256": sha256(output / "review.json"),
        "all_archive_member_hashes_verified": True, "public_release": False,
    })
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--batch", type=Path, required=True)
    parser.add_argument("--reference-batch", type=Path, required=True)
    parser.add_argument("--approval", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--expected-manifest-sha256", required=True)
    args = parser.parse_args()
    review(args.batch.resolve(), args.reference_batch.resolve(), args.approval.resolve(),
           args.out.resolve(), args.expected_manifest_sha256)
    print("PASS: all 1,260 outcomes and 140 canonical references independently verified; complete fixed analysis and export ready.")


if __name__ == "__main__":
    main()
