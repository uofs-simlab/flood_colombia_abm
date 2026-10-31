"""Targeted one-at-a-time robustness experiments using the unchanged model."""

import argparse
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "run"))
from run_study import study_manifest
from study_design import scenario_parameters, seeds
from verify_model import sha256, write_json

ANCHORS = (
    "A_org_0", "A_org_1", "A_social_only",
    "D_cap30_nearby_crossing", "D_cap30_barranquilla",
    "D_fleet1_barranquilla", "D_fleet5_barranquilla",
)
GATES = ("A_org_0", "A_org_1", "D_cap30_nearby_crossing", "D_cap30_barranquilla")
FACTORS = {
    "warning": {"label": "Warning lead time", "parameter": "warning_lead_time_days",
                "baseline": 7, "levels": [1, 3], "unit": "days", "anchors": ANCHORS},
    "hospital": {"label": "Hospital capacity", "parameter": "healthcare_cap_limit",
                 "baseline": 5, "levels": [1, 10], "unit": "%", "anchors": ANCHORS},
    "boat_capacity": {"label": "Boat capacity", "parameter": "boat_capacity_per_trip",
                     "baseline": 6, "levels": [3, 9], "unit": "people/trip", "anchors": ANCHORS},
    "attenuation": {"label": "Depth attenuation", "parameter": "org_depth_attenuation_coef",
                    "baseline": .5, "levels": [.25, .75], "unit": "",
                    "anchors": ("A_org_1",)},
    "social_boost": {"label": "Social-capital boost", "parameter": "org_sct_boost_coef",
                     "baseline": .3, "levels": [.15, .45], "unit": "",
                     "anchors": ("A_org_1", "A_social_only")},
}
CONTRASTS = (
    ("organization", "No organization to full", "A_org_0", "A_org_1"),
    ("social_only", "No organization to social only", "A_org_0", "A_social_only"),
    ("site", "Near to far, 30% capacity", "D_cap30_nearby_crossing", "D_cap30_barranquilla"),
    ("fleet", "One to five boats, far", "D_fleet1_barranquilla", "D_fleet5_barranquilla"),
)


def variant_name(factor, value, anchor):
    return f"S_{factor}_{value:g}_{anchor}"


def variants():
    return {
        variant_name(factor, value, anchor): {
            "factor": factor, "value": value, "anchor": anchor,
            "parameters": {**scenario_parameters(anchor), info["parameter"]: value},
        }
        for factor, info in FACTORS.items()
        for value in info["levels"] for anchor in info["anchors"]
    }


def sensitivity_manifest(replications, seed_base, baseline_manifest_sha256):
    if (len(baseline_manifest_sha256) != 64
            or any(c not in "0123456789abcdef" for c in baseline_manifest_sha256)):
        raise ValueError("A valid primary-cohort manifest fingerprint is required.")
    metadata = study_manifest(replications, seed_base)
    cases = variants()
    metadata.update({
        "study_kind": "sensitivity",
        "purpose": "Targeted one-at-a-time robustness of four primary-study contrasts.",
        "baseline_manifest_sha256": baseline_manifest_sha256,
        "scenarios": {**{name: scenario_parameters(name) for name in GATES},
                      **{name: case["parameters"] for name, case in cases.items()}},
        "scenario_metadata": {
            name: {key: value for key, value in case.items() if key != "parameters"}
            for name, case in cases.items()
        },
        "execution_scenarios": list(cases),
        "analysis_plan": {
            "design": "One factor at a time; two stress levels per factor; seven reference settings; omit inactive parameter-setting combinations.",
            "primary_endpoint": "Integer deaths at hour 408 divided by 972.",
            "baseline": "Reuse independently verified primary-cohort runs at the same seed labels; do not relabel or count them as new runs.",
            "ranges": "Deliberate assumption stress tests, not empirical uncertainty distributions.",
            "contrasts": [list(row) for row in CONTRASTS],
            "effects": "Mean seed-paired second-minus-first mortality change, and its seed-paired difference from the primary contrast, in percentage points.",
            "intervals": "Unadjusted 95% Monte Carlo t intervals across common seed labels; these are finite-seed precision, not empirical or structural uncertainty intervals.",
            "tests": "Exact two-sided sign-flip tests on integer death-count changes in contrasts; Holm correction across all 30 applicable factor-level-contrast modifications.",
            "scope": "Local robustness under specified ranges; no global sensitivity indices, variance attribution or joint-factor robustness claim.",
            "sampling": "Fixed replication budget; no extension or stopping based on significance.",
        },
    })
    for path in [Path(__file__), ROOT / "run" / "hpc" / "transfer.py"]:
        metadata["source_sha256"][str(path.relative_to(ROOT))] = sha256(path)
    return metadata


def reference_batch(batch, metadata):
    return ROOT / "data_collection" / "study" if metadata.get("study_kind") == "sensitivity" else batch


def validate_primary(metadata, primary):
    recorded = json.loads((primary / "manifest.json").read_text(encoding="utf-8"))
    if (sha256(primary / "manifest.json") != metadata["baseline_manifest_sha256"]
            or recorded != study_manifest(recorded["replications"], recorded["seed_base"])
            or not set(metadata["seeds"]) <= set(recorded["seeds"])
            or any(metadata["scenarios"][name] != recorded["scenarios"][name] for name in GATES)):
        raise ValueError("Primary cohort, reference settings or common seeds no longer match.")
    return recorded


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path, default=ROOT / "data_collection" / "sensitivity")
    parser.add_argument("--replications", type=int, default=20)
    parser.add_argument("--seed-base", type=int, default=20261026)
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()
    primary = ROOT / "data_collection" / "study"
    metadata = sensitivity_manifest(args.replications, args.seed_base, sha256(primary / "manifest.json"))
    validate_primary(metadata, primary)
    if not args.dry_run:
        batch = args.out.resolve()
        batch.mkdir(parents=True, exist_ok=True)
        path = batch / "manifest.json"
        if path.exists():
            if json.loads(path.read_text(encoding="utf-8")) != metadata:
                raise ValueError("Sensitivity source, environment or design changed; refusing to merge.")
        elif any(batch.iterdir()):
            raise ValueError("Sensitivity outputs exist without a manifest.")
        else:
            write_json(path, metadata)
    print(f"{len(variants())} new settings x {len(seeds(args.replications, args.seed_base))} seeds "
          f"= {len(variants())*args.replications} new runs; seven primary settings reused.")


if __name__ == "__main__":
    main()
