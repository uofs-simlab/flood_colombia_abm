"""Resume the frozen structural cohort under a separately approved worker increase."""

import argparse
import json
import os
from pathlib import Path
import platform
import sys
import time

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "run"))
from structural import production
from verify_model import complete, sha256, write_json

AUTHORIZATION = {
    "author_approval": "User explicitly selected increasing to 48 workers while preserving completed results.",
    "approved_utc_date": "2026-10-31",
    "origin_execution_manifest_sha256": "2f08f5cbd47e00bc0079f26f3f7ba87eba455be8906d34afb452f53cea164a43",
    "origin_worker_limit": 8,
    "workers": 48,
    "total_runs": 1260,
    "reference_gate_runs": 140,
    "variant_runs": 1120,
    "preserve_completed_runs_and_fingerprints": True,
    "model_rule_or_seed_changes_authorized": False,
    "pilot_reuse_authorized": False,
}


def context(batch, authorization):
    if json.loads(authorization.read_text(encoding="utf-8")) != AUTHORIZATION:
        raise ValueError("Worker scaling authorization differs from the approved controller-only change.")
    fingerprint = sha256(batch / "manifest.json")
    if fingerprint != AUTHORIZATION["origin_execution_manifest_sha256"]:
        raise ValueError("Scaling requires the unchanged original execution manifest.")
    metadata = json.loads((batch / "manifest.json").read_text(encoding="utf-8"))
    reference = Path(metadata["reference_batch"])
    approval = Path(metadata["approval_path"])
    if metadata != production.manifest(reference, approval):
        raise ValueError("Frozen scientific source, inputs, runtime or original approval changed.")
    scenarios = metadata["scenarios"]
    expected = {f"{name}__seed{seed}" for name in scenarios for seed in production.SEEDS}
    for path in batch.glob("*/completion.json"):
        if path.parent.name not in expected or not complete(path.parent, fingerprint):
            raise ValueError(f"Unexpected or changed retained completion: {path}")
    return metadata, reference, approval, fingerprint


def operational_record(authorization, fingerprint):
    return {
        "origin_execution_manifest_sha256": fingerprint,
        "authorization": AUTHORIZATION,
        "authorization_sha256": sha256(authorization),
        "controller_source_sha256": sha256(Path(__file__)),
        "controller_source": Path(__file__).read_text(encoding="utf-8"),
        "meaning": "Supersedes only the original eight-worker operational ceiling. Scientific manifest, seeds, rules, source, outputs and execution identities remain unchanged.",
    }


def resume(batch, authorization, review_out):
    metadata, reference, approval, fingerprint = context(batch, authorization)
    paths = (batch, reference, review_out)
    if any(a.is_relative_to(b) or b.is_relative_to(a)
           for i, a in enumerate(paths) for b in paths[i + 1:]):
        raise ValueError("Outcomes, reference and review destinations must be separate.")
    if review_out.exists() and any(review_out.iterdir()):
        raise ValueError("Completed review evidence exists; refusing to overwrite.")
    record = operational_record(authorization, fingerprint)
    sidecar = batch / "operational_execution.json"
    if sidecar.exists() and json.loads(sidecar.read_text(encoding="utf-8")) != record:
        raise ValueError("Recorded worker-change provenance differs; refusing to mix controllers.")
    lock = batch / "parent.lock"
    with lock.open("x", encoding="utf-8") as handle:
        json.dump({"pid": os.getpid(), "hostname": platform.node(), "started": time.time(),
                   "controller": "scaled", "workers": 48}, handle)
    try:
        if not sidecar.exists():
            write_json(sidecar, record)
        references = [(f"reference__{anchor}", seed)
                      for seed in production.SEEDS for anchor in production.pilot.ANCHORS]
        production.run_phase(batch, references, metadata, fingerprint, 48, reference,
                             approval, "historical_reference_gate")
        if context(batch, authorization)[0] != metadata:
            raise ValueError("Frozen cohort changed during scaled reference execution.")
        gate = production.reference_gate(batch, reference, fingerprint)
        gate_path = batch / "reference_gate.json"
        if gate_path.exists() and json.loads(gate_path.read_text(encoding="utf-8")) != gate:
            raise ValueError("Existing historical gate differs from fresh comparisons.")
        if not gate_path.exists():
            write_json(gate_path, gate)
        print("PASS: all 140 historical references matched exactly; 48-worker variants enabled.", flush=True)
        variants = [(name, seed) for seed in production.SEEDS for name in metadata["scenarios"]
                    if not name.startswith("reference__")]
        production.run_phase(batch, variants, metadata, fingerprint, 48, reference,
                             approval, "stress_variants")
        names = [f"{name}__seed{seed}" for name, seed in references + variants]
        if any(not complete(batch / name, fingerprint) for name in names):
            raise ValueError("Scaled production outcome is incomplete; final validation refused.")
        if context(batch, authorization)[0] != metadata:
            raise ValueError("Frozen source changed during scaled variant execution.")
        if json.loads(sidecar.read_text(encoding="utf-8")) != record:
            raise ValueError("Worker-change provenance changed during execution.")
        write_json(batch / "validation.json", {
            "manifest_sha256": fingerprint, "runs": 1260, "hours_per_run": 408,
            "historical_reference_matches": 140, "variant_runs": 1120,
            "completion_sha256": {name: sha256(batch / name / "completion.json") for name in names},
            "reference_gate_sha256": sha256(gate_path),
            "fresh_runs_only": True, "pilot_runs_reused": 0,
        })
        sys.path.insert(0, str(ROOT / "analysis"))
        from structural_production_results import review
        review(batch, reference, approval, review_out, fingerprint)
        print("PASS: all 1,260 outcomes verified; all 32 contrast changes and export ready. Worker-change provenance is included in the exported cohort.", flush=True)
    finally:
        lock.unlink()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--batch", type=Path, required=True)
    parser.add_argument("--authorization", type=Path, required=True)
    parser.add_argument("--review-out", type=Path, required=True)
    args = parser.parse_args()
    production.require_hpc()
    resume(args.batch.resolve(), args.authorization.resolve(), args.review_out.resolve())


if __name__ == "__main__":
    main()
