"""Run the author-approved, reference-gated structural study on HPC only."""

import argparse
from concurrent.futures import ThreadPoolExecutor, as_completed
import json
import os
from pathlib import Path
import platform
import subprocess
import sys
import time

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "run"))
from structural import pilot
from verify_model import complete, sha256, worker, write_json
from hpc.transfer import execution_fingerprints, run_fingerprint

SEEDS = tuple(range(20261026, 20261046))
APPROVAL = {
    "production_runs_authorized": True,
    "approved_utc_date": "2026-10-31",
    "approval": "User selected the first option approving the fixed 1,260-run replicated structural study.",
    "total_runs": 1260,
    "reference_gate_runs": 140,
    "variant_runs": 1120,
    "seeds": list(SEEDS),
    "anchors": list(pilot.ANCHORS),
    "variants": pilot.VARIANTS,
    "maximum_workers": 8,
    "fresh_runs_only": True,
    "pilot_reuse_authorized": False,
    "hosts": ["richardson", "hundsdorfer"],
    "analysis": "32 paired contrast changes; exact integer-death sign-flip tests with Holm correction across all 32; unadjusted 95% Monte Carlo t intervals and 0.5-pp precision flags.",
    "limits": "Selected one-at-a-time uncalibrated structures; not clinical calibration, empirical policy effects or joint/global robustness.",
}


def require_hpc():
    if platform.node().split(".")[0].lower() not in APPROVAL["hosts"]:
        raise RuntimeError("Full structural production simulations must run on Richardson or Hundsdorfer, not locally.")


def manifest(reference, approval):
    if json.loads(approval.read_text(encoding="utf-8")) != APPROVAL:
        raise ValueError("The structural production authorization differs from the approved fixed design.")
    recorded = json.loads((reference / "manifest.json").read_text(encoding="utf-8"))
    if not set(SEEDS) <= set(recorded["seeds"]):
        raise ValueError("All 20 approved seeds must have historical references.")
    metadata = pilot.manifest(reference)
    metadata.update({
        "study_kind": "structural_production",
        "purpose": "Author-approved replicated one-at-a-time behavioural and clinical sensitivity.",
        "replications": 20, "seed_base": SEEDS[0], "seeds": list(SEEDS),
        "authorization": APPROVAL,
        "approval_sha256": sha256(approval), "approval_path": str(approval),
        "analysis_plan": {
            **metadata["analysis_plan"],
            "budget": "9 configurations x 7 anchors x 20 common seeds = 1,260 fresh runs.",
            "gate": "All 140 fresh inherited-rule outcomes must exactly match historical primary records before the 1,120 variants run.",
            "sampling": "Fixed 20-seed design; no significance-driven extension or variant selection.",
            "inference": APPROVAL["analysis"],
            "sign_flip_assumption": "The paired sign-flip reference distribution assumes exchangeable signs under the null; it is not an empirical randomized intervention test.",
            "next_step": "Independently verify all outcomes and report all 32 prespecified contrast changes; do not overwrite primary, numerical sensitivity or pilot outcomes.",
        },
    })
    for path in (Path(__file__), ROOT / "analysis" / "structural_production_results.py",
                 ROOT / "analysis" / "structural_pilot_review.py"):
        metadata["source_sha256"][str(path.relative_to(ROOT))] = sha256(path)
    return metadata


def reference_gate(batch, reference, fingerprint):
    original = sha256(reference / "manifest.json")
    executions = execution_fingerprints(reference)
    for seed in SEEDS:
        for anchor in pilot.ANCHORS:
            historical = reference / f"{anchor}__seed{seed}"
            directory = batch / f"reference__{anchor}__seed{seed}"
            if not complete(directory, fingerprint):
                raise ValueError(f"Fresh reference is incomplete: {directory}")
            pilot.compare_reference(directory, historical,
                                    run_fingerprint(historical, original, executions))
    return {
        "manifest_sha256": fingerprint, "historical_manifest_sha256": original,
        "exact_reference_matches": 140,
        "comparison": "Every numerical CSV value and scientific summary matches exactly.",
    }


def launch(task_path):
    task = json.loads(task_path.read_text(encoding="utf-8"))
    env = os.environ.copy()
    env.update({"PYTHONHASHSEED": str(task["seed"]), "PYTHONDONTWRITEBYTECODE": "1",
                "OMP_NUM_THREADS": "1", "OPENBLAS_NUM_THREADS": "1",
                "MKL_NUM_THREADS": "1", "NUMEXPR_NUM_THREADS": "1"})
    with (task_path.parent / "worker.log").open("w", encoding="utf-8") as log:
        result = subprocess.run(
            [sys.executable, "-u", "-B", str(Path(__file__)), "--worker", str(task_path)],
            cwd=ROOT, env=env, stdout=log, stderr=subprocess.STDOUT)
    if result.returncode or not complete(task_path.parent, task["manifest_sha256"]):
        raise RuntimeError(f"Structural production worker failed: {task_path.parent / 'worker.log'}")
    return json.loads((task_path.parent / "summary.json").read_text(encoding="utf-8"))


def run_phase(batch, tasks, metadata, fingerprint, workers, reference, approval, stage):
    pending = []
    for name, seed in tasks:
        directory = batch / f"{name}__seed{seed}"
        if complete(directory, fingerprint):
            continue
        directory.mkdir(parents=True, exist_ok=True)
        path = directory / "task.json"
        write_json(path, {"scenario": name, "seed": seed, "manifest_sha256": fingerprint})
        pending.append(path)
    completed = len(tasks) - len(pending)
    print(f"{stage}: {completed}/{len(tasks)} complete; {workers} workers.", flush=True)
    with ThreadPoolExecutor(max_workers=workers) as executor:
        futures = {executor.submit(launch, path): path for path in pending}
        for future in as_completed(futures):
            try:
                summary = future.result()
            except Exception:
                for other in futures:
                    other.cancel()
                raise
            completed += 1
            print(f"COMPLETE {futures[future].parent.name}: deaths={summary['Death_hour408']:.6f}", flush=True)
            write_json(batch / "progress.json", {
                "stage": stage, "completed_in_phase": completed, "phase_total": len(tasks)})
    if metadata != manifest(reference, approval):
        raise ValueError("Structural source, inputs, runtime or approval changed during execution.")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path, default=ROOT / "data_collection" / "structural_production_20261031")
    parser.add_argument("--reference-batch", type=Path, default=ROOT / "data_collection" / "study_hpc")
    parser.add_argument("--approval", type=Path)
    parser.add_argument("--review-out", type=Path)
    parser.add_argument("--workers", type=int, choices=range(1, 9), default=8)
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--worker", type=Path, help=argparse.SUPPRESS)
    args = parser.parse_args()
    if args.worker is not None:
        require_hpc()
        recorded = json.loads((args.worker.parent.parent / "manifest.json").read_text(encoding="utf-8"))
        worker(args.worker, lambda: manifest(Path(recorded["reference_batch"]),
                                            Path(recorded["approval_path"])))
        return
    if not args.dry_run:
        require_hpc()
    if args.approval is None:
        parser.error("--approval is required; pilot completion does not authorize production.")
    reference, approval = args.reference_batch.resolve(), args.approval.resolve()
    metadata = manifest(reference, approval)
    if args.dry_run:
        print(json.dumps({"total_runs": 1260, "reference_gate_runs": 140, "variant_runs": 1120,
                          "seeds": list(SEEDS), "configurations": 9, "settings": 7,
                          "source_files": len(metadata["source_sha256"]),
                          "approval_sha256": metadata["approval_sha256"]}, indent=2))
        return
    batch = args.out.resolve()
    review_out = (args.review_out.resolve() if args.review_out
                  else batch.with_name(batch.name + "_review"))
    paths = (batch, reference, review_out)
    if any(a.is_relative_to(b) or b.is_relative_to(a)
           for i, a in enumerate(paths) for b in paths[i + 1:]):
        raise ValueError("Production, reference and review destinations must be separate.")
    if review_out.exists() and any(review_out.iterdir()):
        raise ValueError("Review evidence already exists; refusing to overwrite.")
    original = sha256(reference / "manifest.json")
    executions = execution_fingerprints(reference)
    for seed in SEEDS:
        for anchor in pilot.ANCHORS:
            historical = reference / f"{anchor}__seed{seed}"
            if not complete(historical, run_fingerprint(historical, original, executions)):
                raise ValueError(f"Historical gate outcome is incomplete: {historical}")
    batch.mkdir(parents=True, exist_ok=True)
    lock = batch / "parent.lock"
    with lock.open("x", encoding="utf-8") as handle:
        json.dump({"pid": os.getpid(), "hostname": platform.node(), "started": time.time()}, handle)
    try:
        path = batch / "manifest.json"
        if path.exists():
            if json.loads(path.read_text(encoding="utf-8")) != metadata:
                raise ValueError("Production contract changed; refusing to mix runs.")
        else:
            if any(p.name != "parent.lock" for p in batch.iterdir()):
                raise ValueError("Production output exists without its manifest.")
            write_json(path, metadata)
        fingerprint = sha256(path)
        references = [(f"reference__{anchor}", seed) for seed in SEEDS for anchor in pilot.ANCHORS]
        run_phase(batch, references, metadata, fingerprint, args.workers, reference, approval,
                  "historical_reference_gate")
        write_json(batch / "reference_gate.json", reference_gate(batch, reference, fingerprint))
        print("PASS: all 140 historical references matched exactly; 1,120 variants enabled.", flush=True)
        variants = [(name, seed) for seed in SEEDS for name in metadata["scenarios"]
                    if not name.startswith("reference__")]
        run_phase(batch, variants, metadata, fingerprint, args.workers, reference, approval, "stress_variants")
        names = [f"{name}__seed{seed}" for name, seed in references + variants]
        if any(not complete(batch / name, fingerprint) for name in names):
            raise ValueError("Production outcome is incomplete; final validation refused.")
        write_json(batch / "validation.json", {
            "manifest_sha256": fingerprint, "runs": 1260, "hours_per_run": 408,
            "historical_reference_matches": 140, "variant_runs": 1120,
            "completion_sha256": {name: sha256(batch / name / "completion.json") for name in names},
            "reference_gate_sha256": sha256(batch / "reference_gate.json"),
            "fresh_runs_only": True, "pilot_runs_reused": 0,
        })
        sys.path.insert(0, str(ROOT / "analysis"))
        from structural_production_results import review
        review(batch, reference, approval, review_out, fingerprint)
        print("PASS: all 1,260 outcomes independently reviewed; all 32 contrast changes and verified export ready.", flush=True)
    finally:
        lock.unlink()


if __name__ == "__main__":
    main()
