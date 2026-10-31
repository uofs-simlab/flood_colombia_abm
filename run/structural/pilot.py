"""HPC-only implementation pilot for the approved behavioural/clinical TODOs."""

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
from run_study import study_manifest
from sensitivity.design import ANCHORS, CONTRASTS
from study_design import scenario_parameters
from verify_model import complete, sha256, worker, write_json
from hpc.transfer import execution_fingerprints, run_fingerprint

SEEDS = (20261026, 20261027)
VARIANTS = {
    "reference": {},
    "evacuation_probit": {"evacuation_success_rule": "probit"},
    "injury_early": {"injury_onset_scale": .75},
    "injury_late": {"injury_onset_scale": 1.25},
    "survival_short": {"survival_deadline_scale": .75},
    "survival_long": {"survival_deadline_scale": 1.25},
    "recovery_075": {"recovery_eligibility": .75},
    "recovery_100": {"recovery_eligibility": 1.0},
    "no_offsite_care": {"offsite_care": False},
}


def require_hpc():
    if platform.node().split(".")[0].lower() not in ("richardson", "hundsdorfer"):
        raise RuntimeError("Full pilot simulations must run on Richardson or Hundsdorfer, not locally.")


def scenarios():
    return {
        f"{variant}__{anchor}": {**scenario_parameters(anchor), **changes}
        for variant, changes in VARIANTS.items() for anchor in ANCHORS
    }


def manifest(reference):
    recorded = json.loads((reference / "manifest.json").read_text(encoding="utf-8"))
    if (recorded["steps"] != 408 or not set(SEEDS) <= set(recorded["seeds"])
            or any(recorded["scenarios"].get(name) != scenario_parameters(name) for name in ANCHORS)):
        raise ValueError("The historical references do not match the approved seeds/settings.")
    metadata = study_manifest(len(SEEDS), SEEDS[0])
    cases = scenarios()
    metadata.update({
        "study_kind": "structural_pilot",
        "purpose": "Author-approved implementation pilot; no policy-effect inference or submission-ready sensitivity claims.",
        "scenarios": cases,
        "execution_scenarios": list(cases),
        "scenario_metadata": {
            f"{variant}__{anchor}": {"variant": variant, "anchor": anchor}
            for variant in VARIANTS for anchor in ANCHORS},
        "reference_manifest_sha256": sha256(reference / "manifest.json"),
        "reference_batch": str(reference),
        "variants": VARIANTS,
        "anchors": list(ANCHORS),
        "contrasts": [list(row) for row in CONTRASTS],
        "analysis_plan": {
            "budget": "9 rule configurations x 7 anchors x 2 seeds = 126 runs.",
            "gate": "All 14 inherited-rule runs must exactly reproduce the historical primary records before variants run.",
            "sampling": "Fixed two-seed implementation pilot, not statistical inference; no p-values or policy conclusions.",
            "behaviour": "Evacuation only: score > independent standard normal; Phi(score) is exploratory, uncalibrated, with a unit noise scale.",
            "clinical": "Separate multipliers for injury onset and the total survival deadline; ceil after original draws. Fixed recovery eligibility, not hourly probability.",
            "clinical_clock": "The survival multiplier applies to the existing common injury deadline and continuous-stranding deadline; it does not separate those two mechanisms.",
            "scope": "One factor at a time; untreated baseline rules remain authoritative for the reported experiments.",
            "next_step": "A production design and replication budget require a separate decision after implementation checks.",
        },
    })
    for path in (Path(__file__), ROOT / "run" / "sensitivity" / "design.py",
                 ROOT / "run" / "hpc" / "transfer.py",
                 ROOT / "analysis" / "paper_results.py",
                 ROOT / "analysis" / "analyze_archive.py"):
        metadata["source_sha256"][str(path.relative_to(ROOT))] = sha256(path)
    return metadata


def contrast_changes(frame):
    import pandas as pd
    expected = {(name, seed) for name in scenarios() for seed in SEEDS}
    identities = list(zip(frame.scenario, frame.seed))
    if len(identities) != len(set(identities)) or set(identities) != expected:
        raise ValueError("Incomplete, duplicated or unexpected pilot outcomes.")
    if any(not 0 <= value <= 972 or int(value) != value for value in frame.Deaths):
        raise ValueError("Pilot death counts must be integer population counts.")
    wide = frame.pivot(index="seed", columns="scenario", values="Deaths")
    rows = []
    for seed in SEEDS:
        for contrast, _, first, second in CONTRASTS:
            baseline = int(wide.loc[seed, f"reference__{second}"]
                           - wide.loc[seed, f"reference__{first}"])
            for variant in VARIANTS:
                if variant == "reference":
                    continue
                changed = int(wide.loc[seed, f"{variant}__{second}"]
                              - wide.loc[seed, f"{variant}__{first}"])
                rows.append({
                    "seed": seed, "contrast": contrast, "variant": variant,
                    "reference_difference_deaths": baseline,
                    "variant_difference_deaths": changed,
                    "change_in_difference_deaths": changed - baseline,
                    "change_in_difference_pp": (changed - baseline) * 100 / 972,
                    "interpretation": "Two-seed implementation pilot only; no policy inference.",
                })
    return pd.DataFrame(rows)


def launch(task_path):
    env = os.environ.copy()
    task = json.loads(task_path.read_text(encoding="utf-8"))
    env.update({"PYTHONHASHSEED": str(task["seed"]), "PYTHONDONTWRITEBYTECODE": "1",
                "OMP_NUM_THREADS": "1", "OPENBLAS_NUM_THREADS": "1",
                "MKL_NUM_THREADS": "1", "NUMEXPR_NUM_THREADS": "1"})
    with (task_path.parent / "worker.log").open("w", encoding="utf-8") as log:
        result = subprocess.run(
            [sys.executable, "-u", "-B", str(Path(__file__)), "--worker", str(task_path)],
            cwd=ROOT, env=env, stdout=log, stderr=subprocess.STDOUT)
    if result.returncode or not complete(task_path.parent, task["manifest_sha256"]):
        raise RuntimeError(f"Pilot worker failed: {task_path.parent / 'worker.log'}")
    return json.loads((task_path.parent / "summary.json").read_text(encoding="utf-8"))


def compare_reference(directory, reference, expected_fingerprint):
    if not complete(reference, expected_fingerprint):
        raise ValueError(f"Historical reference is incomplete: {reference}")
    import pandas as pd
    from verify_model import OUTPUTS
    for name in OUTPUTS[:3]:
        pd.testing.assert_frame_equal(
            pd.read_csv(directory / name), pd.read_csv(reference / name), check_exact=True)
    summaries = [json.loads((p / "summary.json").read_text(encoding="utf-8"))
                 for p in (directory, reference)]
    for record in summaries:
        for key in ("elapsed_seconds", "manifest_sha256", "scenario"):
            del record[key]
    if summaries[0] != summaries[1]:
        raise ValueError(f"Historical scientific summary differs: {directory.name}")


def run_phase(batch, tasks, fingerprint, workers, reference, stage):
    pending = []
    for name, seed in tasks:
        directory = batch / f"{name}__seed{seed}"
        if complete(directory, fingerprint):
            continue
        directory.mkdir(parents=True, exist_ok=True)
        task_path = directory / "task.json"
        write_json(task_path, {"scenario": name, "seed": seed, "manifest_sha256": fingerprint})
        pending.append(task_path)
    print(f"{stage}: {len(tasks)-len(pending)}/{len(tasks)} complete; {workers} workers.", flush=True)
    with ThreadPoolExecutor(max_workers=workers) as executor:
        futures = {executor.submit(launch, p): p for p in pending}
        for future in as_completed(futures):
            try:
                summary = future.result()
            except Exception:
                for other in futures:
                    other.cancel()
                raise
            print(f"COMPLETE {futures[future].parent.name}: "
                  f"deaths={summary['Death_hour408']:.6f}", flush=True)
            write_json(batch / "progress.json", {
                "stage": stage, "completed_in_phase": len(tasks)-len(pending)
                + sum(f.done() and not f.cancelled() and f.exception() is None for f in futures),
                "phase_total": len(tasks),
            })
    if json.loads((batch / "manifest.json").read_text(encoding="utf-8")) != manifest(reference):
        raise ValueError("Pilot source, inputs, runtime or design changed during execution.")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path, default=ROOT / "data_collection" / "structural_pilot")
    parser.add_argument("--reference-batch", type=Path, default=ROOT / "data_collection" / "study_hpc")
    parser.add_argument("--workers", type=int, choices=range(1, 17), default=8)
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--worker", type=Path, help=argparse.SUPPRESS)
    args = parser.parse_args()
    if args.worker is not None:
        require_hpc()
        recorded = json.loads((args.worker.parent.parent / "manifest.json").read_text(encoding="utf-8"))
        worker(args.worker, lambda: manifest(Path(recorded["reference_batch"])))
        return
    if not args.dry_run:
        require_hpc()
    reference = args.reference_batch.resolve()
    metadata = manifest(reference)
    if args.dry_run:
        print(json.dumps(metadata, indent=2, sort_keys=True))
        return
    batch = args.out.resolve()
    if batch == reference or batch.is_relative_to(reference) or reference.is_relative_to(batch):
        raise ValueError("Pilot outputs must be separate from the historical reference cohort.")
    batch.mkdir(parents=True, exist_ok=True)
    lock = batch / "parent.lock"
    with lock.open("x", encoding="utf-8") as handle:
        json.dump({"pid": os.getpid(), "hostname": platform.node(), "started": time.time()}, handle)
    try:
        path = batch / "manifest.json"
        if path.exists():
            if json.loads(path.read_text(encoding="utf-8")) != metadata:
                raise ValueError("Pilot contract changed; refusing to mix runs.")
        else:
            if any(p.name != "parent.lock" for p in batch.iterdir()):
                raise ValueError("Pilot outputs exist without a manifest.")
            write_json(path, metadata)
        fingerprint = sha256(path)
        references = [(f"reference__{name}", seed) for seed in SEEDS for name in ANCHORS]
        run_phase(batch, references, fingerprint, args.workers, reference, "historical_reference_gate")
        executions = execution_fingerprints(reference)
        original_hash = sha256(reference / "manifest.json")
        for name, seed in references:
            anchor = name.removeprefix("reference__")
            directory = reference / f"{anchor}__seed{seed}"
            compare_reference(batch / f"{name}__seed{seed}", directory,
                              run_fingerprint(directory, original_hash, executions))
        write_json(batch / "reference_gate.json", {
            "manifest_sha256": fingerprint, "historical_manifest_sha256": original_hash,
            "exact_reference_matches": len(references),
            "comparison": "All numerical CSV values and scientific summaries match exactly.",
        })
        print("PASS: all 14 historical references reproduced exactly; variants enabled.", flush=True)
        changed = [(name, seed) for seed in SEEDS for name in metadata["scenarios"]
                   if not name.startswith("reference__")]
        run_phase(batch, changed, fingerprint, args.workers, reference, "stress_variants")
        for name, seed in references + changed:
            if not complete(batch / f"{name}__seed{seed}", fingerprint):
                raise ValueError(f"Incomplete pilot run: {name}, {seed}")
        sys.path.insert(0, str(ROOT / "analysis"))
        from paper_results import load_cohort
        frame, _, _ = load_cohort(batch, expected_manifest=metadata,
                                  scenario_names=metadata["scenarios"])
        frame.to_csv(batch / "pilot_run_summary.csv", index=False)
        contrast_changes(frame).to_csv(batch / "pilot_contrast_changes.csv", index=False)
        write_json(batch / "validation.json", {
            "manifest_sha256": fingerprint, "runs": 126, "hours_per_run": 408,
            "historical_reference_matches": 14, "variant_runs": 112,
            "policy_effect_inference": False,
            "production_runs_authorized": False,
            "independent_hourly_output_readback": True,
            "completion_sha256": {
                f"{name}__seed{seed}": sha256(batch / f"{name}__seed{seed}" / "completion.json")
                for name, seed in references + changed},
            "descriptive_outputs_sha256": {
                name: sha256(batch / name)
                for name in ("pilot_run_summary.csv", "pilot_contrast_changes.csv")},
        })
        print("PASS: all 126 pilot runs passed implementation checks; no policy inference.", flush=True)
    finally:
        lock.unlink()


if __name__ == "__main__":
    main()
