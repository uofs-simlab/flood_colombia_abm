"""Run the standard-model study with per-run isolation and verified resumption."""

import argparse
from concurrent.futures import ThreadPoolExecutor, as_completed
import importlib.metadata
import json
import os
from pathlib import Path
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from study_design import SCENARIOS, scenario_parameters, seeds
from verify_model import complete, sha256, write_json, worker


def study_manifest(replications, seed_base):
    source_files = [p for directory in ["agents", "model", "space", "data_collection", "run"]
                    for p in (ROOT / directory).glob("*.py")]
    source_files.extend([ROOT / "requirements.txt", ROOT / "environment-lock.txt"])
    return {
        "purpose": "Standard-model community-organization and rescue-logistics study.",
        "replications": replications, "seed_base": seed_base,
        "seeds": seeds(replications, seed_base), "steps": 408,
        "scenarios": {name: scenario_parameters(name) for name in SCENARIOS},
        "scenario_metadata": {name: {k: v for k, v in info.items() if k != "parameters"}
                              for name, info in SCENARIOS.items()},
        "source_sha256": {str(p.relative_to(ROOT)): sha256(p) for p in sorted(source_files)},
        "spatial_sha256": {str(p.relative_to(ROOT)): sha256(p)
                           for p in sorted((ROOT / "santa_lucia_map_data").rglob("*.zip"))},
        "python": sys.version, "executable": sys.executable,
        "packages": dict(sorted((d.metadata["Name"], d.version)
                               for d in importlib.metadata.distributions() if d.metadata["Name"])),
        "seeding": "Common recorded seeds; fresh interpreter per run; Python, NumPy and Mesa seeded before population creation.",
        "analysis_plan": {
            "primary_endpoint": "Mortality fraction at hour 408.",
            "descriptive": "Median and interquartile range across seeds; hourly process metrics; mean mortality Monte Carlo t intervals.",
            "omnibus": "Friedman test blocked by seed, with Holm correction across seven prespecified families.",
            "pairwise": "All within-family mortality pairs; exact paired sign-flip tests of mean differences with Holm correction within family.",
            "families": "Organization levels; organization channels including baseline and both channels; distance within each of three capacities; fleet within each of two sites.",
            "effects": "Mean seed-paired mortality difference in percentage points with a 95% Monte Carlo t interval; no empirical intervention-effect interpretation.",
            "precision": "Report interval half-widths and flag those exceeding 0.5 percentage points; do not extend sampling based on significance.",
        },
    }


def expected_tasks(batch, manifest):
    fingerprint = sha256(batch / "manifest.json")
    return [{
        "path": batch / f"{scenario}__seed{seed}" / "task.json",
        "scenario": scenario, "seed": seed, "manifest_sha256": fingerprint,
    } for seed in manifest["seeds"] for scenario in SCENARIOS]


def launch(task):
    task_path = task["path"]
    task_path.parent.mkdir(parents=True, exist_ok=True)
    write_json(task_path, {k: v for k, v in task.items() if k != "path"})
    env = os.environ.copy()
    env.update({"PYTHONHASHSEED": str(task["seed"]), "PYTHONDONTWRITEBYTECODE": "1",
                "OMP_NUM_THREADS": "1", "OPENBLAS_NUM_THREADS": "1", "MKL_NUM_THREADS": "1"})
    log_path = task_path.parent / "worker.log"
    with log_path.open("w", encoding="utf-8") as log:
        result = subprocess.run([sys.executable, "-B", str(Path(__file__)),
                                 "--worker", str(task_path)],
                                cwd=ROOT, env=env, stdout=log, stderr=subprocess.STDOUT)
    if result.returncode:
        raise RuntimeError(f"Study worker failed (exit {result.returncode}): {log_path}")
    if not complete(task_path.parent, task["manifest_sha256"]):
        raise RuntimeError(f"Worker did not record verified completion: {log_path}")
    return json.loads((task_path.parent / "summary.json").read_text(encoding="utf-8"))


def main(default_family="all"):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path, default=ROOT / "data_collection" / "study")
    parser.add_argument("--replications", type=int, default=20)
    parser.add_argument("--seed-base", type=int, default=20261026)
    parser.add_argument("--workers", type=int, choices=[1, 2], default=2)
    parser.add_argument("--max-jobs", type=int)
    parser.add_argument("--family", choices=["all", "organization", "ablation", "distance", "fleet", "logistics"],
                        default=default_family)
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--skip-analysis", action="store_true")
    parser.add_argument("--worker", type=Path, help=argparse.SUPPRESS)
    args = parser.parse_args()
    if args.worker is not None:
        batch = args.worker.parent.parent
        recorded = json.loads((batch / "manifest.json").read_text(encoding="utf-8"))
        worker(args.worker, lambda: study_manifest(recorded["replications"], recorded["seed_base"]))
        return
    metadata = study_manifest(args.replications, args.seed_base)
    if args.max_jobs is not None and args.max_jobs < 1:
        parser.error("--max-jobs must be positive.")
    if args.dry_run:
        print(f"{len(SCENARIOS)} settings x {len(metadata['seeds'])} seeds = "
              f"{len(SCENARIOS)*len(metadata['seeds'])} runs.")
        print(json.dumps(metadata["analysis_plan"], indent=2))
        return
    if (importlib.metadata.version("mesa") != "2.1.2"
            or importlib.metadata.version("mesa-geo") != "0.7.1"):
        parser.error("Use the pinned simulation environment.")
    batch = args.out.resolve()
    batch.mkdir(parents=True, exist_ok=True)
    lock = batch / "parent.lock"
    try:
        with lock.open("x", encoding="utf-8") as handle:
            json.dump({"pid": os.getpid(), "purpose": "One parent per study cohort."}, handle)
    except FileExistsError as exc:
        raise RuntimeError(f"Study parent lock exists: {lock}. Check its PID; "
                           "remove a stale lock only after confirming that parent has stopped.") from exc
    try:
        manifest_path = batch / "manifest.json"
        if manifest_path.exists():
            if json.loads(manifest_path.read_text(encoding="utf-8")) != metadata:
                raise ValueError("Study source, inputs, environment or design changed. "
                                 "Do not mix cohorts or overwrite this manifest.")
        else:
            if any(batch.glob("*__seed*")):
                raise ValueError("Study outputs exist without a manifest; refusing to merge.")
            write_json(manifest_path, metadata)
        tasks = expected_tasks(batch, metadata)
        pending = [task for task in tasks
                   if not complete(task["path"].parent, task["manifest_sha256"])]
        selected = [
            task for task in pending if args.family == "all"
            or SCENARIOS[task["scenario"]]["family"] == args.family
            or (args.family == "logistics"
                and SCENARIOS[task["scenario"]]["family"] in {"distance", "fleet"})
        ]
        if args.max_jobs is not None:
            selected = selected[:args.max_jobs]
        print(f"{len(tasks)-len(pending)}/{len(tasks)} verified study runs complete; "
              f"running {len(selected)} with {args.workers} workers.", flush=True)
        with ThreadPoolExecutor(max_workers=args.workers) as executor:
            futures = {executor.submit(launch, task): task for task in selected}
            for future in as_completed(futures):
                try:
                    summary = future.result()
                except Exception:
                    for pending_future in futures:
                        pending_future.cancel()
                    raise
                print(f"COMPLETE {summary['scenario']} seed={summary['seed']} "
                      f"mortality408={summary['Death_hour408']:.6f} "
                      f"admissions={summary['Rescue_Admissions']} "
                      f"seconds={summary['elapsed_seconds']:.1f}", flush=True)
        remaining = [task for task in tasks
                     if not complete(task["path"].parent, task["manifest_sha256"])]
        print(f"{len(tasks)-len(remaining)}/{len(tasks)} study runs verified.", flush=True)
        if not remaining and not args.skip_analysis:
            subprocess.run([sys.executable, "-B", str(ROOT / "analysis" / "paper_results.py"),
                            "--batch", str(batch), "--compile"], cwd=ROOT, check=True)
    finally:
        lock.unlink()


if __name__ == "__main__":
    main()
