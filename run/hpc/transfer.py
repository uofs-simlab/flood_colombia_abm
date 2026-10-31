"""Portable execution of the unchanged study, with a cross-platform repeat gate."""

import argparse
from concurrent.futures import ThreadPoolExecutor, as_completed
import json
import os
from pathlib import Path, PurePosixPath
import platform
import shutil
import subprocess
import sys
import tempfile
import time
import zipfile

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "run"))
from run_study import expected_tasks, study_manifest
from verify_model import OUTPUTS, complete, sha256, worker, write_json

TRANSFER = Path("data_collection") / "hpc_transfer"
REMOTE_BATCH = Path("data_collection") / "study_hpc"
GATE_SCENARIOS = ("A_org_0", "A_org_1",
                  "D_cap30_nearby_crossing", "D_cap30_barranquilla")
RUNTIME_PACKAGES = (
    "mesa", "mesa-geo", "numpy", "pandas", "scipy", "geopandas", "shapely",
    "fiona", "pyproj", "rasterio", "rtree", "networkx", "libpysal", "scikit-learn",
)


def read_json(path):
    return json.loads(path.read_text(encoding="utf-8"))


def current_manifest(recorded):
    if recorded.get("study_kind") == "sensitivity":
        from sensitivity.design import sensitivity_manifest
        return sensitivity_manifest(recorded["replications"], recorded["seed_base"],
                                    recorded["baseline_manifest_sha256"])
    require("study_kind" not in recorded, "Unknown study kind.")
    return study_manifest(recorded["replications"], recorded["seed_base"])


def cohort_tasks(batch, metadata, include_references=False):
    if "execution_scenarios" not in metadata:
        return expected_tasks(batch, metadata)
    names = metadata["scenarios"] if include_references else metadata["execution_scenarios"]
    fingerprint = sha256(batch / "manifest.json")
    return [{"path": batch / f"{name}__seed{seed}" / "task.json",
             "scenario": name, "seed": seed, "manifest_sha256": fingerprint}
            for seed in metadata["seeds"] for name in names]


def reference_directory(batch, metadata):
    if metadata.get("study_kind") == "sensitivity":
        from sensitivity.design import reference_batch, validate_primary
        primary = reference_batch(batch, metadata)
        validate_primary(metadata, primary)
        return primary
    return batch


def reference_fingerprint(plan, name):
    return plan.get("reference_manifest_sha256", {}).get(name, plan["origin_manifest_sha256"])


def require(condition, message):
    if not condition:
        raise ValueError(message)


def relative_path(name):
    path = PurePosixPath(name.replace("\\", "/"))
    require(not path.is_absolute() and ".." not in path.parts
            and path.parts and ":" not in str(path), f"Unsafe relative path: {name}")
    return Path(*path.parts)


def portable_contract(manifest):
    result = {key: value for key, value in manifest.items()
              if key not in {"python", "executable", "packages", "source_sha256",
                             "spatial_sha256", "scenarios", "execution"}}
    for field in ["source_sha256", "spatial_sha256"]:
        result[field] = {str(relative_path(k)).replace("\\", "/"): v
                         for k, v in manifest[field].items()}
    scenarios = {}
    for name, parameters in manifest["scenarios"].items():
        converted = {}
        for key, value in parameters.items():
            if key.endswith("_file") or key.startswith("flood_file_"):
                pieces = value.replace("\\", "/").split("/")
                require(pieces.count("santa_lucia_map_data") == 1,
                        f"Input is outside the canonical spatial directory: {key}")
                value = "/".join(pieces[pieces.index("santa_lucia_map_data"):])
                relative_path(value)
            converted[key] = value
        scenarios[name] = converted
    result["scenarios"] = scenarios
    return result


def require_compatible(origin, execution):
    require(portable_contract(origin) == portable_contract(execution),
            "Source, input hashes, scenario semantics or analysis design differ.")
    require(origin["python"].split()[0].split(".")[:2]
            == execution["python"].split()[0].split(".")[:2] == ["3", "12"],
            "Both executions must use Python 3.12.")
    packages = [{k.lower().replace("_", "-"): v for k, v in m["packages"].items()}
                for m in [origin, execution]]
    for name in RUNTIME_PACKAGES:
        require(name in packages[0] and packages[0][name] == packages[1].get(name),
                f"Scientific runtime package differs or is missing: {name}")


def compare_runs(first, second):
    import pandas as pd

    for name in OUTPUTS[:3]:
        pd.testing.assert_frame_equal(pd.read_csv(first / name),
                                      pd.read_csv(second / name), check_exact=True)
    summaries = [read_json(p / "summary.json") for p in [first, second]]
    for summary in summaries:
        for key in ["elapsed_seconds", "manifest_sha256"]:
            summary.pop(key)
    require(summaries[0] == summaries[1], f"Scientific summaries differ: {first.name}")


def extract_checked(archive, destination):
    with zipfile.ZipFile(archive) as handle:
        names = handle.namelist()
        require(len(names) == len(set(names)), "Duplicate paths in transfer archive.")
        for info in handle.infolist():
            relative_path(info.filename)
            require(not info.is_dir() and (info.external_attr >> 16) & 0o170000 != 0o120000,
                    "Transfer archives must contain regular files, not links/directories.")
        for info in handle.infolist():
            target = destination / relative_path(info.filename)
            target.parent.mkdir(parents=True, exist_ok=True)
            with handle.open(info) as source, target.open("xb") as output:
                shutil.copyfileobj(source, output)


def bundle(batch, archive, transfer=TRANSFER):
    origin = read_json(batch / "manifest.json")
    require(origin == current_manifest(origin),
            "Local cohort no longer matches the executed source/environment.")
    fingerprint = sha256(batch / "manifest.json")
    tasks = cohort_tasks(batch, origin)
    pending = [{k: v for k, v in task.items() if k != "path"} for task in tasks
               if not complete(task["path"].parent, fingerprint)]
    references = [f"{name}__seed{origin['seeds'][0]}" for name in GATE_SCENARIOS]
    reference_root = reference_directory(batch, origin)
    reference_hashes = {}
    for name in references:
        directory = reference_root / name
        reference_hashes[name] = read_json(directory / "completion.json")["manifest_sha256"]
        require(complete(directory, reference_hashes[name]), f"Reference is incomplete: {name}")
    require(pending, "The local study is already complete.")
    files = {relative_path(name): ROOT / relative_path(name)
             for name in [*origin["source_sha256"], *origin["spatial_sha256"]]}
    for directory in ["run/hpc", "analysis", "tests"]:
        for path in (ROOT / relative_path(directory)).glob("*.py"):
            files[path.relative_to(ROOT)] = path
    for suffix in ["*.ps1", "*.sh"]:
        for path in (ROOT / "run" / "hpc").glob(suffix):
            files[path.relative_to(ROOT)] = path
    if origin.get("study_kind") == "sensitivity":
        for path in (ROOT / "run" / "sensitivity").glob("*.py"):
            files[path.relative_to(ROOT)] = path
    files[transfer / "origin_manifest.json"] = batch / "manifest.json"
    for name in references:
        for filename in [*OUTPUTS, "completion.json", "task.json"]:
            files[transfer / "references" / name / filename] = reference_root / name / filename
    plan = {"origin_manifest_sha256": fingerprint, "tasks": pending, "references": references,
            "reference_manifest_sha256": reference_hashes}
    archive.parent.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(archive, "x", compression=zipfile.ZIP_DEFLATED) as handle:
        for name, source in sorted(files.items()):
            if source.suffix == ".sh":
                handle.writestr(name.as_posix(), source.read_bytes().replace(b"\r\n", b"\n"))
            else:
                handle.write(source, name.as_posix())
        handle.writestr((transfer / "plan.json").as_posix(), json.dumps(plan, sort_keys=True))
    print(f"Prepared {len(pending)} pending jobs and four repeat checks: {archive}")


def execution_manifest(transfer=TRANSFER):
    origin = read_json(ROOT / transfer / "origin_manifest.json")
    current = current_manifest(origin)
    require_compatible(origin, current)
    current["execution"] = {
        "origin_manifest_sha256": sha256(ROOT / transfer / "origin_manifest.json"),
        "plan_sha256": sha256(ROOT / transfer / "plan.json"),
        "adapter_sha256": sha256(Path(__file__)),
        "hostname": platform.node(), "platform": platform.platform(),
    }
    return current


def initialize(batch, transfer=TRANSFER):
    metadata = execution_manifest(transfer)
    batch.mkdir(parents=True, exist_ok=True)
    path = batch / "manifest.json"
    if path.exists():
        require(read_json(path) == metadata, "Remote execution manifest changed; refusing to merge.")
    else:
        require(not any(batch.iterdir()), "Remote output exists without an execution manifest.")
        write_json(path, metadata)
    return metadata


def execute(task, transfer=TRANSFER):
    directory = task["path"].parent
    directory.mkdir(parents=True, exist_ok=True)
    write_json(task["path"], {k: v for k, v in task.items() if k != "path"})
    env = os.environ.copy()
    env.update({"PYTHONHASHSEED": str(task["seed"]), "PYTHONDONTWRITEBYTECODE": "1",
                "OMP_NUM_THREADS": "1", "OPENBLAS_NUM_THREADS": "1",
                "MKL_NUM_THREADS": "1", "NUMEXPR_NUM_THREADS": "1"})
    with (directory / "worker.log").open("w", encoding="utf-8") as log:
        command = [sys.executable, "-u", "-B", str(Path(__file__)), "worker", "--task", str(task["path"])]
        if transfer != TRANSFER:
            command.append("--sensitivity")
        result = subprocess.run(command,
                                cwd=ROOT, env=env, stdout=log, stderr=subprocess.STDOUT)
    require(result.returncode == 0, f"Remote worker failed: {directory / 'worker.log'}")
    require(complete(directory, task["manifest_sha256"]), f"Remote job incomplete: {directory}")
    return read_json(directory / "summary.json")


def validate_plan(plan, origin):
    expected = {(name, seed) for name in origin.get("execution_scenarios", origin["scenarios"])
                for seed in origin["seeds"]}
    identities = [(t["scenario"], t["seed"]) for t in plan["tasks"]]
    require(len(identities) == len(set(identities))
            and set(identities) <= expected
            and all(t["manifest_sha256"] == plan["origin_manifest_sha256"]
                    for t in plan["tasks"]), "Invalid or duplicated remote task identities.")
    references = [f"{name}__seed{origin['seeds'][0]}" for name in GATE_SCENARIOS]
    require(plan["references"] == references, "The four required repeat checks differ.")
    if "reference_manifest_sha256" in plan:
        require(set(plan["reference_manifest_sha256"]) == set(references),
                "Reference execution fingerprints are incomplete.")


def validate_bridge(evidence, origin_batch):
    origin = read_json(origin_batch / "manifest.json")
    manifest = read_json(evidence / "manifest.json")
    fingerprint = sha256(evidence / "manifest.json")
    original_hash = sha256(origin_batch / "manifest.json")
    require_compatible(origin, manifest)
    plan = read_json(evidence / "plan.json")
    validate_plan(plan, origin)
    require(plan["origin_manifest_sha256"] == original_hash
            == manifest["execution"]["origin_manifest_sha256"]
            and manifest["execution"]["plan_sha256"] == sha256(evidence / "plan.json"),
            "Execution does not reference the original study and dispatch plan.")
    bridge = read_json(evidence / "bridge.json")
    require(bridge["execution_manifest_sha256"] == fingerprint
            and bridge["origin_manifest_sha256"] == original_hash
            and set(bridge["reference_hashes"]) == set(plan["references"]),
            "Missing or inconsistent cross-platform repeat evidence.")
    for name in plan["references"]:
        local = reference_directory(origin_batch, origin) / name
        remote = evidence / "gates" / name
        require(complete(local, reference_fingerprint(plan, name)) and complete(remote, fingerprint),
                f"Incomplete repeat evidence: {name}")
        require(bridge["reference_hashes"][name] == {
            "local_completion": sha256(local / "completion.json"),
            "remote_completion": sha256(remote / "completion.json"),
        }, f"Repeat-evidence fingerprints differ: {name}")
        compare_runs(local, remote)
    return fingerprint, plan


def execution_fingerprints(batch):
    result = {}
    for path in sorted((batch / "executions").glob("*/manifest.json")):
        fingerprint, plan = validate_bridge(path.parent, batch)
        require(path.parent.name == fingerprint, "Execution evidence directory name differs.")
        result[fingerprint] = {
            f"{t['scenario']}__seed{t['seed']}" for t in plan["tasks"]}
    return result


def verify_gate(batch, plan, transfer=TRANSFER):
    fingerprint = sha256(batch / "manifest.json")
    bridge = read_json(batch / "bridge.json")
    require(bridge["execution_manifest_sha256"] == fingerprint
            and bridge["origin_manifest_sha256"] == plan["origin_manifest_sha256"]
            and set(bridge["reference_hashes"]) == set(plan["references"]),
            "The cross-platform repeat gate is missing or inconsistent.")
    for name in plan["references"]:
        reference = ROOT / transfer / "references" / name
        remote = batch / name
        require(complete(reference, reference_fingerprint(plan, name))
                and complete(remote, fingerprint), f"Repeat evidence is incomplete: {name}")
        require(bridge["reference_hashes"][name] == {
            "local_completion": sha256(reference / "completion.json"),
            "remote_completion": sha256(remote / "completion.json"),
        }, f"Repeat evidence changed: {name}")
        compare_runs(reference, remote)


def run_fingerprint(directory, original_hash, executions):
    sidecar = directory / "execution_origin.json"
    if not sidecar.exists():
        return original_hash
    fingerprint = read_json(sidecar)["execution_manifest_sha256"]
    require(fingerprint in executions and directory.name in executions[fingerprint],
            f"Unverified or unplanned imported execution: {directory}")
    return fingerprint


def run_batch(batch, phase, workers, transfer=TRANSFER):
    origin = read_json(ROOT / transfer / "origin_manifest.json")
    plan = read_json(ROOT / transfer / "plan.json")
    validate_plan(plan, origin)
    metadata = initialize(batch, transfer)
    fingerprint = sha256(batch / "manifest.json")
    lock = batch / "parent.lock"
    with lock.open("x", encoding="utf-8") as handle:
        json.dump({"pid": os.getpid(), "phase": phase}, handle)
    try:
        all_tasks = {t["path"].parent.name: t for t in cohort_tasks(batch, metadata, True)}
        if phase == "gate":
            selected = [all_tasks[name] for name in plan["references"]]
        else:
            require((batch / "bridge.json").is_file(), "Repeat gate has not passed.")
            verify_gate(batch, plan, transfer)
            selected = [all_tasks[f"{t['scenario']}__seed{t['seed']}"] for t in plan["tasks"]]
        pending = [t for t in selected if not complete(t["path"].parent, fingerprint)]
        started = time.time()
        finished = len(selected)-len(pending)
        print(f"{phase}: {finished}/{len(selected)} complete; {workers} workers.", flush=True)
        with ThreadPoolExecutor(max_workers=workers) as executor:
            futures = {executor.submit(execute, task, transfer): task for task in pending}
            for future in as_completed(futures):
                try:
                    summary = future.result()
                except Exception:
                    for other in futures:
                        other.cancel()
                    raise
                finished += 1
                write_json(batch / "progress.json", {
                    "phase": phase, "complete": finished, "total": len(selected),
                    "workers": workers, "elapsed_seconds": time.time()-started,
                })
                print(f"COMPLETE {summary['scenario']} seed={summary['seed']} "
                      f"{finished}/{len(selected)} seconds={summary['elapsed_seconds']:.1f}", flush=True)
        require(metadata == execution_manifest(transfer), "Execution changed while running.")
        if phase == "gate":
            hashes = {}
            for name in plan["references"]:
                reference = ROOT / transfer / "references" / name
                require(complete(reference, reference_fingerprint(plan, name)),
                        f"Changed local reference: {name}")
                compare_runs(reference, batch / name)
                hashes[name] = {"local_completion": sha256(reference / "completion.json"),
                                "remote_completion": sha256(batch / name / "completion.json")}
            write_json(batch / "bridge.json", {
                "origin_manifest_sha256": plan["origin_manifest_sha256"],
                "execution_manifest_sha256": fingerprint, "reference_hashes": hashes,
                "comparison": "Exact parsed numerical CSVs and all scientific summary fields.",
            })
            print("PASS: all four cross-platform references reproduced exactly.", flush=True)
        else:
            write_json(batch / "finished.json", {
                "execution_manifest_sha256": fingerprint, "jobs": len(selected),
            })
    finally:
        lock.unlink()


def export_results(batch, archive, transfer=TRANSFER):
    plan = read_json(ROOT / transfer / "plan.json")
    fingerprint = sha256(batch / "manifest.json")
    require((batch / "finished.json").is_file() and (batch / "bridge.json").is_file(),
            "The remote batch or repeat gate is not complete.")
    require(read_json(batch / "manifest.json") == execution_manifest(transfer),
            "Source or environment changed before exporting.")
    verify_gate(batch, plan, transfer)
    archive.parent.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(archive, "x", compression=zipfile.ZIP_DEFLATED) as handle:
        for filename in ["manifest.json", "bridge.json", "finished.json"]:
            handle.write(batch / filename, "evidence/" + filename)
        handle.write(ROOT / transfer / "plan.json", "evidence/plan.json")
        for category, names in [
            ("evidence/gates", plan["references"]),
            ("runs", [f"{t['scenario']}__seed{t['seed']}" for t in plan["tasks"]]),
        ]:
            for name in names:
                require(complete(batch / name, fingerprint), f"Incomplete export job: {name}")
                for filename in [*OUTPUTS, "completion.json", "task.json", "worker.log"]:
                    handle.write(batch / name / filename, f"{category}/{name}/{filename}")
    print(f"Complete scientific result archive: {archive}")


def import_results(archive, batch):
    require(not (batch / "parent.lock").exists(),
            "Stop the local study parent before importing remote results.")
    origin = read_json(batch / "manifest.json")
    require(origin == current_manifest(origin),
            "Current local source/environment no longer matches the original cohort.")
    original_hash = sha256(batch / "manifest.json")
    lock = batch / "parent.lock"
    with lock.open("x", encoding="utf-8") as handle:
        json.dump({"pid": os.getpid(), "purpose": "Verified HPC result import."}, handle)
    try:
        with tempfile.TemporaryDirectory(prefix="flood_hpc_import_") as temporary:
            incoming = Path(temporary)
            extract_checked(archive, incoming)
            fingerprint, plan = validate_bridge(incoming / "evidence", batch)
            names = {f"{t['scenario']}__seed{t['seed']}" for t in plan["tasks"]}
            require({p.name for p in (incoming / "runs").iterdir()} == names,
                    "Returned jobs differ from the dispatch plan.")
            imported, duplicates = [], []
            for task in plan["tasks"]:
                name = f"{task['scenario']}__seed{task['seed']}"
                source, destination = incoming / "runs" / name, batch / name
                require(complete(source, fingerprint), f"Returned output is incomplete: {name}")
                for filename in ["task.json", "summary.json"]:
                    record = read_json(source / filename)
                    require(record["scenario"] == task["scenario"] and record["seed"] == task["seed"]
                            and record["manifest_sha256"] == fingerprint,
                            f"Returned identity differs: {name}")
                if (destination / "completion.json").exists():
                    existing = read_json(destination / "completion.json")["manifest_sha256"]
                    require(existing in {original_hash, fingerprint} and complete(destination, existing),
                            f"Existing output belongs to another execution: {name}")
                    compare_runs(destination, source)
                    duplicates.append(name)
                else:
                    imported.append(name)
            evidence = batch / "executions" / fingerprint
            if evidence.exists():
                for path in (incoming / "evidence").rglob("*"):
                    if path.is_file():
                        require((evidence / path.relative_to(incoming / "evidence")).is_file()
                                and sha256(path) == sha256(evidence / path.relative_to(incoming / "evidence")),
                                "Existing execution evidence differs from returned evidence.")
            else:
                shutil.copytree(incoming / "evidence", evidence)
            for name in imported:
                destination = batch / name
                if destination.exists():
                    saved = batch / "incomplete_local" / name
                    saved.parent.mkdir(exist_ok=True)
                    require(not saved.exists(), f"Partial-output preservation path exists: {saved}")
                    destination.rename(saved)
                write_json(incoming / "runs" / name / "execution_origin.json",
                           {"execution_manifest_sha256": fingerprint})
                shutil.copytree(incoming / "runs" / name, destination)
            write_json(evidence / "import_report.json", {
                "archive_sha256": sha256(archive), "imported": imported,
                "exact_local_duplicates_preserved": duplicates,
            })
            print(f"Imported {len(imported)} jobs; preserved {len(duplicates)} exact local duplicates.")
    finally:
        lock.unlink()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("action", choices=["bundle", "gate", "run", "worker", "export", "import"])
    parser.add_argument("--batch", type=Path)
    parser.add_argument("--archive", type=Path)
    parser.add_argument("--task", type=Path)
    parser.add_argument("--workers", type=int, default=64)
    parser.add_argument("--sensitivity", action="store_true",
                        help="Use the targeted robustness cohort and isolated transfer/output paths.")
    args = parser.parse_args()
    transfer = TRANSFER / "sensitivity" if args.sensitivity else TRANSFER
    local_batch = ROOT / "data_collection" / ("sensitivity" if args.sensitivity else "study")
    remote_batch = ROOT / "data_collection" / ("sensitivity_hpc" if args.sensitivity else "study_hpc")
    if args.action == "bundle":
        require(args.archive is not None, "--archive is required.")
        bundle(args.batch or local_batch, args.archive, transfer)
    elif args.action == "worker":
        require(args.task is not None, "--task is required.")
        worker(args.task, lambda: execution_manifest(transfer))
    elif args.action in {"gate", "run"}:
        import psutil
        available = psutil.cpu_count(logical=False) or os.cpu_count() or 1
        if hasattr(os, "sched_getaffinity"):
            available = min(available, len(os.sched_getaffinity(0)))
        require(1 <= args.workers <= available, f"Use 1--{available} physical-core workers.")
        run_batch(args.batch or remote_batch, args.action, args.workers, transfer)
    elif args.action == "export":
        require(args.archive is not None, "--archive is required.")
        export_results(args.batch or remote_batch, args.archive, transfer)
    else:
        require(args.archive is not None, "--archive is required.")
        import_results(args.archive, args.batch or local_batch)


if __name__ == "__main__":
    main()
