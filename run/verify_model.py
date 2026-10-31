"""Bounded full-horizon verification of the standard model, not policy inference."""

import argparse
from concurrent.futures import ThreadPoolExecutor, as_completed
import hashlib
import importlib.metadata
import json
import math
import os
from pathlib import Path
import subprocess
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
SEED = 20261026
SCENARIOS = {
    "organization_0": {"community_organization_level": 0.0},
    "organization_1": {"community_organization_level": 1.0},
    "near_capacity30": {
        "shelter_cap_limit": 30, "shelter_route_distance_km": 2.0,
        "shelter_route_is_water_crossing": True,
        "shelter_file": str(ROOT / "santa_lucia_map_data" /
                            "shelter_distance_candidates" / "shelter_nearby_crossing.zip"),
    },
    "far_capacity30": {
        "shelter_cap_limit": 30, "shelter_route_distance_km": 99.9,
        "shelter_file": str(ROOT / "santa_lucia_map_data" /
                            "shelter_distance_candidates" / "shelter_barranquilla.zip"),
    },
}
OUTPUTS = ["hourly_states.csv", "rescue_intervals.csv", "rescue_admissions.csv", "summary.json"]


def sha256(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def write_json(path, value):
    content = json.dumps(value, indent=2, sort_keys=True, allow_nan=False) + "\n"
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(content, encoding="utf-8")
    temporary.replace(path)


def parameters(scenario):
    from run_experimentA_main_sweep import FIXED
    return {**FIXED, **SCENARIOS[scenario], "seed": SEED}


def manifest():
    files = [p for directory in ["agents", "model", "space", "data_collection", "run"]
             for p in (ROOT / directory).glob("*.py")]
    files.extend([ROOT / "requirements.txt", ROOT / "environment-lock.txt"])
    return {
        "purpose": "Software and full-horizon measurement verification; not a policy experiment.",
        "seed": SEED, "steps": 408,
        "scenarios": {name: parameters(name) for name in SCENARIOS},
        "source_sha256": {str(p.relative_to(ROOT)): sha256(p) for p in sorted(files)},
        "spatial_sha256": {str(p.relative_to(ROOT)): sha256(p)
                           for p in sorted((ROOT / "santa_lucia_map_data").rglob("*.zip"))},
        "python": sys.version, "executable": sys.executable,
        "packages": dict(sorted((d.metadata["Name"], d.version)
                               for d in importlib.metadata.distributions() if d.metadata["Name"])),
        "seeding": "Fresh interpreter per job; Python, NumPy and Mesa seeded before population creation.",
    }


def complete(directory, fingerprint):
    marker = directory / "completion.json"
    if not marker.exists():
        return False
    record = json.loads(marker.read_text(encoding="utf-8"))
    if record["manifest_sha256"] != fingerprint or set(record["outputs"]) != set(OUTPUTS):
        raise ValueError(f"Completion does not match the run contract: {directory}")
    for name in OUTPUTS:
        path = directory / name
        if not path.is_file() or sha256(path) != record["outputs"][name]:
            raise ValueError(f"Completed output is missing or changed: {path}")
    return True


def household_diagnostics(model):
    from agents.flood_agents import Person_Agent
    people = [p for p in model.schedule.agents if isinstance(p, Person_Agent)]
    entries = [p for house in model.space.houses for p in house.residents]
    sizes = [len(h.residents) for h in model.space.houses]
    if (len(entries) != model.num_persons or {id(p) for p in entries} != {id(p) for p in people}
            or any(p.household is not house for house in model.space.houses for p in house.residents)
            or any(not any(p.age >= 18 for p in house.residents) for house in model.space.houses)):
        raise ValueError("Invalid household membership or missing adult anchor.")
    base, larger = divmod(len(people), len(sizes))
    expected = sorted([base] * (len(sizes) - larger) + [base + 1] * larger)
    if sorted(sizes) != expected:
        raise ValueError("Household allocation does not match the balanced-size design.")
    return {"people": len(people), "residential_entries": len(entries),
            "houses": len(sizes), "three_person_houses": sizes.count(3),
            "four_person_houses": sizes.count(4), "houses_without_adult": 0,
            "unique_membership": True}


def check_protection(model):
    from agents.flood_agents import Person_Agent
    sheltered = model.shelter.sheltered_agents
    patients = model.healthcare.hospitalized_agents
    shelter_ids = {id(p) for p in sheltered}
    patient_ids = {id(p) for p in patients}
    if (len(shelter_ids) != len(sheltered) or len(patient_ids) != len(patients)
            or shelter_ids & patient_ids
            or len(sheltered) > math.ceil(model.shelter.capacity_limit)
            or len(patients) > math.ceil(model.healthcare.capacity_limit)):
        raise ValueError("Duplicate, overlapping or over-capacity institutional membership.")
    spatial_ids = {id(p) for p in model.space.agents}
    for p in model.schedule.agents:
        if not isinstance(p, Person_Agent):
            continue
        if ((not p.alive and (p.injured or p.stranded))
                or (p.alive and p.injured and p.time_injured >= p.survivability_duration)):
            raise ValueError(f"Inconsistent clinical survival state: person {p.population_index}.")
        if p.evacuated == (id(p) in spatial_ids):
            raise ValueError(f"Evacuation/spatial membership mismatch: person {p.population_index}.")
        for institution, members in [(model.shelter, shelter_ids), (model.healthcare, patient_ids)]:
            if id(p) in members and (not p.alive or p.evacuated or p.stranded
                                     or not institution.geometry.covers(p.geometry)):
                raise ValueError(f"Protected-state/location mismatch: person {p.population_index}.")


def worker(task_path, manifest_factory=None):
    import numpy as np
    import pandas as pd
    from agents.flood_agents import Person_Agent
    from model.flood_model import FloodModel

    task = json.loads(task_path.read_text(encoding="utf-8"))
    batch = task_path.parent.parent
    recorded = json.loads((batch / "manifest.json").read_text(encoding="utf-8"))
    current_manifest = manifest if manifest_factory is None else manifest_factory
    if recorded != current_manifest():
        raise ValueError("Source, spatial inputs or environment changed before worker execution.")
    if task["manifest_sha256"] != sha256(batch / "manifest.json"):
        raise ValueError("Task manifest fingerprint does not match the batch.")
    start = time.monotonic()
    model_parameters = {**recorded["scenarios"][task["scenario"]], "seed": task["seed"]}
    model = FloodModel(**model_parameters)
    households = household_diagnostics(model)
    check_protection(model)
    intervals = []
    for hour in range(408):
        injuries = [(p, p.time_injured) for p in model.schedule.agents
                    if isinstance(p, Person_Agent) and p.alive and p.injured]
        model.step()
        for patient, previous in injuries:
            if patient.alive and patient.injured and patient.time_injured != previous + 1:
                raise ValueError(f"Injury clock did not advance once: person {patient.population_index}.")
        check_protection(model)
        if model._rescues_this_hour > model._rescue_slots_this_hour:
            raise ValueError("Hourly rescue capacity exceeded.")
        if model.rescues_total != len(model.rescue_records):
            raise ValueError("Rescue admission records disagree with cumulative count.")
        intervals.append({
            "Interval_Start_Hour": hour, "Interval_End_Hour": hour + 1,
            "Rescue_Slots": model._rescue_slots_this_hour,
            "Rescue_Admissions": model._rescues_this_hour,
            "Admissions_Cumulative": model.rescues_total,
            "Distinct_Rescued": len(model.rescued_person_ids),
            "Shelter_Occupancy_End": len(model.shelter.sheltered_agents),
            "Hospital_Occupancy_End": len(model.healthcare.hospitalized_agents),
            "Nominal_Rescue_Rate": model._rescue_throughput_per_hour,
        })
    model.datacollector.collect(model)
    states = model.datacollector.get_model_vars_dataframe().reset_index(names="Simulation_Hour")
    fractions = states[["Death", "Stranded", "Injured", "Sheltered", "Hospitalized", "Evacuated"]]
    if (states.Simulation_Hour.tolist() != list(range(409))
            or not np.isfinite(fractions.to_numpy()).all()
            or ((fractions < 0) | (fractions > 1)).any().any()
            or (states.Death.diff().dropna() < -1e-12).any()):
        raise ValueError("Invalid hourly outcomes.")
    if not np.array_equal(states.Rescue_Admissions.to_numpy(),
                          [0] + [e["Admissions_Cumulative"] for e in intervals]):
        raise ValueError("Hourly rescue-state alignment is inconsistent.")
    if model.rescues_total != len(model.rescued_person_ids):
        raise ValueError("Unexpected readmission under a single-footprint protection design.")
    summary = {
        "scenario": task["scenario"], "seed": task["seed"],
        "manifest_sha256": task["manifest_sha256"],
        "households": households, "hourly_checks": len(intervals),
        "Death_hour384": float(states.Death.iloc[384]),
        "Death_hour408": float(states.Death.iloc[408]),
        "Stranded_peak_hourly": float(states.Stranded.max()),
        "Rescue_Admissions": model.rescues_total,
        "Distinct_Rescued": len(model.rescued_person_ids),
        "Shelter_Peak_People": round(float(states.Sheltered.max()) * model.num_persons),
        "Hospital_Peak_People": round(float(states.Hospitalized.max()) * model.num_persons),
        "Stranding_Hours_At_Admission_Mean": (
            float(np.mean(model.rescue_stranding_durations)) if model.rescues_total else None),
        "elapsed_seconds": time.monotonic() - start,
    }
    if recorded != current_manifest():
        raise ValueError("Source, inputs or environment changed during worker execution.")
    for name, frame in [
        ("hourly_states.csv", states), ("rescue_intervals.csv", pd.DataFrame(intervals)),
        ("rescue_admissions.csv", pd.DataFrame(model.rescue_records,
                                             columns=["hour", "person_index", "stranding_hours"])),
    ]:
        path = task_path.parent / name
        temporary = path.with_suffix(".csv.tmp")
        frame.to_csv(temporary, index=False)
        temporary.replace(path)
    write_json(task_path.parent / "summary.json", summary)
    write_json(task_path.parent / "completion.json", {
        "manifest_sha256": task["manifest_sha256"],
        "outputs": {name: sha256(task_path.parent / name) for name in OUTPUTS},
    })
    print(json.dumps(summary, sort_keys=True))


def launch(task_path):
    task = json.loads(task_path.read_text(encoding="utf-8"))
    env = os.environ.copy()
    env.update({"PYTHONHASHSEED": str(SEED), "PYTHONDONTWRITEBYTECODE": "1",
                "OMP_NUM_THREADS": "1", "OPENBLAS_NUM_THREADS": "1", "MKL_NUM_THREADS": "1"})
    log_path = task_path.parent / "worker.log"
    with log_path.open("w", encoding="utf-8") as log:
        result = subprocess.run([sys.executable, "-B", str(Path(__file__)),
                                 "--worker", str(task_path)],
                                cwd=ROOT, env=env, stdout=log, stderr=subprocess.STDOUT)
    if result.returncode:
        raise RuntimeError(f"Verification worker failed (exit {result.returncode}): {log_path}")
    if not complete(task_path.parent, task["manifest_sha256"]):
        raise RuntimeError(f"Worker did not record verified completion: {log_path}")
    return json.loads((task_path.parent / "summary.json").read_text(encoding="utf-8"))


def check_repeats(batch, fingerprint):
    import pandas as pd
    for scenario in ["near_capacity30", "far_capacity30"]:
        first = batch / (scenario + "__repeat1")
        repeated = batch / (scenario + "__repeat2")
        if not complete(first, fingerprint) or not complete(repeated, fingerprint):
            raise ValueError(f"Reproducibility check is incomplete: {scenario}")
        for name in OUTPUTS[:3]:
            pd.testing.assert_frame_equal(pd.read_csv(first / name),
                                          pd.read_csv(repeated / name), check_exact=True)
        summaries = [json.loads((directory / "summary.json").read_text(encoding="utf-8"))
                     for directory in [first, repeated]]
        for record in summaries:
            record.pop("elapsed_seconds")
        if summaries[0] != summaries[1]:
            raise ValueError(f"Scientific summaries differ on repetition: {scenario}")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path, default=ROOT / "data_collection" / "verification")
    parser.add_argument("--workers", type=int, choices=[1, 2], default=2)
    parser.add_argument("--max-jobs", type=int, default=6)
    parser.add_argument("--worker", type=Path, help=argparse.SUPPRESS)
    args = parser.parse_args()
    if args.worker is not None:
        worker(args.worker)
        return
    if args.max_jobs < 1:
        parser.error("--max-jobs must be positive.")
    if (importlib.metadata.version("mesa") != "2.1.2"
            or importlib.metadata.version("mesa-geo") != "0.7.1"):
        parser.error("Use the pinned simulation environment.")
    batch = args.out.resolve()
    batch.mkdir(parents=True, exist_ok=True)
    metadata = manifest()
    manifest_path = batch / "manifest.json"
    if manifest_path.exists():
        if json.loads(manifest_path.read_text(encoding="utf-8")) != metadata:
            raise ValueError("Verification configuration changed; do not mix output cohorts.")
    else:
        write_json(manifest_path, metadata)
    fingerprint = sha256(manifest_path)
    tasks = []
    for scenario, repeat in [(name, 1) for name in SCENARIOS] + [
            ("near_capacity30", 2), ("far_capacity30", 2)]:
        directory = batch / f"{scenario}__repeat{repeat}"
        if complete(directory, fingerprint):
            continue
        directory.mkdir(parents=True, exist_ok=True)
        path = directory / "task.json"
        write_json(path, {"scenario": scenario, "repeat": repeat, "seed": SEED,
                          "manifest_sha256": fingerprint})
        tasks.append(path)
    print(f"{6-len(tasks)}/6 verification runs complete; "
          f"running {min(args.max_jobs, len(tasks))} now.", flush=True)
    with ThreadPoolExecutor(max_workers=args.workers) as executor:
        futures = {executor.submit(launch, p): p for p in tasks[:args.max_jobs]}
        for future in as_completed(futures):
            result = future.result()
            print(f"COMPLETE {futures[future].parent.name}: "
                  f"deaths={result['Death_hour408']:.6f}, "
                  f"admissions={result['Rescue_Admissions']}, "
                  f"distinct={result['Distinct_Rescued']}, "
                  f"seconds={result['elapsed_seconds']:.1f}", flush=True)
    if len(tasks) <= args.max_jobs:
        check_repeats(batch, fingerprint)
        write_json(batch / "validation.json", {
            "runs": 6, "steps_per_run": 408, "states_per_run": 409,
            "manifest_sha256": fingerprint, "exact_full_horizon_repeats": 2,
            "invariants": "Adult-anchored balanced households; exclusive protected states; "
                          "capacity bounds; protected geometry; one admission per person "
                          "under the single-footprint scenario.",
            "policy_effect_inference": False,
        })
        print("All six full-horizon runs and both exact repeat checks passed.", flush=True)


if __name__ == "__main__":
    main()
