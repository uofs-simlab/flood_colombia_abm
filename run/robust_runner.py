"""
Checkpointed, crash-resilient batch runner.

This environment cannot guarantee a long-running background process survives
an idle period between conversation turns (observed directly: a ~4h gap led
to a full sandbox restart that silently killed a `mesa.batchrunner.batch_run`
job partway through, losing all of its in-memory results because they were
only written to disk at the very end).

This runner instead executes one (params, iteration) combination at a time,
appends that single run's results to the output CSV IMMEDIATELY, and records
completed run keys in a JSON checkpoint file next to it. On restart, calling
this again with the same `run_key_fields` skips any run already checkpointed,
so at most one in-progress run is ever lost.
"""
import os
import json
import itertools
import time
import hashlib
import importlib.metadata
from pathlib import Path
import sys
import pandas as pd
from concurrent.futures import ProcessPoolExecutor, as_completed


def _expand_param_grid(param_grid):
    """param_grid: dict of name -> list of values. Returns list of dicts (cartesian product)."""
    keys = list(param_grid.keys())
    combos = []
    for values in itertools.product(*[param_grid[k] for k in keys]):
        combos.append(dict(zip(keys, values)))
    return combos


def _run_key(params, iteration, vary_keys):
    """Stable string key identifying a single run, for checkpointing."""
    parts = [f"{k}={params[k]}" for k in sorted(vary_keys)]
    parts.append(f"iter={iteration}")
    return "|".join(parts)


def _run_one(model_cls, params, iteration, max_steps, data_collection_period):
    """Runs a single model instance to completion and returns its per-step
    model-reporter dataframe (mirrors what mesa.batchrunner.batch_run would
    produce for one run), tagged with the run's own parameters + iteration.
    Runs in a worker process, so must only import top-level picklable stuff.
    """
    model = model_cls(**params)
    for step in range(max_steps):
        model.step()
        # FloodModel.step() already calls self.datacollector.collect(self)
        # every step internally, so we don't need to call it again here.
    model.datacollector.collect(model)
    df = model.datacollector.get_model_vars_dataframe().reset_index().rename(columns={"index": "Step"})
    # Keep only every `data_collection_period`-th row to control file size,
    # matching the batch_run(data_collection_period=...) semantics.
    if data_collection_period > 1:
        df = df[(df["Step"] % data_collection_period == 0) | (df["Step"] == max_steps)].copy()
    for k, v in params.items():
        df[k] = v
    df["iteration"] = iteration
    return df


def run_checkpointed(model_cls, param_grid, vary_keys, iterations, max_steps,
                      out_csv, checkpoint_json, data_collection_period=1,
                      max_workers=2, max_jobs_this_call=None, seed_base=20261026):
    """
    param_grid: dict of ALL model kwargs -> list of values (fixed params have
                a single-element list).
    vary_keys: list of the param names that actually vary (used to build a
               human-readable, stable checkpoint key -- fixed params are
               omitted from the key since they never change across runs in
               a single call).
    Writes/appends to `out_csv` incrementally and tracks progress in
    `checkpoint_json`. Safe to re-invoke after a crash/restart: already
    completed (params, iteration) runs are skipped.
    """
    if "seed" in param_grid:
        raise ValueError("Use seed_base and iteration for paired-seed replication, not a seed grid.")
    if (isinstance(seed_base, bool) or not isinstance(seed_base, int)
            or not 0 <= seed_base < 2**32 or seed_base + iterations > 2**32):
        raise ValueError("Replication seeds must be integers in [0, 2**32).")
    root = Path(__file__).resolve().parents[1]
    source_files = [path for directory in ["agents", "model", "space", "data_collection", "run"]
                    for path in (root / directory).glob("*.py")]
    manifest_path = Path(str(checkpoint_json) + ".manifest.json")
    manifest = {
        "parameters": param_grid, "iterations": iterations, "steps": max_steps,
        "data_collection_period": data_collection_period, "seed_base": seed_base,
        "source_sha256": {str(path.relative_to(root)): hashlib.sha256(path.read_bytes()).hexdigest()
                          for path in sorted(source_files)},
        "spatial_sha256": {str(path.relative_to(root)): hashlib.sha256(path.read_bytes()).hexdigest()
                           for path in sorted((root / "santa_lucia_map_data").rglob("*.zip"))},
        "python": sys.version, "executable": sys.executable,
        "packages": dict(sorted((d.metadata["Name"], d.version)
                               for d in importlib.metadata.distributions() if d.metadata["Name"])),
    }
    if manifest_path.exists():
        if json.loads(manifest_path.read_text(encoding="utf-8")) != manifest:
            raise ValueError("Source, inputs, environment or experiment design changed; "
                             "do not append to an existing cohort.")
    elif os.path.exists(out_csv) or os.path.exists(checkpoint_json):
        raise ValueError("Existing experiment files have no matching seeded source manifest. "
                         "They are preserved; select a new experiment cohort, not these checkpoints.")
    else:
        manifest_path.parent.mkdir(parents=True, exist_ok=True)
        manifest_path.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n",
                                 encoding="utf-8")
    combos = _expand_param_grid(param_grid)
    all_jobs = [(c, it) for c in combos for it in range(iterations)]
    all_jobs = [({**c, "seed": seed_base + it}, it) for c, it in all_jobs]

    done = set()
    if os.path.exists(checkpoint_json):
        with open(checkpoint_json) as f:
            done = set(json.load(f))

    pending = [(c, it) for c, it in all_jobs if _run_key(c, it, vary_keys) not in done]
    total = len(all_jobs)
    print(f"[robust_runner] {len(done)}/{total} runs already checkpointed, {len(pending)} pending.")

    if not pending:
        print("[robust_runner] nothing to do.")
        return {"done": len(done), "total": total, "remaining": 0}

    # This environment does not reliably keep background/nohup'd processes
    # alive across idle gaps between conversation turns (observed directly:
    # a job was killed even within ~1 minute of being backgrounded). So this
    # function is designed to be called repeatedly in the FOREGROUND, each
    # call processing only a small slice of the pending work
    # (max_jobs_this_call) so it finishes well within a single tool-call's
    # execution window. Progress is checkpointed as each run completes, so
    # calling this again just continues where it left off.
    if max_jobs_this_call is not None:
        pending = pending[:max_jobs_this_call]

    write_header = not os.path.exists(out_csv)

    t0 = time.time()
    completed_this_call = 0
    with ProcessPoolExecutor(max_workers=max_workers) as ex:
        futures = {}
        for c, it in pending:
            fut = ex.submit(_run_one, model_cls, c, it, max_steps, data_collection_period)
            futures[fut] = (c, it)

        for fut in as_completed(futures):
            c, it = futures[fut]
            key = _run_key(c, it, vary_keys)
            try:
                df = fut.result()
            except Exception as e:
                print(f"[robust_runner] RUN FAILED {key}: {e}")
                for pending_future in futures:
                    pending_future.cancel()
                raise RuntimeError(f"Experiment run failed: {key}") from e

            df.to_csv(out_csv, mode="a", header=write_header, index=False)
            write_header = False

            done.add(key)
            with open(checkpoint_json, "w") as f:
                json.dump(sorted(done), f)

            completed_this_call += 1
            elapsed = time.time() - t0
            rate = completed_this_call / elapsed if elapsed > 0 else 0
            remaining = len(pending) - completed_this_call
            eta_min = (remaining / rate / 60) if rate > 0 else float("nan")
            print(f"[robust_runner] {len(done)}/{total} done ({key}) "
                  f"elapsed={elapsed/60:.1f}min eta={eta_min:.1f}min")

    remaining = total - len(done)
    print(f"[robust_runner] call finished: {len(done)}/{total} total done, {remaining} remaining.")
    return {"done": len(done), "total": total, "remaining": remaining}
