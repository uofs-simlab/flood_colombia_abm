"""Portability contracts, exact-repeat gates and lossless HPC imports."""

import copy
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import tempfile
import unittest
from unittest.mock import patch
import zipfile

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "run"))
from hpc import transfer
from test_paper_results import write_fixture, refresh_completion


class TransferTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.metadata = transfer.study_manifest(2, 100)

    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.addCleanup(self.temporary.cleanup)
        self.root = Path(self.temporary.name)

    def test_only_platform_paths_and_runtime_location_can_change(self):
        other = copy.deepcopy(self.metadata)
        other["python"] = "3.12.3 (Linux)"
        other["executable"] = "/scratch/user/flood_columbia/.venv/bin/python"
        for field in ["source_sha256", "spatial_sha256"]:
            other[field] = {k.replace("\\", "/"): v for k, v in other[field].items()}
        for parameters in other["scenarios"].values():
            for key, value in parameters.items():
                if key.endswith("_file") or key.startswith("flood_file_"):
                    value = value.replace("\\", "/")
                    parameters[key] = "/scratch/user/flood_columbia/" + value[value.index("santa_lucia_map_data/"):]
        transfer.require_compatible(self.metadata, other)
        for changed in ["source", "population", "package", "python"]:
            invalid = copy.deepcopy(other)
            if changed == "source":
                invalid["source_sha256"][next(iter(invalid["source_sha256"]))] = "changed"
            elif changed == "population":
                invalid["scenarios"]["A_org_0"]["N_persons"] = 973
            elif changed == "package":
                invalid["packages"]["numpy"] = "2.0.0"
            else:
                invalid["python"] = "3.14.0"
            with self.subTest(changed=changed), self.assertRaises(ValueError):
                transfer.require_compatible(self.metadata, invalid)

    def test_archive_extraction_rejects_parent_absolute_and_duplicate_paths(self):
        for index, name in enumerate(["../outside", "/absolute", "C:/outside", "..\\outside"]):
            archive = self.root / f"bad{index}.zip"
            with zipfile.ZipFile(archive, "w") as handle:
                handle.writestr(name, "invalid")
            with self.subTest(path=name), self.assertRaises(ValueError):
                transfer.extract_checked(archive, self.root / "output")

    def test_adapter_stays_outside_the_frozen_model_source_fingerprint(self):
        self.assertNotIn(str(Path("run") / "hpc" / "transfer.py"), self.metadata["source_sha256"])
        self.assertIn(str(Path("run") / "run_study.py"), self.metadata["source_sha256"])

    def shell_pipeline(self, fail_sensitivity_gate=False):
        bash = (Path(os.environ["ProgramFiles"]) / "Git" / "bin" / "bash.exe"
                if os.name == "nt" else Path(shutil.which("bash") or "/missing-bash"))
        if not bash.is_file():
            self.skipTest("Bash is required for shell-orchestration execution tests.")
        script = self.root / "run" / "hpc" / "execute.sh"
        script.parent.mkdir(parents=True)
        shutil.copyfile(ROOT / "run" / "hpc" / "execute.sh", script)
        python = self.root / ".venv" / "bin" / "python"
        python.parent.mkdir(parents=True)
        python.write_text(
            '#!/usr/bin/env bash\n'
            'test "$OMP_NUM_THREADS" = 1 || exit 8\n'
            'printf "%s\\n" "$*" >> calls.log\n'
            'if [ "${FAIL_SENSITIVITY_GATE:-0}" = 1 ] && '
            '[[ " $* " == *" gate "* && " $* " == *" --sensitivity "* ]]; then exit 9; fi\n'
            'exit 0\n', encoding="utf-8")
        python.chmod(0o755)
        env = {**os.environ, "FAIL_SENSITIVITY_GATE": str(int(fail_sensitivity_gate))}
        return subprocess.run([str(bash), str(script), "all", "64"], cwd=self.root,
                              env=env, text=True, capture_output=True)

    def test_all_cohorts_shell_pipeline_runs_gates_before_each_experiment(self):
        result = self.shell_pipeline()
        self.assertEqual(result.returncode, 0, result.stderr)
        calls = (self.root / "calls.log").read_text().splitlines()
        self.assertEqual([call.split()[3] for call in calls],
                         ["gate", "run", "export", "gate", "run", "export"])
        self.assertTrue(all("--sensitivity" not in call for call in calls[:3]))
        self.assertTrue(all("--sensitivity" in call for call in calls[3:]))
        status = self.root / "data_collection" / "hpc_transfer" / "pipeline.status"
        self.assertEqual(status.read_text().strip(), "0")

    def test_all_cohorts_shell_pipeline_stops_and_records_a_failed_gate(self):
        result = self.shell_pipeline(fail_sensitivity_gate=True)
        self.assertEqual(result.returncode, 9, result.stderr)
        calls = (self.root / "calls.log").read_text().splitlines()
        self.assertEqual([call.split()[3] for call in calls], ["gate", "run", "export", "gate"])
        transfer_root = self.root / "data_collection" / "hpc_transfer"
        self.assertEqual((transfer_root / "pipeline.status").read_text().strip(), "9")
        self.assertFalse((transfer_root / "sensitivity" / "run.status").exists())

    def make_return(self):
        batch = self.root / "study"
        batch.mkdir()
        transfer.write_json(batch / "manifest.json", self.metadata)
        write_fixture(batch, self.metadata)
        original_hash = transfer.sha256(batch / "manifest.json")
        returned = self.root / "returned"
        evidence = returned / "evidence"
        evidence.mkdir(parents=True)
        names = ["A_org_0.25__seed101", "D_cap1_nearby_crossing__seed101"]
        plan = {
            "origin_manifest_sha256": original_hash,
            "tasks": [{"scenario": name.split("__seed")[0],
                       "seed": int(name.split("__seed")[1]), "manifest_sha256": original_hash}
                      for name in names],
            "references": [f"{name}__seed100" for name in transfer.GATE_SCENARIOS],
        }
        transfer.write_json(evidence / "plan.json", plan)
        execution = copy.deepcopy(self.metadata)
        execution["python"] = "3.12.3 (Linux)"
        execution["executable"] = "/scratch/example/.venv/bin/python"
        execution["execution"] = {
            "origin_manifest_sha256": original_hash,
            "plan_sha256": transfer.sha256(evidence / "plan.json"),
            "adapter_sha256": transfer.sha256(Path(transfer.__file__)),
        }
        transfer.write_json(evidence / "manifest.json", execution)
        remote_hash = transfer.sha256(evidence / "manifest.json")
        hashes = {}
        for category, settings in [("evidence/gates", plan["references"]), ("runs", names)]:
            for name in settings:
                destination = returned / Path(category) / name
                shutil.copytree(batch / name, destination)
                for filename in ["task.json", "summary.json"]:
                    record = transfer.read_json(destination / filename)
                    record["manifest_sha256"] = remote_hash
                    transfer.write_json(destination / filename, record)
                refresh_completion(destination, remote_hash)
                if category == "evidence/gates":
                    hashes[name] = {
                        "local_completion": transfer.sha256(batch / name / "completion.json"),
                        "remote_completion": transfer.sha256(destination / "completion.json"),
                    }
        transfer.write_json(evidence / "bridge.json", {
            "origin_manifest_sha256": original_hash,
            "execution_manifest_sha256": remote_hash, "reference_hashes": hashes,
        })
        return batch, returned, names, remote_hash

    def archive_return(self, returned):
        archive = self.root / "results.zip"
        with zipfile.ZipFile(archive, "w", compression=zipfile.ZIP_DEFLATED) as handle:
            for path in returned.rglob("*"):
                if path.is_file():
                    handle.write(path, path.relative_to(returned).as_posix())
        return archive

    def test_four_exact_cross_platform_references_are_required(self):
        batch, returned, _, _ = self.make_return()
        transfer.validate_bridge(returned / "evidence", batch)
        gate = returned / "evidence" / "gates" / "A_org_0__seed100"
        record = transfer.read_json(gate / "summary.json")
        record["Death_hour408"] = .9
        transfer.write_json(gate / "summary.json", record)
        refresh_completion(gate, transfer.sha256(returned / "evidence" / "manifest.json"))
        with self.assertRaisesRegex(ValueError, "fingerprints differ"):
            transfer.validate_bridge(returned / "evidence", batch)

    def test_import_preserves_original_records_and_exact_local_duplicates(self):
        batch, returned, names, remote_hash = self.make_return()
        missing = batch / names[0]
        saved = self.root / "original_job"
        missing.rename(saved)
        missing.mkdir()
        (missing / "worker.log").write_text("Interrupted local work, not complete.")
        original_duplicate = (batch / names[1] / "summary.json").read_bytes()
        archive = self.archive_return(returned)
        transfer.import_results(archive, batch)
        self.assertTrue((batch / "incomplete_local" / names[0] / "worker.log").is_file())
        self.assertEqual((batch / names[1] / "summary.json").read_bytes(), original_duplicate)
        self.assertEqual(transfer.read_json(batch / names[0] / "summary.json")["manifest_sha256"],
                         remote_hash)
        executions = transfer.execution_fingerprints(batch)
        self.assertEqual(transfer.run_fingerprint(batch / names[0], "original", executions), remote_hash)
        self.assertEqual(transfer.run_fingerprint(batch / names[1], "original", executions), "original")
        sys.path.insert(0, str(ROOT / "analysis"))
        import paper_results
        frame, _, _ = paper_results.load_cohort(batch)
        self.assertEqual(len(frame), 48)
        transfer.import_results(archive, batch)
        self.assertFalse((batch / "parent.lock").exists())

    def test_import_rejects_conflicting_completed_outputs_without_writing(self):
        batch, returned, names, remote_hash = self.make_return()
        source = returned / "runs" / names[0]
        record = transfer.read_json(source / "summary.json")
        record["Death_hour408"] = .9
        transfer.write_json(source / "summary.json", record)
        refresh_completion(source, remote_hash)
        with self.assertRaisesRegex(ValueError, "Scientific summaries differ"):
            transfer.import_results(self.archive_return(returned), batch)
        self.assertFalse((batch / "executions").exists())
        self.assertFalse((batch / "parent.lock").exists())

    def test_import_refuses_active_local_parent(self):
        batch = self.root / "study"
        batch.mkdir()
        (batch / "parent.lock").write_text("{}")
        with self.assertRaisesRegex(ValueError, "Stop the local study parent"):
            transfer.import_results(self.root / "absent.zip", batch)

    def test_imported_job_requires_verified_execution_sidecar(self):
        job = self.root / "A_org_0__seed100"
        job.mkdir()
        transfer.write_json(job / "execution_origin.json", {"execution_manifest_sha256": "unknown"})
        with self.assertRaisesRegex(ValueError, "Unverified or unplanned"):
            transfer.run_fingerprint(job, "original", {})

    def test_bundle_uses_linux_shell_endings_without_changing_model_bytes(self):
        batch, _, _, _ = self.make_return()
        source_root = self.root / "source"
        for name in [*self.metadata["source_sha256"], *self.metadata["spatial_sha256"]]:
            path = source_root / transfer.relative_path(name)
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_bytes(b"sample\r\n")
        shell = source_root / "run" / "hpc" / "execute.sh"
        shell.parent.mkdir(parents=True)
        shell.write_bytes(b"#!/bin/bash\r\nset -euo pipefail\r\n")
        archive = self.root / "deployment.zip"
        incomplete = batch / "A_org_0.25__seed101"
        (incomplete / "completion.json").unlink()
        with patch.object(transfer, "ROOT", source_root), patch.object(
                transfer, "study_manifest", return_value=self.metadata):
            transfer.bundle(batch, archive)
        with zipfile.ZipFile(archive) as handle:
            self.assertNotIn(b"\r", handle.read("run/hpc/execute.sh"))
            self.assertEqual(handle.read("model/flood_model.py"), b"sample\r\n")

    def test_dispatch_plan_rejects_out_of_design_and_duplicate_tasks(self):
        origin_hash = "origin"
        task = {"scenario": "A_org_0", "seed": 100, "manifest_sha256": origin_hash}
        plan = {"origin_manifest_sha256": origin_hash, "tasks": [task, task],
                "references": [f"{name}__seed100" for name in transfer.GATE_SCENARIOS]}
        with self.assertRaisesRegex(ValueError, "duplicated"):
            transfer.validate_plan(plan, self.metadata)
        plan["tasks"] = [{**task, "seed": 999}]
        with self.assertRaisesRegex(ValueError, "Invalid"):
            transfer.validate_plan(plan, self.metadata)


if __name__ == "__main__":
    unittest.main()
