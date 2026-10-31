"""Controller-only scaling preserves the frozen scientific execution identity."""

import copy
import json
from pathlib import Path
import sys
import tempfile
import unittest
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "run"))
from structural import scaled
from verify_model import sha256, write_json


class ScalingTests(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.addCleanup(self.temporary.cleanup)
        self.root = Path(self.temporary.name)
        self.reference = self.root / "reference"
        self.reference.mkdir()
        write_json(self.reference / "manifest.json", scaled.production.pilot.study_manifest(20, 20261026))
        self.approval = self.root / "original_approval.json"
        write_json(self.approval, scaled.production.APPROVAL)
        self.metadata = scaled.production.manifest(self.reference, self.approval)
        self.batch = self.root / "batch"
        self.batch.mkdir()
        write_json(self.batch / "manifest.json", self.metadata)
        self.fingerprint = sha256(self.batch / "manifest.json")
        self.authorization = self.root / "scaling_authorization.json"
        self.authorized = {**scaled.AUTHORIZATION, "origin_execution_manifest_sha256": self.fingerprint}
        write_json(self.authorization, self.authorized)
        self.patch = patch.object(scaled, "AUTHORIZATION", self.authorized)
        self.patch.start()
        self.addCleanup(self.patch.stop)
        self.output = self.root / "review"

    def test_frozen_context_requires_same_manifest_and_source(self):
        value = scaled.context(self.batch, self.authorization)
        self.assertEqual(value, (self.metadata, self.reference, self.approval, self.fingerprint))
        changed = copy.deepcopy(self.metadata)
        changed["seeds"] = changed["seeds"][:-1]
        write_json(self.batch / "manifest.json", changed)
        with self.assertRaisesRegex(ValueError, "unchanged original execution manifest"):
            scaled.context(self.batch, self.authorization)

    def test_different_authorization_is_rejected(self):
        changed = copy.deepcopy(self.authorized)
        changed["workers"] = 64
        write_json(self.authorization, changed)
        with self.assertRaisesRegex(ValueError, "authorization differs"):
            scaled.context(self.batch, self.authorization)

    def test_local_controller_cannot_execute(self):
        with patch.object(scaled.production.platform, "node", return_value="workstation"):
            with patch.object(sys, "argv", ["scaled", "--batch", str(self.batch),
                                           "--authorization", str(self.authorization),
                                           "--review-out", str(self.output)]):
                with self.assertRaisesRegex(RuntimeError, "not locally"):
                    scaled.main()

    def test_failed_reference_gate_prevents_variant_execution(self):
        manifest_before = (self.batch / "manifest.json").read_bytes()
        approval_before = self.approval.read_bytes()
        with patch.object(scaled.production, "run_phase") as phase:
            with patch.object(scaled.production, "reference_gate", side_effect=ValueError("gate mismatch")):
                with self.assertRaisesRegex(ValueError, "gate mismatch"):
                    scaled.resume(self.batch, self.authorization, self.output)
        phase.assert_called_once()
        self.assertEqual(phase.call_args.args[4], 48)
        self.assertEqual(phase.call_args.args[-1], "historical_reference_gate")
        self.assertEqual(len(phase.call_args.args[1]), 140)
        self.assertEqual((self.batch / "manifest.json").read_bytes(), manifest_before)
        self.assertEqual(self.approval.read_bytes(), approval_before)
        self.assertFalse((self.batch / "parent.lock").exists())
        record = json.loads((self.batch / "operational_execution.json").read_text())
        self.assertEqual(record["origin_execution_manifest_sha256"], self.fingerprint)
        self.assertEqual(record["authorization"]["workers"], 48)
        self.assertEqual(record["controller_source_sha256"], sha256(Path(scaled.__file__)))
        self.assertEqual(record["controller_source"], Path(scaled.__file__).read_text(encoding="utf-8"))

    def test_source_changes_are_rejected_even_with_same_saved_manifest(self):
        changed = copy.deepcopy(self.metadata)
        changed["source_sha256"]["model/flood_model.py"] = "a" * 64
        with patch.object(scaled.production, "manifest", return_value=changed):
            with self.assertRaisesRegex(ValueError, "scientific source"):
                scaled.context(self.batch, self.authorization)

    def test_existing_controller_lock_not_removed(self):
        lock = self.batch / "parent.lock"
        lock.write_text('{"pid":123}\n')
        with self.assertRaises(FileExistsError):
            scaled.resume(self.batch, self.authorization, self.output)
        self.assertEqual(lock.read_text(), '{"pid":123}\n')
        self.assertFalse((self.batch / "operational_execution.json").exists())

    def test_changed_operational_provenance_is_rejected(self):
        write_json(self.batch / "operational_execution.json", {"workers": 64})
        with self.assertRaisesRegex(ValueError, "worker-change provenance differs"):
            scaled.resume(self.batch, self.authorization, self.output)
        self.assertFalse((self.batch / "parent.lock").exists())

    def test_unknown_completion_directory_is_rejected(self):
        unexpected = self.batch / "unknown__seed1"
        unexpected.mkdir()
        write_json(unexpected / "completion.json", {})
        with self.assertRaisesRegex(ValueError, "Unexpected or changed retained completion"):
            scaled.context(self.batch, self.authorization)

    def test_successful_controller_runs_both_phases_and_the_original_review(self):
        sys.path.insert(0, str(ROOT / "analysis"))
        import structural_production_results
        manifest_before = (self.batch / "manifest.json").read_bytes()
        with patch.object(scaled.production, "run_phase") as phase:
            with patch.object(scaled.production, "reference_gate", return_value={"exact_reference_matches": 140}):
                with patch.object(scaled, "complete", return_value=True):
                    with patch.object(scaled, "sha256", wraps=sha256) as digest:
                        digest.side_effect = lambda p: "b" * 64 if p.name == "completion.json" else sha256(p)
                        with patch.object(structural_production_results, "review", return_value={}) as review:
                            scaled.resume(self.batch, self.authorization, self.output)
        self.assertEqual(phase.call_count, 2)
        self.assertEqual([len(call.args[1]) for call in phase.call_args_list], [140, 1120])
        self.assertTrue(all(call.args[4] == 48 for call in phase.call_args_list))
        review.assert_called_once_with(self.batch, self.reference, self.approval,
                                       self.output, self.fingerprint)
        validation = json.loads((self.batch / "validation.json").read_text())
        self.assertEqual(len(validation["completion_sha256"]), 1260)
        self.assertEqual(validation["manifest_sha256"], self.fingerprint)
        self.assertEqual(validation["pilot_runs_reused"], 0)
        self.assertEqual((self.batch / "manifest.json").read_bytes(), manifest_before)
        self.assertFalse((self.batch / "parent.lock").exists())


if __name__ == "__main__":
    unittest.main()
