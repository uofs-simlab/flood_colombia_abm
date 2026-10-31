"""Fixed replicated structural contract and saved-output tests; no simulations."""

import copy
import json
from pathlib import Path
import shutil
import sys
import tempfile
import unittest
from unittest.mock import patch
import zipfile

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "run"))
sys.path.insert(0, str(ROOT / "analysis"))
from structural import production
import structural_production_results as results
from verify_model import sha256, write_json
from test_paper_results import refresh_completion, write_fixture


class ContractAndEffectTests(unittest.TestCase):
    def frame(self):
        rows = []
        for seed in production.SEEDS:
            for name in production.pilot.scenarios():
                variant, anchor = name.split("__")
                deaths = 100 + production.pilot.ANCHORS.index(anchor) * 2
                if variant != "reference" and anchor == "A_org_1":
                    deaths += 1
                rows.append({"scenario": name, "seed": seed, "Deaths": deaths})
        return pd.DataFrame(rows)

    def test_fixed_fresh_budget_and_all_alternatives(self):
        self.assertEqual(len(production.SEEDS), 20)
        self.assertEqual(len(production.pilot.scenarios()) * len(production.SEEDS), 1260)
        self.assertEqual(len(production.pilot.ANCHORS) * len(production.SEEDS), 140)
        self.assertTrue(production.APPROVAL["fresh_runs_only"])
        self.assertFalse(production.APPROVAL["pilot_reuse_authorized"])
        self.assertEqual(production.APPROVAL["maximum_workers"], 8)
        self.assertEqual(production.APPROVAL["variants"], production.pilot.VARIANTS)

    def test_local_execution_and_local_workers_are_refused(self):
        for args in ([], ["--worker", "unused.json"],
                     ["--dry-run", "--worker", "unused.json"]):
            with self.subTest(args=args), patch.object(production.platform, "node", return_value="workstation"):
                with patch.object(sys, "argv", ["production", *args]):
                    with self.assertRaisesRegex(RuntimeError, "not locally"):
                        production.main()

    def test_all_32_adjusted_changes_use_exact_integer_pairs(self):
        result, changes = results.effects(self.frame())
        self.assertEqual(len(result), 36)
        self.assertEqual(len(changes), 640)
        self.assertEqual(result.p_change_holm_32.notna().sum(), 32)
        organization = result[(result.contrast == "organization") & result.variant.ne("reference")]
        self.assertTrue((changes.loc[changes.contrast == "organization",
                                     "change_in_difference_deaths"] == 1).all())
        np.testing.assert_allclose(organization.change_from_reference_pp, 100 / 972,
                                   rtol=0, atol=1e-15)
        self.assertTrue((organization.p_change_raw == 2 / 2**20).all())
        self.assertTrue((organization.p_change_holm_32 == 32 * 2 / 2**20).all())
        self.assertTrue((organization.change_mc_half_width_pp < 1e-12).all())
        unchanged = result[result.contrast != "organization"]
        self.assertTrue((unchanged.p_change_raw.dropna() == 1).all())
        self.assertTrue((result.negative_difference_seeds
                         + result.positive_difference_seeds + result.zero_difference_seeds == 20).all())

    def test_pairing_order_does_not_change_effects(self):
        frame = self.frame()
        a, changes_a = results.effects(frame)
        b, changes_b = results.effects(frame.sample(frac=1, random_state=1))
        pd.testing.assert_frame_equal(a, b, check_exact=True)
        pd.testing.assert_frame_equal(changes_a, changes_b, check_exact=True)

    def test_partial_duplicate_and_noninteger_counts_are_rejected(self):
        frame = self.frame()
        invalid = [frame.iloc[:-1], pd.concat([frame, frame.iloc[:1]])]
        for value in (.5, float("nan"), float("inf"), -1, 973):
            changed = frame.copy()
            changed["Deaths"] = changed.Deaths.astype(float)
            changed.loc[0, "Deaths"] = value
            invalid.append(changed)
        for changed in invalid:
            with self.subTest(size=len(changed)), self.assertRaises(ValueError):
                results.effects(changed)


class CompleteReviewTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.temporary = tempfile.TemporaryDirectory()
        cls.root = Path(cls.temporary.name)
        cls.reference = cls.root / "historical"
        cls.reference.mkdir()
        write_json(cls.reference / "manifest.json", production.pilot.study_manifest(20, 20261026))
        cls.approval = cls.root / "approved_design.json"
        write_json(cls.approval, production.APPROVAL)
        cls.metadata = production.manifest(cls.reference, cls.approval)
        cls.batch = cls.root / "production"
        cls.batch.mkdir()
        write_json(cls.batch / "manifest.json", cls.metadata)
        cls.fingerprint = sha256(cls.batch / "manifest.json")
        write_fixture(cls.batch, cls.metadata)
        historical_hash = sha256(cls.reference / "manifest.json")
        for seed in production.SEEDS:
            for anchor in production.pilot.ANCHORS:
                source = cls.batch / f"reference__{anchor}__seed{seed}"
                target = cls.reference / f"{anchor}__seed{seed}"
                shutil.copytree(source, target)
                for name in ("task.json", "summary.json"):
                    record = results.read_json(target / name)
                    record.update({"scenario": anchor, "manifest_sha256": historical_hash})
                    write_json(target / name, record)
                refresh_completion(target, historical_hash)
        write_json(cls.batch / "reference_gate.json",
                   production.reference_gate(cls.batch, cls.reference, cls.fingerprint))
        write_json(cls.batch / "validation.json", {
            "manifest_sha256": cls.fingerprint, "runs": 1260, "hours_per_run": 408,
            "historical_reference_matches": 140, "variant_runs": 1120,
            "completion_sha256": {
                p.parent.name: sha256(p) for p in cls.batch.glob("*/completion.json")},
            "reference_gate_sha256": sha256(cls.batch / "reference_gate.json"),
            "fresh_runs_only": True, "pilot_runs_reused": 0,
        })

    @classmethod
    def tearDownClass(cls):
        cls.temporary.cleanup()

    def output(self):
        return self.root / self.id().rsplit(".", 1)[-1]

    def run_review(self, fingerprint=None):
        return results.review(self.batch, self.reference, self.approval, self.output(),
                              self.fingerprint if fingerprint is None else fingerprint)

    def test_complete_1260_run_review_analysis_and_export(self):
        report = self.run_review()
        self.assertEqual(report["verified_runs"], 1260)
        self.assertEqual(report["exact_canonical_reference_matches"], 140)
        self.assertEqual(report["effect_modification_comparisons"], 32)
        self.assertEqual(report["paired_seedwise_contrast_changes"], 640)
        self.assertFalse(report["clinical_or_empirical_validation"])
        self.assertFalse(report["global_or_joint_robustness"])
        self.assertEqual(report["pilot_runs_reused"], 0)
        saved = pd.read_csv(self.output() / "production_effects.csv", float_precision="round_trip")
        self.assertEqual(len(saved), 36)
        self.assertEqual(saved.p_change_holm_32.notna().sum(), 32)
        self.assertTrue((saved.negative_difference_seeds + saved.positive_difference_seeds
                         + saved.zero_difference_seeds == 20).all())
        for name, digest in report["analysis_outputs_sha256"].items():
            self.assertEqual(sha256(self.output() / name), digest)
            self.assertNotIn(b"\r\n", (self.output() / name).read_bytes())
        export = results.read_json(self.output() / "export.json")
        self.assertEqual(sha256(self.output() / export["archive"]), export["archive_sha256"])
        with zipfile.ZipFile(self.output() / export["archive"]) as z:
            inventory = json.loads(z.read("retrieval_inventory.json"))
            self.assertEqual(len(inventory), export["archived_files"])
            self.assertIn("authorization/approved_design.json", inventory)
            self.assertIn("source/run/structural/production.py", inventory)
        with self.assertRaisesRegex(ValueError, "already exists"):
            self.run_review()

    def test_changed_approval_rejected(self):
        original = self.approval.read_bytes()
        try:
            value = copy.deepcopy(production.APPROVAL)
            value["total_runs"] = 100
            write_json(self.approval, value)
            with self.assertRaisesRegex(ValueError, "authorization differs"):
                self.run_review()
        finally:
            self.approval.write_bytes(original)
        self.assertFalse(self.output().exists())

    def test_unapproved_fingerprint_rejected(self):
        with self.assertRaisesRegex(ValueError, "approved fingerprint"):
            self.run_review("a" * 64)
        self.assertFalse(self.output().exists())

    def test_partial_final_validation_rejected(self):
        path = self.batch / "validation.json"
        original = path.read_bytes()
        try:
            value = results.read_json(path)
            value["completion_sha256"].pop(next(iter(value["completion_sha256"])))
            write_json(path, value)
            with self.assertRaisesRegex(ValueError, "1,260 fresh runs"):
                self.run_review()
        finally:
            path.write_bytes(original)
        self.assertFalse(self.output().exists())

    def test_changed_hourly_output_rejected(self):
        path = next(self.batch.glob("*/hourly_states.csv"))
        original = path.read_bytes()
        try:
            path.write_bytes(original + b"\n")
            with self.assertRaisesRegex(ValueError, "missing or changed"):
                self.run_review()
        finally:
            path.write_bytes(original)

    def test_failed_reference_gate_never_enables_variants(self):
        argv = ["production", "--out", str(self.batch), "--reference-batch", str(self.reference),
                "--approval", str(self.approval), "--review-out", str(self.output())]
        with patch.object(production.platform, "node", return_value="richardson"):
            with patch.object(sys, "argv", argv), patch.object(production, "run_phase") as phase:
                with patch.object(production, "reference_gate", side_effect=ValueError("reference mismatch")):
                    with self.assertRaisesRegex(ValueError, "reference mismatch"):
                        production.main()
                phase.assert_called_once()
                self.assertEqual(phase.call_args.args[-1], "historical_reference_gate")
        self.assertFalse((self.batch / "parent.lock").exists())

    def test_complete_cohort_without_final_validation_is_rejected(self):
        path = self.batch / "validation.json"
        original = path.read_bytes()
        try:
            path.unlink()
            with self.assertRaises(FileNotFoundError):
                self.run_review()
        finally:
            path.write_bytes(original)
        self.assertFalse(self.output().exists())


if __name__ == "__main__":
    unittest.main()
