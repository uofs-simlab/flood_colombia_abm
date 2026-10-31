"""Complete-pilot review gates using synthetic outcomes, never model runs."""

import json
from pathlib import Path
import shutil
import sys
import tempfile
import unittest
import zipfile

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "analysis"))
import structural_pilot_review as review
from test_paper_results import refresh_completion, write_fixture


class ReviewTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.temporary = tempfile.TemporaryDirectory()
        cls.root = Path(cls.temporary.name)
        cls.reference = cls.root / "historical"
        cls.reference.mkdir()
        review.write_json(cls.reference / "manifest.json", review.pilot.study_manifest(20, 20261026))
        cls.metadata = review.pilot.manifest(cls.reference)
        cls.batch = cls.root / "pilot"
        cls.batch.mkdir()
        review.write_json(cls.batch / "manifest.json", cls.metadata)
        cls.fingerprint = review.sha256(cls.batch / "manifest.json")
        write_fixture(cls.batch, cls.metadata)
        historical_hash = review.sha256(cls.reference / "manifest.json")
        for seed in review.pilot.SEEDS:
            for anchor in review.pilot.ANCHORS:
                source = cls.batch / f"reference__{anchor}__seed{seed}"
                target = cls.reference / f"{anchor}__seed{seed}"
                shutil.copytree(source, target)
                for filename in ("task.json", "summary.json"):
                    value = review.read_json(target / filename)
                    value.update({"scenario": anchor, "manifest_sha256": historical_hash})
                    review.write_json(target / filename, value)
                refresh_completion(target, historical_hash)
        frame, _, _ = review.primary.load_cohort(
            cls.batch, expected_manifest=cls.metadata, scenario_names=cls.metadata["scenarios"])
        frame.to_csv(cls.batch / "pilot_run_summary.csv", index=False)
        review.pilot.contrast_changes(frame).to_csv(cls.batch / "pilot_contrast_changes.csv", index=False)
        review.write_json(cls.batch / "reference_gate.json", {
            "manifest_sha256": cls.fingerprint, "historical_manifest_sha256": historical_hash,
            "exact_reference_matches": 14,
        })
        review.write_json(cls.batch / "validation.json", {
            "manifest_sha256": cls.fingerprint, "runs": 126, "hours_per_run": 408,
            "historical_reference_matches": 14, "variant_runs": 112,
            "independent_hourly_output_readback": True,
            "policy_effect_inference": False, "production_runs_authorized": False,
            "completion_sha256": {
                path.parent.name: review.sha256(path)
                for path in cls.batch.glob("*/completion.json")},
            "descriptive_outputs_sha256": {
                name: review.sha256(cls.batch / name)
                for name in ("pilot_run_summary.csv", "pilot_contrast_changes.csv")},
        })

    @classmethod
    def tearDownClass(cls):
        cls.temporary.cleanup()

    def output(self):
        return self.root / self.id().rsplit(".", 1)[-1]

    def run_review(self, fingerprint=None):
        return review.review(self.batch, self.reference, self.output(),
                             self.fingerprint if fingerprint is None else fingerprint)

    def test_complete_review_exports_every_file_and_does_not_authorize_production(self):
        report = self.run_review()
        self.assertEqual(report["verified_runs"], 126)
        self.assertEqual(report["exact_canonical_reference_matches"], 14)
        self.assertEqual(report["variant_contrast_descriptions"], 32)
        self.assertFalse(report["production_runs_authorized"])
        self.assertFalse(report["policy_effect_or_robustness_inference"])
        self.assertEqual(report["next_design_for_author_decision"]["candidate_total_runs"], 1260)
        overview = pd.read_csv(self.output() / "descriptive_overview.csv")
        self.assertEqual(len(overview), 32)
        self.assertNotIn("p_value", overview)
        record = review.read_json(self.output() / "export.json")
        archive = self.output() / record["archive"]
        self.assertEqual(review.sha256(archive), record["archive_sha256"])
        with zipfile.ZipFile(archive) as handle:
            inventory = json.loads(handle.read("retrieval_inventory.json"))
            self.assertEqual(len(inventory), record["archived_files"])
            self.assertIn("review/review.json", inventory)
            self.assertIn("source/model/flood_model.py", inventory)
            self.assertIn("source/analysis/structural_pilot_review.py", inventory)
        with self.assertRaisesRegex(ValueError, "already contains"):
            self.run_review()

    def test_unapproved_execution_fingerprint_is_rejected(self):
        with self.assertRaisesRegex(ValueError, "approved fingerprint"):
            self.run_review("a" * 64)
        self.assertFalse(self.output().exists())

    def test_partial_pilot_cannot_produce_a_review(self):
        path = self.batch / "validation.json"
        content = path.read_bytes()
        try:
            path.unlink()
            with self.assertRaises(FileNotFoundError):
                self.run_review()
        finally:
            path.write_bytes(content)
        self.assertFalse(self.output().exists())

    def test_final_validation_must_cover_all_126_completion_records(self):
        path = self.batch / "validation.json"
        content = path.read_bytes()
        try:
            value = review.read_json(path)
            value["completion_sha256"].pop(next(iter(value["completion_sha256"])))
            review.write_json(path, value)
            with self.assertRaisesRegex(ValueError, "every planned completion"):
                self.run_review()
        finally:
            path.write_bytes(content)

    def test_changed_raw_output_is_not_accepted(self):
        path = next(self.batch.glob("*/hourly_states.csv"))
        content = path.read_bytes()
        try:
            path.write_bytes(content + b"\n")
            with self.assertRaisesRegex(ValueError, "missing or changed"):
                self.run_review()
        finally:
            path.write_bytes(content)

    def test_modified_descriptive_output_is_not_accepted(self):
        path = self.batch / "pilot_contrast_changes.csv"
        content = path.read_bytes()
        try:
            path.write_bytes(content + b"\n")
            with self.assertRaisesRegex(ValueError, "description changed"):
                self.run_review()
        finally:
            path.write_bytes(content)

    def test_production_authorization_cannot_be_inferred_from_pilot(self):
        path = self.batch / "validation.json"
        content = path.read_bytes()
        try:
            value = review.read_json(path)
            value["production_runs_authorized"] = True
            review.write_json(path, value)
            with self.assertRaisesRegex(ValueError, "inconsistent"):
                self.run_review()
        finally:
            path.write_bytes(content)


if __name__ == "__main__":
    unittest.main()
