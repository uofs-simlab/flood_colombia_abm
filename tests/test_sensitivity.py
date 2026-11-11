"""Design and portability checks for the targeted robustness study."""

import copy
import contextlib
import inspect
import io
from pathlib import Path
import shutil
import sys
import tempfile
import unittest
from unittest.mock import patch
import zipfile

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "run"))
from hpc import transfer
from sensitivity import design
from model.flood_model import FloodModel
sys.path.insert(0, str(ROOT / "analysis"))
import sensitivity_results as analysis
from test_paper_results import assert_external_legend, refresh_completion, write_fixture


class SensitivityDesignTests(unittest.TestCase):
    def test_exact_budget_and_no_inactive_combinations(self):
        cases = design.variants()
        self.assertEqual(len(cases), 48)
        self.assertEqual(len(design.ANCHORS), 7)
        for case in cases.values():
            info = design.FACTORS[case["factor"]]
            self.assertIn(case["anchor"], info["anchors"])
            self.assertIn(case["value"], info["levels"])
            original = design.scenario_parameters(case["anchor"])
            changes = {key for key, value in case["parameters"].items()
                       if key not in original or original[key] != value}
            self.assertEqual(changes, {info["parameter"]})
            inspect.signature(FloodModel).bind(**case["parameters"], seed=100)

    def test_thirty_applicable_contrast_modifications(self):
        count = sum(
            2 for info in design.FACTORS.values()
            for _, _, first, second in design.CONTRASTS
            if first in info["anchors"] or second in info["anchors"]
        )
        self.assertEqual(count, 30)

    def test_reference_jobs_are_not_dispatched_as_primary_sensitivity_runs(self):
        metadata = design.sensitivity_manifest(20, 20261026, "a"*64)
        with tempfile.TemporaryDirectory() as folder:
            batch = Path(folder)
            transfer.write_json(batch / "manifest.json", metadata)
            tasks = transfer.cohort_tasks(batch, metadata)
            all_tasks = transfer.cohort_tasks(batch, metadata, True)
        self.assertEqual(len(tasks), 960)
        self.assertEqual(len(all_tasks), 1040)
        self.assertFalse(set(design.GATES) & {task["scenario"] for task in tasks})

    def test_primary_source_contract_remains_unchanged(self):
        primary = transfer.study_manifest(2, 100)
        metadata = design.sensitivity_manifest(2, 100, "a"*64)
        self.assertNotIn(str(Path("run") / "sensitivity" / "design.py"), primary["source_sha256"])
        self.assertIn(str(Path("run") / "sensitivity" / "design.py"), metadata["source_sha256"])
        self.assertIn(str(Path("run") / "hpc" / "transfer.py"), metadata["source_sha256"])
        for name in design.GATES:
            self.assertEqual(primary["scenarios"][name], metadata["scenarios"][name])

    def test_secondary_contract_is_portable_and_protects_design(self):
        metadata = design.sensitivity_manifest(2, 100, "a"*64)
        remote = copy.deepcopy(metadata)
        remote["python"] = "3.12.3 (Linux)"
        remote["executable"] = "/scratch/example/.venv/bin/python"
        for field in ["source_sha256", "spatial_sha256"]:
            remote[field] = {key.replace("\\", "/"): value for key, value in remote[field].items()}
        for parameters in remote["scenarios"].values():
            for key, value in parameters.items():
                if key.endswith("_file") or key.startswith("flood_file_"):
                    portable = value.replace("\\", "/")
                    parameters[key] = "/scratch/example/" + portable[portable.index("santa_lucia_map_data/"):]
        transfer.require_compatible(metadata, remote)
        remote["baseline_manifest_sha256"] = "b"*64
        with self.assertRaises(ValueError):
            transfer.require_compatible(metadata, remote)

    def test_reference_execution_fingerprint_can_differ_from_secondary_manifest(self):
        plan = {"origin_manifest_sha256": "new", "reference_manifest_sha256": {"gate": "primary"}}
        self.assertEqual(transfer.reference_fingerprint(plan, "gate"), "primary")
        self.assertEqual(transfer.reference_fingerprint({"origin_manifest_sha256": "old"}, "gate"), "old")

    def test_invalid_fingerprint_and_seed_budget_are_rejected(self):
        for fingerprint in ["", "x"*64, "a"*63]:
            with self.assertRaises(ValueError):
                design.sensitivity_manifest(20, 20261026, fingerprint)
        with self.assertRaises(ValueError):
            design.sensitivity_manifest(1, 20261026, "a"*64)

    def test_reference_settings_cannot_enter_sensitivity_dispatch_plan(self):
        metadata = design.sensitivity_manifest(2, 100, "a"*64)
        plan = {"origin_manifest_sha256": "fingerprint",
                "tasks": [{"scenario": "A_org_0", "seed": 100, "manifest_sha256": "fingerprint"}],
                "references": [f"{name}__seed100" for name in design.GATES]}
        with self.assertRaisesRegex(ValueError, "Invalid or duplicated"):
            transfer.validate_plan(plan, metadata)


def statistical_frames():
    seeds = [10, 11, 12, 13]
    deaths = dict(zip(design.ANCHORS, [60, 2, 60, 30, 40, 50, 45]))
    reference = pd.DataFrame([
        {"scenario": name, "seed": seed, "Deaths": count+i}
        for i, seed in enumerate(seeds) for name, count in deaths.items()
    ])
    sensitivity = pd.DataFrame([
        {"scenario": name, "seed": seed,
         "Deaths": deaths[case["anchor"]]+i
         + (2 if case["anchor"] == "A_org_1" else 0)
         + (7 if case["factor"] == "hospital"
            and case["anchor"] in {"D_cap30_nearby_crossing", "D_cap30_barranquilla"} else 0)}
        for i, seed in enumerate(seeds) for name, case in design.variants().items()
    ])
    return reference, sensitivity, seeds


class SensitivityStatisticsTests(unittest.TestCase):
    def test_seed_paired_difference_of_differences_not_independent_intervals(self):
        reference, sensitivity, seeds = statistical_frames()
        result = analysis.effects(reference, sensitivity, seeds)
        self.assertEqual(len(result), 34)
        changed = result.loc[result.contrast.eq("organization") & result.factor.ne("baseline")]
        self.assertEqual(len(changed), 10)
        for row in changed.itertuples():
            self.assertAlmostEqual(row.change_from_reference_pp, 2*100/972)
            self.assertAlmostEqual(row.change_mc95_low_pp, 2*100/972)
            self.assertAlmostEqual(row.change_mc95_high_pp, 2*100/972)
        self.assertEqual(result.p_change_holm_30.notna().sum(), 30)

    def test_equal_shifts_in_both_policy_arms_cancel(self):
        reference, sensitivity, seeds = statistical_frames()
        result = analysis.effects(reference, sensitivity, seeds)
        rows = result.loc[result.contrast.eq("site") & result.factor.eq("hospital")]
        self.assertEqual(len(rows), 2)
        self.assertTrue(rows.change_from_reference_pp.eq(0).all())
        self.assertTrue(rows.change_mc95_low_pp.eq(0).all())
        self.assertTrue(rows.p_change_raw.eq(1).all())
        baseline = result.loc[result.contrast.eq("site") & result.factor.eq("baseline")].iloc[0]
        for row in rows.itertuples():
            self.assertAlmostEqual(row.first_mean_mortality_pct-baseline.first_mean_mortality_pct,
                                   7*100/972)
            self.assertAlmostEqual(row.second_mean_mortality_pct-baseline.second_mean_mortality_pct,
                                   7*100/972)

    def test_pairing_is_independent_of_row_order(self):
        reference, sensitivity, seeds = statistical_frames()
        expected = analysis.effects(reference, sensitivity, seeds)
        actual = analysis.effects(reference.sample(frac=1, random_state=1),
                                  sensitivity.sample(frac=1, random_state=2), seeds)
        pd.testing.assert_frame_equal(actual, expected)

    def test_missing_duplicate_and_noninteger_outcomes_are_rejected(self):
        reference, sensitivity, seeds = statistical_frames()
        for frame in [sensitivity.iloc[:-1], pd.concat([sensitivity, sensitivity.iloc[:1]])]:
            with self.assertRaisesRegex(ValueError, "seed pairing"):
                analysis.effects(reference, frame, seeds)
        fractional = sensitivity.astype({"Deaths": float})
        fractional.loc[0, "Deaths"] = 1.5
        with self.assertRaisesRegex(ValueError, "integer"):
            analysis.effects(reference, fractional, seeds)

    def test_clinical_sensitivity_reports_capacity_outcome_and_assumptions(self):
        reference, sensitivity, seeds = statistical_frames()
        name = design.variant_name("hospital", 1, "A_org_0")
        sensitivity.loc[sensitivity.scenario.eq(name), "Deaths"] -= 10
        result = analysis.effects(reference, sensitivity, seeds)
        metadata = {"execution_scenarios": list(design.variants()), "replications": len(seeds)}
        with tempfile.TemporaryDirectory() as folder:
            output = Path(folder)
            with analysis.plt.rc_context(analysis.primary.FIGURE_STYLE):
                analysis.generate_assets(result, metadata, output)
            narrative = (output / "sensitivity_results_text.tex").read_text()
            discussion = (output / "sensitivity_discussion_text.tex").read_text()
        self.assertIn("reducing nominal hospital capacity", narrative)
        self.assertIn("from 5\\% to 1\\%", narrative)
        self.assertIn("common injury clock", discussion)
        self.assertIn("off-site care", discussion)

    def test_detected_fleet_modification_retains_both_per_trip_comparisons(self):
        reference, sensitivity, seeds = statistical_frames()
        result = analysis.effects(reference, sensitivity, seeds)
        detected = result.contrast.eq("fleet") & result.factor.eq("boat_capacity") & result.value.eq(3)
        result.loc[detected, "p_change_holm_30"] = .01
        unresolved = result.contrast.eq("fleet") & result.factor.eq("boat_capacity") & result.value.eq(9)
        result.loc[unresolved, ["mc95_low_pp", "mc95_high_pp"]] = [-.75, .17]
        metadata = {"execution_scenarios": list(design.variants()), "replications": len(seeds)}
        legend_labels = []
        savefig = analysis.plt.Figure.savefig

        def checked_save(figure, filename, *args, **kwargs):
            legend_labels.extend(assert_external_legend(self, figure))
            return savefig(figure, filename, *args, **kwargs)

        with tempfile.TemporaryDirectory() as folder:
            output = Path(folder)
            with analysis.plt.rc_context(analysis.primary.FIGURE_STYLE), patch.object(
                    analysis.plt.Figure, "savefig", checked_save):
                analysis.generate_assets(result, metadata, output)
            discussion = (output / "sensitivity_discussion_text.tex").read_text()
            results = (output / "sensitivity_results_text.tex").read_text()
            abstract = (output / "sensitivity_abstract.tex").read_text()
            methods = (output / "sensitivity_methods_text.tex").read_text()
            result.loc[detected, "p_change_holm_30"] = 0
            with analysis.plt.rc_context(analysis.primary.FIGURE_STYLE):
                analysis.generate_assets(result, metadata, output)
            small_probability = (output / "sensitivity_discussion_text.tex").read_text()
        self.assertEqual(set(legend_labels), {
            "Primary reference", "95% Monte Carlo interval", "Primary mean",
            "No difference (zero)", *[info["label"] for info in design.FACTORS.values()]})
        self.assertIn("At three people per boat trip", discussion)
        self.assertIn("At nine people per trip", discussion)
        self.assertIn("with an interval spanning zero", discussion)
        self.assertIn("0.0100", discussion)
        self.assertIn(r"$p_{\mathrm{Holm}}=0.0100$", discussion)
        self.assertIn(r"$p_{\mathrm{Holm}}$", results)
        self.assertNotIn("$p=$", discussion)
        self.assertNotIn("$p$", results)
        self.assertIn(r"$p_{\mathrm{Holm}}<0.0001$", small_probability)
        self.assertIn("Student's t distribution", methods)
        self.assertNotIn("$t$", methods)
        self.assertIn("mortality levels in each scenario", results)
        self.assertIn("at the near site", results)
        self.assertIn("Full organization continues to lower modeled mortality", abstract)
        self.assertIn("relative to no organization", abstract)
        self.assertIn("nearer shelter also continues to have lower mortality", abstract)
        self.assertIn("30\\% shelter capacity", abstract)
        self.assertIn("with an interval spanning zero", discussion)
        self.assertNotIn("retain their directions", abstract)
        self.assertNotIn("fleet conclusion is conditional", abstract)

    def test_abstract_does_not_claim_consistency_when_comparisons_are_unresolved(self):
        reference, sensitivity, seeds = statistical_frames()
        result = analysis.effects(reference, sensitivity, seeds)
        unresolved = result.factor.eq("hospital") & result.value.eq(1)
        result.loc[unresolved & result.contrast.eq("organization"), "mc95_high_pp"] = .1
        result.loc[unresolved & result.contrast.eq("site"), "mc95_low_pp"] = -.1
        metadata = {"execution_scenarios": list(design.variants()), "replications": len(seeds)}
        with tempfile.TemporaryDirectory() as folder:
            output = Path(folder)
            with analysis.plt.rc_context(analysis.primary.FIGURE_STYLE):
                analysis.generate_assets(result, metadata, output)
            abstract = (output / "sensitivity_abstract.tex").read_text()
        self.assertIn("between full and no organization is not clearly resolved", abstract)
        self.assertIn("between the near and far shelters is not clearly resolved", abstract)
        self.assertNotIn("continues to lower", abstract)
        self.assertNotIn("continues to have lower", abstract)


class SensitivityPipelineTests(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.addCleanup(self.temporary.cleanup)
        self.root = Path(self.temporary.name)
        self.baseline = self.root / "study"
        self.baseline.mkdir()
        primary_metadata = transfer.study_manifest(2, 100)
        transfer.write_json(self.baseline / "manifest.json", primary_metadata)
        write_fixture(self.baseline, primary_metadata)
        self.batch = self.root / "sensitivity"
        self.batch.mkdir()
        self.metadata = design.sensitivity_manifest(2, 100, transfer.sha256(self.baseline / "manifest.json"))
        transfer.write_json(self.batch / "manifest.json", self.metadata)

    def test_incomplete_sensitivity_cannot_write_publication_results(self):
        output = self.root / "assets"
        with patch.object(sys, "argv", ["sensitivity_results", "--batch", str(self.batch),
                                       "--primary", str(self.baseline), "--out", str(output)]):
            with self.assertRaisesRegex(ValueError, "Incomplete or unexpected cohort"):
                analysis.main()
        self.assertFalse(output.exists())

    def test_remote_import_reuses_primary_references_without_relabeling_them(self):
        write_fixture(self.batch, self.metadata)
        name = f"{self.metadata['execution_scenarios'][0]}__seed100"
        original_hash = transfer.sha256(self.batch / "manifest.json")
        primary_hash = transfer.sha256(self.baseline / "manifest.json")
        returned = self.root / "returned"
        evidence = returned / "evidence"
        evidence.mkdir(parents=True)
        plan = {
            "origin_manifest_sha256": original_hash,
            "tasks": [{"scenario": self.metadata["execution_scenarios"][0],
                       "seed": 100, "manifest_sha256": original_hash}],
            "references": [f"{scenario}__seed100" for scenario in design.GATES],
            "reference_manifest_sha256": {
                f"{scenario}__seed100": primary_hash for scenario in design.GATES},
        }
        transfer.write_json(evidence / "plan.json", plan)
        remote = copy.deepcopy(self.metadata)
        remote["python"] = "3.12.3 (Linux)"
        remote["executable"] = "/scratch/example/.venv/bin/python"
        remote["execution"] = {
            "origin_manifest_sha256": original_hash,
            "plan_sha256": transfer.sha256(evidence / "plan.json"),
            "adapter_sha256": transfer.sha256(Path(transfer.__file__)),
        }
        transfer.write_json(evidence / "manifest.json", remote)
        remote_hash = transfer.sha256(evidence / "manifest.json")
        hashes = {}
        for source, destination in [
            *[(self.baseline / gate, evidence / "gates" / gate) for gate in plan["references"]],
            (self.batch / name, returned / "runs" / name),
        ]:
            shutil.copytree(source, destination)
            for filename in ["task.json", "summary.json"]:
                record = transfer.read_json(destination / filename)
                record["manifest_sha256"] = remote_hash
                transfer.write_json(destination / filename, record)
            refresh_completion(destination, remote_hash)
            if source.parent == self.baseline:
                hashes[source.name] = {
                    "local_completion": transfer.sha256(source / "completion.json"),
                    "remote_completion": transfer.sha256(destination / "completion.json"),
                }
        transfer.write_json(evidence / "bridge.json", {
            "origin_manifest_sha256": original_hash,
            "execution_manifest_sha256": remote_hash, "reference_hashes": hashes,
        })
        (self.batch / name).rename(self.root / "held_fixture")
        archive = self.root / "results.zip"
        with zipfile.ZipFile(archive, "w", compression=zipfile.ZIP_DEFLATED) as handle:
            for path in returned.rglob("*"):
                if path.is_file():
                    handle.write(path, path.relative_to(returned).as_posix())
        before = {str(path.relative_to(self.baseline)): transfer.sha256(path)
                  for path in self.baseline.rglob("*") if path.is_file()}
        with patch.object(transfer, "reference_directory", return_value=self.baseline):
            transfer.import_results(archive, self.batch)
            frame, _, _ = analysis.primary.load_cohort(
                self.batch, expected_manifest=self.metadata,
                scenario_names=self.metadata["execution_scenarios"])
        self.assertEqual(len(frame), 96)
        self.assertEqual(transfer.read_json(self.batch / name / "summary.json")["manifest_sha256"],
                         remote_hash)
        self.assertEqual(before, {str(path.relative_to(self.baseline)): transfer.sha256(path)
                                  for path in self.baseline.rglob("*") if path.is_file()})

    @unittest.skipUnless(
        shutil.which("pdflatex") and (analysis.primary.PAPER / "flood_columbia.tex").is_file(),
        "The LaTeX toolchain and manuscript working copy are required")
    def test_complete_pipeline_preserves_primary_assets_and_builds_sensitivity_article(self):
        write_fixture(self.batch, self.metadata)
        paper = self.root / "paper"
        paper.mkdir()
        shutil.copyfile(analysis.primary.PAPER / "flood_columbia.tex", paper / "flood_columbia.tex")
        output = self.root / "assets"
        shutil.copytree(analysis.primary.PUBLICATION_OUTPUT, output)
        with patch.object(analysis.primary, "PAPER", paper), patch.object(
                analysis.primary, "PUBLICATION_OUTPUT", output), contextlib.redirect_stdout(io.StringIO()):
            with patch.object(sys, "argv", ["paper_results", "--batch", str(self.baseline)]):
                analysis.primary.main()
            before = {name: transfer.sha256(output / name) for name in analysis.primary.PUBLICATION_ASSETS}
            with patch.object(sys, "argv", ["sensitivity_results", "--batch", str(self.batch),
                                           "--primary", str(self.baseline), "--compile"]):
                analysis.main()
        self.assertEqual(before, {name: transfer.sha256(output / name) for name in before})
        validation = transfer.read_json(output / "sensitivity_validation.json")
        self.assertEqual(validation["new_runs"], 96)
        self.assertEqual(len(validation["baseline_completion_hashes"]), 14)
        self.assertEqual(set(validation["generated_sha256"]), set(analysis.ASSETS))
        for name, digest in validation["generated_sha256"].items():
            self.assertEqual(transfer.sha256(output / name), digest)
        self.assertTrue((output / "sensitivity_ready.tex").is_file())
        self.assertGreater((paper / "flood_columbia.pdf").stat().st_size, 120000)
        self.assertFalse((paper / "analysis").exists())
        text = (paper / "flood_columbia.tex").read_text(encoding="utf-8")
        self.assertNotIn(r"\input{analysis/", text)
        self.assertNotIn("GENERATED:", text)
        self.assertIn(r"\newcommand{\SensitivityRuns}{96}", text)


if __name__ == "__main__":
    unittest.main()
