"""Opt-in rule boundaries and the bounded HPC-only TODO pilot."""

import copy
import inspect
import json
import math
from pathlib import Path
import random
import sys
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "run"))
from agents import decision_making_module as decisions
from model.flood_model import FloodModel
from run_experimentA_main_sweep import FIXED
from structural import pilot
from test_model import person, support_model
from verify_model import OUTPUTS, sha256, write_json


class RuleTests(unittest.TestCase):
    def test_inherited_evacuation_keeps_exact_gaussian_call(self):
        agent = SimpleNamespace(model=SimpleNamespace(evacuation_success_rule="inherited"))
        for score in (-10, -.2, 0, .2, 10):
            with self.subTest(score=score), patch.object(
                    decisions.random, "gauss", return_value=score) as draw:
                self.assertFalse(decisions.evacuation_succeeds(agent, score))
                draw.assert_called_once_with(score, score)

    def test_probit_threshold_is_independent_of_score(self):
        agent = SimpleNamespace(model=SimpleNamespace(evacuation_success_rule="probit"))
        for score, threshold, expected in [(-1, 0, False), (0, 0, False), (1, 0, True),
                                            (0, -.1, True), (.2, .3, False)]:
            with self.subTest(score=score, threshold=threshold), patch.object(
                    decisions.random, "gauss", return_value=threshold) as draw:
                self.assertEqual(decisions.evacuation_succeeds(agent, score), expected)
                draw.assert_called_once_with(0, 1)

    def test_all_four_evacuation_actions_use_the_selected_rule(self):
        agent = SimpleNamespace(
            model=SimpleNamespace(evacuation_success_rule="probit"),
            severity=.5, costs=.2, self_efficacy=.6, social_trust=.8,
            social_capital_score=3, worldview="fatalist",
        )
        for function in (decisions.PMT_decide_on_evacuation, decisions.TPB_decide_on_evacuation,
                         decisions.SCT_decide_on_evacuation, decisions.CRT_decide_on_evacuation):
            with self.subTest(rule=function.__name__), patch.object(
                    decisions, "evacuation_succeeds", return_value=False) as test:
                function(agent)
                test.assert_called_once()
                self.assertFalse(agent.evacuated)

    def test_invalid_rule_does_not_silently_select_a_fallback(self):
        agent = SimpleNamespace(model=SimpleNamespace(evacuation_success_rule="unknown"))
        with self.assertRaisesRegex(ValueError, "Unknown"):
            decisions.evacuation_succeeds(agent, 1)

    def test_recovery_cutoff_is_fixed_not_redrawn_hourly(self):
        model = support_model()
        model.recovery_eligibility = .75
        for propensity, expected in ((.75, True), (.750001, False), (.9, False)):
            patient = person()
            patient.injured = True
            patient.recovery_rate = propensity
            for _ in range(3):
                with patch.object(random, "uniform", side_effect=AssertionError("Unexpected draw")):
                    self.assertEqual(model.recover_injury(patient), expected)
                patient.injured = True
        model.recovery_eligibility = 1
        patient.recovery_rate = 1
        self.assertTrue(model.recover_injury(patient))

    def test_no_offsite_care_retains_clock_but_does_not_disable_local_care(self):
        model = support_model()
        model.offsite_care = False
        patient = person()
        patient.injured = True
        patient.evacuated = True
        patient.time_injured = 99
        model._advance_injury(patient)
        self.assertFalse(patient.alive)
        self.assertEqual(patient.time_injured, 100)
        patient = person()
        patient.injured = True
        self.assertTrue(model.recover_injury(patient))

    def test_invalid_parameters_fail_before_spatial_initialization(self):
        changes = [
            {"evacuation_success_rule": "other"}, {"injury_onset_scale": 0},
            {"injury_onset_scale": True}, {"injury_onset_scale": float("inf")},
            {"survival_deadline_scale": -.5}, {"survival_deadline_scale": "1"},
            {"recovery_eligibility": -1}, {"recovery_eligibility": 1.1},
            {"recovery_eligibility": float("nan")}, {"recovery_eligibility": True},
            {"offsite_care": 1},
        ]
        for change in changes:
            with self.subTest(change=change), patch(
                    "model.flood_model.StudyArea", side_effect=AssertionError("Unexpected initialization")):
                with self.assertRaises(ValueError):
                    FloodModel(**FIXED, **change)


class ThresholdInitializationTests(unittest.TestCase):
    def initialize(self, **kwargs):
        model = FloodModel(**FIXED, seed=20261026, **kwargs)
        people = {p.population_index: p for h in model.space.houses for p in h.residents}
        return model, people, random.getstate()

    def test_explicit_baseline_matches_defaults_and_preserves_rng_state(self):
        first, a, rng = self.initialize()
        second, b, repeated_rng = self.initialize(
            evacuation_success_rule="inherited", injury_onset_scale=1,
            survival_deadline_scale=1, recovery_eligibility=.9, offsite_care=True)
        self.assertEqual(rng, repeated_rng)
        for index in a:
            self.assertEqual(
                (a[index].injury_duration, a[index].survivability_duration, a[index].recovery_rate),
                (b[index].injury_duration, b[index].survivability_duration, b[index].recovery_rate))
        pd.testing.assert_frame_equal(
            first.datacollector.get_model_vars_dataframe(),
            second.datacollector.get_model_vars_dataframe(), check_exact=True)

    def test_scales_are_separate_and_consume_no_extra_random_draws(self):
        _, baseline, rng = self.initialize()
        for parameter, scale in (("injury_onset_scale", .75), ("injury_onset_scale", 1.25),
                                 ("survival_deadline_scale", .75), ("survival_deadline_scale", 1.25)):
            with self.subTest(parameter=parameter, scale=scale):
                _, changed, changed_rng = self.initialize(**{parameter: scale})
                self.assertEqual(rng, changed_rng)
                for index, person_before in baseline.items():
                    after = changed[index]
                    self.assertEqual(after.injury_duration, math.ceil(
                        person_before.injury_duration * (scale if parameter == "injury_onset_scale" else 1)))
                    self.assertEqual(after.survivability_duration, math.ceil(
                        person_before.survivability_duration
                        * (scale if parameter == "survival_deadline_scale" else 1)))
                    self.assertEqual(after.recovery_rate, person_before.recovery_rate)
                    self.assertLess(after.injury_duration, after.survivability_duration)


class PilotContractTests(unittest.TestCase):
    def test_complete_pilot_has_independent_hourly_readback(self):
        from test_paper_results import analysis, write_fixture
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            reference = root / "reference"
            reference.mkdir()
            write_json(reference / "manifest.json", pilot.study_manifest(20, pilot.SEEDS[0]))
            metadata = pilot.manifest(reference)
            batch = root / "pilot"
            batch.mkdir()
            write_json(batch / "manifest.json", metadata)
            write_fixture(batch, metadata)
            frame, _, _ = analysis.load_cohort(
                batch, expected_manifest=metadata, scenario_names=metadata["scenarios"])
            self.assertEqual(len(frame), 126)
            self.assertEqual(len(pilot.contrast_changes(frame)), 64)

    def test_pilot_descriptions_are_seed_paired_without_inferential_claims(self):
        frame = pd.DataFrame([
            {"scenario": f"{variant}__{anchor}", "seed": seed,
             "Deaths": index + (index * 2 if variant == "evacuation_probit" else 0)}
            for seed in pilot.SEEDS for variant in pilot.VARIANTS
            for index, anchor in enumerate(pilot.ANCHORS)
        ])
        result = pilot.contrast_changes(frame)
        self.assertEqual(len(result), 64)
        row = result.loc[(result.contrast == "organization")
                         & (result.variant == "evacuation_probit")].iloc[0]
        self.assertEqual(row.reference_difference_deaths, 1)
        self.assertEqual(row.variant_difference_deaths, 3)
        self.assertEqual(row.change_in_difference_deaths, 2)
        self.assertNotIn("p_value", result)
        self.assertNotIn("mc95_low_pp", result)
        with self.assertRaisesRegex(ValueError, "Incomplete"):
            pilot.contrast_changes(frame.iloc[1:])
        frame.loc[0, "Deaths"] = 1000
        with self.assertRaisesRegex(ValueError, "integer"):
            pilot.contrast_changes(frame)

    def test_exact_pilot_budget_and_one_factor_at_a_time(self):
        cases = pilot.scenarios()
        self.assertEqual(len(cases), 63)
        self.assertEqual(len(cases) * len(pilot.SEEDS), 126)
        for variant, changes in pilot.VARIANTS.items():
            self.assertLessEqual(len(changes), 1)
            for anchor in pilot.ANCHORS:
                parameters = cases[f"{variant}__{anchor}"]
                self.assertEqual(parameters, {**pilot.scenario_parameters(anchor), **changes})
                inspect.signature(FloodModel).bind(**parameters, seed=100)

    def test_host_gate_blocks_local_full_simulations(self):
        with patch.object(pilot.platform, "node", return_value="workstation"):
            with self.assertRaisesRegex(RuntimeError, "not locally"):
                pilot.require_hpc()
            with patch.object(sys, "argv", ["pilot"]):
                with self.assertRaisesRegex(RuntimeError, "not locally"):
                    pilot.main()
        for name in ("richardson", "hundsdorfer", "richardson.example"):
            with patch.object(pilot.platform, "node", return_value=name):
                pilot.require_hpc()

    def test_manifest_freezes_reference_and_implementation(self):
        with tempfile.TemporaryDirectory() as folder:
            reference = Path(folder)
            original = pilot.study_manifest(20, pilot.SEEDS[0])
            write_json(reference / "manifest.json", original)
            metadata = pilot.manifest(reference)
            self.assertEqual(metadata["reference_manifest_sha256"], sha256(reference / "manifest.json"))
            self.assertEqual(metadata["seeds"], list(pilot.SEEDS))
            self.assertEqual(len(metadata["scenarios"]), 63)
            self.assertIn(str(Path("run") / "structural" / "pilot.py"), metadata["source_sha256"])
            changed = copy.deepcopy(original)
            changed["scenarios"][pilot.ANCHORS[0]]["N_persons"] = 1
            write_json(reference / "manifest.json", changed)
            with self.assertRaisesRegex(ValueError, "do not match"):
                pilot.manifest(reference)

    def test_reference_comparison_ignores_only_execution_labels_and_elapsed_time(self):
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            reference, current = root / "reference", root / "current"
            for path in (reference, current):
                path.mkdir()
                for name in OUTPUTS[:3]:
                    pd.DataFrame({"value": [1, 2]}).to_csv(path / name, index=False)
                write_json(path / "summary.json", {
                    "scenario": path.name, "manifest_sha256": path.name,
                    "elapsed_seconds": len(path.name), "Death_hour408": .1,
                })
            write_json(reference / "completion.json", {
                "manifest_sha256": "old",
                "outputs": {name: sha256(reference / name) for name in OUTPUTS},
            })
            pilot.compare_reference(current, reference, "old")
            summary = json.loads((current / "summary.json").read_text())
            summary["Death_hour408"] = .2
            write_json(current / "summary.json", summary)
            with self.assertRaisesRegex(ValueError, "summary differs"):
                pilot.compare_reference(current, reference, "old")
            with self.assertRaisesRegex(ValueError, "contract"):
                pilot.compare_reference(current, reference, "wrong")


if __name__ == "__main__":
    unittest.main()
