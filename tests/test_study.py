"""Prespecified study design and safe scheduling."""

import json
from pathlib import Path
import sys
import tempfile
import unittest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "run"))
from run_study import expected_tasks, study_manifest
from study_design import SCENARIOS, scenario_parameters, seeds
from verify_model import write_json


class StudyDesignTests(unittest.TestCase):
    def test_exact_setting_counts(self):
        counts = {family: sum(row["family"] == family for row in SCENARIOS.values())
                  for family in ["organization", "ablation", "distance", "fleet"]}
        self.assertEqual(counts, {"organization": 5, "ablation": 3, "distance": 12, "fleet": 4})
        self.assertEqual(len(SCENARIOS), 24)

    def test_all_settings_use_the_same_population_and_horizon(self):
        for name in SCENARIOS:
            p = scenario_parameters(name)
            self.assertEqual(p["N_persons"], 972)
            self.assertEqual(p["pre_flood_days"] + p["flood_days"] + p["post_flood_days"], 17)
            for key, value in p.items():
                if key.endswith("_file") or key.startswith("flood_file_"):
                    self.assertTrue(Path(value).is_file(), (name, key))

    def test_setting_parameters_are_distinct(self):
        encoded = [json.dumps(scenario_parameters(name), sort_keys=True) for name in SCENARIOS]
        self.assertEqual(len(set(encoded)), 24)

    def test_twenty_seeds_are_recorded_and_reused(self):
        self.assertEqual(seeds(20, 20261026), list(range(20261026, 20261046)))

    def test_invalid_seed_design_is_rejected(self):
        for n, base in [(1, 0), (True, 1), (20, -1), (20, 2**32-1)]:
            with self.assertRaises(ValueError):
                seeds(n, base)

    def test_480_unique_jobs_are_interleaved_by_seed(self):
        with tempfile.TemporaryDirectory() as temporary:
            batch = Path(temporary)
            manifest = study_manifest(20, 20261026)
            write_json(batch / "manifest.json", manifest)
            tasks = expected_tasks(batch, manifest)
            self.assertEqual(len(tasks), 480)
            self.assertEqual(len({t["path"] for t in tasks}), 480)
            self.assertEqual({t["seed"] for t in tasks[:24]}, {20261026})
            self.assertEqual({t["scenario"] for t in tasks[:24]}, set(SCENARIOS))
            self.assertEqual({t["seed"] for t in tasks[-24:]}, {20261045})


if __name__ == "__main__":
    unittest.main()
