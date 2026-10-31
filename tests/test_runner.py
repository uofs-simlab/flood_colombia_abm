"""Cohort integrity, full-horizon measurement and deterministic run keys."""

import json
from pathlib import Path
import sys
import tempfile
from types import SimpleNamespace
import unittest

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "run"))
from robust_runner import _expand_param_grid, _run_key, _run_one, run_checkpointed
from verify_model import complete, OUTPUTS, sha256, write_json


class RunnerTests(unittest.TestCase):
    def test_grid_expansion_preserves_cartesian_product(self):
        self.assertEqual(_expand_param_grid({"a": [1, 2], "b": [3]}),
                         [{"a": 1, "b": 3}, {"a": 2, "b": 3}])

    def test_key_ignores_dictionary_order_but_not_replication(self):
        a = _run_key({"a": 1, "b": 2}, 0, ["a", "b"])
        self.assertEqual(a, _run_key({"b": 2, "a": 1}, 0, ["b", "a"]))
        self.assertNotEqual(a, _run_key({"a": 1, "b": 2}, 1, ["a", "b"]))

    def test_full_endpoint_is_kept_even_with_subsampling(self):
        class Collector:
            def __init__(self):
                self.rows = []

            def collect(self, model):
                self.rows.append({"Death": model.hour / 100})

            def get_model_vars_dataframe(self):
                return pd.DataFrame(self.rows)

        class Model:
            def __init__(self, seed):
                self.hour = 0
                self.datacollector = Collector()

            def step(self):
                self.datacollector.collect(self)
                self.hour += 1

        frame = _run_one(Model, {"seed": 1}, 0, 5, 2)
        self.assertEqual(frame.Step.tolist(), [0, 2, 4, 5])
        self.assertAlmostEqual(frame.Death.iloc[-1], .05)

    def test_old_output_is_rejected_without_overwriting(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            output = root / "results.csv"
            checkpoint = root / "checkpoint.json"
            output.write_text("original data\n", encoding="utf-8")
            checkpoint.write_text("[]", encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "no matching seeded source manifest"):
                run_checkpointed(None, {"a": [1]}, ["a"], 1, 1, str(output), str(checkpoint))
            self.assertEqual(output.read_text(encoding="utf-8"), "original data\n")
            self.assertEqual(checkpoint.read_text(encoding="utf-8"), "[]")
            self.assertFalse(Path(str(checkpoint) + ".manifest.json").exists())

    def test_explicit_seed_grid_rejected_before_writing(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            with self.assertRaisesRegex(ValueError, "seed_base"):
                run_checkpointed(None, {"seed": [1]}, [], 1, 1,
                                 str(root / "out.csv"), str(root / "check.json"))
            self.assertEqual(list(root.iterdir()), [])

    def test_incomplete_output_is_not_completed(self):
        with tempfile.TemporaryDirectory() as temporary:
            self.assertFalse(complete(Path(temporary), "fingerprint"))

    def test_output_hashes_and_required_files_are_checked(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            for name in OUTPUTS:
                (root / name).write_text("data", encoding="utf-8")
            write_json(root / "completion.json", {
                "manifest_sha256": "fingerprint",
                "outputs": {name: sha256(root / name) for name in OUTPUTS},
            })
            self.assertTrue(complete(root, "fingerprint"))
            (root / OUTPUTS[0]).write_text("changed", encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "missing or changed"):
                complete(root, "fingerprint")

    def test_incomplete_completion_contract_is_rejected(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            write_json(root / "completion.json", {
                "manifest_sha256": "fingerprint", "outputs": {},
            })
            with self.assertRaisesRegex(ValueError, "run contract"):
                complete(root, "fingerprint")

    def test_nonfinite_json_does_not_replace_valid_output(self):
        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary) / "manifest.json"
            write_json(path, {"value": 1})
            with self.assertRaises(ValueError):
                write_json(path, {"value": float("nan")})
            self.assertEqual(json.loads(path.read_text(encoding="utf-8")), {"value": 1})


if __name__ == "__main__":
    unittest.main()
