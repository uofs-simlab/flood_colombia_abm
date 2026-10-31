"""Focused tests for the paper re-analysis, not validation of flood realism."""

import unittest
from pathlib import Path
import sys

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "analysis"))
from analyze_archive import cliffs_delta, holm_adjust, pairwise, validate_raw


class AnalysisTests(unittest.TestCase):
    def test_holm_preserves_input_order(self):
        np.testing.assert_allclose(holm_adjust([.04, .01, .03]), [.06, .03, .06])

    def test_holm_rejects_invalid_values(self):
        for values in [[np.nan], [-.1], [1.1]]:
            with self.assertRaises(ValueError):
                holm_adjust(values)

    def test_cliffs_delta_sign_and_ties(self):
        self.assertEqual(cliffs_delta([1, 2], [3, 4]), -1)
        self.assertEqual(cliffs_delta([3, 4], [1, 2]), 1)
        self.assertEqual(cliffs_delta([1, 1], [1, 1]), 0)

    def test_cliffs_delta_rejects_empty_sample(self):
        with self.assertRaises(ValueError):
            cliffs_delta([], [1])

    def test_pairwise_family_adjustment(self):
        rows = pairwise({"a": [0, 1, 2], "b": [2, 3, 4],
                         "c": [5, 6, 7]}, "example")
        self.assertEqual(len(rows), 3)
        self.assertTrue(all(row["p_holm"] >= row["p_raw"] for row in rows))

    def raw_fixture(self):
        return pd.DataFrame({
            "scenario": ["test"] * 17, "iteration": [0] * 17,
            "Step": list(range(0, 408, 24)), "N_persons": [972] * 17,
            "pre_flood_days": [7] * 17, "flood_days": [5] * 17,
            "post_flood_days": [5] * 17, "healthcare_cap_limit": [5] * 17,
            **{column: np.zeros(17) for column in
               ["Death", "Stranded", "Injured", "Sheltered", "Hospitalized",
                "Evacuated", "Houses_Flooded"]}})

    def test_valid_daily_series(self):
        validate_raw(self.raw_fixture(), ["scenario"], 1)

    def test_duplicate_observations_are_rejected(self):
        df = self.raw_fixture()
        with self.assertRaisesRegex(ValueError, "Duplicate"):
            validate_raw(pd.concat([df, df.iloc[[0]]]), ["scenario"], 1)

    def test_missing_observations_are_rejected(self):
        with self.assertRaisesRegex(ValueError, "Incomplete"):
            validate_raw(self.raw_fixture().iloc[:-1], ["scenario"], 1)

    def test_nonmonotone_deaths_are_rejected(self):
        df = self.raw_fixture()
        df.loc[1, "Death"] = .1
        with self.assertRaisesRegex(ValueError, "Non-monotone"):
            validate_raw(df, ["scenario"], 1)

    def test_invalid_fractions_are_rejected(self):
        for value in [np.nan, -.1, 1.1]:
            df = self.raw_fixture()
            df.loc[1, "Death"] = value
            with self.assertRaisesRegex(ValueError, "out-of-range"):
                validate_raw(df, ["scenario"], 1)


if __name__ == "__main__":
    unittest.main()
