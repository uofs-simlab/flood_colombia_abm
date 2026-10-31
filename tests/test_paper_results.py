"""Statistical and publication-pipeline checks using temporary synthetic fixtures."""

import contextlib
import io
import itertools
import json
import math
from pathlib import Path
import shutil
import sys
import tempfile
import unittest
from unittest.mock import patch

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "analysis"))
import paper_results as analysis


def assert_external_legend(test, figure):
    figure.canvas.draw()
    renderer = figure.canvas.get_renderer()
    test.assertEqual(len(figure.legends), 1)
    legend = figure.legends[0]
    bounds = legend.get_window_extent(renderer)
    test.assertGreaterEqual(bounds.x0, figure.bbox.x0)
    test.assertGreaterEqual(bounds.y0, figure.bbox.y0)
    test.assertLessEqual(bounds.x1, figure.bbox.x1)
    test.assertLessEqual(bounds.y1, figure.bbox.y1)
    for axis in figure.axes:
        test.assertIsNone(axis.get_legend())
        test.assertFalse(bounds.overlaps(axis.get_tightbbox(renderer)),
                         "The legend overlaps a plot, axis label or panel title.")
    return [text.get_text() for text in legend.get_texts()]


def statistical_fixture():
    return pd.DataFrame([
        {"scenario": name, "seed": seed,
         "Deaths": 2 * index + offset + (index % 3) * (offset % 2)}
        for offset, seed in enumerate([10, 11, 12, 13])
        for index, name in enumerate(analysis.SCENARIOS)
    ])


def write_fixture(batch, metadata):
    """Fabricated outcomes for validation, never real study/publication data."""
    fingerprint = analysis.sha256(batch / "manifest.json")
    for seed_index, seed in enumerate(metadata["seeds"]):
        for index, name in enumerate(metadata.get("execution_scenarios", analysis.SCENARIOS)):
            directory = batch / f"{name}__seed{seed}"
            directory.mkdir()
            parameters = metadata["scenarios"][name]
            distance = parameters.get("shelter_route_distance_km", 0)
            roundtrip = {0: .2, 2: .8, 38.4: 1.83, 58: 149/60, 99.9: 3.88}[distance]
            rate = (parameters.get("rescue_boats_available", 2)
                    * parameters.get("boat_capacity_per_trip", 6) / roundtrip)
            rescued = int((index + seed_index) % 3 != 0)
            cumulative = np.zeros(409, dtype=int)
            cumulative[201:] = rescued
            sheltered = np.zeros(409)
            sheltered[201:269] = rescued / 972
            mortality = np.zeros(409)
            mortality[400:] = (index + seed_index) / 972
            stranded = np.zeros(409)
            stranded[199:201] = rescued / 972
            states = pd.DataFrame({
                "Simulation_Hour": range(409), "Death": mortality,
                "Stranded": stranded, "Injured": np.zeros(409),
                "Sheltered": sheltered, "Hospitalized": np.zeros(409),
                "Evacuated": np.zeros(409), "Rescue_Admissions": cumulative,
                "Distinct_Rescued": cumulative,
            })
            allowance = np.floor(np.arange(1, 409) * rate + 1e-10)
            events = pd.DataFrame({
                "Interval_Start_Hour": range(408), "Interval_End_Hour": range(1, 409),
                "Rescue_Slots": np.diff(np.r_[0, allowance]),
                "Rescue_Admissions": np.diff(cumulative),
                "Admissions_Cumulative": cumulative[1:],
                "Distinct_Rescued": cumulative[1:],
                "Shelter_Occupancy_End": np.rint(sheltered[1:] * 972).astype(int),
                "Hospital_Occupancy_End": np.zeros(408, dtype=int),
                "Nominal_Rescue_Rate": rate,
            })
            admissions = pd.DataFrame(
                [{"hour": 200, "person_index": 0, "stranding_hours": 2}] if rescued else [],
                columns=["hour", "person_index", "stranding_hours"])
            task = {"scenario": name, "seed": seed, "manifest_sha256": fingerprint}
            summary = {
                **task, "households": {
                    "people": 972, "residential_entries": 972, "houses": 286,
                    "three_person_houses": 172, "four_person_houses": 114,
                    "houses_without_adult": 0, "unique_membership": True,
                },
                "hourly_checks": 408, "Death_hour384": 0,
                "Death_hour408": mortality[-1], "Stranded_peak_hourly": stranded.max(),
                "Rescue_Admissions": rescued, "Distinct_Rescued": rescued,
                "Shelter_Peak_People": rescued, "Hospital_Peak_People": 0,
                "Stranding_Hours_At_Admission_Mean": 2.0 if rescued else None,
                "elapsed_seconds": 1.0,
            }
            for filename, frame in [
                ("hourly_states.csv", states), ("rescue_intervals.csv", events),
                ("rescue_admissions.csv", admissions),
            ]:
                frame.to_csv(directory / filename, index=False)
            analysis.write_json(directory / "task.json", task)
            analysis.write_json(directory / "summary.json", summary)
            refresh_completion(directory, fingerprint)


def refresh_completion(directory, fingerprint):
    analysis.write_json(directory / "completion.json", {
        "manifest_sha256": fingerprint,
        "outputs": {name: analysis.sha256(directory / name) for name in [
            "hourly_states.csv", "rescue_intervals.csv", "rescue_admissions.csv", "summary.json",
        ]},
    })


class StatisticsTests(unittest.TestCase):
    def test_exact_sign_flip_matches_brute_force_with_zeros_and_ties(self):
        for differences in [[0, 0], [1, 1], [1, -1], [1, 2, -3],
                            [3, 3, 5, -2], [0, 2, -4, 4]]:
            observed = abs(sum(differences))
            samples = [abs(sum(sign * value for sign, value in zip(signs, differences)))
                       for signs in itertools.product([-1, 1], repeat=len(differences))]
            expected = sum(total >= observed for total in samples) / len(samples)
            self.assertEqual(analysis.paired_sign_flip(differences), expected)

    def test_sign_flip_rejects_noninteger_or_missing_differences(self):
        for values in [[], [1], [1.5, 2], [np.nan, 2], [[1, 2], [3, 4]]]:
            with self.subTest(values=values), self.assertRaises(ValueError):
                analysis.paired_sign_flip(values)

    def test_mean_interval_matches_known_student_t_value(self):
        mean, low, high = analysis.mean_interval([1, 2, 3])
        self.assertEqual(mean, 2)
        half = 4.302652729911275 / math.sqrt(3)
        self.assertAlmostEqual(low, 2-half)
        self.assertAlmostEqual(high, 2+half)
        self.assertEqual(analysis.mean_interval([0, 0, 0]), (0, 0, 0))

    def test_mean_interval_rejects_invalid_outcomes(self):
        for values in [[], [1], [1, np.inf], [np.nan, 0], [[1, 2], [3, 4]]]:
            with self.subTest(values=values), self.assertRaises(ValueError):
                analysis.mean_interval(values)

    def test_seven_families_and_44_contrasts(self):
        families = analysis.families()
        self.assertEqual(len(families), 7)
        self.assertEqual(sum(math.comb(len(names), 2) for names in families.values()), 44)
        tests, pairs = analysis.analyze_pairs(statistical_fixture(), [10, 11, 12, 13])
        self.assertEqual(len(tests), 7)
        self.assertEqual(len(pairs), 44)
        self.assertTrue(pairs.p_holm_family.between(0, 1).all())
        self.assertTrue(tests.p_holm_seven_families.between(0, 1).all())
        self.assertTrue((pairs.p_holm_family >= pairs.p_raw).all())

    def test_pairing_uses_seed_not_input_row_order(self):
        frame = statistical_fixture()
        original = analysis.analyze_pairs(frame, [10, 11, 12, 13])
        reordered = analysis.analyze_pairs(frame.sample(frac=1, random_state=5),
                                           [13, 10, 12, 11])
        for first, second in zip(original, reordered):
            pd.testing.assert_frame_equal(first, second)
        contrast = original[1].iloc[0]
        wide = frame.pivot(index="seed", columns="scenario", values="Deaths")
        expected = (wide[contrast.b] - wide[contrast.a]).mean() * 100 / 972
        self.assertAlmostEqual(contrast.mean_b_minus_a_pp, expected)

    def test_all_within_seed_ties_are_zero_separation(self):
        frame = statistical_fixture()
        frame["Deaths"] = frame.seed - 10
        tests, pairs = analysis.analyze_pairs(frame, [10, 11, 12, 13])
        self.assertTrue(tests.Friedman_statistic.eq(0).all())
        self.assertTrue(tests.p_raw.eq(1).all())
        self.assertTrue(tests.Kendall_W.eq(0).all())
        self.assertTrue(pairs.p_raw.eq(1).all())
        self.assertTrue(pairs.mean_b_minus_a_pp.eq(0).all())
        self.assertTrue(pairs.mc95_low_pp.eq(0).all())
        self.assertTrue(pairs.mc95_high_pp.eq(0).all())

    def test_invalid_or_incomplete_seed_pairing_is_rejected(self):
        frame = statistical_fixture()
        bad_counts = frame.copy()
        bad_counts["Deaths"] = bad_counts.Deaths.astype(float)
        bad_counts.loc[0, "Deaths"] = .5
        for invalid in [frame.iloc[1:], frame.loc[frame.seed != 10],
                        pd.concat([frame, frame.iloc[:1]]), bad_counts]:
            with self.subTest(rows=len(invalid)), self.assertRaises(ValueError):
                analysis.analyze_pairs(invalid, [10, 11, 12, 13])

    def test_table_intervals_reject_nonfinite_or_empty_outcomes(self):
        for values in [[], [np.nan], [1, np.inf]]:
            with self.subTest(values=values), self.assertRaises(ValueError):
                analysis.interval(values)

    def test_graphical_effects_reuse_the_exact_prespecified_statistics(self):
        _, pairs = analysis.analyze_pairs(statistical_fixture(), [10, 11, 12, 13])
        selected = analysis.selected_contrasts(pairs)
        self.assertEqual(len(selected), 7)
        self.assertEqual(selected.label.tolist(), [row[0] for row in analysis.CONTRASTS])
        for row in selected.itertuples():
            match = pairs.loc[(pairs.family == row.family) & (pairs.a == row.a) & (pairs.b == row.b)]
            self.assertEqual(len(match), 1)
            for column in ["mean_b_minus_a_pp", "mc95_low_pp", "mc95_high_pp", "p_holm_family"]:
                self.assertEqual(getattr(row, column), match.iloc[0][column])
        with self.assertRaisesRegex(ValueError, "Missing prespecified contrast"):
            analysis.selected_contrasts(pairs.loc[~(
                pairs.family.eq("organization") & pairs.a.eq("A_org_0") & pairs.b.eq("A_org_1"))])

    def test_seed_connectors_align_by_label_instead_of_row_order(self):
        first = pd.Series([1., 3.], index=[100, 101])
        second = pd.Series([4., 2.], index=[101, 100])
        figure, axis = analysis.plt.subplots()
        self.addCleanup(analysis.plt.close, figure)
        analysis.plot_groups(axis, [first, second], ["First", "Second"], "Outcome")
        np.testing.assert_array_equal(axis.lines[0].get_ydata(), [1., 2.])
        np.testing.assert_array_equal(axis.lines[1].get_ydata(), [3., 4.])
        with self.assertRaisesRegex(ValueError, "same complete seed blocks"):
            analysis.plot_groups(axis, [first, second.rename(index={100: 102})],
                                 ["First", "Second"], "Outcome")


class PublicationLayoutTests(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.addCleanup(self.temporary.cleanup)
        self.root = Path(self.temporary.name)
        self.paper = self.root / "paper"
        self.paper.mkdir()
        self.output = self.root / "analysis"
        self.output.mkdir()
        self.manuscript = self.paper / "flood_columbia.tex"
        self.original = (
            "Author-edited introduction.\n"
            "Old result.\n"
            "Author-edited discussion.\n")
        self.manuscript.write_text(self.original, encoding="utf-8")
        (self.output / "result.tex").write_text("Old result.\n", encoding="utf-8")
        (self.output / "plot.pdf").write_bytes(b"fixture graph")
        (self.output / "summary.csv").write_text("value\n7.72\n", encoding="utf-8")
        self.names = ("result.tex", "plot.pdf", "summary.csv")

    def test_publication_embeds_text_and_copies_only_graphs(self):
        with patch.object(analysis, "PAPER", self.paper):
            snapshot = analysis.publication_snapshot(self.output, self.names)
            (self.output / "result.tex").write_text(r"Mortality is 7.72\%." + "\n", encoding="utf-8")
            analysis.publish_assets(self.output, self.names, snapshot)
            first = self.manuscript.read_bytes()
            analysis.publish_assets(
                self.output, self.names, analysis.publication_snapshot(self.output, self.names))
        self.assertEqual(self.manuscript.read_bytes(), first)
        text = self.manuscript.read_text(encoding="utf-8")
        self.assertIn(r"Mortality is 7.72\%.", text)
        self.assertIn("Author-edited introduction.", text)
        self.assertIn("Author-edited discussion.", text)
        self.assertNotIn(r"\input", text)
        self.assertNotIn("GENERATED:", text)
        self.assertFalse((self.paper / "analysis").exists())
        self.assertFalse((self.paper / "summary.csv").exists())
        self.assertEqual((self.paper / "figures" / "plot.pdf").read_bytes(), b"fixture graph")

    def test_missing_or_ambiguous_text_fails_without_changing_manuscript(self):
        for text in ["Different manuscript.\n", self.original * 2]:
            with self.subTest(text=text):
                self.manuscript.write_text(text, encoding="utf-8")
                with patch.object(analysis, "PAPER", self.paper), self.assertRaisesRegex(
                        ValueError, "exactly one occurrence of previous publication text"):
                    analysis.publication_snapshot(self.output, self.names)
                self.assertEqual(self.manuscript.read_text(encoding="utf-8"), text)
                self.assertFalse((self.paper / "figures").exists())

    def test_overlapping_text_is_rejected(self):
        (self.output / "overlap.tex").write_text("Old result", encoding="utf-8")
        with patch.object(analysis, "PAPER", self.paper), self.assertRaisesRegex(
                ValueError, "fragments overlap"):
            analysis.publication_snapshot(self.output, (*self.names, "overlap.tex"))
        self.assertEqual(self.manuscript.read_text(encoding="utf-8"), self.original)

    def test_concurrent_author_edits_are_not_overwritten(self):
        with patch.object(analysis, "PAPER", self.paper):
            snapshot = analysis.publication_snapshot(self.output, self.names)
            edited = self.original + "New author paragraph.\n"
            self.manuscript.write_text(edited, encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "manuscript changed during analysis"):
                analysis.publish_assets(self.output, self.names, snapshot)
        self.assertEqual(self.manuscript.read_text(encoding="utf-8"), edited)
        self.assertFalse((self.paper / "figures").exists())

    def test_missing_assets_and_empty_text_fail_before_publication(self):
        with patch.object(analysis, "PAPER", self.paper):
            snapshot = analysis.publication_snapshot(self.output, self.names)
            with self.assertRaisesRegex(ValueError, "Missing generated publication asset"):
                analysis.publish_assets(self.output, (*self.names, "missing.pdf"), snapshot)
            (self.output / "result.tex").write_text("\n", encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "Generated manuscript content is empty"):
                analysis.publish_assets(self.output, self.names, snapshot)
        self.assertEqual(self.manuscript.read_text(encoding="utf-8"), self.original)
        self.assertFalse((self.paper / "figures").exists())

    def test_missing_or_empty_previous_text_is_rejected(self):
        with patch.object(analysis, "PAPER", self.paper):
            with self.assertRaisesRegex(ValueError, "Previous publication text is required"):
                analysis.publication_snapshot(self.output, (*self.names, "missing.tex"))
            (self.output / "result.tex").write_text("\n", encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "Previous publication text is empty"):
                analysis.publication_snapshot(self.output, self.names)


class CompilationTests(unittest.TestCase):
    def setUp(self):
        temporary = tempfile.TemporaryDirectory()
        self.addCleanup(temporary.cleanup)
        self.paper = Path(temporary.name)

    def build(self, logs):
        def compile_once(*args, **kwargs):
            (self.paper / "flood_columbia.log").write_text(next(logs), encoding="utf-8")
            return analysis.subprocess.CompletedProcess(args[0], 0, stdout="")

        with patch.object(analysis, "PAPER", self.paper), patch.object(
                analysis.subprocess, "run", side_effect=compile_once) as runner:
            analysis.compile_paper()
        return runner.call_count

    def test_build_uses_at_least_two_passes(self):
        self.assertEqual(self.build(iter(["", ""])), 2)

    def test_build_continues_until_labels_settle(self):
        warning = "LaTeX Warning: Label(s) may have changed. Rerun to get cross-references right."
        self.assertEqual(self.build(iter([warning, warning, ""])), 3)

    def test_nonconverging_labels_fail_after_five_passes(self):
        warning = "LaTeX Warning: Label(s) may have changed. Rerun to get cross-references right."
        with self.assertRaisesRegex(ValueError, "Label"):
            self.build(iter([warning] * 5))

    def test_layout_warnings_are_not_suppressed(self):
        with self.assertRaisesRegex(ValueError, "Overfull"):
            self.build(iter(["", "Overfull \\\\hbox"]))


class CohortTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.metadata = analysis.study_manifest(2, 100)

    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.addCleanup(self.temporary.cleanup)
        self.batch = Path(self.temporary.name) / "fixture"
        self.batch.mkdir()
        analysis.write_json(self.batch / "manifest.json", self.metadata)

    def test_incomplete_cohort_fails_before_creating_publication_assets(self):
        output = Path(self.temporary.name) / "assets"
        with patch.object(sys, "argv", ["paper_results", "--batch", str(self.batch),
                                       "--out", str(output)]):
            with self.assertRaisesRegex(ValueError, "Incomplete or unexpected cohort"):
                analysis.main()
        self.assertFalse(output.exists())

    def test_manifest_changes_are_rejected(self):
        altered = {**self.metadata, "steps": 407}
        analysis.write_json(self.batch / "manifest.json", altered)
        with self.assertRaisesRegex(ValueError, "differs from current"):
            analysis.load_cohort(self.batch)

    def test_synthetic_complete_cohort_regenerates_every_expected_asset(self):
        write_fixture(self.batch, self.metadata)
        output = Path(self.temporary.name) / "assets"
        output.mkdir()
        (output / "unrelated_archive.csv").write_text("Not a final-study asset\n")
        legends = {}
        savefig = analysis.plt.Figure.savefig

        def checked_save(figure, filename, *args, **kwargs):
            legends[Path(filename).name] = assert_external_legend(self, figure)
            return savefig(figure, filename, *args, **kwargs)

        with patch.object(sys, "argv", ["paper_results", "--batch", str(self.batch),
                                       "--out", str(output)]), patch.object(
                analysis.plt.Figure, "savefig", checked_save), contextlib.redirect_stdout(io.StringIO()):
            analysis.main()
        self.assertEqual(len(legends), 6)
        for name in ["organization.pdf", "ablation.pdf"]:
            self.assertEqual(legends[name], [
                "Individual seed outcome", "Same-seed connection", "Median and IQR"])
        self.assertEqual(set(legends["logistics.pdf"]), {
            "a: 1% shelter (10 places)", "a: 10% shelter (98 places)",
            "a: 30% shelter (292 places)", "b: Near (2.0 km)", "b: Far (99.9 km)",
            "Individual seed outcome"})
        self.assertIn("IQR band (site colour)", legends["rescue_process.pdf"])
        self.assertIn("Active flood footprint", legends["rescue_process.pdf"])
        self.assertIn("Facility capacity", legends["rescue_process.pdf"])
        self.assertIn("95% Monte Carlo interval", legends["paired_effects.pdf"])
        self.assertIn("No difference (zero)", legends["paired_effects.pdf"])
        self.assertTrue((output / "study_ready.tex").is_file())
        validation = json.loads((output / "validation.json").read_text())
        self.assertEqual(validation["runs"], 48)
        self.assertEqual(validation["settings"], 24)
        self.assertEqual(len(validation["source_completion_hashes"]), 48)
        self.assertEqual(set(validation["generated_sha256"]), set(analysis.PUBLICATION_ASSETS))
        for name, fingerprint in validation["generated_sha256"].items():
            self.assertEqual(analysis.sha256(output / name), fingerprint)
            self.assertGreater((output / name).stat().st_size, 0)
        for name in [n for n in analysis.PUBLICATION_ASSETS if n.endswith("_table.tex")]:
            text = (output / name).read_text(encoding="ascii")
            self.assertIn(r"\begin{tabular}", text)
            self.assertIn(r"\end{tabular}", text)
        self.assertEqual(len(pd.read_csv(output / "run_summary.csv")), 48)
        self.assertEqual(len(pd.read_csv(output / "pairwise_tests.csv")), 44)
        self.assertEqual(len(pd.read_csv(output / "omnibus_tests.csv")), 7)
        self.assertEqual(len(pd.read_csv(output / "monte_carlo_precision.csv")), 24)
        self.assertIn(r"\newcommand{\StudyRuns}{48}",
                      (output / "study_design_numbers.tex").read_text())
        self.assertEqual(sum(name.endswith(".pdf") for name in analysis.PUBLICATION_ASSETS), 6)
        self.assertEqual(sum(name.endswith("_table.tex") for name in analysis.PUBLICATION_ASSETS), 10)
        self.assertIn("48 unique primary runs", (output / "design_table.tex").read_text())
        self.assertIn("Hours full", (output / "process_table.tex").read_text())
        parameters = (output / "parameters_table.tex").read_text()
        for symbol in [r"\alpha_{\mathrm{depth}}", r"\alpha_{\mathrm{social}}",
                       r"\operatorname{Unif}(10,55)", r"g_{\mathrm{org}}"]:
            self.assertIn(symbol, parameters)
        self.assertNotIn("$c_d", parameters)
        self.assertNotIn("$c_s", parameters)
        design = (output / "design_table.tex").read_text()
        self.assertIn(r"$g_{\mathrm{org}}=0,0.25,0.5,0.75,1$", design)
        self.assertIn(r"$g_{\mathrm{org}}=1$", design)
        self.assertEqual(design.count(r"$g_{\mathrm{org}}=0$"), 2)
        self.assertNotIn("$o", parameters + design)
        self.assertNotIn(r"\leq o", parameters)
        test_table = (output / "tests_table.tex").read_text()
        for symbol in ["Q_{\\mathrm{Fr}}", "A_{\\mathrm{rank}}",
                       "p_{\\mathrm{raw}}", "p_{\\mathrm{Holm}}"]:
            self.assertIn(symbol, test_table)
        self.assertNotIn("$Q$", test_table)
        self.assertNotIn("$W$", test_table)
        for filename in ["study_logistics_text.tex", "study_inference_text.tex",
                         "study_discussion_logistics.tex"]:
            text = (output / filename).read_text()
            self.assertIn(r"$p_{\mathrm{Holm}}", text)
            self.assertNotIn("$p=", text)
            self.assertNotIn("$p$", text)
        self.assertIn("95\\% Monte Carlo interval", (output / "study_organization_text.tex").read_text())
        abstract = (output / "study_abstract.tex").read_text()
        self.assertIn("random seeds per setting", abstract)
        self.assertIn("at the nearer site", abstract)
        self.assertIn("with one boat", abstract)
        self.assertIn("with five boats", abstract)
        self.assertNotIn("near/far medians", abstract)
        self.assertNotIn("one/five boats yield", abstract)

    def test_study_map_uses_verified_projected_inputs(self):
        layers = analysis.spatial_layers(self.metadata)
        self.assertEqual(len(layers["houses"]), 286)
        self.assertEqual(len(layers["businesses"]), 4)
        self.assertEqual(len(layers["schools"]), 3)
        self.assertTrue(all(layer.crs.to_epsg() == 32618 for layer in layers.values()))
        flood = layers["flood"].unary_union
        self.assertTrue(layers["houses"].geometry.within(flood).all())
        for name in analysis.SITES:
            self.assertFalse(layers[name].geometry.intersects(flood).any())

    def test_hash_valid_but_inconsistent_measurements_are_rejected(self):
        write_fixture(self.batch, self.metadata)
        directory = self.batch / "A_org_0.25__seed100"
        fingerprint = analysis.sha256(self.batch / "manifest.json")
        changes = [
            ("hourly_states.csv", "Death", 408, 1.5 / 972, "integer people"),
            ("hourly_states.csv", "Sheltered", 201, 2 / 972, "occupancy/state"),
            ("rescue_intervals.csv", "Rescue_Slots", 0, 0, "Rescue-credit"),
            ("rescue_intervals.csv", "Nominal_Rescue_Rate", 0, 1, "nominal rescue"),
            ("rescue_admissions.csv", "person_index", 0, .5, "identities/durations"),
            ("rescue_admissions.csv", "stranding_hours", 0, 2.5, "identities/durations"),
        ]
        for filename, column, index, value, message in changes:
            path = directory / filename
            original = path.read_bytes()
            with self.subTest(column=column):
                frame = pd.read_csv(path)
                frame[column] = frame[column].astype(float)
                frame.loc[index, column] = value
                frame.to_csv(path, index=False)
                refresh_completion(directory, fingerprint)
                with self.assertRaisesRegex(ValueError, message):
                    analysis.load_cohort(self.batch)
                path.write_bytes(original)
                refresh_completion(directory, fingerprint)

    def test_unexpected_completed_run_is_rejected(self):
        write_fixture(self.batch, self.metadata)
        extra = self.batch / "unplanned__seed100"
        extra.mkdir()
        (extra / "completion.json").write_text("{}")
        with self.assertRaisesRegex(ValueError, "Incomplete or unexpected cohort"):
            analysis.load_cohort(self.batch)

    def test_compile_rejects_nonstandard_asset_destination(self):
        output = Path(self.temporary.name) / "assets"
        with patch.object(sys, "argv", ["paper_results", "--batch", str(self.batch),
                                       "--out", str(output), "--compile"]):
            with self.assertRaisesRegex(ValueError, "standard publication-asset destination"):
                analysis.main()
        self.assertFalse(output.exists())

    def test_analysis_cannot_write_into_the_manuscript_working_copy(self):
        output = analysis.PAPER / "analysis" / "paper_generated"
        with patch.object(sys, "argv", ["paper_results", "--batch", str(self.batch),
                                       "--out", str(output)]):
            with self.assertRaisesRegex(ValueError, "outside the manuscript working copy"):
                analysis.main()

    def test_archive_reanalysis_cannot_overwrite_final_publication_assets(self):
        from analyze_archive import OUT
        self.assertNotEqual(OUT.resolve(), analysis.PUBLICATION_OUTPUT.resolve())
        self.assertFalse(OUT.resolve().is_relative_to(analysis.PAPER.resolve()))

    @unittest.skipUnless(
        shutil.which("pdflatex") and (analysis.PAPER / "flood_columbia.tex").is_file(),
        "The LaTeX toolchain and manuscript working copy are required")
    def test_complete_pipeline_compiles_the_actual_manuscript_with_fixture_assets(self):
        write_fixture(self.batch, self.metadata)
        paper = Path(self.temporary.name) / "paper"
        paper.mkdir()
        shutil.copyfile(analysis.PAPER / "flood_columbia.tex", paper / "flood_columbia.tex")
        shutil.copytree(analysis.PAPER / "figures", paper / "figures")
        output = Path(self.temporary.name) / "assets"
        shutil.copytree(analysis.PUBLICATION_OUTPUT, output)
        with patch.object(analysis, "PAPER", paper), patch.object(
                analysis, "PUBLICATION_OUTPUT", output), patch.object(
                sys, "argv", ["paper_results", "--batch", str(self.batch),
                              "--out", str(output), "--compile"]
        ), contextlib.redirect_stdout(io.StringIO()):
            analysis.main()
        self.assertTrue((output / "study_ready.tex").is_file())
        self.assertGreater((paper / "flood_columbia.pdf").stat().st_size, 50000)
        self.assertFalse((paper / "analysis").exists())
        text = (paper / "flood_columbia.tex").read_text(encoding="utf-8")
        self.assertNotIn(r"\input{analysis/", text)
        self.assertNotIn("GENERATED:", text)
        self.assertIn(r"\newcommand{\StudyRuns}{48}", text)


if __name__ == "__main__":
    unittest.main()
