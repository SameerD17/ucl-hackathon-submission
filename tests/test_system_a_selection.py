"""Scientific regression checks; run with pytest or standard-library unittest.

python -m unittest discover -s tests -p test_system_a_selection.py
"""

import contextlib
import copy
import io
import json
from pathlib import Path
import sys
import tempfile
import unittest
from unittest.mock import patch

import numpy as np
import pandas as pd
from scipy.stats import norm

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import validate_system_a_selection as selection


class SystemASelectionTests(unittest.TestCase):
    def test_deviance_matches_independent_gaussian_log_density(self):
        relative = np.tile([0.02, 0.05], 41)
        y = np.geomspace(1e-6, 2, len(relative))
        mean = y*np.geomspace(1e-3, 1e3, len(y))
        optimum = 2*y/(1+np.sqrt(1+4*relative**2))
        residual = selection.deviance_residuals(y, mean, relative)
        expected = -norm.logpdf(y, loc=mean, scale=relative*mean)
        expected += norm.logpdf(y, loc=optimum, scale=relative*optimum)
        np.testing.assert_allclose(0.5*residual**2, expected, rtol=1e-11, atol=1e-8)

    def test_deviance_is_smooth_and_zero_at_individual_optimum(self):
        relative = np.array([0.02, 0.05, 0.02, 0.05])
        y = np.array([1e-5, 0.002, 0.3, 1.2])
        a = (1+np.sqrt(1+4*relative**2))/2
        optimum = y/a
        np.testing.assert_allclose(selection.deviance_residuals(y, optimum, relative), 0, atol=1e-12)
        step = 1e-6
        left = selection.deviance_residuals(y, optimum/(1-step), relative)
        right = selection.deviance_residuals(y, optimum/(1+step), relative)
        np.testing.assert_allclose((right-left)/(2*step), np.sqrt((a/relative)**2+1), rtol=1e-8)

    def test_m1_likelihood_matches_original_independent_implementation(self):
        runs = selection.reference.load_runs()
        x = np.array([-1.6, 1.1, -2.5, 1.4])
        problem = selection.FitProblem("M1", runs)
        expected = selection.reference.negative_log_likelihood(x, runs)
        self.assertAlmostEqual(problem.nll(x), expected, places=8)
        residuals = problem.residuals(x)
        self.assertAlmostEqual(float(0.5*residuals@residuals+problem.constant), expected, places=7)

    def test_training_partitions_exclude_entire_experiment_before_full_fits(self):
        runs = selection.reference.load_runs()
        calls = []

        def fit(model, training):
            calls.append((model, tuple(run["id"] for run in training)))
            return {"training_run_ids": [run["id"] for run in training]}

        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            with patch.object(selection, "ROOT", root), patch.object(selection, "RESULTS", root / "results"), \
                    patch.object(selection, "source_manifest", return_value={"sentinel": "unchanged"}), \
                    patch.object(selection, "fit_model", side_effect=fit), \
                    patch.object(selection, "regenerate", return_value={}), contextlib.redirect_stdout(io.StringIO()):
                selection.run_analysis()
            report = json.loads((root / "results" / selection.FIT_FILE).read_text())
        self.assertEqual(len(calls), 36)
        for index, excluded in enumerate(runs):
            expected_ids = tuple(run["id"] for run in runs if run["id"] != excluded["id"])
            for model, actual_ids in calls[index*6:(index+1)*6]:
                self.assertEqual(actual_ids, expected_ids)
                self.assertNotIn(excluded["id"], report["fold_fits"][model][excluded["id"]]["training_run_ids"])
        self.assertTrue(all(len(ids) == 5 for _, ids in calls[30:]))

    def test_excluded_scores_cannot_change_training_only_choices(self):
        runs = selection.reference.load_runs()
        report = {"full_fits": {}, "fold_fits": {}, "elapsed_fitting_seconds": 0, "source_sha256": {}}
        expected = {run["id"]: selection.MODELS[index] for index, run in enumerate(runs)}
        for model in selection.MODELS:
            base = {"parameters": {}, "n_parameters": len(selection.CANDIDATES[model].param_names),
                    "parameters_at_bounds": [], "successful_start_nll_span": 0.0,
                    "n_observations": 244, "negative_log_likelihood": 10.0*int(model[1])}
            report["full_fits"][model] = base
            report["fold_fits"][model] = {
                run["id"]: {**base, "negative_log_likelihood": 0.0 if expected[run["id"]] == model else 100.0,
                            "n_observations": 244-int(run["use"].sum())*4,
                            "training_run_ids": [other["id"] for other in runs if other["id"] != run["id"]]}
                for run in runs
            }

        def score_with_favoured(favoured):
            def score(model, run, _params):
                return pd.DataFrame([
                    {"model": model, "run_id": run["id"], "time_min": float(run["times"][index]),
                     "species": species, "standardized_error": 0.0,
                     "negative_log_likelihood": -100.0 if model == favoured else 100.0}
                    for index in np.flatnonzero(run["use"]) for species in selection.SPECIES
                ])
            return score

        summaries = []
        for favoured in ["M1", "M6"]:
            with patch.object(selection, "scored_predictions", side_effect=score_with_favoured(favoured)), \
                    patch.object(selection, "digest", return_value="test"):
                summary, *_ = selection.analyse(report, runs)
            actual = {row["heldout_run_id"]: row["selected_model"] for row in summary["fold_selections"]}
            self.assertEqual(actual, expected)
            self.assertEqual(summary["fixed_candidate_CV_NLL_leader"], favoured)
            summaries.append(summary)
        self.assertEqual(summaries[0]["reference_selected_folds"], 1)
        self.assertEqual(summaries[1]["reference_selected_folds"], 1)

    def test_saved_fit_check_accepts_measured_windows_numerical_differences(self):
        # Observed with identical sources/data and pinned dependencies, using
        # Windows/Python 3.13.7 to check Linux/Python 3.12.14 saved parameters.
        # Within each model: all runs, then exclude A1, A2, A3, A4, A5.
        measured_deltas = {
            "M1": [3.9491852e-8, 2.6460953e-8, 6.9017915e-8, -3.1968739e-10, 1.6216063e-8, -1.5654678e-8],
            "M2": [1.4406186e-5, 1.552921e-6, 2.1348714e-6, 8.7324336e-6, 1.9869208e-6, 2.3331414e-6],
            "M3": [1.3087585e-5, 7.1048853e-6, -4.4206681e-6, 2.1525193e-7, -4.5401976e-9, 6.9849193e-9],
            "M4": [6.1965997e-5, 5.8181282e-5, 0.00011195105, 0.00013455352, 4.5836978e-6, -1.809226e-6],
            "M5": [0.00040538365, 2.0459993e-6, 5.0696544e-6, 2.2951281e-7, -1.0016374e-5, 1.0750664e-6],
            "M6": [-1.3250974e-7, -1.8626451e-9, 5.9575541e-8, -1.3969839e-9, -1.0966323e-7, -3.0704541e-9],
        }
        report = json.loads((selection.RESULTS / selection.FIT_FILE).read_text(encoding="utf-8"))
        runs = selection.reference.load_runs()
        scores = {}
        for model in selection.MODELS:
            entries = [("all", runs, report["full_fits"][model])]
            entries += [(held_out["id"], [run for run in runs if run["id"] != held_out["id"]],
                         report["fold_fits"][model][held_out["id"]]) for held_out in runs]
            for (label, training, fit), delta in zip(entries, measured_deltas[model]):
                with self.subTest(model=model, partition=label):
                    actual = fit["negative_log_likelihood"] + delta
                    with patch.object(selection.FitProblem, "nll", return_value=actual):
                        selection.checked_fit(model, training, fit)
                    scores.setdefault(label, []).append((model, selection.aicc(fit),
                        selection.aicc({**fit, "negative_log_likelihood": actual})))
        for label, values in scores.items():
            with self.subTest(partition=label):
                self.assertEqual(sorted(values, key=lambda row: (row[1], row[0]))[0][0],
                                 sorted(values, key=lambda row: (row[2], row[0]))[0][0])

    def test_saved_fit_check_rejects_material_and_nonfinite_changes(self):
        report = json.loads((selection.RESULTS / selection.FIT_FILE).read_text(encoding="utf-8"))
        runs = selection.reference.load_runs()
        for model in ("M1", "M5"):
            fit = report["full_fits"][model]
            for actual in (fit["negative_log_likelihood"]+0.01, float("inf"), float("nan")):
                with self.subTest(model=model, actual=actual), \
                        patch.object(selection.FitProblem, "nll", return_value=actual):
                    with self.assertRaisesRegex(ValueError, "allowed absolute difference=0.001"):
                        selection.checked_fit(model, runs, fit)
        changed = copy.deepcopy(report["full_fits"]["M1"])
        changed["parameters"]["k1"] *= 1.01
        with self.assertRaisesRegex(ValueError, "likelihood disagrees"):
            selection.checked_fit("M1", runs, changed)

    def test_saved_start_records_still_require_exact_agreement(self):
        report = json.loads((selection.RESULTS / selection.FIT_FILE).read_text(encoding="utf-8"))
        runs = selection.reference.load_runs()
        changed = copy.deepcopy(report["full_fits"]["M1"])
        changed["negative_log_likelihood"] += 0.0001
        with self.assertRaisesRegex(ValueError, "successful starting fits"):
            selection.checked_fit("M1", runs, changed)

    def test_changed_source_or_data_is_rejected_before_reintegration(self):
        report = json.loads((selection.RESULTS / selection.FIT_FILE).read_text(encoding="utf-8"))
        for name in ("data/system_A/measurements.csv", "validate_system_a_selection.py"):
            changed = {**report["source_sha256"], name: "changed-for-test"}
            with self.subTest(source=name), \
                    patch.object(selection, "source_manifest", return_value=changed), \
                    patch.object(selection.FitProblem, "nll") as evaluate:
                with self.assertRaisesRegex(ValueError, "source/data changed"):
                    selection.checked_report()
                evaluate.assert_not_called()


if __name__ == "__main__":
    unittest.main()
