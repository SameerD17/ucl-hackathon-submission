"""Regenerate figures from saved fit parameters, checking them against the data.

Run: python replicate_figures.py

Keep results/ in the submitted repository alongside all analysis scripts. The
stored fit parameters and starting-fit records are reproducible intermediate
results, not new measurements. This script recomputes predictions, residuals,
scores, intervals and figures; it avoids repeating expensive optimization just
to redraw figures. Missing fitting results are regenerated using the original
fit/validation scripts. Mismatched likelihoods or source hashes stop the run.

To repeat optimization deliberately, run fit_system_a.py, validate_system_a.py,
validate_system_a_selection.py, fit_system_b.py and validate_system_b.py before running this script. A clean
clone WITH the saved results is the intended quick repository-contract path.
"""

from __future__ import annotations

import json
import subprocess
import sys
import time
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pandas as pd

import fit_system_a as fit_a
import validate_system_a as cv_a
import validate_system_a_selection as selection_a
import fit_system_b as fit_b
import validate_system_b as cv_b
import load_system_b as loader
import predict_system_b as predictor

ROOT = Path(__file__).resolve().parent
RESULTS = ROOT / "results"


def require_files(names, generate):
    if not all((RESULTS / name).is_file() for name in names):
        print("Saved results are missing; running the corresponding fit stage.", flush=True)
        generate()


def assert_close(actual, expected, description):
    if not np.isclose(actual, expected, atol=1e-5, rtol=1e-8):
        raise RuntimeError(f"Saved results disagree with current data/model for {description}. Rerun its fitting script.")


def recreate_system_a():
    require_files(["system_A_M1_fit.json", "system_A_M1_starts.csv"], fit_a.main)
    require_files(["system_A_cv_summary.json", "system_A_cv_starts.csv"], cv_a.main)
    runs = fit_a.load_runs()
    full = json.loads((RESULTS / "system_A_M1_fit.json").read_text(encoding="utf-8"))
    params = full["parameters"]
    x = predictor.coordinates("M1", params)
    nll = fit_a.negative_log_likelihood(x, runs)
    assert_close(nll, full["negative_log_likelihood"], "System A full fit")
    starts = pd.read_csv(RESULTS / "system_A_M1_starts.csv")
    fit_a.save_outputs(runs, SimpleNamespace(x=x, fun=nll), starts.to_dict(orient="records"))
    report = json.loads((RESULTS / "system_A_cv_summary.json").read_text(encoding="utf-8"))
    starts = pd.read_csv(RESULTS / "system_A_cv_starts.csv")
    folds = pd.DataFrame(report["folds"])
    if set(folds.heldout_run_id) != {run["id"] for run in runs} or folds.heldout_run_id.duplicated().any():
        raise RuntimeError("System A validation has missing or duplicate experiments.")
    all_predictions, parameters = [], {}
    for run in runs:
        row = folds[folds.heldout_run_id.eq(run["id"])].iloc[0]
        training = [other for other in runs if other["id"] != run["id"]]
        ids = [other["id"] for other in training]
        if set(row.training_run_ids.split(",")) != set(ids):
            raise RuntimeError("System A saved training partition is inconsistent.")
        params = {name: float(row[name]) for name in fit_a.PARAMETER_NAMES}
        parameters[run["id"]] = params
        x = predictor.coordinates("M1", params)
        successful = starts[starts.heldout_run_id.eq(run["id"]) & starts.success]
        if successful.empty:
            raise RuntimeError("System A validation lacks a successful saved starting fit.")
        assert_close(fit_a.negative_log_likelihood(x, training), successful.negative_log_likelihood.min(),
                     f"System A fold {run['id']} training likelihood")
        test = cv_a.observation_predictions(run, params, ids)
        assert_close(cv_a.scores(test)["standardized_rms"], row.heldout_standardized_rms,
                     f"System A fold {run['id']} held-out score")
        all_predictions.append(test)
    predictions = pd.concat(all_predictions, ignore_index=True)
    cv_a.check_prediction_keys(runs, predictions)
    cv_a.save_results(runs, predictions, folds, starts, parameters, report["elapsed_fitting_seconds"])
    print("Regenerated System A fits and validation figures from checked parameters.", flush=True)


def recreate_system_b():
    loader.main()
    require_files(["system_B_fits.json", "system_B_fit_starts.csv"], fit_b.main)
    full = predictor.checked_report(RESULTS / "system_B_fits.json")
    require_files(["system_B_cv_summary.json", "system_B_cv_folds.csv", "system_B_cv_starts.csv"], cv_b.main)
    report = predictor.checked_report(RESULTS / "system_B_cv_summary.json", cross_validation=True)
    runs = loader.load_runs()
    if set(full["models"]) != set(fit_b.MODELS) or set(report["fold_fits"]) != set(fit_b.MODELS):
        raise RuntimeError("Saved System B fits must include all six candidates.")
    for model, fit in full["models"].items():
        problem = fit_b.FitProblem(model, runs)
        assert_close(problem.nll(predictor.coordinates(model, fit["parameters"])),
                     fit["negative_log_likelihood"], f"System B {model} full fit")
    # Recalculate all fitted curves and rankings, not just copy saved images.
    predictions = pd.concat([fit_b.prediction_table(model, runs, fit["parameters"])
                             for model, fit in full["models"].items()], ignore_index=True)
    comparison = fit_b.compare_models(full["models"], predictions)
    residuals = fit_b.residual_summary(predictions)
    fit_b.make_figures(runs, full["models"], comparison, predictions, residuals)
    fold_rows = pd.read_csv(RESULTS / "system_B_cv_folds.csv")
    if len(fold_rows) != len(fit_b.MODELS)*len(runs) or fold_rows.duplicated(["model", "heldout_run_id"]).any():
        raise RuntimeError("System B saved folds are missing or duplicated.")
    all_predictions = []
    for model in fit_b.MODELS:
        for run in runs:
            training = [other for other in runs if other["id"] != run["id"]]
            ids = [other["id"] for other in training]
            fit = report["fold_fits"][model][run["id"]]
            problem = fit_b.FitProblem(model, training)
            assert_close(problem.nll(predictor.coordinates(model, fit["parameters"])),
                         fit["negative_log_likelihood"], f"System B {model} fold {run['id']}")
            row = fold_rows[fold_rows.model.eq(model) & fold_rows.heldout_run_id.eq(run["id"])]
            if len(row) != 1 or set(row.iloc[0].training_run_ids.split(",")) != set(ids):
                raise RuntimeError("System B saved training partition is inconsistent.")
            predicted = cv_b.scored_predictions(model, run, fit["parameters"], ids)
            assert_close(cv_b.summarize_predictions(predicted)["plugin_NLL"], row.iloc[0].heldout_plugin_NLL,
                         f"System B {model} fold {run['id']} excluded measurements")
            all_predictions.append(predicted)
    predicted = pd.concat(all_predictions, ignore_index=True)
    cv_b.check_prediction_keys(runs, fit_b.MODELS, predicted)
    summary = cv_b.model_summary(predicted, fold_rows)
    by_species = cv_b.species_summary(predicted)
    pairs = cv_b.paired_comparison(fold_rows)
    cv_b.make_figures(runs, summary, by_species, fold_rows, pairs, report["fold_fits"])
    print("Regenerated System B comparison and validation figures from checked parameters.", flush=True)


def main():
    started = time.perf_counter()
    print("Regenerating figures from saved fits, with data/model consistency checks.", flush=True)
    subprocess.run([sys.executable, str(ROOT / "baseline" / "replicate.py")], check=True, cwd=ROOT)
    recreate_system_a()
    require_files([selection_a.FIT_FILE], selection_a.run_analysis)
    selection_a.regenerate()
    recreate_system_b()
    # Recompute intervals and their coverage every time, including covariance.
    subprocess.run([sys.executable, str(ROOT / "predict_system_b.py"), "--check-intervals"], check=True, cwd=ROOT)
    print(f"\nAll current analysis figures regenerated in {time.perf_counter()-started:.1f} seconds.")
    print("Keep the analysis scripts and results/ in the repository for the quick reproduction path.")
    print("Run the original fit/validation scripts when you want to repeat optimization.")


if __name__ == "__main__":
    main()
