"""Check the current submission and optionally run the four required recipes.

    python check_submission.py                 # check existing files
    python check_submission.py --run-workflow  # install, predict, suggest, replicate, check

Use --run-workflow with the project's virtual environment active. Installation
may use the network; the scientific recipes require no network. This command
checks the current working copy. Repeat it after cloning your own submitted
repository into a fresh directory to verify that every required file was saved.

These checks verify file formats, consistency, budgets, figure presence and
workflow execution. They do not establish scientific correctness, calibration
at new initial charges, experimental success, or a finished presentation.
"""

from __future__ import annotations

import argparse
import importlib.metadata
import json
import math
import shutil
import struct
import subprocess
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent
RESULTS = ROOT / "results"
FIT_FILES = [
    "system_A_M1_fit.json", "system_A_M1_starts.csv",
    "system_A_cv_summary.json", "system_A_cv_starts.csv",
    "system_A_selection_fits.json",
    "system_B_fits.json", "system_B_fit_starts.csv",
    "system_B_cv_summary.json", "system_B_cv_folds.csv", "system_B_cv_starts.csv",
]
FIGURES = [
    "system_B_overview.png", "system_A_M1_fit.png", "system_A_M1_residuals.png",
    "system_A_cv_predictions.png", "system_A_cv_diagnostics.png",
    "system_A_selection_comparison.png", "system_A_selection_folds.png",
    "system_B_measurements.png", "system_B_model_comparison.png",
    "system_B_leading_fits.png", "system_B_leading_residuals.png",
    "system_B_cv_comparison.png", "system_B_cv_leading_predictions.png",
    "system_B_cv_diagnostics.png", "system_B_heldout_Q_intervals.png",
    "system_B_interval_coverage.png", "system_B_campaign_predictions.png",
    "system_B_campaign_design.png",
]


def require(condition, message):
    if not condition:
        raise ValueError(message)


def read_json(name):
    path = RESULTS / name
    require(path.is_file(), f"Missing {path.relative_to(ROOT)}; run the required recipes first.")
    return json.loads(path.read_text(encoding="utf-8"))


def run_workflow():
    require(sys.prefix != sys.base_prefix,
            "Activate the project's .venv before using --run-workflow.")
    executable = shutil.which("just")
    require(executable is not None, "just was not found. Run this from the terminal where just replicate worked.")
    timings = []
    started = time.perf_counter()
    for recipe in ["install", "predict", "suggest", "replicate"]:
        print(f"\nRunning just {recipe} ...", flush=True)
        command_started = time.perf_counter()
        completed = subprocess.run([executable, recipe], cwd=ROOT, check=False)
        duration = time.perf_counter()-command_started
        timings.append({"command": f"just {recipe}", "seconds": duration, "exit_code": completed.returncode})
        require(completed.returncode == 0, f"just {recipe} failed. Resolve the error above before continuing.")
    return timings, time.perf_counter()-started


def check_files():
    # Import scientific dependencies AFTER optional installation, so the checker
    # itself can start inside a fresh virtual environment.
    import numpy as np
    import pandas as pd
    import yaml
    import predict_system_b as predictor
    import suggest_campaign as designer
    import validate_system_a_selection as selection_a

    dependencies = {}
    for line in (ROOT / "requirements.txt").read_text(encoding="utf-8").splitlines():
        if not line.strip() or line.lstrip().startswith("#"):
            continue
        name, expected = line.strip().split("==")
        actual = importlib.metadata.version(name)
        require(actual == expected, f"{name} is {actual}, but requirements.txt pins {expected}. Run just install.")
        dependencies[name] = actual
    missing = [name for name in FIT_FILES if not (RESULTS / name).is_file()]
    require(not missing, "Missing saved fits for fast reproduction: "+", ".join(missing))
    reference_fits, reference_runs = selection_a.checked_report()
    reference_summary = read_json(selection_a.SUMMARY_FILE)
    require(reference_summary["fit_report_sha256"] == selection_a.digest(RESULTS / selection_a.FIT_FILE),
            "System A selection summary used different saved fits; run just replicate.")
    require(reference_summary["n_models"] == 6 and reference_summary["n_fold_fits"] == 30
            and reference_summary["n_positive_time_measurements"] == 244,
            "System A selection comparison has missing models, folds or observations.")
    expected_leader = min(reference_fits["full_fits"],
                          key=lambda model: (selection_a.aicc(reference_fits["full_fits"][model]), model))
    require(reference_summary["full_data_AICc_leader"] == expected_leader,
            "System A full-data model ranking disagrees with the saved likelihoods.")
    expected_choices = {run["id"]: min(selection_a.MODELS,
                        key=lambda model: (selection_a.aicc(reference_fits["fold_fits"][model][run["id"]]), model))
                        for run in reference_runs}
    choices = reference_summary["fold_selections"]
    require(len(choices) == 5 and {row["heldout_run_id"]: row["selected_model"] for row in choices} == expected_choices,
            "System A fold selections disagree with the training-only AICc rule.")
    require(reference_summary["reference_selected_folds"] == sum(model == "M1" for model in expected_choices.values()),
            "System A reference-selection count is inconsistent.")

    with (ROOT / "submission" / "hypotheses.yaml").open(encoding="utf-8") as handle:
        hypotheses = yaml.safe_load(handle)
    require(isinstance(hypotheses, dict) and set(hypotheses) == {"ranking"}, "Invalid hypotheses.yaml top-level schema.")
    ranking = hypotheses["ranking"]
    require(isinstance(ranking, list) and len(ranking) == 6, "Rank all six hypotheses exactly once.")
    require(all(isinstance(row, dict) and set(row) == {"model", "weight", "justification"} for row in ranking),
            "Every hypothesis needs model, weight and justification.")
    names = [row["model"] for row in ranking]
    require(set(names) == set(predictor.fitter.MODELS) and len(set(names)) == 6, "Missing or duplicate hypothesis model.")
    weights = {row["model"]: row["weight"] for row in ranking}
    require(all(isinstance(value, (int, float)) and not isinstance(value, bool)
                and math.isfinite(value) and value >= 0 for value in weights.values()),
            "Hypothesis weights must be finite nonnegative numbers.")
    require(math.isclose(sum(weights.values()), 1.0, abs_tol=1e-10), "Hypothesis weights do not sum to one.")
    require(all(isinstance(row["justification"], str) and row["justification"].strip()
                and "placeholder" not in row["justification"].lower() for row in ranking),
            "A hypothesis justification is missing or still a placeholder.")
    positive_weights = {model: value for model, value in weights.items() if value > 0}
    implemented_weights = dict(zip(predictor.MODELS, predictor.WEIGHTS.tolist()))
    require(positive_weights == implemented_weights, "Hypothesis weights disagree with the implemented prediction mixture.")
    cv = read_json("system_B_cv_summary.json")
    require(names == [row["model"] for row in sorted(cv["models"], key=lambda row: row["plugin_NLL"])],
            "Hypothesis ordering disagrees with the stated held-out-NLL ranking rule.")

    conditions, _ = predictor.load_conditions()
    predictions = pd.read_csv(ROOT / "submission" / "predictions.csv")
    require(list(predictions.columns) == predictor.OUTPUT_COLUMNS, "predictions.csv has the wrong columns.")
    predictor.check_submission(predictions, conditions)
    detailed = pd.read_csv(RESULTS / "system_B_prediction_details.csv")
    keys = ["run_id", "time_min", "species"]
    left = predictions.set_index(keys).sort_index()
    right = detailed.set_index(keys).sort_index()
    require(left.index.equals(right.index), "Submission and detailed prediction keys disagree.")
    require(np.allclose(left[["prediction", "lower90", "upper90"]], right[["prediction", "lower90", "upper90"]],
                        rtol=1e-9, atol=1e-12), "Submission predictions disagree with the detailed model output.")

    with (ROOT / "submission" / "campaign.yaml").open(encoding="utf-8") as handle:
        campaign = yaml.safe_load(handle)
    require(isinstance(campaign, dict) and set(campaign) == {"rationale", "experiments"}, "Invalid campaign.yaml top-level schema.")
    require(isinstance(campaign["rationale"], str) and campaign["rationale"].strip()
            and "placeholder" not in campaign["rationale"].lower(), "Missing campaign rationale.")
    expected_keys = {"run_id", "temperature_C", "A0_M", "B0_M", "sample_times_min", "assay"}
    require(all(set(row) == expected_keys for row in campaign["experiments"]), "A campaign experiment has unexpected or missing fields.")
    budget = designer.audit_budget(campaign["experiments"])

    # Validate saved provenance, accepting only LF/CRLF representation changes
    # when comparing source files cloned between Windows and Linux.
    for name in ["system_B_prediction_summary.json", "system_B_interval_cv_summary.json", "system_B_campaign_summary.json",
                 "system_A_selection_summary.json"]:
        report = read_json(name)
        for relative, digest in report["source_sha256"].items():
            require((ROOT / relative).is_file() and predictor.matches_source(ROOT / relative, digest),
                    f"{name} is stale relative to {relative}; regenerate its outputs.")
        for relative, digest in report.get("artifact_sha256", {}).items():
            require((ROOT / relative).is_file() and predictor.matches_source(ROOT / relative, digest),
                    f"{relative} changed after {name} was written; regenerate its outputs.")
    forecast = read_json("system_B_prediction_summary.json")
    require(positive_weights == forecast["model_weights"], "Saved prediction weights disagree with hypotheses.yaml.")
    require(predictor.matches_source(RESULTS / "system_B_fits.json", forecast["fit_report_sha256"]),
            "Predictions were made with a different saved full-data fit.")
    interval_report = read_json("system_B_interval_cv_summary.json")
    require(predictor.matches_source(RESULTS / "system_B_cv_summary.json", interval_report["cv_report_sha256"]),
            "Interval validation used a different saved set of cross-validation fits.")
    design_report = read_json("system_B_campaign_summary.json")
    require(all(positive_weights == row["model_comparison_weights"] for row in design_report["simulation_forecasts"]),
            "Campaign simulation weights disagree with hypotheses.yaml.")
    require(budget == design_report["budget"], "The submitted campaign and saved budget report disagree.")

    for name in FIGURES:
        path = ROOT / "submission" / "figures" / name
        require(path.is_file(), f"Missing presentation figure {name}; run just replicate.")
        header = path.read_bytes()[:24]
        require(len(header) == 24 and header[:8] == b"\x89PNG\r\n\x1a\n", f"{name} is not a PNG file.")
        width, height = struct.unpack(">II", header[16:24])
        require(width >= 100 and height >= 100, f"{name} has invalid dimensions.")
    return {"hypotheses": len(ranking), "weight_sum": sum(weights.values()), "predictive_weights": weights,
            "prediction_rows": len(predictions), "heldout_runs": int(predictions.run_id.nunique()),
            "campaign_budget": budget, "required_figures": len(FIGURES),
            "system_A_reference_selected_folds": reference_summary["reference_selected_folds"],
            "system_A_full_data_AICc_leader": expected_leader,
            "pinned_dependencies": dependencies, "saved_fitting_files": len(FIT_FILES)}


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--run-workflow", action="store_true", help="run and time the four required just recipes before checking")
    args = parser.parse_args()
    try:
        timings, elapsed = run_workflow() if args.run_workflow else ([], None)
        summary = check_files()
        summary.update({"all_file_checks_passed": True, "workflow_commands": timings,
                        "total_workflow_seconds": elapsed,
                        "within_ten_minutes": elapsed < 600 if elapsed is not None else None,
                        "python_version": sys.version.split()[0],
                        "scope": "working-copy file and workflow checks; not a scientific-validation guarantee or a completed presentation"})
        RESULTS.mkdir(exist_ok=True)
        (RESULTS / "submission_check.json").write_text(json.dumps(summary, indent=2, allow_nan=False)+"\n", encoding="utf-8")
        print("\nSubmission file checks passed.")
        print(f"Hypotheses: {summary['hypotheses']} unique candidates; weights sum to {summary['weight_sum']:.1f}.")
        print("M2/M4 weights match predictions and campaign design.")
        print(f"System A: all 6 candidates and 30 validation fits checked; training AICc selected M1 in {summary['system_A_reference_selected_folds']}/5 folds.")
        print(f"Predictions: {summary['prediction_rows']} complete, unique rows across {summary['heldout_runs']} runs; intervals valid.")
        budget = summary["campaign_budget"]
        print(f"Campaign: {budget['experiments']} experiments, {budget['A_consumed_g']:.3f}/5 g A, {budget['analytical_samples']}/60 samples; all limits passed.")
        print(f"Figures: all {summary['required_figures']} required PNGs found.")
        print("Pinned dependencies, saved-fit files and output provenance checked.")
        if elapsed is None:
            print("Workflow execution and runtime were not measured in this check.")
        else:
            for row in timings:
                print(f"  {row['command']}: {row['seconds']:.1f} seconds; passed")
            print(f"Total workflow time, including install: {elapsed:.1f} seconds (limit 600).")
            require(elapsed < 600, "The recipes completed, but exceeded the ten-minute limit on this machine.")
        print("Saved results/submission_check.json.")
        print("Keep analysis scripts and results/ with the submission; a clean clone of your own repository is the final packaging check.")
    except (ValueError, KeyError, FileNotFoundError, ImportError, TypeError) as error:
        print(f"\nSubmission check FAILED: {error}", file=sys.stderr)
        raise SystemExit(1) from error


if __name__ == "__main__":
    main()
