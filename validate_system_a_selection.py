"""Six-candidate reference check using only the released System A measurements.

Run once: python validate_system_a_selection.py
Regenerate checked scores/figures: python validate_system_a_selection.py --figures-only

Each candidate is fitted to all five experiments and to each four-experiment
training partition, from four fixed starts. In every partition the model with
the smallest TRAINING AICc is selected before the excluded run is scored.
The disclosed answer is used only to assess the results, never to select a
candidate, initialize parameters or break a tie. This is a retrospective
known-answer benchmark, not an independently blinded discovery exercise.

The likelihood is Gaussian with SD=2% of the model mean for A/B/P and 5% for Q.
Every candidate uses the same positive-time observations. There is no absolute
assay-noise floor, censoring, deletion of outliers or reuse of System B noise.
Numerical protection at 1e-12 M is checked to be inactive at accepted fits.

For efficiency, the exact likelihood is represented by signed deviance
residuals. For an observation y>0 and relative SD r, the unrestricted MLE of
its mean is mu*=2*y/(1+sqrt(1+4*r*r)). Write a=y/mu* and d=mu*/mu-1. Then
NLL(mu)-NLL(mu*) = d-log(1+d) + a*a*d*d/(2*r*r).
Thus half the residual sum of squares equals NLL minus a data-only constant.
This retains log(SD); ordinary weighted least squares would not be equivalent.

Fixed-candidate CV scores and training-selected CV scores are reported
separately. Scores include assay noise only, not parameter uncertainty; they
do not validate 90% intervals. AICc is approximate for nonlinear models and
particularly tentative at boundaries or weakly identified limiting cases.
Four starts do not certify a global optimum. All output is separate from B.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import time
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.optimize import least_squares

import fit_system_a as reference
from candidates import CANDIDATES, simulate

ROOT = Path(__file__).resolve().parent
RESULTS = ROOT / "results"
FIGURES = ROOT / "submission" / "figures"
FIT_FILE = "system_A_selection_fits.json"
SUMMARY_FILE = "system_A_selection_summary.json"
FIGURE_NAMES = ("system_A_selection_comparison.png", "system_A_selection_folds.png")
MODELS = tuple(CANDIDATES)
SPECIES = tuple(reference.SPECIES)
RELATIVE_SD = np.array([0.02, 0.02, 0.02, 0.05])
MEAN_PROTECTION = 1e-12
# Saved-fit verification only: allow up to 0.001 in the SUMMED NLL when the
# same parameters are reintegrated on another platform. The Windows/Linux
# diagnostic found differences up to 0.000406 across all 36 fits. This fixed
# absolute bound corresponds to at most 0.002 in an individual AICc score;
# it does not alter the likelihood, optimiser tolerances or assay-noise model.
SAVED_NLL_ATOL = 1e-3
# Broad search assumptions, in each reaction's own rate units at 75 C.
# The high ceiling allows an intermediate to turn over faster than sampling.
RATE_BOUNDS = (1e-9, 1e3)
EA_BOUNDS = (0.0, 180.0)
BASE_RATES = {
    "M1": [0.01, 0.001], "M2": [0.02, 0.1, 0.01],
    "M3": [0.01, 0.001], "M4": [0.02, 0.1, 0.01],
    "M5": [0.01, 0.001], "M6": [0.02, 0.1, 0.01],
}
SOURCE_FILES = [
    "data/system_A/measurements.csv", "fit_system_a.py",
    "validate_system_a_selection.py", "src/candidates/__init__.py",
    "src/candidates/models.py", "src/candidates/simulate.py",
]


def require(condition, message):
    if not condition:
        raise ValueError(message)


def digest(path):
    # Git's normal LF/CRLF conversion must not invalidate a Windows clone.
    return hashlib.sha256(Path(path).read_bytes().replace(b"\r\n", b"\n")).hexdigest()


def source_manifest():
    return {name: digest(ROOT / name) for name in SOURCE_FILES}


def parameters(model, x):
    return {name: float(10**value if name.startswith("k") else 50*value)
            for name, value in zip(CANDIDATES[model].param_names, x)}


def coordinates(model, params):
    return np.array([np.log10(params[name]) if name.startswith("k") else params[name]/50
                     for name in CANDIDATES[model].param_names])


def parameter_bounds(model):
    names = CANDIDATES[model].param_names
    return tuple(np.array([np.log10(rate) if name.startswith("k") else ea/50
                           for name in names])
                 for rate, ea in zip(RATE_BOUNDS, EA_BOUNDS))


def starting_points(model):
    # Fixed independently of observations, full-data fits and the reference answer.
    variations = [([0, 0, 0], [50, 50, 50]),
                  ([-0.8, 0.6, -0.4], [35, 85, 55]),
                  ([0.7, -0.8, 0.8], [95, 40, 100]),
                  ([-0.2, 1.7, 1.7], [70, 80, 90])]
    return [np.array([item for j, rate in enumerate(BASE_RATES[model])
                      for item in (np.log10(rate)+offsets[j], energies[j]/50)])
            for offsets, energies in variations]


def likelihood_terms(y, mean, relative):
    protected = np.maximum(mean, MEAN_PROTECTION)
    sigma = relative*protected
    return 0.5*((y-protected)/sigma)**2 + np.log(sigma) + 0.5*np.log(2*np.pi)


def deviance_residuals(y, mean, relative):
    a = (1+np.sqrt(1+4*relative**2))/2
    d = y/(a*np.maximum(mean, MEAN_PROTECTION))-1
    # Stable evaluation of (d-log1p(d))/d^2 near the exact minimum.
    near = np.abs(d) < 1e-3
    factor = np.empty_like(d)
    dn = d[near]
    factor[near] = 0.5-dn/3+dn**2/4-dn**3/5+dn**4/6-dn**5/7
    factor[~near] = (d[~near]-np.log1p(d[~near]))/d[~near]**2
    return d*np.sqrt((a/relative)**2+2*factor)


class FitProblem:
    def __init__(self, model, runs):
        self.model, self.runs = model, runs
        self.y = np.concatenate([run["values"][run["use"]].ravel() for run in runs])
        self.relative = np.tile(RELATIVE_SD, len(self.y)//len(SPECIES))
        require(len(self.y) > 0 and np.all(self.y > 0), "System A needs positive numeric measurements.")
        best_mean = 2*self.y/(1+np.sqrt(1+4*self.relative**2))
        self.constant = float(likelihood_terms(self.y, best_mean, self.relative).sum())
        self.failed_evaluations = 0

    def means(self, x):
        params = parameters(self.model, x)
        values = np.concatenate([
            simulate(self.model, run["condition"], params)[list(SPECIES)].to_numpy(float)[run["use"]].ravel()
            for run in self.runs
        ])
        if not np.isfinite(values).all() or (values < 0).any():
            raise FloatingPointError("Non-finite or negative ODE prediction.")
        return values

    def residuals(self, x):
        try:
            return deviance_residuals(self.y, self.means(x), self.relative)
        except (RuntimeError, FloatingPointError, OverflowError):
            self.failed_evaluations += 1
            return np.full(len(self.y), 1e15)

    def nll(self, x):
        return float(likelihood_terms(self.y, self.means(x), self.relative).sum())


def fit_model(model, runs):
    problem = FitProblem(model, runs)
    lower, upper = parameter_bounds(model)
    records, solutions = [], []
    for number, start in enumerate(starting_points(model), start=1):
        started = time.perf_counter()
        solution = least_squares(problem.residuals, start, bounds=(lower, upper),
                                 method="trf", diff_step=1e-4, x_scale="jac", max_nfev=600,
                                 ftol=1e-9, xtol=1e-9, gtol=1e-5)
        nfev = int(solution.nfev)
        continued = False
        if not solution.success:
            # The continuation still uses exactly this training partition.
            continued = True
            solution = least_squares(problem.residuals, solution.x, bounds=(lower, upper),
                                     method="trf", diff_step=1e-4, x_scale="jac", max_nfev=1200,
                                     ftol=1e-9, xtol=1e-9, gtol=1e-5)
            nfev += int(solution.nfev)
        nll = problem.nll(solution.x)
        require(np.isclose(nll, solution.cost+problem.constant, rtol=1e-9, atol=1e-6),
                "Direct likelihood and deviance objective disagree.")
        successful = bool(solution.success and np.isfinite(nll) and nll < 1e25)
        records.append({"start": number, "success": successful, "negative_log_likelihood": nll,
                        "nfev": nfev, "continued": continued, "optimality": float(solution.optimality),
                        "message": str(solution.message), "elapsed_seconds": time.perf_counter()-started,
                        "parameters": parameters(model, solution.x)})
        print(f"  Start {number}/4: NLL={nll:.4f}; converged={successful}; "
              f"{records[-1]['elapsed_seconds']:.1f} s", flush=True)
        if successful:
            solutions.append((nll, solution))
    require(bool(solutions), f"No {model} start converged; investigate the optimisation.")
    nll, best = min(solutions, key=lambda pair: pair[0])
    require(not any(row["negative_log_likelihood"] < nll-0.01 for row in records),
            f"{model} has a better unfinished start; investigate before ranking.")
    require(np.all(problem.means(best.x) > MEAN_PROTECTION), "Numerical mean protection affected a fitted result.")
    flags = np.isclose(best.x, lower, rtol=0, atol=1e-3) | np.isclose(best.x, upper, rtol=0, atol=1e-3)
    successful_nll = [row["negative_log_likelihood"] for row in records if row["success"]]
    return {"parameters": parameters(model, best.x), "negative_log_likelihood": nll,
            "training_run_ids": [run["id"] for run in runs], "n_observations": len(problem.y),
            "n_parameters": len(best.x), "successful_starts": len(successful_nll),
            "parameters_at_bounds": [name for name, flag in zip(CANDIDATES[model].param_names, flags) if flag],
            "successful_start_nll_span": float(max(successful_nll)-min(successful_nll)),
            "optimality": float(best.optimality), "failed_simulation_evaluations": problem.failed_evaluations,
            "starts": records}


def aicc(fit):
    n, k = fit["n_observations"], fit["n_parameters"]
    require(n > k+1, "Too few observations for AICc.")
    return float(2*fit["negative_log_likelihood"]+2*k+2*k*(k+1)/(n-k-1))


def scored_predictions(model, run, params):
    values = simulate(model, run["condition"], params)[list(SPECIES)].to_numpy(float)
    rows = []
    for index in np.flatnonzero(run["use"]):
        for column, species in enumerate(SPECIES):
            mean = float(values[index, column])
            require(mean > MEAN_PROTECTION, "Numerical mean protection affected a scored prediction.")
            observed = float(run["values"][index, column])
            sd = float(RELATIVE_SD[column]*mean)
            rows.append({"model": model, "run_id": run["id"], "time_min": float(run["times"][index]),
                         "species": species, "observed_M": observed, "prediction_M": mean,
                         "measurement_sd_M": sd, "standardized_error": (observed-mean)/sd,
                         "negative_log_likelihood": float(likelihood_terms(observed, mean, RELATIVE_SD[column]))})
    return pd.DataFrame(rows)


def check_prediction_keys(runs, predictions):
    expected = {(model, run["id"], float(run["times"][i]), species)
                for model in MODELS for run in runs for i in np.flatnonzero(run["use"]) for species in SPECIES}
    keys = list(predictions[["model", "run_id", "time_min", "species"]].itertuples(index=False, name=None))
    require(len(keys) == len(expected) and set(keys) == expected,
            "Missing or duplicated System A model/observation predictions.")


def checked_fit(model, training, fit):
    require(set(fit["training_run_ids"]) == {run["id"] for run in training}
            and len(fit["training_run_ids"]) == len(training), "Saved training partition is inconsistent.")
    problem = FitProblem(model, training)
    require(fit["n_observations"] == len(problem.y), "Saved observation count is inconsistent.")
    require(fit["n_parameters"] == len(CANDIDATES[model].param_names), "Saved parameter count is inconsistent.")
    actual = problem.nll(coordinates(model, fit["parameters"]))
    saved = fit["negative_log_likelihood"]
    require(np.isfinite(actual) and np.isfinite(saved)
            and np.isclose(actual, saved, rtol=0, atol=SAVED_NLL_ATOL),
            f"Saved {model} likelihood disagrees for training runs "
            f"{', '.join(run['id'] for run in training)}: "
            f"saved={saved:.12g}, current={actual:.12g}, difference={actual-saved:+.6g}, "
            f"allowed absolute difference={SAVED_NLL_ATOL:g}. "
            "Refit System A selection if source/data or fitted parameters changed.")
    starts = fit["starts"]
    require(len(starts) == 4 and {row["start"] for row in starts} == {1, 2, 3, 4}, "Invalid starting-fit records.")
    good = [row["negative_log_likelihood"] for row in starts if row["success"]]
    # These two values come from the same fitting run and JSON file. Unlike
    # reintegration across platforms, this comparison needs no tolerance.
    require(bool(good) and saved == min(good),
            "Saved selected fit disagrees with successful starting fits.")


def checked_report():
    path = RESULTS / FIT_FILE
    require(path.is_file(), "Run python validate_system_a_selection.py first and keep its results/ files.")
    report = json.loads(path.read_text(encoding="utf-8"))
    require(report["source_sha256"] == source_manifest(),
            "System A selection source/data changed; rerun python validate_system_a_selection.py.")
    require(set(report["full_fits"]) == set(MODELS) and set(report["fold_fits"]) == set(MODELS),
            "Saved reference comparison must include all six candidates.")
    runs = reference.load_runs()
    for model in MODELS:
        checked_fit(model, runs, report["full_fits"][model])
        require(set(report["fold_fits"][model]) == {run["id"] for run in runs}, "Saved folds are incomplete.")
        for held_out in runs:
            training = [run for run in runs if run["id"] != held_out["id"]]
            checked_fit(model, training, report["fold_fits"][model][held_out["id"]])
    return report, runs


def analyse(report, runs):
    full_rows, fold_rows, predictions = [], [], []
    for model in MODELS:
        fit = report["full_fits"][model]
        full_rows.append({"model": model, "training_NLL": fit["negative_log_likelihood"],
                          "training_AICc": aicc(fit), "n_parameters": fit["n_parameters"],
                          "full_fit_bounds": ",".join(fit["parameters_at_bounds"]) or "none",
                          "full_fit_start_NLL_span": fit["successful_start_nll_span"]})
        for held_out in runs:
            fit = report["fold_fits"][model][held_out["id"]]
            table = scored_predictions(model, held_out, fit["parameters"])
            table["training_run_ids"] = ",".join(fit["training_run_ids"])
            predictions.append(table)
            fold_rows.append({"model": model, "heldout_run_id": held_out["id"],
                              "training_run_ids": ",".join(fit["training_run_ids"]),
                              "training_AICc": aicc(fit), "heldout_NLL": float(table.negative_log_likelihood.sum()),
                              "heldout_standardized_RMS": float(np.sqrt(np.mean(table.standardized_error**2))),
                              "n_heldout_observations": len(table),
                              "parameters_at_bounds": ",".join(fit["parameters_at_bounds"]) or "none",
                              "start_NLL_span": fit["successful_start_nll_span"]})
    predictions = pd.concat(predictions, ignore_index=True)
    check_prediction_keys(runs, predictions)
    folds = pd.DataFrame(fold_rows)
    folds["delta_training_AICc"] = folds.training_AICc-folds.groupby("heldout_run_id").training_AICc.transform("min")
    selections, selected_predictions = [], []
    for run in runs:
        partition = folds[folds.heldout_run_id.eq(run["id"])].sort_values(["training_AICc", "model"])
        winner = partition.iloc[0]
        near = partition[partition.delta_training_AICc <= 2].model.tolist()
        selections.append({"heldout_run_id": run["id"], "selected_model": winner.model,
                           "training_run_ids": winner.training_run_ids,
                           "models_within_2_training_AICc": ",".join(near),
                           "runner_up_training_AICc_gap": float(partition.iloc[1].training_AICc-winner.training_AICc),
                           "heldout_NLL": float(winner.heldout_NLL),
                           "heldout_standardized_RMS": float(winner.heldout_standardized_RMS)})
        selected_predictions.append(predictions[predictions.run_id.eq(run["id"]) & predictions.model.eq(winner.model)])
    selected = pd.concat(selected_predictions, ignore_index=True)
    selected_keys = selected[["run_id", "time_min", "species"]]
    require(len(selected) == sum(int(run["use"].sum())*4 for run in runs)
            and not selected_keys.duplicated().any(), "Selected-model predictions are incomplete or duplicated.")
    selection_table = pd.DataFrame(selections)
    comparison = pd.DataFrame(full_rows)
    for index, row in comparison.iterrows():
        subset = predictions[predictions.model.eq(row.model)]
        model_folds = folds[folds.model.eq(row.model)]
        comparison.loc[index, "heldout_NLL"] = subset.negative_log_likelihood.sum()
        comparison.loc[index, "heldout_standardized_RMS"] = np.sqrt(np.mean(subset.standardized_error**2))
        comparison.loc[index, "folds_selected_by_training_AICc"] = int(selection_table.selected_model.eq(row.model).sum())
        comparison.loc[index, "folds_with_bound_hits"] = int(model_folds.parameters_at_bounds.ne("none").sum())
        comparison.loc[index, "folds_with_distinct_minima"] = int(model_folds.start_NLL_span.gt(0.1).sum())
    comparison["delta_training_AICc"] = comparison.training_AICc-comparison.training_AICc.min()
    comparison["delta_heldout_NLL"] = comparison.heldout_NLL-comparison.heldout_NLL.min()
    comparison = comparison.sort_values(["training_AICc", "model"]).reset_index(drop=True)
    comparison.insert(0, "training_AICc_rank", np.arange(1, len(comparison)+1))
    summary = {
        "system": "A", "reference_model": "M1", "reference_used_for_selection": False,
        "selection_rule": "lowest training AICc in each four-experiment partition; model ID breaks exact ties",
        "n_models": len(MODELS), "n_folds": len(runs), "n_full_fits": len(MODELS),
        "n_fold_fits": len(MODELS)*len(runs), "n_positive_time_measurements": len(selected),
        "n_fixed_candidate_predictions": len(predictions),
        "full_data_AICc_leader": comparison.iloc[0].model,
        "fixed_candidate_CV_NLL_leader": comparison.sort_values("heldout_NLL").iloc[0].model,
        "reference_selected_folds": int(selection_table.selected_model.eq("M1").sum()),
        "selection_CV_NLL": float(selected.negative_log_likelihood.sum()),
        "selection_CV_standardized_RMS": float(np.sqrt(np.mean(selected.standardized_error**2))),
        "comparison": comparison.to_dict(orient="records"), "fold_selections": selections,
        "elapsed_fitting_seconds": report["elapsed_fitting_seconds"],
        "source_sha256": report["source_sha256"], "fit_report_sha256": digest(RESULTS / FIT_FILE),
        "limitations": [
            "Retrospective benchmark against a disclosed mechanism, not a blinded discovery experiment.",
            "Only five experiments; folds share training data and are not independent repetitions.",
            "AICc uses nominal parameter counts and is approximate, especially at boundaries or weak identifiability.",
            "Reported CV likelihoods use point parameter estimates and assay noise only; no interval coverage is established.",
            "All observations, including the large A4 discrepancies, are retained.",
            "Fast-intermediate mechanisms can approximate direct routes at finite sampling resolution.",
            "Four starting fits and boundary diagnostics do not prove a global optimum.",
            "This reference check neither determines nor changes the mechanism weights or deliverables for System B.",
        ],
    }
    return summary, comparison, folds, selection_table, predictions, selected


def make_figures(comparison, folds, selection_table):
    FIGURES.mkdir(parents=True, exist_ok=True)
    plt.rcParams.update({"font.size": 10})
    fig, ax = plt.subplots(figsize=(12.8, 4.7))
    ax.axis("off")
    cells = [[row.model, f"{row.delta_training_AICc:.2f}", f"{row.heldout_NLL:.2f}",
              f"{row.delta_heldout_NLL:.2f}", f"{row.heldout_standardized_RMS:.3f}",
              str(int(row.folds_selected_by_training_AICc)), str(int(row.folds_with_bound_hits))]
             for row in comparison.itertuples()]
    table = ax.table(cellText=cells, colLabels=["Model", "Full-data\nDelta AICc", "Held-out\nNLL",
                     "Held-out\nDelta NLL", "Held-out\nRMS / assay SD", "Folds selected\nby training AICc", "Folds with\nbound hits"],
                     cellLoc="center", loc="center")
    table.auto_set_font_size(False)
    table.set_fontsize(10)
    table.scale(1, 2.2)
    for (row, _), cell in table.get_celld().items():
        if row == 0:
            cell.set_facecolor("#DCEBF2")
        elif row % 2:
            cell.set_facecolor("#F2F5F7")
    fig.suptitle("System A: comparison of all six mechanisms", fontsize=16, y=0.97)
    ax.set_title("244 positive-time measurements per candidate; each experiment excluded once", fontsize=11, pad=8)
    fig.text(0.5, 0.055, "Lower NLL/AICc is better. Disclosed reference: M1. AICc is approximate; inspect boundary and start diagnostics.",
             ha="center", fontsize=9)
    fig.text(0.5, 0.018, "Training AICc makes each fold's selection; excluded measurements only score it. Scores do not establish interval coverage.",
             ha="center", fontsize=9)
    fig.tight_layout(rect=(0.015, 0.11, 0.985, 0.90))
    fig.savefig(FIGURES / FIGURE_NAMES[0], dpi=160)
    plt.close(fig)

    run_ids = selection_table.heldout_run_id.tolist()
    matrix = folds.pivot(index="model", columns="heldout_run_id", values="delta_training_AICc").loc[list(MODELS), run_ids]
    fig, axes = plt.subplots(1, 2, figsize=(12.8, 5.2), gridspec_kw={"width_ratios": [1.05, 1]})
    chart = axes[0].imshow(np.log10(1+matrix.to_numpy()), aspect="auto", cmap="YlOrRd")
    axes[0].set_xticks(range(len(run_ids)), run_ids)
    axes[0].set_yticks(range(len(MODELS)), MODELS)
    axes[0].set_xlabel("Excluded experiment")
    axes[0].set_title("Model choice uses training data only")
    for i in range(len(MODELS)):
        for j in range(len(run_ids)):
            value = float(matrix.iloc[i, j])
            color = "white" if chart.norm(np.log10(1+value)) > 0.65 else "black"
            axes[0].text(j, i, f"{value:.1f}", ha="center", va="center", fontsize=9, color=color)
    fig.colorbar(chart, ax=axes[0], label="log10(1 + training Delta AICc)", shrink=0.8)
    axes[1].axis("off")
    cells = [[row.heldout_run_id, row.selected_model, f"{row.runner_up_training_AICc_gap:.2f}",
              f"{row.heldout_standardized_RMS:.3f}"] for row in selection_table.itertuples()]
    table = axes[1].table(cellText=cells, colLabels=["Excluded\nrun", "Selected\nmodel", "Runner-up\nAICc gap", "Held-out\nRMS / SD"],
                          cellLoc="center", loc="center")
    table.auto_set_font_size(False)
    table.set_fontsize(10)
    table.scale(1, 2.1)
    axes[1].set_title("Then score the selected model")
    fig.suptitle("System A: validate the selection procedure", fontsize=16)
    fig.text(0.5, 0.025, "Reference M1 was disclosed. Five folds share training data; selection frequency is not a probability of mechanism identity.",
             ha="center", fontsize=9)
    fig.tight_layout(rect=(0, 0.07, 1, 0.93))
    fig.savefig(FIGURES / FIGURE_NAMES[1], dpi=160)
    plt.close(fig)


def regenerate():
    report, runs = checked_report()
    summary, comparison, folds, selection_table, predictions, selected = analyse(report, runs)
    comparison.to_csv(RESULTS / "system_A_selection_comparison.csv", index=False)
    folds.to_csv(RESULTS / "system_A_selection_folds.csv", index=False)
    selection_table.to_csv(RESULTS / "system_A_selection_choices.csv", index=False)
    predictions.to_csv(RESULTS / "system_A_selection_predictions.csv", index=False)
    selected.to_csv(RESULTS / "system_A_selection_chosen_predictions.csv", index=False)
    diagnostic = predictions[predictions.model.eq("M1")].copy()
    diagnostic["absolute_standardized_error"] = diagnostic.standardized_error.abs()
    diagnostic.sort_values("absolute_standardized_error", ascending=False).head(12).to_csv(
        RESULTS / "system_A_selection_largest_errors.csv", index=False)
    starts = []
    for model in MODELS:
        for scope, fit in [("all", report["full_fits"][model]), *report["fold_fits"][model].items()]:
            for row in fit["starts"]:
                record = {"model": model, "excluded_run": scope, **{k: v for k, v in row.items() if k != "parameters"},
                          **row["parameters"]}
                starts.append(record)
    pd.DataFrame(starts).to_csv(RESULTS / "system_A_selection_starts.csv", index=False)
    make_figures(comparison, folds, selection_table)
    summary["artifact_sha256"] = {str(path.relative_to(ROOT)).replace("\\", "/"): digest(path)
                                  for path in sorted(RESULTS.glob("system_A_selection_*.csv"))}
    (RESULTS / SUMMARY_FILE).write_text(json.dumps(summary, indent=2, allow_nan=False)+"\n", encoding="utf-8")
    print("\nSystem A model-selection validation complete.")
    print(f"Compared 6 full-data fits and 30 independently trained excluded-experiment fits.")
    print(f"Scored {summary['n_fixed_candidate_predictions']} observations: 244 per candidate, with no missing or duplicate predictions.")
    columns = ["model", "delta_training_AICc", "heldout_NLL", "delta_heldout_NLL", "heldout_standardized_RMS",
               "folds_selected_by_training_AICc", "folds_with_bound_hits"]
    print(comparison[columns].to_string(index=False, float_format=lambda x: f"{x:.4f}"))
    print(f"\nFull-data AICc leader: {summary['full_data_AICc_leader']}.")
    print(f"Lowest fixed-candidate held-out NLL: {summary['fixed_candidate_CV_NLL_leader']}.")
    print(f"Training-only selection chose disclosed M1 in {summary['reference_selected_folds']}/5 folds.")
    print(f"Selected-model held-out standardized RMS: {summary['selection_CV_standardized_RMS']:.3f}.")
    print("AICc is approximate; boundary fits and fast-intermediate limits need cautious interpretation.")
    print("All measurements were retained, including the large A4 discrepancies. No prediction intervals were assessed here.")
    print("This is a known-answer reference check; System B results and deliverables are unchanged.")
    print(f"Saved results/{FIT_FILE}, summary, seven diagnostic CSVs and two new figures.")
    return summary


def run_analysis():
    started = time.perf_counter()
    manifest = source_manifest()
    runs = reference.load_runs()
    require(len(runs) == 5, "Expected five released System A experiments.")
    report = {"system": "A", "source_sha256": manifest,
              "noise_assumption": "Independent Gaussian assay errors, SD=0.02*mean for A/B/P and 0.05*mean for Q",
              "rate_bounds": RATE_BOUNDS, "activation_energy_bounds_kJ_mol": EA_BOUNDS,
              "full_fits": {}, "fold_fits": {model: {} for model in MODELS}}
    checkpoint = ROOT / "scratch" / "system_A_selection_checkpoint.json"
    if checkpoint.is_file():
        saved = json.loads(checkpoint.read_text(encoding="utf-8"))
        if saved.get("source_sha256") == manifest:
            report = saved
            print("Resuming completed fits from the same source and data.", flush=True)
    previous_seconds = report.get("elapsed_fitting_seconds", 0.0)

    def save_checkpoint():
        report["elapsed_fitting_seconds"] = previous_seconds+time.perf_counter()-started
        checkpoint.parent.mkdir(exist_ok=True)
        checkpoint.write_text(json.dumps(report, indent=2, allow_nan=False)+"\n", encoding="utf-8")

    # Fit folds first: none can read any full-data fit or ranking.
    for held_out in runs:
        training = [run for run in runs if run["id"] != held_out["id"]]
        print(f"\nExclude {held_out['id']}; fit only {', '.join(run['id'] for run in training)}", flush=True)
        for model in MODELS:
            print(f" {model}", flush=True)
            if held_out["id"] in report["fold_fits"][model]:
                checked_fit(model, training, report["fold_fits"][model][held_out["id"]])
                print("  Reused completed, checked starting fits.", flush=True)
            else:
                report["fold_fits"][model][held_out["id"]] = fit_model(model, training)
                save_checkpoint()
    print("\nFull-data fits for the separate descriptive AICc comparison", flush=True)
    for model in MODELS:
        print(f" {model}", flush=True)
        if model in report["full_fits"]:
            checked_fit(model, runs, report["full_fits"][model])
            print("  Reused completed, checked starting fits.", flush=True)
        else:
            report["full_fits"][model] = fit_model(model, runs)
            save_checkpoint()
    report["elapsed_fitting_seconds"] = previous_seconds+time.perf_counter()-started
    require(manifest == source_manifest(), "Source/data changed during fitting; rerun with stable files.")
    RESULTS.mkdir(exist_ok=True)
    (RESULTS / FIT_FILE).write_text(json.dumps(report, indent=2, allow_nan=False)+"\n", encoding="utf-8")
    summary = regenerate()
    checkpoint.unlink(missing_ok=True)
    print(f"Fitting and output work took {time.perf_counter()-started:.1f} seconds on this computer.")
    return summary


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--figures-only", action="store_true", help="check saved fits and regenerate scores/figures without optimization")
    args = parser.parse_args()
    if args.figures_only:
        regenerate()
    else:
        run_analysis()


if __name__ == "__main__":
    main()
