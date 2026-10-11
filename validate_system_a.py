"""Predict entire System A experiments that were excluded from fitting.

Put this file beside fit_system_a.py and run: python validate_system_a.py

For each of the five runs, fit the disclosed M1 model to the OTHER four runs
using the existing fitter and its fixed starting guesses. Then predict the
excluded run from its temperature and initial concentrations. No saved fit
parameters or full-data residuals are used to train these models.

All positive-time observations are scored exactly once. Time-zero observations
are plotted but excluded from fitting and scoring because their predictions
are fixed by the specified initial conditions.

The standardized errors use the System A measurement-noise assumption only:
SD=2% of predicted concentration for A/B/P and 5% for Q. They do NOT include
parameter uncertainty and are NOT a test of calibrated prediction intervals.
The five-fold design also does not identify the mechanism of System B.
"""

from __future__ import annotations

import json
import time
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

try:
    import fit_system_a as reference
except ModuleNotFoundError as error:
    if error.name != "fit_system_a":
        raise
    raise SystemExit(
        "Save validate_system_a.py beside fit_system_a.py in the main project folder."
    ) from error

ROOT = Path(__file__).resolve().parent
RESULTS = ROOT / "results"
FIGURES = ROOT / "submission" / "figures"
SPECIES = reference.SPECIES


def observation_predictions(run, params, training_ids):
    """Score only positive-time measurements, after a model has been fitted."""
    predicted = reference.predict_run(run, params)
    condition = run["condition"]
    rows = []
    for index in np.flatnonzero(run["use"]):
        for column, species in enumerate(SPECIES):
            mean = float(predicted[index, column])
            observed = float(run["values"][index, column])
            sigma = float(reference.RELATIVE_SD[column]*max(mean, 1e-12))
            rows.append({
                "run_id": run["id"], "training_run_ids": ",".join(training_ids),
                "temperature_C": condition.temperature_c,
                "A0_M": condition.A0, "B0_M": condition.B0,
                "time_min": float(run["times"][index]), "species": species,
                "observed_M": observed, "prediction_M": mean,
                "error_M": observed-mean, "measurement_sd_M": sigma,
                "standardized_error": (observed-mean)/sigma,
            })
    return pd.DataFrame(rows)


def scores(frame):
    errors = frame["error_M"].to_numpy(float)
    z = frame["standardized_error"].to_numpy(float)
    return {
        "n_observations": int(len(frame)),
        "rmse_M": float(np.sqrt(np.mean(errors**2))),
        "mean_error_M": float(np.mean(errors)),
        "standardized_rms": float(np.sqrt(np.mean(z**2))),
        "mean_standardized_error": float(np.mean(z)),
        "max_abs_standardized_error": float(np.max(np.abs(z))),
    }


def condition_ranges(training, held_out):
    """Flag marginal extrapolation; being in each range is not a convex-hull test."""
    records, outside = {}, []
    for attribute, label in [("temperature_c", "temperature_C"),
                             ("A0", "A0_M"), ("B0", "B0_M")]:
        values = [getattr(run["condition"], attribute) for run in training]
        low, high = float(min(values)), float(max(values))
        target = float(getattr(held_out["condition"], attribute))
        records[f"training_{label}_min"] = low
        records[f"training_{label}_max"] = high
        records[f"heldout_{label}"] = target
        if target < low or target > high:
            outside.append(label)
    records["outside_training_ranges"] = ",".join(outside) if outside else "none"
    return records


def validate(runs, fit_function=None):
    """Fit each training partition independently; score the withheld run afterward."""
    if fit_function is None:
        fit_function = reference.fit
    identifiers = [run["id"] for run in runs]
    if len(runs) < 3 or len(set(identifiers)) != len(identifiers):
        raise ValueError("Validation needs at least three uniquely identified runs.")
    prediction_tables, fold_rows, start_rows = [], [], []
    fold_parameters = {}

    for number, held_out in enumerate(runs, start=1):
        training = [run for run in runs if run["id"] != held_out["id"]]
        training_ids = [run["id"] for run in training]
        if held_out["id"] in training_ids or len(training) != len(runs)-1:
            raise RuntimeError("Invalid validation split.")
        print(f"\nFold {number}/{len(runs)}: predict {held_out['id']}; "
              f"fit {', '.join(training_ids)}", flush=True)

        # The fitter receives only this fold's training data. It uses fixed
        # starts from fit_system_a.py, never estimates from the full dataset.
        best, starts = fit_function(training)
        params = reference.parameters(best.x)
        fold_parameters[held_out["id"]] = params
        test_predictions = observation_predictions(held_out, params, training_ids)
        training_predictions = pd.concat(
            [observation_predictions(run, params, training_ids) for run in training],
            ignore_index=True,
        )
        test_scores = scores(test_predictions)
        training_scores = scores(training_predictions)
        successful = [row for row in starts if row["success"]]
        nlls = [row["negative_log_likelihood"] for row in successful]
        at_bound = bool(
            np.any(np.isclose(best.x, reference.LOWER, atol=1e-4, rtol=0)) or
            np.any(np.isclose(best.x, reference.UPPER, atol=1e-4, rtol=0))
        )
        fold_rows.append({
            "heldout_run_id": held_out["id"], "training_run_ids": ",".join(training_ids),
            "n_training_observations": training_scores["n_observations"],
            "n_heldout_observations": test_scores["n_observations"],
            "training_standardized_rms": training_scores["standardized_rms"],
            "heldout_standardized_rms": test_scores["standardized_rms"],
            "heldout_rmse_M": test_scores["rmse_M"],
            "heldout_mean_error_M": test_scores["mean_error_M"],
            "successful_starts": len(successful), "attempted_starts": len(starts),
            "successful_start_nll_span": float(max(nlls)-min(nlls)),
            "parameter_at_search_bound": at_bound,
            **params, **condition_ranges(training, held_out),
        })
        start_rows.extend({"heldout_run_id": held_out["id"], **row} for row in starts)
        prediction_tables.append(test_predictions)
        print(f"  Training standardized RMS: {training_scores['standardized_rms']:.3f}; "
              f"held-out RMS: {test_scores['standardized_rms']:.3f}", flush=True)
        if at_bound:
            print("  NOTE: a parameter reached a search bound; investigate this fold.", flush=True)

    predictions = pd.concat(prediction_tables, ignore_index=True)
    check_prediction_keys(runs, predictions)
    return predictions, pd.DataFrame(fold_rows), pd.DataFrame(start_rows), fold_parameters


def check_prediction_keys(runs, predictions):
    """Verify that every eligible original observation was scored exactly once."""
    expected = {
        (run["id"], float(run["times"][index]), species)
        for run in runs for index in np.flatnonzero(run["use"]) for species in SPECIES
    }
    actual = list(predictions[["run_id", "time_min", "species"]].itertuples(index=False, name=None))
    if len(actual) != len(set(actual)) or set(actual) != expected:
        raise RuntimeError("Cross-validation predictions contain missing or duplicate observations.")
    for row in predictions.itertuples(index=False):
        if row.run_id in row.training_run_ids.split(","):
            raise RuntimeError("A held-out run appears in its own training set.")


def per_group_scores(predictions, column):
    return pd.DataFrame([
        {column: group_name, **scores(group)}
        for group_name, group in predictions.groupby(column, sort=True)
    ])


def make_figures(runs, fold_parameters, predictions, folds, by_species):
    FIGURES.mkdir(parents=True, exist_ok=True)
    plt.rcParams.update({"font.size": 9})
    fig, axes = plt.subplots(len(runs), len(SPECIES), figsize=(13, 2.15*len(runs)), squeeze=False)
    for row, run in enumerate(runs):
        condition = run["condition"]
        times = np.linspace(0, run["times"].max(), 260)
        curve = reference.simulate(
            "M1", reference.Conditions(condition.temperature_c, condition.A0, condition.B0, times),
            fold_parameters[run["id"]],
        )
        for column, species in enumerate(SPECIES):
            ax = axes[row, column]
            ax.plot(times, curve[species], color="#176B87", lw=1.6, label="Held-out prediction")
            ax.scatter(run["times"], run["values"][:, column], color="#D97941", s=15,
                       zorder=3, label="Excluded measurements")
            if row == 0:
                ax.set_title(species, fontweight="bold")
            if column == 0:
                ax.set_ylabel(f"{run['id']} held out ({condition.temperature_c:g} C)\nConcentration [mol/L]")
            if row == len(runs)-1:
                ax.set_xlabel("Time [min]")
            ax.grid(alpha=0.15)
    axes[0, 0].legend(fontsize=7)
    fig.suptitle("System A: predict each experiment using only the other four", fontsize=14)
    fig.tight_layout(rect=(0, 0, 1, 0.97))
    fig.savefig(FIGURES / "system_A_cv_predictions.png", dpi=160)
    plt.close(fig)

    fig, axes = plt.subplots(1, 2, figsize=(11.5, 4.8))
    x = np.arange(len(folds))
    axes[0].bar(x-0.19, folds.training_standardized_rms, width=0.38,
                color="#8BAAB6", label="Training runs")
    axes[0].bar(x+0.19, folds.heldout_standardized_rms, width=0.38,
                color="#D97941", label="Held-out run")
    axes[0].set_xticks(x, folds.heldout_run_id)
    axes[0].set_xlabel("Experiment excluded from fitting")
    axes[0].set_title("Training and held-out errors by fold")
    axes[0].legend(fontsize=8)
    values = by_species.set_index("species").loc[SPECIES, "standardized_rms"]
    axes[1].bar(SPECIES, values, color="#176B87", width=0.55)
    axes[1].set_title("Held-out errors by species")
    axes[1].set_xlabel("Species")
    maximum = max(float(folds.training_standardized_rms.max()),
                  float(folds.heldout_standardized_rms.max()), float(values.max()), 1.0)
    for ax in axes:
        ax.axhline(1, color="#666666", ls="--", lw=1)
        ax.set_ylim(0, maximum*1.25)
        ax.set_ylabel("RMS of error / assumed measurement SD")
        ax.grid(axis="y", alpha=0.15)
        ax.set_axisbelow(True)
    fig.suptitle("System A: leave-one-run-out validation", fontsize=14)
    fig.text(0.5, 0.025,
             "The dashed line marks the assumed measurement-noise scale. Parameter uncertainty is not included.",
             ha="center", fontsize=8, color="#444444")
    fig.tight_layout(rect=(0, 0.07, 1, 0.95))
    fig.savefig(FIGURES / "system_A_cv_diagnostics.png", dpi=160)
    plt.close(fig)


def save_results(runs, predictions, folds, starts, fold_parameters, elapsed):
    RESULTS.mkdir(exist_ok=True)
    by_species = per_group_scores(predictions, "species")
    by_run_species = pd.DataFrame([
        {"run_id": run_id, "species": species, **scores(group)}
        for (run_id, species), group in predictions.groupby(["run_id", "species"], sort=True)
    ])
    predictions.to_csv(RESULTS / "system_A_cv_predictions.csv", index=False)
    folds.to_csv(RESULTS / "system_A_cv_folds.csv", index=False)
    starts.to_csv(RESULTS / "system_A_cv_starts.csv", index=False)
    by_species.to_csv(RESULTS / "system_A_cv_by_species.csv", index=False)
    by_run_species.to_csv(RESULTS / "system_A_cv_by_run_species.csv", index=False)
    worst = predictions.assign(abs_z=predictions.standardized_error.abs()).nlargest(5, "abs_z")
    report = {
        "system": "A", "model": "M1 (disclosed)", "method": "leave-one-run-out",
        "n_folds": len(runs), "elapsed_fitting_seconds": elapsed,
        "noise_assumption": "Independent Gaussian errors: SD=0.02*c for A/B/P and 0.05*c for Q",
        "overall_heldout_scores": scores(predictions),
        "per_species": by_species.to_dict(orient="records"),
        "folds": folds.to_dict(orient="records"),
        "largest_standardized_errors": worst.drop(columns="abs_z").to_dict(orient="records"),
        "parameter_units": {"k1": "L mol^-1 min^-1 at 75 C", "k2": "L mol^-1 min^-1 at 75 C",
                            "Ea1": "kJ/mol", "Ea2": "kJ/mol"},
        "limitations": [
            "Only the disclosed M1 mechanism is evaluated; no mechanism selection is tested.",
            "Only five experimental runs are available, so generalization evidence is limited.",
            "Standardized errors include measurement noise only, not parameter uncertainty.",
            "This exercise does not calculate prediction intervals or establish 90% coverage.",
            "Time-zero rows are excluded from scoring because their predictions are fixed initial conditions.",
            "Training range flags describe individual variables, not joint-design interpolation.",
            "System A validation does not establish performance or the mechanism for System B.",
        ],
    }
    (RESULTS / "system_A_cv_summary.json").write_text(json.dumps(report, indent=2)+"\n", encoding="utf-8")
    make_figures(runs, fold_parameters, predictions, folds, by_species)
    return by_species, worst


def main():
    print("System A validation: fit four experiments, predict the fifth.", flush=True)
    runs = reference.load_runs()
    if len(runs) != 5:
        raise ValueError(f"Expected the five released System A runs, found {len(runs)}.")
    started = time.perf_counter()
    predictions, folds, starts, fold_parameters = validate(runs)
    elapsed = time.perf_counter()-started
    by_species, worst = save_results(runs, predictions, folds, starts, fold_parameters, elapsed)
    overall = scores(predictions)
    print("\nValidation complete: all five experiments were held out once.")
    print(f"Scored {len(predictions)} positive-time measurements; no missing or duplicate predictions.")
    print(f"Overall held-out standardized RMS: {overall['standardized_rms']:.3f}")
    print("\nHeld-out standardized RMS by experiment:")
    for row in folds.itertuples(index=False):
        print(f"  {row.heldout_run_id}: {row.heldout_standardized_rms:.3f}")
    print("\nHeld-out errors by species:")
    for row in by_species.itertuples(index=False):
        print(f"  {row.species}: RMSE={row.rmse_M:.6g} mol/L; standardized RMS={row.standardized_rms:.3f}")
    first = worst.iloc[0]
    print(f"\nLargest standardized error: {first.run_id}, {first.species}, "
          f"t={first.time_min:g} min: {first.standardized_error:+.2f} assumed measurement SD.")
    print("All observations were retained. Large discrepancies need investigation.")
    print("Standardization uses measurement noise only; these are not calibrated prediction intervals.")
    print("\nSaved results/system_A_cv_summary.json and five diagnostic CSV files.")
    print("Saved submission/figures/system_A_cv_predictions.png")
    print("Saved submission/figures/system_A_cv_diagnostics.png")
    print(f"Fitting and scoring took {elapsed:.1f} seconds on this computer.")


if __name__ == "__main__":
    main()
