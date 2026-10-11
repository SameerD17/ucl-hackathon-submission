"""Leave one complete System B experiment out, for every candidate mechanism.

Save beside fit_system_b.py and load_system_b.py, then run:
    python validate_system_b.py

Each of six mechanisms is fitted from four fixed starts to four experiments;
only afterward are the excluded experiment's measurements used for scoring.
No saved full-data fit, full-data ranking, or heldout_conditions.csv is read.

The negative log score uses the same censored Gaussian assay likelihood as
fit_system_b.py, evaluated at point-estimated parameters. It includes the
documented measurement noise but NOT parameter uncertainty or model averaging.
Consequently this is a plug-in prediction check, not a test of calibrated 90%
prediction intervals. Numeric zeros remain numeric, Q <LOQ remains censored,
and absent assays contribute nothing. All positive-time rows are scored once
per model. Initial conditions are fixed, so time-zero rows are only plotted.

All released runs share the same initial charge. Leaving temperatures out
does not establish performance at new A:B ratios. Fold scores share training
data and must not be treated as five independent tests of model identity.
"""

from __future__ import annotations

import hashlib
import json
import time
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm
from matplotlib.lines import Line2D
import numpy as np
import pandas as pd
from scipy.special import log_ndtr

import fit_system_b as fitter
import load_system_b as data_loader
from candidates import Conditions, simulate

ROOT = Path(__file__).resolve().parent
SPECIES = data_loader.SPECIES
COLORS = {"M1": "#777777", "M2": "#9C3F74", "M3": "#B7822B",
          "M4": "#176B87", "M5": "#59883E", "M6": "#AC443A"}


def scored_predictions(model, held_out, params, training_ids):
    table = fitter.prediction_table(model, [held_out], params)
    table = table.loc[table.use_for_fit].drop(columns="use_for_fit").copy()
    table = table.rename(columns={"standardized_residual": "standardized_error"})
    table["training_run_ids"] = ",".join(training_ids)
    numeric = ~table.censored.to_numpy(bool)
    mean = table.prediction_M.to_numpy(float)
    sigma = table.measurement_sd_M.to_numpy(float)
    scores = np.empty(len(table), dtype=float)
    z = table.standardized_error.to_numpy(float)
    scores[numeric] = 0.5*z[numeric]**2+np.log(sigma[numeric])+0.5*fitter.LOG_2PI
    scores[~numeric] = -log_ndtr(
        (table.loq_M.to_numpy(float)[~numeric]-mean[~numeric])/sigma[~numeric]
    )
    if not np.isfinite(scores).all():
        raise RuntimeError(f"Nonfinite held-out scores for {model}, {held_out['id']}.")
    table["plugin_negative_log_score"] = scores
    return table.reset_index(drop=True)


def summarize_predictions(table):
    numeric = table.loc[~table.censored]
    return {
        "n_observations": len(table), "n_numeric": len(numeric),
        "n_censored": int(table.censored.sum()),
        "plugin_NLL": float(table.plugin_negative_log_score.sum()),
        "numeric_standardized_RMS": float(np.sqrt(np.mean(numeric.standardized_error**2))),
        "numeric_RMSE_M": float(np.sqrt(np.mean(numeric.error_M**2))),
        "numeric_mean_standardized_error": float(numeric.standardized_error.mean()),
    }


def temperature_range(training, held_out):
    temperatures = [run["condition"].temperature_c for run in training]
    low, high = float(min(temperatures)), float(max(temperatures))
    target = float(held_out["condition"].temperature_c)
    return {"temperature_C": target,
            "training_temperature_min_C": low, "training_temperature_max_C": high,
            "temperature_outside_training_range": target < low or target > high}


def validate(runs, models=fitter.MODELS, fit_function=None):
    if fit_function is None:
        fit_function = fitter.fit_model
    identifiers = [run["id"] for run in runs]
    if len(runs) < 3 or len(set(identifiers)) != len(identifiers):
        raise ValueError("Validation needs at least three uniquely identified experiments.")
    if len(set(models)) != len(models) or not models:
        raise ValueError("Choose at least one model, with no duplicates.")
    all_predictions, fold_rows, start_rows, fold_fits = [], [], [], {}
    for model_number, model in enumerate(models, start=1):
        print(f"\nModel {model_number}/{len(models)}: {model}", flush=True)
        fold_fits[model] = {}
        for fold_number, held_out in enumerate(runs, start=1):
            training = [run for run in runs if run["id"] != held_out["id"]]
            training_ids = [run["id"] for run in training]
            print(f" Fold {fold_number}/{len(runs)}: predict {held_out['id']}; "
                  f"fit {', '.join(training_ids)}", flush=True)
            # Only these four runs reach the fitter, including every starting fit.
            fit, starts = fit_function(model, training)
            fold_fits[model][held_out["id"]] = fit
            test = scored_predictions(model, held_out, fit["parameters"], training_ids)
            test_scores = summarize_predictions(test)
            training_table = fitter.prediction_table(model, training, fit["parameters"])
            training_numeric = training_table[training_table.use_for_fit & ~training_table.censored]
            training_rms = float(np.sqrt(np.mean(training_numeric.standardized_residual**2)))
            fold_rows.append({
                "model": model, "heldout_run_id": held_out["id"],
                "training_run_ids": ",".join(training_ids),
                "n_training_observations": fit["n_observations"],
                "n_heldout_observations": test_scores["n_observations"],
                "n_heldout_numeric": test_scores["n_numeric"],
                "n_heldout_censored": test_scores["n_censored"],
                "training_NLL": fit["negative_log_likelihood"],
                "heldout_plugin_NLL": test_scores["plugin_NLL"],
                "training_numeric_standardized_RMS": training_rms,
                "heldout_numeric_standardized_RMS": test_scores["numeric_standardized_RMS"],
                "successful_starts": fit["successful_starts"],
                "attempted_starts": fit["attempted_starts"],
                "start_NLL_span": fit["successful_start_nll_span"],
                "parameters_at_bounds": ",".join(fit["parameters_at_bounds"]) or "none",
                **temperature_range(training, held_out), **fit["parameters"],
            })
            start_rows.extend({"heldout_run_id": held_out["id"], **record} for record in starts)
            all_predictions.append(test)
            print(f"  Held-out NLL: {test_scores['plugin_NLL']:.3f}; "
                  f"numeric standardized RMS: {test_scores['numeric_standardized_RMS']:.3f}", flush=True)
    predictions = pd.concat(all_predictions, ignore_index=True)
    check_prediction_keys(runs, models, predictions)
    return predictions, pd.DataFrame(fold_rows), pd.DataFrame(start_rows), fold_fits


def check_prediction_keys(runs, models, predictions):
    expected = {
        (model, run["id"], float(row.time_min), row.species)
        for model in models for run in runs
        for row in run["observations"].loc[run["use"]].itertuples(index=False)
    }
    actual = list(predictions[["model", "run_id", "time_min", "species"]].itertuples(index=False, name=None))
    if len(actual) != len(set(actual)) or set(actual) != expected:
        raise RuntimeError("Missing or duplicate held-out prediction keys.")
    identifiers = {run["id"] for run in runs}
    for row in predictions.itertuples(index=False):
        if set(row.training_run_ids.split(",")) != identifiers-{row.run_id}:
            raise RuntimeError("A validation prediction has an invalid training partition.")


def model_summary(predictions, folds):
    rows = []
    for model, group in predictions.groupby("model", sort=True):
        subset = folds[folds.model.eq(model)]
        rows.append({
            "model": model, **summarize_predictions(group), "n_folds": len(subset),
            "max_fold_numeric_standardized_RMS": float(subset.heldout_numeric_standardized_RMS.max()),
            "folds_with_bound_hits": int(subset.parameters_at_bounds.ne("none").sum()),
            "folds_with_distinct_minima": int(subset.start_NLL_span.gt(0.1).sum()),
        })
    summary = pd.DataFrame(rows).sort_values("plugin_NLL").reset_index(drop=True)
    summary.insert(0, "rank", np.arange(1, len(summary)+1))
    summary["delta_plugin_NLL"] = summary.plugin_NLL-summary.plugin_NLL.min()
    return summary


def species_summary(predictions):
    return pd.DataFrame([
        {"model": model, "species": species, **summarize_predictions(group)}
        for (model, species), group in predictions.groupby(["model", "species"], sort=True)
    ])


def paired_comparison(folds):
    selected = folds[folds.model.isin(["M2", "M4"])]
    nll = selected.pivot(index="heldout_run_id", columns="model", values="heldout_plugin_NLL")
    rms = selected.pivot(index="heldout_run_id", columns="model", values="heldout_numeric_standardized_RMS")
    conditions = selected.drop_duplicates("heldout_run_id").set_index("heldout_run_id")
    pairs = pd.DataFrame({
        "heldout_run_id": nll.index,
        "temperature_C": conditions.loc[nll.index, "temperature_C"].to_numpy(),
        "M2_plugin_NLL": nll.M2.to_numpy(), "M4_plugin_NLL": nll.M4.to_numpy(),
        "M4_minus_M2_plugin_NLL": (nll.M4-nll.M2).to_numpy(),
        "M2_numeric_standardized_RMS": rms.M2.to_numpy(),
        "M4_numeric_standardized_RMS": rms.M4.to_numpy(),
    })
    return pairs


def make_figures(runs, summary, by_species, folds, pairs, fold_fits):
    figures = ROOT / "submission" / "figures"
    figures.mkdir(parents=True, exist_ok=True)
    plt.rcParams.update({"font.size": 9})
    order = summary.model.tolist()
    fig, axes = plt.subplots(1, 2, figsize=(12.5, 4.8), gridspec_kw={"width_ratios": [1.2, 1]})
    axes[0].axis("off")
    cells = [[row.model, f"{row.plugin_NLL:.2f}", f"{row.delta_plugin_NLL:.2f}",
              f"{row.numeric_standardized_RMS:.3f}"] for row in summary.itertuples()]
    display = axes[0].table(cellText=cells, colLabels=["Model", "Held-out NLL", "NLL gap", "Numeric RMS / SD"],
                            cellLoc="center", loc="center", colWidths=[0.16, 0.28, 0.22, 0.34])
    display.auto_set_font_size(False)
    display.set_fontsize(9)
    display.scale(1, 1.7)
    axes[0].set_title("Total held-out plug-in NLL (lower is better)")
    values = by_species.pivot(index="model", columns="species", values="numeric_standardized_RMS").loc[order, list(SPECIES)]
    chart = axes[1].imshow(values.to_numpy(), aspect="auto", cmap="YlOrRd",
                           norm=LogNorm(vmin=0.5, vmax=max(2, float(values.max().max()))))
    axes[1].set_xticks(np.arange(4), SPECIES)
    axes[1].set_yticks(np.arange(len(order)), order)
    axes[1].set_title("Held-out numeric RMS / assay SD")
    for row in range(len(order)):
        for column in range(4):
            value = values.iloc[row, column]
            axes[1].text(column, row, f"{value:.2f}", ha="center", va="center",
                         color="white" if chart.norm(value) > 0.7 else "black")
    fig.colorbar(chart, ax=axes[1], label="Standardized RMS (log colour scale)", shrink=0.85)
    fig.suptitle("System B: each experiment predicted from the other four", fontsize=14)
    fig.text(0.5, 0.025,
             "Scores use measurement noise at fitted parameter values. Parameter uncertainty is not included; RMS excludes censored rows.",
             ha="center", fontsize=8)
    fig.tight_layout(rect=(0, 0.07, 1, 0.95))
    fig.savefig(figures / "system_B_cv_comparison.png", dpi=160)
    plt.close(fig)

    leading = order[:2]
    fig, axes = plt.subplots(len(runs), 4, figsize=(13, 2.15*len(runs)), squeeze=False)
    for row, run in enumerate(runs):
        condition = run["condition"]
        times = np.unique(np.r_[0, np.geomspace(0.1, run["times"].max(), 260)])
        curves = {model: simulate(model, Conditions(condition.temperature_c, condition.A0,
                                                    condition.B0, times),
                                  fold_fits[model][run["id"]]["parameters"]) for model in leading}
        for column, species in enumerate(SPECIES):
            ax = axes[row, column]
            for model in leading:
                ax.plot(times, curves[model][species], color=COLORS[model],
                        ls="--" if model == "M2" else "-", lw=1.6)
            observations = run["observations"]
            observations = observations[observations.species.eq(species)]
            numeric = observations[~observations.censored]
            ax.scatter(numeric.time_min, numeric.value_M, s=15, color="#333333", zorder=3)
            if species == "Q":
                censored = observations[observations.censored]
                ax.scatter(censored.time_min, censored.loq_M, marker="v", s=28,
                           color="#D97941", zorder=4)
                ax.axhline(data_loader.Q_LOQ_M, color="#D97941", ls=":", lw=0.8)
                ax.ticklabel_format(axis="y", style="sci", scilimits=(0, 0))
            if row == 0:
                ax.set_title(species, fontweight="bold")
            if column == 0:
                ax.set_ylabel(f"{run['id']} held out: {condition.temperature_c:g} C\nConcentration [mol/L]")
            if row == len(runs)-1:
                ax.set_xlabel("Time [min]")
            ax.set_ylim(bottom=-0.04*ax.get_ylim()[1])
            ax.grid(alpha=0.15)
    handles = [Line2D([], [], color=COLORS[model], ls="--" if model == "M2" else "-", label=model)
               for model in leading]
    handles += [Line2D([], [], color="#333333", marker="o", ls="", label="Excluded measurement"),
                Line2D([], [], color="#D97941", marker="v", ls="", label="Excluded Q <LOQ")]
    fig.suptitle("System B: held-out predictions from the two leading candidates", fontsize=14, y=0.99)
    fig.legend(handles=handles, loc="upper center", bbox_to_anchor=(0.5, 0.967), ncol=4)
    fig.text(0.5, 0.012, "Each row uses parameters fitted without that experiment. Curves are point predictions; no intervals are shown.",
             ha="center", fontsize=9)
    fig.tight_layout(rect=(0, 0.03, 1, 0.96))
    fig.savefig(figures / "system_B_cv_leading_predictions.png", dpi=160)
    plt.close(fig)

    fig, axes = plt.subplots(1, 2, figsize=(12, 4.8))
    x = np.arange(len(pairs))
    for shift, model in [(-0.19, "M2"), (0.19, "M4")]:
        axes[0].bar(x+shift, pairs[f"{model}_numeric_standardized_RMS"], width=0.38,
                    color=COLORS[model], label=model)
    axes[0].axhline(1, color="#666666", ls="--", lw=0.8)
    axes[0].set_xticks(x, pairs.heldout_run_id)
    axes[0].set_ylabel("Held-out numeric RMS / assay SD")
    axes[0].set_title("Held-out error by experiment")
    axes[0].legend()
    differences = pairs.M4_minus_M2_plugin_NLL
    axes[1].bar(x, differences, color=[COLORS["M2"] if value >= 0 else COLORS["M4"] for value in differences])
    axes[1].axhline(0, color="#666666", lw=0.8)
    axes[1].set_xticks(x, pairs.heldout_run_id)
    axes[1].set_ylabel("Held-out NLL difference: M4 minus M2")
    axes[1].set_title("Positive favours M2; negative favours M4")
    for ax in axes:
        ax.set_xlabel("Experiment excluded from fitting")
        ax.grid(axis="y", alpha=0.15)
        ax.set_axisbelow(True)
    fig.suptitle("System B: comparing M2 and M4 across excluded experiments", fontsize=14)
    fig.text(0.5, 0.025, "B1 and B5 lie outside their training temperature ranges. All folds keep the same initial A:B charge.",
             ha="center", fontsize=8)
    fig.tight_layout(rect=(0, 0.07, 1, 0.95))
    fig.savefig(figures / "system_B_cv_diagnostics.png", dpi=160)
    plt.close(fig)


def save_outputs(runs, predictions, folds, starts, fold_fits, elapsed):
    results = ROOT / "results"
    results.mkdir(exist_ok=True)
    summary = model_summary(predictions, folds)
    by_species = species_summary(predictions)
    pairs = paired_comparison(folds)
    predictions.to_csv(results / "system_B_cv_predictions.csv", index=False)
    folds.to_csv(results / "system_B_cv_folds.csv", index=False)
    starts.to_csv(results / "system_B_cv_starts.csv", index=False)
    summary.to_csv(results / "system_B_cv_models.csv", index=False)
    by_species.to_csv(results / "system_B_cv_by_species.csv", index=False)
    pairs.to_csv(results / "system_B_cv_M2_vs_M4.csv", index=False)
    report = {
        "system": "B", "method": "leave-one-run-out for every candidate mechanism",
        "source_sha256": hashlib.sha256(data_loader.DATA_PATH.read_bytes()).hexdigest(),
        "fitter_sha256": hashlib.sha256(Path(fitter.__file__).read_bytes()).hexdigest(),
        "validator_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        "n_experiments": len(runs), "n_models": len(summary),
        "n_fits": len(folds), "elapsed_fitting_seconds": elapsed,
        "models": summary.to_dict(orient="records"),
        "M2_vs_M4_by_experiment": pairs.to_dict(orient="records"),
        "total_M4_minus_M2_plugin_NLL": float(pairs.M4_minus_M2_plugin_NLL.sum()),
        "fold_fits": fold_fits,
        "limitations": [
            "Scores use point-estimated parameters and measurement noise only, not full predictive uncertainty.",
            "No prediction intervals or 90% coverage estimates are calculated.",
            "Only five experiments are available and every fold shares training runs with other folds.",
            "The released design varies only temperature; performance at different initial charges is untested.",
            "Held-out log scores are compared directly; no AIC penalty or probability interpretation is added.",
            "Gaussian assay errors and the numeric treatment of reported zeros remain assumptions.",
            "Finite-start optimization, fixed search bounds, and possible local minima remain limitations.",
            "This evaluates candidates already specified in the brief; it is not an untouched final test set.",
        ],
    }
    (results / "system_B_cv_summary.json").write_text(json.dumps(report, indent=2, allow_nan=False)+"\n", encoding="utf-8")
    make_figures(runs, summary, by_species, folds, pairs, fold_fits)
    return summary, pairs


def main():
    runs = data_loader.load_runs()
    if len(runs) != 5:
        raise ValueError(f"Expected the five released System B experiments; found {len(runs)}.")
    print("System B validation: 6 models x 5 held-out experiments, with 4 starts per fit.", flush=True)
    print("Each fit receives only its four training experiments. Allow several minutes.", flush=True)
    started = time.perf_counter()
    predictions, folds, starts, fold_fits = validate(runs)
    elapsed = time.perf_counter()-started
    summary, pairs = save_outputs(runs, predictions, folds, starts, fold_fits, elapsed)
    print("\nSystem B validation complete.")
    print(f"Completed {len(folds)} model/experiment fits and {len(starts)} starting fits.")
    print(f"Scored {len(predictions)} observations in total; each positive-time observation once per model.")
    print("Each model: 161 observations, comprising 159 numeric and 2 censored Q results.")
    print("\nHeld-out comparison (lower NLL is better):")
    columns = ["rank", "model", "plugin_NLL", "delta_plugin_NLL", "numeric_standardized_RMS",
               "folds_with_bound_hits", "folds_with_distinct_minima"]
    print(summary[columns].to_string(index=False, float_format=lambda value: f"{value:.4f}"))
    print("\nM2 versus M4 by excluded experiment:")
    print(pairs[["heldout_run_id", "temperature_C", "M4_minus_M2_plugin_NLL"]].to_string(
        index=False, float_format=lambda value: f"{value:.4f}"))
    print(f"Total NLL difference, M4 minus M2: {pairs.M4_minus_M2_plugin_NLL.sum():+.4f}")
    print("Positive favours M2; negative favours M4. This difference is not a probability.")
    print("\nB1 and B5 test temperature extrapolation; new initial charges remain untested.")
    print("Scores include assay noise only. Calibrated 90% prediction intervals remain to be assessed.")
    print("Saved results/system_B_cv_summary.json and six diagnostic CSV files.")
    print("Saved system_B_cv_comparison.png, system_B_cv_leading_predictions.png,")
    print("  and system_B_cv_diagnostics.png in submission/figures.")
    print(f"Fitting and scoring took {elapsed:.1f} seconds on this computer.")


if __name__ == "__main__":
    main()
