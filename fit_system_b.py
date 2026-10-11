"""Fit all six System B mechanisms with an explicit censored likelihood.

Save beside load_system_b.py and run: python fit_system_b.py

Assumptions: independent Gaussian assay errors with the documented
SD=sqrt((rel*model_mean)^2+abs^2). A numeric observation contributes a
Gaussian log density, including log(SD). A Q <LOQ observation contributes
log(P(noisy assay < LOQ)), computed with scipy.special.log_ndtr. Numeric
zeros remain numeric observations; sensitivity to possible reporting/clipping
is a later check. Unassayed Q has no likelihood term. Fixed initial conditions
make t=0 contributions constant, so all candidates use the same 161 t>0 rows.

The Gaussian negative log likelihood can be minimized exactly with a
least-squares solver because each assay has a positive absolute SD floor:
for numeric rows use residuals z and sqrt(log(1+(rel*mean/abs)^2)); for
censored rows use sqrt(-2*log(P(assay < LOQ))). Half their squared sum
differs from NLL only by a parameter-independent constant. The extra
optimization residuals are NOT additional observations for AIC/AICc/BIC.

Scores and Akaike weights are provisional training-data comparisons, not
posterior mechanism probabilities or validation of 90% prediction intervals.
The AICc correction is an approximation for this nonlinear censored problem.
No submission predictions, campaign, or hypotheses are changed by this script.
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
from scipy.optimize import least_squares
from scipy.special import log_ndtr

import load_system_b as data_loader
from candidates import CANDIDATES, Conditions, simulate

ROOT = Path(__file__).resolve().parent
SPECIES = data_loader.SPECIES
MODELS = tuple(CANDIDATES)
LOG_2PI = float(np.log(2*np.pi))
# Broad numerical search assumptions, not disclosed true parameter ranges.
RATE_BOUNDS = (1e-9, 10.0)
EA_BOUNDS = (0.0, 180.0)  # kJ/mol
BASE_RATES = {
    "M1": [0.01, 1e-5], "M2": [0.025, 0.02, 2e-5],
    "M3": [0.01, 1e-6], "M4": [0.025, 0.02, 2e-5],
    "M5": [0.01, 1e-6], "M6": [0.001, 0.1, 1e-5],
}
RATE_UNITS = {
    "M1": ["L mol^-1 min^-1", "L mol^-1 min^-1"],
    "M2": ["L mol^-1 min^-1", "min^-1", "min^-1"],
    "M3": ["L mol^-1 min^-1", "min^-1"],
    "M4": ["L mol^-1 min^-1", "min^-1", "L mol^-1 min^-1"],
    "M5": ["L mol^-1 min^-1", "L mol^-1 min^-1"],
    "M6": ["min^-1", "L mol^-1 min^-1", "min^-1"],
}


def parameters(model, x):
    return {name: float(10**value if name.startswith("k") else 50*value)
            for name, value in zip(CANDIDATES[model].param_names, x)}


def parameter_bounds(model):
    names = CANDIDATES[model].param_names
    lower = np.array([np.log10(RATE_BOUNDS[0]) if name.startswith("k")
                      else EA_BOUNDS[0]/50 for name in names])
    upper = np.array([np.log10(RATE_BOUNDS[1]) if name.startswith("k")
                      else EA_BOUNDS[1]/50 for name in names])
    return lower, upper


def starting_points(model):
    # Fixed guesses only: future validation can reuse these without full-data leakage.
    variations = [([0, 0, 0], [50, 50, 50]),
                  ([-0.8, 0.6, -0.4], [35, 85, 55]),
                  ([0.7, -0.8, 0.8], [95, 40, 100]),
                  ([-0.2, 0.8, -0.8], [70, 110, 30])]
    starts = []
    for offsets, energies in variations:
        start = []
        for index, rate in enumerate(BASE_RATES[model]):
            start.extend([np.log10(rate)+offsets[index], energies[index]/50])
        starts.append(np.asarray(start))
    return starts


def observation_means(model, run, params):
    prediction = simulate(model, run["condition"], params)
    values = prediction.loc[:, list(SPECIES)].to_numpy(float)
    return values[run["time_index"], run["species_index"]]


class FitProblem:
    """One model fitted to exactly the runs provided; no access to other runs."""

    def __init__(self, model, runs):
        self.model = model
        self.runs = runs
        self.observations = pd.concat(
            [run["observations"].loc[run["use"]] for run in runs], ignore_index=True
        )
        if self.observations.empty:
            raise ValueError("No positive-time observations to fit.")
        self.y = self.observations.value_M.to_numpy(float)
        self.censored = self.observations.censored.to_numpy(bool)
        self.numeric = ~self.censored
        self.loq = self.observations.loq_M.to_numpy(float)
        self.relative = self.observations.relative_sd.to_numpy(float)
        self.absolute = self.observations.absolute_sd_M.to_numpy(float)
        self.constant = float(np.sum(np.log(self.absolute[self.numeric])+0.5*LOG_2PI))
        self.failed_evaluations = 0

    def means(self, x):
        params = parameters(self.model, x)
        values = np.concatenate([
            observation_means(self.model, run, params)[run["use"]] for run in self.runs
        ])
        if not np.isfinite(values).all() or (values < 0).any():
            raise FloatingPointError("Invalid simulated concentration.")
        return values

    def residuals(self, x):
        try:
            mean = self.means(x)
        except (RuntimeError, FloatingPointError, OverflowError):
            self.failed_evaluations += 1
            return np.full(2*int(self.numeric.sum())+int(self.censored.sum()), 1e15)
        sigma = np.hypot(self.relative*mean, self.absolute)
        numeric_z = (self.y[self.numeric]-mean[self.numeric])/sigma[self.numeric]
        variance_term = np.sqrt(np.log1p(
            (self.relative[self.numeric]*mean[self.numeric]/self.absolute[self.numeric])**2
        ))
        log_probability = log_ndtr((self.loq[self.censored]-mean[self.censored])/sigma[self.censored])
        censor_term = np.sqrt(-2*log_probability)
        return np.concatenate([numeric_z, variance_term, censor_term])

    def nll(self, x):
        # Independently expressed log likelihood, also used to check the solver cost.
        mean = self.means(x)
        sigma = np.hypot(self.relative*mean, self.absolute)
        z = (self.y[self.numeric]-mean[self.numeric])/sigma[self.numeric]
        numeric_nll = np.sum(0.5*z**2+np.log(sigma[self.numeric])+0.5*LOG_2PI)
        censor_nll = -np.sum(log_ndtr(
            (self.loq[self.censored]-mean[self.censored])/sigma[self.censored]
        ))
        return float(numeric_nll+censor_nll)


def fit_model(model, runs):
    problem = FitProblem(model, runs)
    lower, upper = parameter_bounds(model)
    solutions, records = [], []
    for number, start in enumerate(starting_points(model), start=1):
        started = time.perf_counter()
        solution = least_squares(
            problem.residuals, start, bounds=(lower, upper), method="trf",
            diff_step=1e-4, x_scale="jac", max_nfev=400,
            ftol=1e-9, xtol=1e-8, gtol=1e-5,
        )
        nll = problem.nll(solution.x)
        if not np.isclose(nll, solution.cost+problem.constant, rtol=1e-9, atol=1e-6):
            raise RuntimeError("Optimization objective and direct likelihood disagree.")
        successful = bool(solution.success and np.isfinite(nll) and nll < 1e25)
        records.append({
            "model": model, "start": number, "success": successful,
            "negative_log_likelihood": nll, "nfev": int(solution.nfev),
            "optimality": float(solution.optimality), "message": str(solution.message),
            "elapsed_seconds": time.perf_counter()-started, **parameters(model, solution.x),
        })
        print(f"  Start {number}/4: NLL={nll:.4f}; converged={successful}; "
              f"{records[-1]['elapsed_seconds']:.1f} s", flush=True)
        if successful:
            solutions.append((nll, solution))
    if not solutions:
        raise RuntimeError(f"No {model} fit converged. Inspect the printed starts.")
    nll, best = min(solutions, key=lambda pair: pair[0])
    # Do not silently prefer a worse converged solution to an unfinished better one.
    if any(row["negative_log_likelihood"] < nll-0.01 for row in records):
        raise RuntimeError(f"{model} has a better unconverged start; investigate before ranking.")
    names = CANDIDATES[model].param_names
    near_bound = np.isclose(best.x, lower, atol=1e-3, rtol=0) | np.isclose(best.x, upper, atol=1e-3, rtol=0)
    bound_names = [name for name, flag in zip(names, near_bound) if flag]
    successes = [row["negative_log_likelihood"] for row in records if row["success"]]
    report = {
        "parameters": parameters(model, best.x), "negative_log_likelihood": nll,
        "n_observations": len(problem.y), "n_numeric": int(problem.numeric.sum()),
        "n_censored": int(problem.censored.sum()), "n_parameters": len(best.x),
        "successful_starts": len(successes), "attempted_starts": len(records),
        "successful_start_nll_span": float(max(successes)-min(successes)),
        "parameters_at_bounds": bound_names, "optimality": float(best.optimality),
        "failed_simulation_evaluations": problem.failed_evaluations,
        "parameter_units": {
            name: RATE_UNITS[model][index//2]+" at 75 C" if name.startswith("k") else "kJ/mol"
            for index, name in enumerate(names)
        },
    }
    if bound_names:
        print(f"  NOTE: parameter(s) at search bounds: {', '.join(bound_names)}", flush=True)
    if report["successful_start_nll_span"] > 0.1:
        print("  NOTE: starts reached different minima; use the comparison cautiously.", flush=True)
    return report, records


def prediction_table(model, runs, params):
    tables = []
    for run in runs:
        table = run["observations"].copy()
        mean = observation_means(model, run, params)
        sigma = np.hypot(table.relative_sd.to_numpy()*mean, table.absolute_sd_M.to_numpy())
        table.insert(0, "model", model)
        table["prediction_M"] = mean
        table["measurement_sd_M"] = sigma
        table["error_M"] = table.value_M-mean
        table["standardized_residual"] = table.error_M/sigma
        table["probability_below_LOQ"] = np.nan
        is_q = table.species.eq("Q")
        table.loc[is_q, "probability_below_LOQ"] = np.exp(log_ndtr(
            (table.loc[is_q, "loq_M"].to_numpy()-mean[is_q])/sigma[is_q]
        ))
        tables.append(table)
    return pd.concat(tables, ignore_index=True)


def compare_models(fits, predictions):
    rows = []
    for model, fit in fits.items():
        n, k = fit["n_observations"], fit["n_parameters"]
        aic = 2*fit["negative_log_likelihood"]+2*k
        if n <= k+1:
            raise ValueError("Insufficient observations for the AICc correction.")
        numeric = predictions[(predictions.model == model) & predictions.use_for_fit & ~predictions.censored]
        rows.append({
            "model": model, "n_observations": n, "n_parameters": k,
            "negative_log_likelihood": fit["negative_log_likelihood"],
            "AIC": aic, "AICc": aic+2*k*(k+1)/(n-k-1),
            "BIC": 2*fit["negative_log_likelihood"]+k*np.log(n),
            "numeric_standardized_RMS": float(np.sqrt(np.mean(numeric.standardized_residual**2))),
            "successful_starts": fit["successful_starts"],
            "start_NLL_span": fit["successful_start_nll_span"],
            "parameters_at_bounds": ",".join(fit["parameters_at_bounds"]) or "none",
        })
    table = pd.DataFrame(rows).sort_values("AICc").reset_index(drop=True)
    table.insert(0, "rank", np.arange(1, len(table)+1))
    table["delta_AICc"] = table.AICc-table.AICc.min()
    relative_support = np.exp(-0.5*table.delta_AICc)
    table["provisional_Akaike_weight"] = relative_support/relative_support.sum()
    return table


def residual_summary(predictions):
    numeric = predictions[predictions.use_for_fit & ~predictions.censored]
    return pd.DataFrame([
        {"model": model, "species": species, "n_numeric": len(group),
         "RMSE_M": float(np.sqrt(np.mean(group.error_M**2))),
         "mean_standardized_residual": float(group.standardized_residual.mean()),
         "standardized_RMS": float(np.sqrt(np.mean(group.standardized_residual**2))),
         "max_abs_standardized_residual": float(group.standardized_residual.abs().max())}
        for (model, species), group in numeric.groupby(["model", "species"], sort=True)
    ])


def make_figures(runs, fits, comparison, predictions, residuals):
    figures = ROOT / "submission" / "figures"
    figures.mkdir(parents=True, exist_ok=True)
    plt.rcParams.update({"font.size": 9})
    order = comparison.model.tolist()
    fig, axes = plt.subplots(1, 2, figsize=(12.5, 4.8), gridspec_kw={"width_ratios": [1.2, 1]})
    axes[0].axis("off")
    cells = [[row.model, f"{row.negative_log_likelihood:.2f}", f"{row.delta_AICc:.2f}",
              f"{row.provisional_Akaike_weight:.3f}"] for row in comparison.itertuples()]
    display = axes[0].table(cellText=cells, colLabels=["Model", "NLL", "Delta AICc", "Weight"],
                            cellLoc="center", loc="center")
    display.auto_set_font_size(False)
    display.set_fontsize(10)
    display.scale(1, 1.7)
    axes[0].set_title("Approximate AICc ranking (lower is better)")
    values = residuals.pivot(index="model", columns="species", values="standardized_RMS").loc[order, list(SPECIES)]
    chart = axes[1].imshow(values.to_numpy(), aspect="auto", cmap="YlOrRd",
                           norm=LogNorm(vmin=0.5, vmax=max(2, float(values.max().max()))))
    axes[1].set_xticks(np.arange(4), SPECIES)
    axes[1].set_yticks(np.arange(len(order)), order)
    axes[1].set_title("Numeric residual RMS / assay SD")
    for row in range(len(order)):
        for column in range(4):
            value = values.iloc[row, column]
            color = "white" if chart.norm(value) > 0.7 else "black"
            axes[1].text(column, row, f"{value:.2f}", ha="center", va="center", color=color)
    fig.colorbar(chart, ax=axes[1], label="Standardized RMS (log colour scale)", shrink=0.85)
    fig.suptitle("System B: provisional comparison of all six mechanisms", fontsize=14)
    fig.text(0.5, 0.025,
             "Training data only. Weights are relative Akaike support, not posterior probabilities; RMS excludes censored rows.",
             ha="center", fontsize=8)
    fig.tight_layout(rect=(0, 0.07, 1, 0.95))
    fig.savefig(figures / "system_B_model_comparison.png", dpi=160)
    plt.close(fig)

    leading = order[:2]
    colors = ["#176B87", "#9C3F74"]
    styles = ["-", "--"]
    fig, axes = plt.subplots(len(runs), 4, figsize=(13, 2.15*len(runs)), squeeze=False)
    for row, run in enumerate(runs):
        condition = run["condition"]
        times = np.unique(np.r_[0, np.geomspace(0.1, run["times"].max(), 260)])
        curves = {model: simulate(model, Conditions(condition.temperature_c, condition.A0,
                                                    condition.B0, times), fits[model]["parameters"])
                  for model in leading}
        for column, species in enumerate(SPECIES):
            ax = axes[row, column]
            for model, color, style in zip(leading, colors, styles):
                ax.plot(times, curves[model][species], color=color, ls=style, lw=1.6)
            measured = run["observations"]
            measured = measured[measured.species.eq(species)]
            numeric = measured[~measured.censored]
            ax.scatter(numeric.time_min, numeric.value_M, s=15, color="#333333", zorder=3)
            if species == "Q":
                censored = measured[measured.censored]
                ax.scatter(censored.time_min, censored.loq_M, marker="v", s=28,
                           color="#D97941", zorder=4)
                ax.axhline(data_loader.Q_LOQ_M, color="#D97941", ls=":", lw=0.8)
                ax.ticklabel_format(axis="y", style="sci", scilimits=(0, 0))
            if row == 0:
                ax.set_title(species, fontweight="bold")
            if column == 0:
                ax.set_ylabel(f"{run['id']}: {condition.temperature_c:g} C\nConcentration [mol/L]")
            if row == len(runs)-1:
                ax.set_xlabel("Time [min]")
            ax.set_ylim(bottom=-0.04*ax.get_ylim()[1])
            ax.grid(alpha=0.15)
    handles = [Line2D([], [], color=color, ls=style, label=model)
               for model, color, style in zip(leading, colors, styles)]
    handles += [Line2D([], [], color="#333333", marker="o", ls="", label="Numeric measurement"),
                Line2D([], [], color="#D97941", marker="v", ls="", label="Q <LOQ (upper limit)")]
    fig.suptitle("System B: the two leading fits on the training experiments", fontsize=14, y=0.99)
    fig.legend(handles=handles, loc="upper center", bbox_to_anchor=(0.5, 0.967), ncol=4)
    fig.text(0.5, 0.012, "Curves use all five released runs for fitting; these are not held-out predictions.",
             ha="center", fontsize=9)
    fig.tight_layout(rect=(0, 0.03, 1, 0.96))
    fig.savefig(figures / "system_B_leading_fits.png", dpi=160)
    plt.close(fig)

    fig, axes = plt.subplots(2, 4, figsize=(13, 6.5), sharey=True)
    for row, model in enumerate(leading):
        for column, species in enumerate(SPECIES):
            ax = axes[row, column]
            subset = predictions[(predictions.model == model) & (predictions.species == species)
                                 & predictions.use_for_fit & ~predictions.censored]
            for run_id, group in subset.groupby("run_id"):
                ax.scatter(group.time_min, group.standardized_residual, s=17, label=run_id)
            ax.axhline(0, color="#444444", lw=0.8)
            for line in [-2, 2]:
                ax.axhline(line, color="#888888", ls="--", lw=0.8)
            ax.set_title(f"{model}: {species}")
            ax.set_xlabel("Time [min]")
            if column == 0:
                ax.set_ylabel("(Measured - fitted) / assay SD")
    axes[0, -1].legend(fontsize=7)
    fig.suptitle("System B: numeric training residuals for the two leading fits", fontsize=14)
    fig.text(0.5, 0.015,
             "Censored Q is included in the likelihood, but has no exact numeric residual to plot.",
             ha="center", fontsize=9)
    fig.tight_layout(rect=(0, 0.05, 1, 0.95))
    fig.savefig(figures / "system_B_leading_residuals.png", dpi=160)
    plt.close(fig)


def save_outputs(runs, fits, starts, elapsed):
    predictions = pd.concat([prediction_table(model, runs, fit["parameters"])
                             for model, fit in fits.items()], ignore_index=True)
    comparison = compare_models(fits, predictions)
    residuals = residual_summary(predictions)
    results = ROOT / "results"
    results.mkdir(exist_ok=True)
    comparison.to_csv(results / "system_B_model_comparison.csv", index=False)
    pd.DataFrame(starts).to_csv(results / "system_B_fit_starts.csv", index=False)
    predictions.to_csv(results / "system_B_training_predictions.csv", index=False)
    residuals.to_csv(results / "system_B_residual_summary.csv", index=False)
    report = {
        "system": "B", "stage": "provisional training-data comparison",
        "source_sha256": hashlib.sha256(data_loader.DATA_PATH.read_bytes()).hexdigest(),
        "fitter_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        "training_run_ids": [run["id"] for run in runs], "elapsed_fitting_seconds": elapsed,
        "noise_distribution_assumption": "Independent Gaussian assay errors before Q reporting censoring",
        "noise_sd": "sqrt((rel*predicted_mean)^2+abs^2), using the documented species-specific constants",
        "censored_log_likelihood": "log(P(noisy assay < LOQ)) = log_ndtr((LOQ-model_mean)/SD)",
        "search_bounds": {"rate_at_75C_in_reaction_specific_units": RATE_BOUNDS,
                          "activation_energy_kJ_per_mol": EA_BOUNDS},
        "models": fits, "ranking": comparison.to_dict(orient="records"),
        "limitations": [
            "The ranking uses the same measurements used for fitting; run-wise validation remains necessary.",
            "The Gaussian error distribution and independent errors are assumptions, beyond the specified SD formula.",
            "Numeric zeros are treated as numeric Gaussian observations; sensitivity to reporting/clipping remains to check.",
            "AICc is an approximate small-sample correction for nonlinear models with censoring and fixed noise scales.",
            "Akaike weights are relative support within these six fitted candidates, not posterior model probabilities.",
            "Only temperature varies in the released design; stoichiometry extrapolation remains uncertain.",
            "Finite starts do not guarantee a global optimum; bound hits and distinct minima need scrutiny.",
            "No parameter confidence intervals or calibrated 90% prediction intervals have been estimated.",
        ],
    }
    (results / "system_B_fits.json").write_text(json.dumps(report, indent=2)+"\n", encoding="utf-8")
    make_figures(runs, fits, comparison, predictions, residuals)
    return comparison


def main():
    runs = data_loader.load_runs()
    started = time.perf_counter()
    fits, all_starts = {}, []
    print("System B: fit all six candidate mechanisms.", flush=True)
    print("Using positive-time measurements, including censored Q; fixed initial conditions.", flush=True)
    for number, model in enumerate(MODELS, start=1):
        print(f"\nModel {number}/{len(MODELS)}: {model}", flush=True)
        fits[model], starts = fit_model(model, runs)
        all_starts.extend(starts)
    elapsed = time.perf_counter()-started
    comparison = save_outputs(runs, fits, all_starts, elapsed)
    print("\nSystem B fitting complete. Provisional ranking:")
    columns = ["rank", "model", "negative_log_likelihood", "delta_AICc",
               "provisional_Akaike_weight", "numeric_standardized_RMS", "parameters_at_bounds"]
    print(comparison[columns].to_string(index=False, float_format=lambda value: f"{value:.4f}"))
    print("\nAll models used the same 161 positive-time observations (159 numeric and 2 censored).")
    print("Training-data ranking only: next check predictions on excluded experiments.")
    print("Saved results/system_B_fits.json and four diagnostic CSV files.")
    print("Saved three figures in submission/figures: system_B_model_comparison.png,")
    print("  system_B_leading_fits.png, system_B_leading_residuals.png")
    print(f"Fitting took {elapsed:.1f} seconds on this computer.")


if __name__ == "__main__":
    main()
