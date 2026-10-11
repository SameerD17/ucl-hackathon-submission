"""First fitting exercise: estimate the disclosed M1 kinetics of System A.

Save this file beside justfile, then run: python fit_system_a.py

This fits M1 because System A's mechanism is disclosed. It does not select a
mechanism for System B or estimate prediction intervals. Relative Gaussian
measurement errors are assumed: 2% for A/B/P and 5% for Q, as documented for
System A. All time-zero rows are excluded from fitting: their predicted values
are fixed by the initial conditions and contain no rate-parameter information.

There are two optimisation stages. Weighted least squares finds starting
estimates; maximum likelihood then accounts for noise that depends on the
predicted concentration. Several starts help check for local optima.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.optimize import least_squares, minimize

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT / "src"))
from candidates import Conditions, simulate

SPECIES = ["A", "B", "P", "Q"]
RELATIVE_SD = np.array([0.02, 0.02, 0.02, 0.05])
PARAMETER_NAMES = ["k1", "Ea1", "k2", "Ea2"]
# Optimiser variables are log10(k1), Ea1/50, log10(k2), Ea2/50.
# These broad numerical bounds are assumptions, not disclosed true values.
LOWER = np.array([-6.0, 0.0, -7.0, 0.0])
UPPER = np.array([1.0, 3.0, 1.0, 3.0])
STARTS = [
    [-1.5, 1.2, -3.0, 1.2],
    [-2.0, 0.8, -2.5, 1.6],
    [-1.0, 1.6, -2.0, 0.8],
    [-2.5, 1.0, -3.5, 1.0],
]


def parameters(x):
    """Convert scaled optimisation variables to the candidate library's units."""
    return {"k1": float(10**x[0]), "Ea1": float(50*x[1]),
            "k2": float(10**x[2]), "Ea2": float(50*x[3])}


def load_runs():
    path = ROOT / "data" / "system_A" / "measurements.csv"
    if not path.exists():
        raise FileNotFoundError("Save fit_system_a.py in the main project folder, beside justfile.")
    data = pd.read_csv(path)
    if data["censored"].astype(str).str.lower().eq("true").any():
        raise ValueError("This reference exercise expects uncensored System A data.")
    data["value"] = pd.to_numeric(data["value"], errors="raise")
    runs = []
    for run_id, group in data.groupby("run_id", sort=True):
        design = group[["temperature_C", "A0_M", "B0_M"]].drop_duplicates()
        if len(design) != 1:
            raise ValueError(f"Inconsistent conditions in {run_id}")
        table = group.pivot(index="time_min", columns="species", values="value")
        table = table.reindex(columns=SPECIES).sort_index()
        if table.isna().any().any():
            raise ValueError(f"Incomplete reference measurements in {run_id}")
        times = table.index.to_numpy(float)
        values = table.to_numpy(float)
        if not np.isfinite(values).all() or (values < 0).any():
            raise ValueError(f"Invalid concentrations in {run_id}")
        use = times > 0
        if not use.any() or (values[use] <= 0).any():
            raise ValueError("The relative-error reference fit needs positive observations at t > 0.")
        row = design.iloc[0]
        condition = Conditions(float(row.temperature_C), float(row.A0_M),
                               float(row.B0_M), times)
        runs.append({"id": run_id, "condition": condition, "times": times,
                     "values": values, "use": use})
    return runs


def predict_run(run, params):
    return simulate("M1", run["condition"], params)[SPECIES].to_numpy(float)


def starting_residuals(x, runs):
    """Observed-value weights are used only to initialise the final likelihood fit."""
    params = parameters(x)
    residuals = []
    for run in runs:
        use = run["use"]
        observed = run["values"][use]
        predicted = predict_run(run, params)[use]
        residuals.extend(((predicted-observed)/(RELATIVE_SD*observed)).ravel())
    return np.asarray(residuals)


def negative_log_likelihood(x, runs):
    """Gaussian likelihood including its parameter-dependent log(sigma) term."""
    params = parameters(x)
    total = 0.0
    for run in runs:
        use = run["use"]
        observed = run["values"][use]
        predicted = predict_run(run, params)[use]
        # Numerical protection only; this is not an assumed assay noise floor.
        sigma = RELATIVE_SD * np.maximum(predicted, 1e-12)
        residual = (observed-predicted)/sigma
        total += np.sum(0.5*residual**2 + np.log(sigma) + 0.5*np.log(2*np.pi))
    return float(total)


def fit(runs):
    records, solutions = [], []
    for number, start in enumerate(STARTS, start=1):
        initial = least_squares(starting_residuals, np.array(start), args=(runs,),
                                bounds=(LOWER, UPPER), max_nfev=180,
                                ftol=1e-9, xtol=1e-9, gtol=1e-7)
        # Powell avoids differentiating the ODE solver's numerical output.
        result = minimize(negative_log_likelihood, initial.x, args=(runs,),
                          method="Powell", bounds=list(zip(LOWER, UPPER)),
                          options={"maxiter": 300, "ftol": 1e-10, "xtol": 1e-7})
        records.append({"start": number, "success": bool(result.success),
                        "negative_log_likelihood": float(result.fun),
                        "message": str(result.message), **parameters(result.x)})
        print(f"Start {number}/{len(STARTS)}: NLL={result.fun:.4f}; "
              f"converged={result.success}", flush=True)
        if result.success and np.isfinite(result.fun):
            solutions.append(result)
    if not solutions:
        raise RuntimeError("No likelihood optimisation converged. Inspect the printed messages.")
    best = min(solutions, key=lambda result: result.fun)
    return best, records


def save_outputs(runs, best, starts):
    params = parameters(best.x)
    results_dir = ROOT / "results"
    figures_dir = ROOT / "submission" / "figures"
    results_dir.mkdir(exist_ok=True)
    figures_dir.mkdir(parents=True, exist_ok=True)
    residual_rows = []
    for run in runs:
        predicted = predict_run(run, params)
        for index in np.flatnonzero(run["use"]):
            for column, species in enumerate(SPECIES):
                mean = predicted[index, column]
                observed = run["values"][index, column]
                sigma = RELATIVE_SD[column]*max(mean, 1e-12)
                residual_rows.append({"run_id": run["id"],
                    "time_min": run["times"][index], "species": species,
                    "observed_M": observed, "fitted_M": mean, "sigma_M": sigma,
                    "standardized_residual": (observed-mean)/sigma})
    residuals = pd.DataFrame(residual_rows)
    residuals.to_csv(results_dir / "system_A_M1_residuals.csv", index=False)
    pd.DataFrame(starts).to_csv(results_dir / "system_A_M1_starts.csv", index=False)
    rms = float(np.sqrt(np.mean(residuals.standardized_residual**2)))
    at_bound = bool(np.any(np.isclose(best.x, LOWER, atol=1e-4, rtol=0)) or
                    np.any(np.isclose(best.x, UPPER, atol=1e-4, rtol=0)))
    report = {"system": "A", "model": "M1 (disclosed)", "parameters": params,
        "units": {"k1": "L mol^-1 min^-1 at 75 C", "k2": "L mol^-1 min^-1 at 75 C",
                  "Ea1": "kJ/mol", "Ea2": "kJ/mol"},
        "noise_assumption": "Independent Gaussian errors, SD=0.02*c for A/B/P and 0.05*c for Q",
        "fitted_observations": len(residuals), "free_parameters": 4,
        "negative_log_likelihood": float(best.fun), "standardized_residual_rms": rms,
        "parameter_at_search_bound": at_bound,
        "limitations": ["M1 is given; this is not a comparison of the six mechanisms.",
            "Plots and residuals describe the training data, not held-out validation.",
            "No parameter or prediction intervals are estimated in this first exercise.",
            "Time-zero observations carry no kinetic information and are excluded from fitting."]}
    (results_dir / "system_A_M1_fit.json").write_text(json.dumps(report, indent=2)+"\n", encoding="utf-8")

    # One row per experimental run; one column per observed species.
    plt.rcParams.update({"font.size": 9})
    fig, axes = plt.subplots(len(runs), 4, figsize=(13, 2.15*len(runs)), squeeze=False)
    for row, run in enumerate(runs):
        condition = run["condition"]
        times = np.linspace(0, run["times"].max(), 260)
        curve = simulate("M1", Conditions(condition.temperature_c, condition.A0,
                                          condition.B0, times), params)
        for column, species in enumerate(SPECIES):
            ax = axes[row, column]
            ax.plot(times, curve[species], color="#176B87", lw=1.6, label="M1 fit")
            ax.scatter(run["times"], run["values"][:, column], color="#D97941",
                       s=15, zorder=3, label="Measured")
            if row == 0:
                ax.set_title(species, fontweight="bold")
            if column == 0:
                ax.set_ylabel(f"{run['id']}: {condition.temperature_c:g} C\nConcentration [mol/L]")
            if row == len(runs)-1:
                ax.set_xlabel("Time [min]")
            ax.grid(alpha=0.15)
    axes[0, 0].legend(fontsize=8)
    fig.suptitle("System A: fitting the disclosed M1 mechanism", fontsize=15)
    fig.tight_layout(rect=(0, 0, 1, 0.97))
    fig.savefig(figures_dir / "system_A_M1_fit.png", dpi=160)
    plt.close(fig)

    fig, axes = plt.subplots(1, 4, figsize=(13, 3.6), sharey=True)
    for ax, species in zip(axes, SPECIES):
        for run_id, group in residuals[residuals.species == species].groupby("run_id"):
            ax.scatter(group.time_min, group.standardized_residual, s=17, label=run_id)
        ax.axhline(0, color="#333333", lw=1)
        ax.axhline(2, color="#999999", lw=0.8, ls="--")
        ax.axhline(-2, color="#999999", lw=0.8, ls="--")
        ax.set_title(species)
        ax.set_xlabel("Time [min]")
    axes[0].set_ylabel("(Measured - fitted) / assumed SD")
    axes[-1].legend(fontsize=7)
    fig.suptitle("System A: residuals from the training-data fit")
    fig.tight_layout()
    fig.savefig(figures_dir / "system_A_M1_residuals.png", dpi=160)
    plt.close(fig)
    return params, rms, at_bound


def main():
    print("Fitting the disclosed System A mechanism: M1", flush=True)
    runs = load_runs()
    best, starts = fit(runs)
    params, rms, at_bound = save_outputs(runs, best, starts)
    print("\nEstimated parameters:")
    for name in PARAMETER_NAMES:
        unit = "L/(mol min) at 75 C" if name.startswith("k") else "kJ/mol"
        print(f"  {name:3s} = {params[name]:.6g} {unit}")
    print(f"\nStandardized residual RMS: {rms:.3f}")
    print("A value near 1 is a useful noise-scale check; also inspect the residual pattern.")
    if at_bound:
        print("NOTE: a fitted parameter is at a search bound; investigate before interpreting it.")
    print("\nSaved results/system_A_M1_fit.json and the residual/start CSV files.")
    print("Saved submission/figures/system_A_M1_fit.png")
    print("Saved submission/figures/system_A_M1_residuals.png")
    print("This first exercise does not identify System B or establish held-out calibration.")


if __name__ == "__main__":
    main()
