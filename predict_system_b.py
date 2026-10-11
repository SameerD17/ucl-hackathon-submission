"""Predict System B with approximate, pointwise 90% prediction intervals.

    python predict_system_b.py                    # submission predictions
    python predict_system_b.py --check-intervals  # also assess released-run coverage

Save beside fit_system_b.py, validate_system_b.py and load_system_b.py.
No new dependencies, random simulation, or runtime network access is needed.
Existing fits are checked and reused; missing fits are generated automatically.

Method:
* Retain M2 and M4 with fixed equal predictive weights. These are a deliberately
  cautious weighting choice after their near tie, NOT posterior probabilities.
* In scaled coordinates (log10 rates, activation energies/50), invert the observed
  Hessian of the actual censored negative log likelihood at each optimum. Check
  positive definiteness and stability using two finite-difference step sizes.
  Do NOT use the Jacobian of the fitter's artificial optimization residuals as
  a covariance matrix, and do NOT estimate an extra residual-noise multiplier.
* Propagate that covariance through each ODE prediction using a local Jacobian.
  Add the documented assay variance, evaluated at the fitted concentration.
* Find the 5th and 95th percentiles of the two-component normal mixture, rather
  than averaging endpoints or replacing separated mechanisms with one normal.
  Intersect intervals with the nonnegative concentration domain. This preserves
  coverage for nonnegative outcomes and does not imply a truncated-normal fit.
  Point predictions are weighted fitted concentrations, not censored reports.

These are local normal/delta-method approximations, not MCMC or a guarantee of
90% coverage. They target individual assays, not just noiseless ODE trajectories;
see data/README.md's requirement to include assay noise in interval calibration.
Known initial charges are fixed, but a new time-zero assay still has assay noise.
Intervals are pointwise, not simultaneous bands for an entire trajectory.

Coverage uses each excluded run's saved four-run fit and covariance calculated
from those SAME four runs. No width or weight is tuned on excluded values. The
choice to retain M2/M4 used the released data, so this remains a retrospective
check, not an untouched final test. Numeric zeros remain numeric observations.
For Q <LOQ, report bounds on coverage using [0, LOQ), never fabricate a value.
Missing assays contribute nothing. All released charges are identical: coverage
at different initial charges remains untested, even with model averaging.

Background: https://www.itl.nist.gov/div898/handbook/pmd/section1/pmd132.htm
Root solver: https://docs.scipy.org/doc/scipy-1.15.3/reference/generated/scipy.optimize.brentq.html
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
from scipy.optimize import brentq
from scipy.special import ndtr

import fit_system_b as fitter
import load_system_b as data_loader
from candidates import CANDIDATES, Conditions, simulate

ROOT = Path(__file__).resolve().parent
RESULTS = ROOT / "results"
FIGURES = ROOT / "submission" / "figures"
HELDOUT = ROOT / "data" / "heldout_conditions.csv"
MODELS = ("M2", "M4")
WEIGHTS = np.array([0.5, 0.5])
SPECIES = data_loader.SPECIES
OUTPUT_COLUMNS = ["run_id", "time_min", "species", "prediction", "lower90", "upper90"]


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def matches_source(path, saved_digest):
    # Earlier scripts stored byte hashes. Git may convert LF/CRLF when cloning
    # between Windows and Linux; accept that one representation change only.
    raw = Path(path).read_bytes()
    lf = raw.replace(b"\r\n", b"\n")
    return saved_digest in {hashlib.sha256(value).hexdigest()
                            for value in (raw, lf, lf.replace(b"\n", b"\r\n"))}


def save_json(path, value):
    path.write_text(json.dumps(value, indent=2, allow_nan=False)+"\n", encoding="utf-8")


def provenance():
    paths = [data_loader.DATA_PATH, HELDOUT, Path(__file__),
             Path(fitter.__file__), Path(data_loader.__file__),
             ROOT / "src" / "candidates" / "models.py",
             ROOT / "src" / "candidates" / "simulate.py"]
    return {str(path.relative_to(ROOT)): digest(path) for path in paths}


def checked_report(path, cross_validation=False):
    """Reject stale data/fitter results; never silently use an incompatible fit."""
    if not path.exists():
        print(f"{path.name} is missing; generating the required fits first.", flush=True)
        if cross_validation:
            import validate_system_b as validator
            validator.main()
        else:
            fitter.main()
    report = json.loads(path.read_text(encoding="utf-8"))
    expected = {"source_sha256": data_loader.DATA_PATH,
                "fitter_sha256": fitter.__file__}
    if cross_validation:
        expected["validator_sha256"] = ROOT / "validate_system_b.py"
    for field, source_path in expected.items():
        if not matches_source(source_path, report.get(field)):
            command = "python validate_system_b.py" if cross_validation else "python fit_system_b.py"
            raise RuntimeError(f"{path.name} no longer matches {field}. Run {command}, then try again.")
    return report


def coordinates(model, params):
    return np.array([np.log10(params[name]) if name.startswith("k") else params[name]/50
                     for name in CANDIDATES[model].param_names], dtype=float)


def central_hessian(function, x, step):
    """Observed NLL curvature, including parameter-dependent noise and censoring."""
    shifts = np.eye(len(x))*step
    matrix = np.empty((len(x), len(x)))
    base = function(x)
    for i, ei in enumerate(shifts):
        matrix[i, i] = (function(x+ei)-2*base+function(x-ei))/step**2
        for j in range(i):
            ej = shifts[j]
            matrix[i, j] = matrix[j, i] = (
                function(x+ei+ej)-function(x+ei-ej)
                -function(x-ei+ej)+function(x-ei-ej)
            )/(4*step**2)
    return matrix


def uncertainty(model, training, fit):
    """This function receives training runs only; it never reads target assays."""
    if fit["parameters_at_bounds"]:
        raise RuntimeError(f"{model}: a boundary optimum needs a different uncertainty method.")
    if fit["successful_start_nll_span"] > 0.1:
        raise RuntimeError(f"{model}: distinct fit minima need investigation before local intervals.")
    problem = fitter.FitProblem(model, training)
    x = coordinates(model, fit["parameters"])
    if len(problem.y) != fit["n_observations"]:
        raise RuntimeError(f"{model}: saved fit has the wrong number of training observations.")
    nll = problem.nll(x)
    if not np.isclose(nll, fit["negative_log_likelihood"], atol=1e-5, rtol=1e-9):
        raise RuntimeError(f"{model}: saved likelihood does not match the supplied training runs.")
    covariances, eigenvalues = [], []
    for step in (2e-3, 1e-3):
        hessian = central_hessian(problem.nll, x, step)
        eig = np.linalg.eigvalsh(hessian)
        if not np.isfinite(eig).all() or eig.min() <= 0 or eig.max()/eig.min() > 1e8:
            raise RuntimeError(f"{model}: likelihood curvature is singular or unreliable; stop and investigate.")
        covariances.append(np.linalg.inv(hessian))
        eigenvalues.append(eig)
    covariance = covariances[-1]
    sd = np.sqrt(np.diag(covariance))
    sd_change = float(np.max(np.abs(np.sqrt(np.diag(covariances[0]))/sd-1)))
    # Check correlations as well as marginal SDs, on a dimensionless scale.
    covariance_change = float(np.max(np.abs(covariances[0]-covariance)/np.outer(sd, sd)))
    if max(sd_change, covariance_change) > 0.03:
        raise RuntimeError(f"{model}: covariance changes materially with differentiation step size.")
    lower, upper = fitter.parameter_bounds(model)
    bound_distance = float(np.min(np.minimum(x-lower, upper-x)/sd))
    if bound_distance < 3:
        raise RuntimeError(f"{model}: local normal uncertainty reaches a parameter bound; investigate.")
    return {
        "model": model, "training_run_ids": [run["id"] for run in training],
        "parameter_order": list(CANDIDATES[model].param_names),
        "coordinate_definition": "log10(k at 75 C); Ea in kJ/mol divided by 50",
        "center": x.tolist(), "covariance": covariance.tolist(),
        "coordinate_sd": sd.tolist(), "minimum_hessian_eigenvalue": float(eigenvalues[-1].min()),
        "hessian_condition_number": float(eigenvalues[-1].max()/eigenvalues[-1].min()),
        "max_relative_SD_change_on_halving_step": sd_change,
        "max_scaled_covariance_change_on_halving_step": covariance_change,
        "closest_parameter_bound_in_marginal_SD": bound_distance,
        "training_negative_log_likelihood": nll,
    }


def load_conditions(path=HELDOUT):
    table = pd.read_csv(path)
    columns = ["run_id", "temperature_C", "A0_M", "B0_M", "time_min"]
    if list(table.columns) != columns or table.empty or table.isna().any().any():
        raise ValueError("Held-out conditions must have the documented five-column schema and no missing values.")
    if table.duplicated(["run_id", "time_min"]).any():
        raise ValueError("Duplicate held-out run/time conditions.")
    numbers = table[columns[1:]].to_numpy(float)
    if not np.isfinite(numbers).all() or (table.time_min < 0).any():
        raise ValueError("Conditions must be finite, with nonnegative times.")
    if (table[["A0_M", "B0_M"]] <= 0).any().any() or (table.temperature_C <= -273.15).any():
        raise ValueError("Invalid temperature or initial charge.")
    conditions = []
    for run_id, group in table.groupby("run_id", sort=True):
        if (group[["temperature_C", "A0_M", "B0_M"]].nunique() != 1).any():
            raise ValueError(f"Inconsistent conditions within {run_id}.")
        row = group.iloc[0]
        times = np.sort(group.time_min.to_numpy(float))
        conditions.append((str(run_id), Conditions(float(row.temperature_C), float(row.A0_M),
                                                  float(row.B0_M), times)))
    return table, conditions


def component(model, condition, estimate):
    x = np.asarray(estimate["center"])
    covariance = np.asarray(estimate["covariance"])
    def curve(vector):
        # Integrate from zero even if the requested times do not include it.
        times = np.unique(np.r_[0.0, condition.times_min])
        if len(times) == 1:
            return np.array([[condition.A0, condition.B0, 0.0, 0.0]])
        full = simulate(model, Conditions(condition.temperature_c, condition.A0, condition.B0, times),
                        fitter.parameters(model, vector))
        return full.loc[:, list(SPECIES)].to_numpy(float)[np.searchsorted(times, condition.times_min)]
    mean = curve(x)
    jacobian = np.empty((*mean.shape, len(x)))
    step = 2e-4
    for i, shift in enumerate(np.eye(len(x))*step):
        jacobian[:, :, i] = (curve(x+shift)-curve(x-shift))/(2*step)
    parameter_variance = np.einsum("...i,ij,...j->...", jacobian, covariance, jacobian)
    if not np.isfinite(parameter_variance).all() or (parameter_variance < -1e-15).any():
        raise RuntimeError("Invalid propagated parameter variance.")
    parameter_variance = np.maximum(parameter_variance, 0)
    assay_variance = np.column_stack([
        data_loader.measurement_sd(species, mean[:, index])**2
        for index, species in enumerate(SPECIES)
    ])
    return mean, parameter_variance, assay_variance


def mixture_quantile(probability, means, sds):
    """Solve the mixture CDF itself; averaging component quantiles is incorrect."""
    left = float(np.min(means-12*sds))
    right = float(np.max(means+12*sds))
    return brentq(lambda value: float(WEIGHTS @ ndtr((value-means)/sds))-probability,
                  left, right, xtol=1e-13, rtol=1e-12)


def predict_condition(run_id, condition, estimates):
    parts = [component(model, condition, estimates[model]) for model in MODELS]
    means = np.stack([part[0] for part in parts])
    parameter_variances = np.stack([part[1] for part in parts])
    assay_variances = np.stack([part[2] for part in parts])
    total_sds = np.sqrt(parameter_variances+assay_variances)
    rows = []
    for index, time_min in enumerate(condition.times_min):
        for column, species in enumerate(SPECIES):
            mu, sd = means[:, index, column], total_sds[:, index, column]
            prediction = float(WEIGHTS @ mu)
            lower = max(0.0, mixture_quantile(0.05, mu, sd))
            upper = max(0.0, mixture_quantile(0.95, mu, sd))
            row = {"run_id": run_id, "time_min": float(time_min), "species": species,
                   "temperature_C": float(condition.temperature_c), "A0_M": float(condition.A0),
                   "B0_M": float(condition.B0), "prediction": prediction,
                   "lower90": lower, "upper90": upper,
                   "within_model_parameter_variance_M_squared": float(WEIGHTS @ parameter_variances[:, index, column]),
                   "assay_variance_M_squared": float(WEIGHTS @ assay_variances[:, index, column]),
                   "between_model_variance_M_squared": float(WEIGHTS @ (mu-prediction)**2)}
            for m, model in enumerate(MODELS):
                row[f"{model}_prediction"] = float(mu[m])
                row[f"{model}_parameter_sd"] = float(np.sqrt(parameter_variances[m, index, column]))
                row[f"{model}_assay_sd"] = float(np.sqrt(assay_variances[m, index, column]))
                row[f"{model}_total_sd"] = float(sd[m])
            if species == "Q":
                row["probability_below_LOQ"] = float(WEIGHTS @ ndtr((data_loader.Q_LOQ_M-mu)/sd))
            rows.append(row)
    return pd.DataFrame(rows)


def check_submission(table, conditions):
    expected = {(str(row.run_id), float(row.time_min), species)
                for row in conditions.itertuples(index=False) for species in SPECIES}
    keys = list(table[["run_id", "time_min", "species"]].itertuples(index=False, name=None))
    if len(keys) != len(set(keys)) or set(keys) != expected:
        raise RuntimeError("Missing or duplicate submission prediction keys.")
    values = table[["prediction", "lower90", "upper90"]].to_numpy(float)
    if not np.isfinite(values).all() or (values < 0).any():
        raise RuntimeError("Nonfinite or negative submission predictions.")
    if ((table.lower90 > table.prediction) | (table.prediction > table.upper90)).any():
        raise RuntimeError("An interval does not contain its point estimate.")


def coverage_summary(table):
    numeric = table.loc[~table.censored]
    censored = table.loc[table.censored]
    covered = int(((numeric.value_M >= numeric.lower90) & (numeric.value_M <= numeric.upper90)).sum())
    # A censored concentration lies somewhere in [0, LOQ). These bounds do not
    # assume it is uniform in that interval and do not assign a numeric value.
    definitely = int(((censored.lower90 <= 0) & (censored.upper90 >= censored.loq_M)).sum())
    possibly = int(((censored.lower90 < censored.loq_M) & (censored.upper90 >= 0)).sum())
    width = numeric.upper90-numeric.lower90
    interval_score = width+20*np.maximum(numeric.lower90-numeric.value_M, 0)+20*np.maximum(numeric.value_M-numeric.upper90, 0)
    return {"n_observations": len(table), "n_numeric": len(numeric),
            "numeric_covered": covered, "numeric_coverage": covered/len(numeric) if len(numeric) else None,
            "n_censored": len(censored), "censored_definitely_covered": definitely,
            "censored_possibly_covered": possibly,
            "all_observation_coverage_lower_bound": (covered+definitely)/len(table),
            "all_observation_coverage_upper_bound": (covered+possibly)/len(table),
            "numeric_mean_width_M": float(width.mean()) if len(numeric) else None,
            "numeric_mean_interval_score_M": float(interval_score.mean()) if len(numeric) else None}


def validate_intervals(runs, report):
    predictions, uncertainties = [], {}
    identifiers = {run["id"] for run in runs}
    for held_out in runs:
        run_id = held_out["id"]
        training = [run for run in runs if run["id"] != run_id]
        print(f"  Interval check: {run_id} excluded; covariance uses the other four runs.", flush=True)
        estimates = {model: uncertainty(model, training, report["fold_fits"][model][run_id]) for model in MODELS}
        uncertainties[run_id] = estimates
        # Compute intervals BEFORE looking at the target values.
        predicted = predict_condition(run_id, held_out["condition"], estimates)
        measured = held_out["observations"].loc[held_out["use"]].copy()
        merge_columns = [name for name in predicted if name not in measured or name in ["run_id", "time_min", "species"]]
        table = measured.merge(predicted[merge_columns], on=["run_id", "time_min", "species"],
                               how="left", validate="one_to_one")
        if len(table) != len(measured) or table[OUTPUT_COLUMNS].isna().any().any():
            raise RuntimeError(f"Missing interval predictions for {run_id}.")
        table["training_run_ids"] = ",".join(run["id"] for run in training)
        predictions.append(table)
    combined = pd.concat(predictions, ignore_index=True)
    expected = {(run["id"], float(row.time_min), row.species)
                for run in runs for row in run["observations"].loc[run["use"]].itertuples(index=False)}
    actual = list(combined[["run_id", "time_min", "species"]].itertuples(index=False, name=None))
    if len(actual) != len(set(actual)) or set(actual) != expected:
        raise RuntimeError("Coverage check has missing or duplicate observations.")
    for row in combined.itertuples(index=False):
        if set(row.training_run_ids.split(",")) != identifiers-{row.run_id}:
            raise RuntimeError("Held-out data entered its own training partition.")
    return combined, uncertainties


def make_prediction_figure(table):
    runs = list(table.run_id.unique())
    fig, axes = plt.subplots(int(np.ceil(len(runs)/4)), 4, figsize=(13, 3.0*np.ceil(len(runs)/4)), squeeze=False)
    for ax, run_id in zip(axes.flat, runs):
        group = table[table.run_id.eq(run_id) & table.species.eq("Q")].sort_values("time_min")
        ax.fill_between(group.time_min, group.lower90, group.upper90, color="#D2E3EA", label="Mixture 90% interval")
        ax.plot(group.time_min, group.M2_prediction, color="#9C3F74", ls="--", lw=1.2, label="M2 fitted mean")
        ax.plot(group.time_min, group.M4_prediction, color="#176B87", ls=":", lw=1.6, label="M4 fitted mean")
        ax.plot(group.time_min, group.prediction, color="#243746", lw=1.1, label="Equal-weight mean")
        row = group.iloc[0]
        ax.set_title(f"{run_id}: {row.temperature_C:g} C; A0={row.A0_M:g}, B0={row.B0_M:g} M", fontsize=9)
        ax.set_xlabel("Time [min]")
        ax.set_ylabel("Q [mol/L]")
        ax.ticklabel_format(axis="y", style="sci", scilimits=(0, 0))
        ax.set_ylim(bottom=0)
        ax.grid(alpha=0.15)
    for ax in list(axes.flat)[len(runs):]:
        ax.axis("off")
    handles, labels = axes.flat[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="upper center", bbox_to_anchor=(0.5, 0.965), ncol=4, fontsize=8)
    fig.suptitle("System B: Q predictions expose uncertainty between mechanisms", fontsize=13)
    fig.text(0.5, 0.013, "Approximate pointwise intervals include parameter and assay uncertainty. New initial charges remain unvalidated; lines join requested times.",
             ha="center", fontsize=8)
    fig.tight_layout(rect=(0, 0.035, 1, 0.93))
    fig.savefig(FIGURES / "system_B_heldout_Q_intervals.png", dpi=160)
    plt.close(fig)


def make_coverage_figure(by_species, by_run):
    fig, axes = plt.subplots(1, 2, figsize=(11.5, 4.7))
    for ax, table, key, title in [(axes[0], by_species, "species", "Numeric measurements by species"),
                                   (axes[1], by_run, "run_id", "Numeric measurements by excluded run")]:
        ax.bar(table[key], 100*table.numeric_coverage, color="#176B87", width=0.58)
        ax.axhline(90, color="#D97941", ls="--", label="Nominal 90%")
        for index, row in enumerate(table.itertuples(index=False)):
            ax.text(index, 100*row.numeric_coverage+2, f"{row.numeric_covered}/{row.n_numeric}", ha="center", fontsize=9)
        ax.set_ylim(0, 112)
        ax.set_ylabel("Excluded numeric observations inside interval [%]")
        ax.set_title(title)
        ax.grid(axis="y", alpha=0.15)
        ax.set_axisbelow(True)
        ax.legend(loc="lower left", fontsize=8)
    fig.suptitle("System B: retrospective coverage of approximate 90% intervals", fontsize=13)
    fig.text(0.5, 0.025, "Censored Q is assessed separately. Five runs share training data; new initial charges and simultaneous trajectory coverage are untested.",
             ha="center", fontsize=8)
    fig.tight_layout(rect=(0, 0.065, 1, 0.94))
    fig.savefig(FIGURES / "system_B_interval_coverage.png", dpi=160)
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--check-intervals", action="store_true", help="also assess coverage using the saved leave-one-run-out fits")
    args = parser.parse_args()
    started = time.perf_counter()
    RESULTS.mkdir(exist_ok=True)
    FIGURES.mkdir(parents=True, exist_ok=True)
    runs = data_loader.load_runs()
    fit_path = RESULTS / "system_B_fits.json"
    report = checked_report(fit_path)
    if set(report["training_run_ids"]) != {run["id"] for run in runs}:
        raise RuntimeError("Saved full-data fits use different experiments.")
    print("System B prediction: equal-weight M2/M4 mixture.", flush=True)
    estimates = {}
    for model in MODELS:
        print(f"  Checking {model} parameter uncertainty on all five released runs.", flush=True)
        estimates[model] = uncertainty(model, runs, report["models"][model])
    condition_table, conditions = load_conditions()
    detailed = pd.concat([predict_condition(run_id, condition, estimates) for run_id, condition in conditions], ignore_index=True)
    check_submission(detailed, condition_table)
    submission = detailed[OUTPUT_COLUMNS]
    submission.to_csv(ROOT / "submission" / "predictions.csv", index=False, float_format="%.12g")
    detailed.to_csv(RESULTS / "system_B_prediction_details.csv", index=False)
    temperatures = [run["condition"].temperature_c for run in runs]
    charges = {(run["condition"].A0, run["condition"].B0) for run in runs}
    outside_temperature = [run_id for run_id, c in conditions if not min(temperatures) <= c.temperature_c <= max(temperatures)]
    new_charges = [run_id for run_id, c in conditions if (c.A0, c.B0) not in charges]
    limitations = [
        "Local likelihood-normal and linear propagation approximations; not exact posterior predictive intervals.",
        "Gaussian independent assay errors are assumptions; the documented SD formula is used without an extra fitted multiplier.",
        "Equal M2/M4 weights are a transparent predictive choice after a near tie, not posterior model probabilities.",
        "Weights and interval scale are not tuned on excluded values; the retained pair was chosen using released data.",
        "All released runs share one initial charge, so calibration for changed initial concentrations remains untested.",
        "Interval endpoints are restricted to nonnegative concentrations; fitting still treats numeric zeros as numeric Gaussian observations.",
        "No additional model discrepancy term is estimated; mechanisms outside the retained pair are not represented.",
        "Initial charges and temperatures are treated as exact; intervals cover individual future assays, not whole trajectories.",
    ]
    save_json(RESULTS / "system_B_prediction_summary.json", {
        "method": "equal-weight normal mixture, observed likelihood curvature and delta-method propagation",
        "nominal_pointwise_coverage": 0.9, "model_weights": dict(zip(MODELS, WEIGHTS.tolist())),
        "source_sha256": provenance(), "fit_report_sha256": digest(fit_path),
        "n_predictions": len(submission), "n_heldout_runs": len(conditions),
        "runs_outside_released_temperature_range": outside_temperature,
        "runs_with_unvalidated_initial_charges": new_charges,
        "parameter_uncertainty": estimates, "limitations": limitations,
    })
    make_prediction_figure(detailed)
    print(f"\nWrote submission/predictions.csv: {len(submission)} rows; no missing or duplicate keys.", flush=True)
    print(f"Unvalidated initial charges: {', '.join(new_charges) or 'none'}.")
    print(f"Outside released temperature range: {', '.join(outside_temperature) or 'none'}.")
    if args.check_intervals:
        cv_path = RESULTS / "system_B_cv_summary.json"
        cv_report = checked_report(cv_path, cross_validation=True)
        predicted, fold_uncertainties = validate_intervals(runs, cv_report)
        overall = coverage_summary(predicted)
        by_species = pd.DataFrame([{"species": species, **coverage_summary(group)}
                                   for species, group in predicted.groupby("species", sort=True)])
        by_run = pd.DataFrame([{"run_id": run_id, **coverage_summary(group)}
                               for run_id, group in predicted.groupby("run_id", sort=True)])
        predicted.to_csv(RESULTS / "system_B_interval_cv_predictions.csv", index=False)
        by_species.to_csv(RESULTS / "system_B_interval_coverage_by_species.csv", index=False)
        by_run.to_csv(RESULTS / "system_B_interval_coverage_by_run.csv", index=False)
        save_json(RESULTS / "system_B_interval_cv_summary.json", {
            "method": "retrospective leave-one-run-out assessment of fixed equal-weight M2/M4 intervals",
            "source_sha256": provenance(), "cv_report_sha256": digest(cv_path),
            "overall": overall, "by_species": by_species.to_dict(orient="records"),
            "by_run": by_run.to_dict(orient="records"), "fold_parameter_uncertainty": fold_uncertainties,
            "limitations": limitations+["Only five experiments; overlapping training sets and related measurements preclude an independent-binomial interpretation."]})
        make_coverage_figure(by_species, by_run)
        print("\nSystem B interval assessment complete.")
        print(f"Scored {len(predicted)} positive-time measurements once: {overall['n_numeric']} numeric, {overall['n_censored']} censored.")
        print(f"Numeric coverage: {overall['numeric_covered']}/{overall['n_numeric']} = {100*overall['numeric_coverage']:.1f}% (nominal 90%).")
        columns = ["species", "n_numeric", "numeric_covered", "numeric_coverage", "n_censored"]
        print("\nCoverage by species:")
        print(by_species[columns].to_string(index=False, float_format=lambda value: f"{value:.4f}"))
        print(f"\nIncluding censoring, coverage lies between {100*overall['all_observation_coverage_lower_bound']:.1f}% and {100*overall['all_observation_coverage_upper_bound']:.1f}%.")
        print("No censored value was replaced by zero or by the LOQ.")
        print("No interval-width multiplier was tuned on excluded measurements.")
        print("This check does not establish 90% coverage at new initial charges.")
        print("Saved interval CV summary, three diagnostic CSV files and system_B_interval_coverage.png.")
    print("Saved prediction summary, detailed predictions and system_B_heldout_Q_intervals.png.")
    print(f"Prediction and interval work took {time.perf_counter()-started:.1f} seconds on this computer.")


if __name__ == "__main__":
    main()
