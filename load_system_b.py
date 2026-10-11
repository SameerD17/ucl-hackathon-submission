"""Validate and inspect System B without discarding censored or missing assays.

Save this file beside justfile and run: python load_system_b.py

Reusable entry points for the later model fitter:
    load_measurements() -> one row per REPORTED observation
    load_runs()         -> conditions, observations, and simulation row indices
    measurement_sd(species, concentration) -> the documented assay SD

Numeric zeros remain numeric observations. A <LOQ row has value_M=NaN,
censored=True and loq_M=3e-5. An absent assay has NO observation row.
The separate coverage table describes gaps; it is never used as fitting data.
Time-zero observations are retained and marked use_for_fit=False, because
initial conditions are fixed in the supplied candidate models.

This script does not fit, rank mechanisms, or produce prediction intervals.
"""

from __future__ import annotations

import hashlib
import json
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT / "src"))
from candidates import Conditions

DATA_PATH = ROOT / "data" / "system_B" / "measurements.csv"
SPECIES = ("A", "B", "P", "Q")
COLUMNS = ("system", "run_id", "temperature_C", "A0_M", "B0_M", "time_min",
           "species", "value", "unit", "censored")
NOISE = {"A": (0.03, 3.5e-3), "B": (0.03, 3.5e-3),
         "P": (0.03, 3.5e-3), "Q": (0.15, 2.2e-5)}
Q_LOQ_M = 3.0e-5


def measurement_sd(species, concentration):
    """Evaluate sqrt((rel*c)^2 + abs^2), using model means when fitting.

    This gives a noise scale, not a calibrated prediction interval. No error
    distribution or treatment of censored likelihoods is chosen by this helper.
    """
    if species not in NOISE:
        raise ValueError(f"Unknown measured species: {species!r}")
    values = np.asarray(concentration, dtype=float)
    if not np.isfinite(values).all() or (values < 0).any():
        raise ValueError("Concentrations for the noise model must be finite and nonnegative.")
    relative, absolute = NOISE[species]
    return np.hypot(relative * values, absolute)


def load_measurements(path=DATA_PATH):
    """Load the documented long format, preserving the meaning of every row."""
    path = Path(path)
    if not path.is_file():
        raise FileNotFoundError(
            f"Measurement file not found: {path}. Save load_system_b.py beside justfile."
        )
    data = pd.read_csv(path, dtype=str, keep_default_na=False)
    if len(data.columns) != len(COLUMNS) or set(data.columns) != set(COLUMNS):
        raise ValueError(f"Expected exactly these columns: {', '.join(COLUMNS)}")
    if data.empty:
        raise ValueError("The measurement file is empty.")
    data = data.loc[:, list(COLUMNS)].copy()
    for column in COLUMNS:
        data[column] = data[column].str.strip()
        if data[column].eq("").any():
            raise ValueError(f"Blank {column!r} entry: an absent assay must have no row.")
    if not data.system.eq("B").all():
        raise ValueError("This loader expects only System B measurements.")
    if not data.unit.eq("mol/L").all():
        raise ValueError("All measurement units must be mol/L.")
    if not data.species.isin(SPECIES).all():
        raise ValueError("The only measured species in the released schema are A, B, P, Q.")

    numeric_columns = ["temperature_C", "A0_M", "B0_M", "time_min"]
    for column in numeric_columns:
        data[column] = pd.to_numeric(data[column], errors="raise").astype(float)
    if not np.isfinite(data[numeric_columns].to_numpy()).all():
        raise ValueError("Temperatures, initial concentrations and times must be finite.")
    if (data.temperature_C <= -273.15).any():
        raise ValueError("Temperatures must be above absolute zero.")
    if (data[["A0_M", "B0_M"]] <= 0).any().any() or (data.time_min < 0).any():
        raise ValueError("Initial concentrations must be positive and times nonnegative.")

    flags = data.censored.str.lower()
    if not flags.isin(["true", "false"]).all():
        raise ValueError("Censored flags must be true or false.")
    data["censored"] = flags.eq("true")
    if not data.censored.eq(data.value.eq("<LOQ")).all():
        raise ValueError("Each <LOQ value must have censored=true, and conversely.")
    if (data.censored & data.species.ne("Q")).any():
        raise ValueError("The released data contract permits censoring only for Q.")

    # Numeric conversion is deliberately separate from the original text value.
    data["value_M"] = pd.to_numeric(data.value.mask(data.censored), errors="raise")
    observed = data.loc[~data.censored, "value_M"].to_numpy(float)
    if not np.isfinite(observed).all() or (observed < 0).any():
        raise ValueError("Uncensored concentrations must be finite and nonnegative.")
    if (data.species.eq("Q") & ~data.censored & data.value_M.lt(Q_LOQ_M)).any():
        raise ValueError("A numeric Q below its documented LOQ must be reported as <LOQ.")
    if data.duplicated(["run_id", "time_min", "species"]).any():
        raise ValueError("Duplicate run/time/species keys would double-count observations.")
    for run_id, group in data.groupby("run_id"):
        if len(group[["temperature_C", "A0_M", "B0_M"]].drop_duplicates()) != 1:
            raise ValueError(f"Inconsistent experimental conditions within run {run_id}.")

    data["loq_M"] = np.where(data.species.eq("Q"), Q_LOQ_M, np.nan)
    data["measurement_kind"] = np.where(data.censored, "left_censored", "numeric")
    data["use_for_fit"] = data.time_min.gt(0)
    data["relative_sd"] = data.species.map({key: value[0] for key, value in NOISE.items()})
    data["absolute_sd_M"] = data.species.map({key: value[1] for key, value in NOISE.items()})
    return data.sort_values(["run_id", "time_min", "species"]).reset_index(drop=True)


def load_runs(path=DATA_PATH):
    """Keep the sparse observation table plus indices into a dense simulation.

    In each run, a simulation array whose columns are SPECIES can be indexed
    with [time_index, species_index] to get one prediction per observation.
    The fit mask does not discard positive-time censored observations.
    """
    data = load_measurements(path)
    runs = []
    for run_id, group in data.groupby("run_id", sort=True):
        observations = group.reset_index(drop=True).copy()
        times = np.unique(observations.time_min.to_numpy(float))
        first = observations.iloc[0]
        runs.append({
            "id": run_id,
            "condition": Conditions(float(first.temperature_C), float(first.A0_M),
                                    float(first.B0_M), times),
            "times": times,
            "observations": observations,
            "time_index": np.searchsorted(times, observations.time_min.to_numpy(float)),
            "species_index": observations.species.map(
                {species: index for index, species in enumerate(SPECIES)}
            ).to_numpy(int),
            "use": observations.use_for_fit.to_numpy(bool),
        })
    return runs


def diagnostic_tables(data):
    """Describe missing assays separately; never fill the measurement table."""
    coverage_rows, run_rows = [], []
    for run_id, group in data.groupby("run_id", sort=True):
        times = np.sort(group.time_min.unique())
        first = group.iloc[0]
        lookup = group.set_index(["time_min", "species"]).measurement_kind.to_dict()
        for sample_time in times:
            for species in SPECIES:
                coverage_rows.append({"run_id": run_id, "time_min": sample_time,
                                      "species": species,
                                      "status": lookup.get((sample_time, species), "not_assayed")})
        q = group[group.species.eq("Q")]
        run_rows.append({
            "run_id": run_id, "temperature_C": float(first.temperature_C),
            "A0_M": float(first.A0_M), "B0_M": float(first.B0_M),
            "sample_times": len(times), "measurement_rows": len(group),
            "numeric_rows": int((~group.censored).sum()),
            "censored_rows": int(group.censored.sum()),
            "numeric_zeros": int(group.value_M.eq(0).sum()),
            "positive_time_rows": int(group.use_for_fit.sum()),
            "Q_assays": len(q), "Q_not_assayed": len(times)-len(q),
        })
    coverage = pd.DataFrame(coverage_rows)
    species_rows = []
    for species in SPECIES:
        group = data[data.species.eq(species)]
        species_rows.append({
            "species": species, "measurement_rows": len(group),
            "numeric_rows": int((~group.censored).sum()),
            "censored_rows": int(group.censored.sum()),
            "not_assayed": int((coverage.species.eq(species) &
                                coverage.status.eq("not_assayed")).sum()),
            "numeric_zeros": int(group.value_M.eq(0).sum()),
        })
    return coverage, pd.DataFrame(run_rows), pd.DataFrame(species_rows)


def make_figure(data, output):
    groups = list(data.groupby("run_id", sort=True))
    plt.rcParams.update({"font.size": 9})
    fig, axes = plt.subplots(len(groups), 4, figsize=(13, 2.15*len(groups)), squeeze=False)
    for row, (run_id, run) in enumerate(groups):
        for column, species in enumerate(SPECIES):
            ax = axes[row, column]
            group = run[run.species.eq(species)]
            numeric = group[~group.censored]
            ax.scatter(numeric.time_min, numeric.value_M, s=18, color="#176B87", zorder=3)
            if species == "Q":
                censored = group[group.censored]
                ax.axhspan(0, Q_LOQ_M, color="#D97941", alpha=0.12)
                ax.axhline(Q_LOQ_M, color="#D97941", ls="--", lw=0.8)
                ax.scatter(censored.time_min, np.full(len(censored), Q_LOQ_M),
                           marker="v", s=32, color="#D97941", zorder=4)
                ax.ticklabel_format(axis="y", style="sci", scilimits=(0, 0))
            if row == 0:
                ax.set_title(species, fontweight="bold")
            if column == 0:
                ax.set_ylabel(f"{run_id}: {run.temperature_C.iloc[0]:g} C\nConcentration [mol/L]")
            if row == len(groups)-1:
                ax.set_xlabel("Time [min]")
            # Leave enough room below zero to display reported-zero markers fully.
            ax.set_ylim(bottom=-0.04*ax.get_ylim()[1])
            ax.grid(alpha=0.15)
    handles = [Line2D([], [], marker="o", ls="", color="#176B87", label="Numeric measurement"),
               Line2D([], [], marker="v", ls="", color="#D97941", label="Q <LOQ (upper limit)"),
               Line2D([], [], ls="--", color="#D97941", label="Q LOQ = 3e-5 mol/L")]
    fig.suptitle("System B: released measurements, including censored Q assays", fontsize=14, y=0.99)
    fig.legend(handles=handles, loc="upper center", bbox_to_anchor=(0.5, 0.967), ncol=3, fontsize=9)
    fig.text(0.5, 0.012,
             "Triangles mark upper limits, not measured concentrations. Unassayed Q times have no point.",
             ha="center", fontsize=9)
    fig.tight_layout(rect=(0, 0.03, 1, 0.96))
    fig.savefig(output, dpi=160)
    plt.close(fig)


def main():
    data = load_measurements()
    coverage, by_run, by_species = diagnostic_tables(data)
    results = ROOT / "results"
    figures = ROOT / "submission" / "figures"
    results.mkdir(exist_ok=True)
    figures.mkdir(parents=True, exist_ok=True)
    data.to_csv(results / "system_B_measurements_checked.csv", index=False)
    coverage.to_csv(results / "system_B_assay_coverage.csv", index=False)
    by_run.to_csv(results / "system_B_data_by_run.csv", index=False)
    by_species.to_csv(results / "system_B_data_by_species.csv", index=False)
    report = {
        "system": "B", "source": "data/system_B/measurements.csv",
        "source_sha256": hashlib.sha256(DATA_PATH.read_bytes()).hexdigest(),
        "runs": int(data.run_id.nunique()), "measurement_rows": len(data),
        "numeric_rows": int((~data.censored).sum()), "censored_rows": int(data.censored.sum()),
        "numeric_zeros": int(data.value_M.eq(0).sum()),
        "positive_time_rows": int(data.use_for_fit.sum()),
        "positive_time_censored_rows": int((data.censored & data.use_for_fit).sum()),
        "unassayed_species_time_pairs": int(coverage.status.eq("not_assayed").sum()),
        "Q_LOQ_M": Q_LOQ_M,
        "noise_sd_formula": "sqrt((relative * predicted_concentration)^2 + absolute^2)",
        "noise": {species: {"relative": relative, "absolute_M": absolute}
                  for species, (relative, absolute) in NOISE.items()},
        "by_run": by_run.to_dict(orient="records"),
        "by_species": by_species.to_dict(orient="records"),
        "interpretation": [
            "Censored Q values are preserved as upper-limit information, never replaced by zero.",
            "The coverage table is diagnostic metadata, not additional observations.",
            "Numeric zeros remain uncensored numeric observations as supplied.",
            "Time-zero rows are retained, but do not inform rates when initial conditions are fixed.",
            "All released runs use the same initial charge; temperature is the only varied input.",
            "A censored likelihood and predictive uncertainty still need to be implemented and assessed.",
            "No mechanism ranking is produced by this inspection step.",
        ],
    }
    (results / "system_B_data_summary.json").write_text(json.dumps(report, indent=2)+"\n", encoding="utf-8")
    make_figure(data, figures / "system_B_measurements.png")
    print("System B data checks passed.")
    print(f"Runs: {report['runs']}; measurement rows: {len(data)}")
    print(f"Numeric measurements: {report['numeric_rows']}; <LOQ measurements: {report['censored_rows']}")
    print(f"Numeric zeros retained: {report['numeric_zeros']}")
    print(f"Unassayed Q times: {int(by_run.Q_not_assayed.sum())} (no observations added)")
    print(f"Positive-time rows for later fitting: {report['positive_time_rows']}, "
          f"including {report['positive_time_censored_rows']} censored Q rows")
    print("\nCounts by species:")
    print(by_species.to_string(index=False))
    print("\nAll released runs start at A0=1.0 M and B0=0.25 M; temperatures span 65-85 C.")
    print("No candidate mechanism has been fitted or selected by this script.")
    print("Saved results/system_B_data_summary.json and four diagnostic CSV files.")
    print("Saved submission/figures/system_B_measurements.png")


if __name__ == "__main__":
    main()
