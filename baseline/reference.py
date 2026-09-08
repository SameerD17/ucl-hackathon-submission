"""Shared helpers for the BASELINE reference solution.

This baseline is deliberately naive -- one candidate fitted crudely, wide fixed intervals,
throwaway experiments. Its only job is to make `just install/predict/suggest/replicate` run
end-to-end so teams don't lose an evening to plumbing. Beat it with actual reasoning.

Pure numpy/scipy/pandas -- nothing outside the five pinned packages.
"""
from __future__ import annotations

import os
import sys

import numpy as np
import pandas as pd
from scipy.integrate import solve_ivp
from scipy.optimize import least_squares

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "src"))
from candidates import CANDIDATES  # noqa: E402

DATA_B = os.path.join(ROOT, "data", "system_B", "measurements.csv")
HELDOUT = os.path.join(ROOT, "data", "heldout_conditions.csv")
SUBMISSION = os.path.join(ROOT, "submission")

BASELINE_MODEL = "M1"     # deliberately a rejected model -- the baseline is meant to be bad
OBSERVED = ["A", "B", "P", "Q"]


def load_system_b() -> list[dict]:
    df = pd.read_csv(DATA_B)
    runs = []
    for rid, g in df.groupby("run_id"):
        obs = {}
        for sp, gg in g.groupby("species"):
            gg = gg[gg["value"] != "<LOQ"]          # naive: just drop censored points
            obs[sp] = (gg["time_min"].to_numpy(float),
                       gg["value"].astype(float).to_numpy())
        runs.append({"run_id": rid, "T": float(g["temperature_C"].iloc[0]),
                     "A0": float(g["A0_M"].iloc[0]), "B0": float(g["B0_M"].iloc[0]), "obs": obs})
    return runs


def _sim(model, params, T, A0, B0, times):
    cand = CANDIDATES[model]
    y0 = np.array([A0 if s == "A" else B0 if s == "B" else 0.0 for s in cand.species])
    sol = solve_ivp(lambda t, y: cand.rhs(y, params, T), (0, float(times.max())), y0,
                    t_eval=times, method="LSODA", rtol=1e-7, atol=1e-10)
    return {s: np.clip(sol.y[i], 0, None) for i, s in enumerate(cand.species)}


def fit_baseline(model: str = BASELINE_MODEL) -> dict[str, float]:
    """Crude single-start fit, unweighted, censored points dropped. Intentionally weak."""
    runs = load_system_b()
    cand = CANDIDATES[model]
    names = cand.param_names

    def resid(x):
        params = {n: (10.0 ** x[i] if n.startswith("k") else x[i]) for i, n in enumerate(names)}
        out = []
        for r in runs:
            tall = np.unique(np.concatenate([v[0] for v in r["obs"].values()]))
            sim = _sim(model, params, r["T"], r["A0"], r["B0"], tall)
            idx = {t: i for i, t in enumerate(tall)}
            for sp, (ts, vs) in r["obs"].items():
                if sp not in sim:
                    continue
                for t, v in zip(ts, vs):
                    out.append(sim[sp][idx[t]] - v)   # unweighted (naive)
        return np.array(out)

    x0 = np.array([-1.5 if n.startswith("k") else 60.0 for n in names])
    lb = np.array([-6.0 if n.startswith("k") else 20.0 for n in names])
    ub = np.array([2.0 if n.startswith("k") else 130.0 for n in names])
    sol = least_squares(resid, x0, bounds=(lb, ub), max_nfev=300)
    return {n: (10.0 ** sol.x[i] if n.startswith("k") else sol.x[i]) for i, n in enumerate(names)}


def predict(params: dict[str, float], model: str = BASELINE_MODEL,
            rel_halfwidth: float = 0.5) -> pd.DataFrame:
    """Predict held-out conditions with absurdly wide fixed relative intervals."""
    cond = pd.read_csv(HELDOUT)
    rows = []
    for rid, g in cond.groupby("run_id"):
        times = g["time_min"].to_numpy(float)
        T, A0, B0 = float(g["temperature_C"].iloc[0]), float(g["A0_M"].iloc[0]), float(g["B0_M"].iloc[0])
        sim = _sim(model, params, T, A0, B0, times)
        for sp in OBSERVED:
            for t, v in zip(times, sim.get(sp, np.zeros_like(times))):
                lo = max(0.0, v * (1 - rel_halfwidth))
                hi = v * (1 + rel_halfwidth) + 1e-4
                rows.append({"run_id": rid, "time_min": round(float(t), 4), "species": sp,
                             "prediction": f"{v:.6g}", "lower90": f"{lo:.6g}", "upper90": f"{hi:.6g}"})
    return pd.DataFrame(rows)


def ensure_submission_dir():
    os.makedirs(SUBMISSION, exist_ok=True)
    os.makedirs(os.path.join(SUBMISSION, "figures"), exist_ok=True)
