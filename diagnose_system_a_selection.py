"""Read-only diagnostic for saved System A selection likelihoods.

Save beside validate_system_a_selection.py and run:
    python diagnose_system_a_selection.py

This does not fit models, overwrite results, or change any validation checks.
"""

from __future__ import annotations

import json
import platform
from importlib.metadata import version

import numpy as np

import validate_system_a_selection as selection


def main():
    print("System A saved-fit diagnostic (read only)", flush=True)
    print(f"Python {platform.python_version()}; {platform.system()} {platform.machine()}")
    for package in ("numpy", "scipy", "pandas", "matplotlib", "PyYAML"):
        print(f"  {package}: {version(package)}")

    path = selection.RESULTS / selection.FIT_FILE
    report = json.loads(path.read_text(encoding="utf-8"))
    current_manifest = selection.source_manifest()
    saved_manifest = report["source_sha256"]
    changed = sorted(name for name in set(current_manifest) | set(saved_manifest)
                     if current_manifest.get(name) != saved_manifest.get(name))
    if changed:
        print("Source/data checks FAILED for: " + ", ".join(changed))
        print("Stopped: likelihood differences cannot be attributed to numerical portability.")
        return
    print("Source/data hashes: all match (normal LF/CRLF differences accepted).")

    runs = selection.reference.load_runs()
    failures, errors, rows = [], [], []
    print("\nModel  Training set       Saved NLL      Current NLL       Difference      Tolerance  Check")
    for model in selection.MODELS:
        entries = [("all runs", runs, report["full_fits"][model])]
        entries.extend((f"exclude {held_out['id']}",
                        [run for run in runs if run["id"] != held_out["id"]],
                        report["fold_fits"][model][held_out["id"]])
                       for held_out in runs)
        for label, training, fit in entries:
            try:
                if sorted(fit["training_run_ids"]) != sorted(run["id"] for run in training):
                    raise ValueError("Saved training-run identifiers do not match.")
                problem = selection.FitProblem(model, training)
                if fit["n_observations"] != len(problem.y):
                    raise ValueError("Saved observation count does not match.")
                x = selection.coordinates(model, fit["parameters"])
                saved = float(fit["negative_log_likelihood"])
                actual = problem.nll(x)
                if not np.isfinite(saved) or not np.isfinite(actual):
                    raise ValueError("Non-finite likelihood.")
                delta = actual - saved
                tolerance = selection.SAVED_NLL_ATOL
                passes = bool(np.isclose(actual, saved, rtol=0, atol=tolerance))
                row = {"model": model, "label": label, "fit": fit,
                       "actual": actual, "delta": delta, "tolerance": tolerance}
                rows.append(row)
                print(f"{model:<6} {label:<13} {saved:14.7f} {actual:16.7f} "
                      f"{delta:+16.8g} {tolerance:14.8g}  {'PASS' if passes else 'FAIL'}",
                      flush=True)
                if not passes:
                    repeated = problem.nll(x)
                    failures.append(row)
                    print(f"       Repeated calculation minus first: {repeated - actual:+.12g}",
                          flush=True)
            except Exception as exc:
                errors.append((model, label, str(exc)))
                print(f"{model:<6} {label:<13} ERROR: {type(exc).__name__}: {exc}", flush=True)

    print(f"\nEvaluated {len(rows)}/36 saved fits; {len(failures)} outside the existing tolerance; "
          f"{len(errors)} evaluation errors.")
    if rows:
        largest = max(rows, key=lambda row: abs(row["delta"]))
        print(f"Largest absolute NLL difference: {abs(largest['delta']):.12g} "
              f"({largest['model']}, {largest['label']}).")

    print("\nTraining AICc selections using saved versus recalculated likelihoods:")
    for label in ["all runs"] + [f"exclude {run['id']}" for run in runs]:
        subset = [row for row in rows if row["label"] == label]
        if len(subset) != len(selection.MODELS):
            print(f"  {label}: incomplete; cannot compare.")
            continue
        saved_scores = sorted((selection.aicc(row["fit"]), row["model"]) for row in subset)
        actual_scores = sorted((selection.aicc({**row["fit"], "negative_log_likelihood": row["actual"]}),
                                row["model"]) for row in subset)
        saved_winner = saved_scores[0][1]
        actual_winner = actual_scores[0][1]
        saved_gap = saved_scores[1][0] - saved_scores[0][0]
        actual_gap = actual_scores[1][0] - actual_scores[0][0]
        print(f"  {label:<13} saved={saved_winner}, current={actual_winner}; "
              f"runner-up gaps={saved_gap:.8f}, {actual_gap:.8f}")

    print("\nDiagnostic complete. No analysis outputs or validation tolerances were changed.")
    print("This diagnostic does not replace the submission check.")


if __name__ == "__main__":
    main()
