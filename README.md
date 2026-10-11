# Reaction Mechanism Discovery Challenge

A student modelling challenge: infer which of six candidate reaction networks produced a
blind kinetic dataset, predict held-out conditions with calibrated intervals, and design a
confirmatory experimental campaign inside a materials budget.

**The brief is the handout: [`UCL_Hackathon_Brief.pdf`](UCL_Hackathon_Brief.pdf)** — read it
first. This README is the repo tour and reference (install, layout, the candidate library);
`data/README.md` is the data reference (schema, LOQ, censoring, noise model, budget).

## Quickstart

Prerequisites: Python 3.10 or later and the `just` command runner. Use an active
virtual environment. On Windows with Git Bash:

```bash
python -m venv .venv
source .venv/Scripts/activate
```

On Linux/macOS, activate with `source .venv/bin/activate` instead. If `just` is
not already available, install its packaged binary inside the environment:

```bash
python -m pip install rust-just
just --version
```

From the repository root, run the four required recipes:

```bash
just install     # deps (five pinned packages) + the local `candidates` package
just predict     # -> submission/predictions.csv
just suggest    # -> submission/campaign.yaml
just replicate  # -> submission/figures/*, including System A model selection
```

To run those four commands, time the workflow and check output consistency:

```bash
python check_submission.py --run-workflow
```

Keep the analysis scripts, `results/` and `submission/` in the repository.
The reproduction path uses saved fitted parameters and starting-fit records.
It checks their provenance and likelihoods against the data, then recomputes
predictions, scores, interval assessments and figures. It does not simply copy
old images. Source/data mismatches stop regeneration; missing fitting files
trigger the corresponding full fit, which can take substantially longer.

The ten-minute recipe budget is tested with the saved fits included. Repeating
all optimisation from scratch is a separate, longer verification path. To test
the submission, clone it into a fresh directory, create a new environment and
run the four recipes there. Dependency installation may use the network; the
scientific recipes require no network access. `.venv/` is not submitted.

## Analysis and full refitting

| Command | Purpose |
|---------|---------|
| `python fit_system_a.py` | Fit the disclosed M1 reference model. |
| `python validate_system_a.py` | Predict each excluded System A experiment using M1. |
| `python validate_system_a_selection.py` | Fit M1-M6 on System A, compare fixed candidates and validate training-only AICc selection. |
| `python load_system_b.py` | Check numeric, censored and absent assays. |
| `python fit_system_b.py` | Fit all six candidates with the censored assay likelihood. |
| `python validate_system_b.py` | Compare predictions on entire excluded System B experiments. |
| `python predict_system_b.py --check-intervals` | Generate held-out predictions and assess intervals on released experiments. |
| `python suggest_campaign.py` | Design and audit the proposed experimental campaign. |
| `just replicate` | Regenerate every analysis figure from checked saved results. |

System A selection uses 2% relative assay SD for A/B/P and 5% for Q. Each
candidate is fitted from four fixed starts to the same observations. The
training-only AICc winner is chosen within each four-experiment partition and
then scored on its excluded experiment. The full-data AICc ranking and the
fixed-candidate cross-validation scores are also reported separately. Time-zero
rows are excluded because their predictions are fixed initial conditions.
All positive-time observations, including the A4 discrepancies, are retained.
The new selection fits and their starting records are stored together in
`results/system_A_selection_fits.json`. Regeneration does not reoptimise them.

Reintegrating identical saved parameters can produce tiny likelihood differences
between platforms. The System A selection check allows an absolute difference
of at most 0.001 in the summed negative log likelihood (at most 0.002 per AICc
score). A Windows/Python 3.13.7 versus Linux/Python 3.12.14 diagnostic, with the
same pinned dependencies and identical sources/data, found a maximum difference
of 0.000406 across 36 fits and no change in any model choice. The tolerance only
affects saved-fit verification; source/data hashes are still checked, and saved
selected likelihoods must exactly match their best successful starting record.
Fitting objectives, optimiser settings and the noise model are unchanged.

The System A comparison has nine regression checks for its exact likelihood,
agreement with the original M1 implementation, experiment-level separation,
training-only model choice, and the numerical portability checks. They cover
the reported Windows differences and rejection of changed source/data,
materially different likelihoods and inconsistent starting records.
Run them without an extra test dependency:

```bash
python -m unittest discover -s tests -p test_system_a_selection.py
```

This is a retrospective reference check against a disclosed mechanism, not a
blinded discovery experiment. AICc is approximate, especially for boundary fits
and weakly identified parameters. Fast-intermediate mechanisms may closely
approximate direct pathways at the available sampling times. Selection
frequency across five overlapping folds is not a probability of mechanism
identity. Neither these point-parameter scores nor the original System A
validation establish 90% interval coverage.

System B retains its own noise and censoring model. Predictions and campaign
forecasts use equal M2/M4 weights, expressing the unresolved distinction in the
released data. System A reference results do not change those weights. System B
interval assessment on the released initial charge does not establish coverage
at unseen initial charges. Campaign accuracy figures are simulations, not new
experimental validation. See the diagnostic JSON/CSV files in `results/`.

## Layout

```
data/
  system_A/    disclosed reference reaction (MECHANISM.md) + measurements.csv
  system_B/    the blind system -- measurements.csv (one species is never measured)
  heldout_conditions.csv   conditions to predict; README.md documents schema, LOQ, budget
src/candidates/   the six candidate models (M1-M6) and one simulate() interface
baseline/        original starter implementation; overview figure still used
getting_started.py   orientation: prints a tour, writes figures to scratch/
results/         fitted parameters, provenance, validation scores and diagnostics
submission/      hypotheses, held-out predictions, proposed campaign and figures
tests/           sanity tests for the candidate library
```

## The candidate library

```python
from candidates import CANDIDATES, Conditions, simulate
cond = Conditions(temperature_c=75.0, A0=1.0, B0=1.0, times_min=[0, 60, 120, 240, 480])
# illustrative parameters only -- estimating the real ones is your job
df = simulate("M2", cond, {"k1":0.012,"Ea1":58,"k2":0.006,"Ea2":52,"k3":7e-4,"Ea3":66})
```

Rate constants are given at a **reference temperature of 75 °C** plus an activation energy
(kJ/mol); concentrations are mol/L, time is minutes. See `data/README.md` for the data schema,
the LOQ / censoring convention, and the campaign budget.

Dependencies: `numpy`, `scipy`, `pandas`, `matplotlib`, `pyyaml` (pinned in `requirements.txt`).
No network access is required at run time.
