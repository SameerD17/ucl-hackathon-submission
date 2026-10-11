"""Design a single-block campaign to distinguish the surviving M2/M4 mechanisms.

    python suggest_campaign.py                 # design, audit, simulate, plot
    python suggest_campaign.py --figures-only  # redraw checked saved results

No new packages or runtime network access are required. This proposes future
experiments; all synthetic outcomes used for the design audit stay separate
from the released measurements, fitted models and held-out submission forecasts.

Mechanistic argument at fixed temperature:
  M2: dQ/dP = k3/k2, so Q/P is constant for zero initial P and Q.
  M4: dQ/dP = (k3/k2)*A(t), so changing A and its depletion changes Q/P.
Temperature already varies in the released data. Fix new runs at 75 C and
choose three complementary charges: A-depleted, balanced, and A-rich. There
are no interventions or measurements of the unobserved intermediate I.

Search: a disclosed 7 x 7 charge grid at 75 C, a fixed time grid up to 960 min,
and exactly three runs (the smallest legal campaign). Each run receives time
zero, an early product-formation sample, a mid-course sample, and four samples
chosen for discrimination. Enumerate every feasible triple of the three roles.
The screening score is the sum of marginal Bhattacharyya distances between
M2/M4 predictive assays, with local parameter uncertainty and the exact stated
noise SD. Q's below-LOQ probability is one reporting category, not a zero value.
This score is a transparent HEURISTIC: it is not joint information gain, ignores
cross-time covariance during screening, and does not prove global optimality
over the full continuous design space or all possible numbers of experiments.

After selecting the campaign, forecast discrimination with nonlinear ODE draws
and censored assay likelihoods. A single parameter vector is shared across ALL
experiments in each synthetic campaign; inner integration draws are independent
of the truth draws. This preserves dependence due to parameter uncertainty.
Compare with three temperature-only runs at the historical charge and the same
number of samples. Also stress the proposed design with doubled parameter SD
and 1.5 times Q assay SD. These are sensitivity assumptions, not recalibration.
No synthetic result is an experimental result or a guarantee of identification.

A finite local-normal parameter ensemble approximates uncertainty; no MCMC,
unmodelled kinetics, inter-assay correlation or additional numerical-zero
reporting rule is included. The campaign targets discrimination, not production
optimisation or proof of the isolated-API 15 ppm w/w impurity specification.
Q/P concentration ratios alone cannot establish that mass-based specification.

General model-discrimination design background:
https://proceedings.mlr.press/v80/olofsson18a.html
"""

from __future__ import annotations

import argparse
import itertools
import json
import time
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.patches import Patch
import numpy as np
import pandas as pd
from scipy.special import log_ndtr, logsumexp, ndtr
import yaml

import predict_system_b as predictor
import fit_system_b as fitter
import load_system_b as loader
from candidates import Conditions, simulate

ROOT = Path(__file__).resolve().parent
RESULTS = ROOT / "results"
FIGURES = ROOT / "submission" / "figures"
CAMPAIGN = ROOT / "submission" / "campaign.yaml"
MODELS = ("M2", "M4")
SPECIES = ("A", "B", "P", "Q")
CHARGES = np.array([0.1, 0.25, 0.5, 0.75, 1.0, 1.25, 1.5])
TIMES = np.array([0, 2, 5, 10, 20, 30, 45, 60, 90, 120, 180, 240, 360, 480, 720, 960], dtype=float)
ROLES = ("A-depleted", "balanced", "A-rich")
SAMPLES_PER_RUN = 7
OUTER_DRAWS = 256
INNER_DRAWS = 128
SEED = 271828
MAX_A_G = 5.0
MAX_SAMPLES = 60


def a_consumed(a0):
    return 0.005*float(a0)*250.0


def role_for(a0, b0):
    if b0 >= 2*a0-1e-12:
        return "A-depleted"
    if np.isclose(a0, b0):
        return "balanced"
    # Concentrated A-rich runs make a new contrast, not another historical run.
    if a0 >= 1.25 and a0 >= 3*b0-1e-12:
        return "A-rich"
    return None


def bhattacharyya_distance(mean1, sd1, mean2, sd2, censor_limit=None):
    """Exact scalar-normal affinity, with optional left-censored reporting.

    The Q affinity is sqrt(p1_censored*p2_censored) plus the integral of
    sqrt(f1*f2) above LOQ. Censoring cannot create discrimination information.
    """
    mean1, sd1, mean2, sd2 = np.broadcast_arrays(mean1, sd1, mean2, sd2)
    if (sd1 <= 0).any() or (sd2 <= 0).any():
        raise ValueError("Positive predictive standard deviations are required.")
    variance_sum = sd1**2+sd2**2
    affinity = np.sqrt(2*sd1*sd2/variance_sum)*np.exp(-(mean1-mean2)**2/(4*variance_sum))
    if censor_limit is not None:
        product_mean = (mean1*sd2**2+mean2*sd1**2)/variance_sum
        product_sd = np.sqrt(2*sd1**2*sd2**2/variance_sum)
        censored = np.sqrt(ndtr((censor_limit-mean1)/sd1)*ndtr((censor_limit-mean2)/sd2))
        affinity = censored+affinity*ndtr((product_mean-censor_limit)/product_sd)
    return -np.log(np.clip(affinity, 1e-300, 1.0))


def select_sample_indices(mean_product, time_scores):
    # Preserve early and middle kinetics as well as highly separating late data.
    positive = np.flatnonzero(TIMES > 0)
    selected = [0]
    for fraction in (0.1, 0.5):
        order = sorted(positive, key=lambda i: (abs(mean_product[i]-fraction*mean_product[-1]), TIMES[i]))
        selected.append(next(i for i in order if i not in selected))
    ranked = sorted(positive, key=lambda i: (-time_scores[i], TIMES[i]))
    while len(selected) < SAMPLES_PER_RUN:
        # A modest spacing rule prevents an impractical cluster of late samples.
        acceptable = [i for i in ranked if i not in selected and all(abs(TIMES[i]-TIMES[j]) >= 15 for j in selected)]
        if not acceptable:
            raise RuntimeError("The time grid cannot satisfy the planned sampling rule.")
        selected.append(acceptable[0])
    return np.asarray(sorted(selected), dtype=int)


def screen_candidates(estimates):
    records = []
    for a0, b0 in itertools.product(CHARGES, repeat=2):
        role = role_for(a0, b0)
        if role is None:
            continue
        condition = Conditions(75.0, float(a0), float(b0), TIMES)
        parts = [predictor.component(model, condition, estimates[model]) for model in MODELS]
        means = [part[0] for part in parts]
        sds = [np.sqrt(part[1]+part[2]) for part in parts]
        distance = np.column_stack([
            bhattacharyya_distance(means[0][:, j], sds[0][:, j], means[1][:, j], sds[1][:, j],
                                    loader.Q_LOQ_M if species == "Q" else None)
            for j, species in enumerate(SPECIES)
        ])
        indices = select_sample_indices(0.5*(means[0][:, 2]+means[1][:, 2]), distance.sum(axis=1))
        records.append({"candidate_id": f"C{len(records)+1:02d}", "role": role,
                        "temperature_C": 75.0, "A0_M": float(a0), "B0_M": float(b0),
                        "A_consumed_g": a_consumed(a0), "sample_times_min": TIMES[indices].tolist(),
                        "screening_score": float(distance[indices].sum()),
                        "Q_screening_score": float(distance[indices, 3].sum()),
                        "M2_final_Q_M": float(means[0][-1, 3]), "M4_final_Q_M": float(means[1][-1, 3]),
                        "M2_final_Q_below_LOQ_probability": float(ndtr((loader.Q_LOQ_M-means[0][-1, 3])/sds[0][-1, 3])),
                        "M4_final_Q_below_LOQ_probability": float(ndtr((loader.Q_LOQ_M-means[1][-1, 3])/sds[1][-1, 3]))})
    return records


def select_campaign(candidates):
    pools = [[row for row in candidates if row["role"] == role] for role in ROLES]
    if any(not pool for pool in pools):
        raise RuntimeError("A mechanistic contrast has no candidates.")
    feasible = []
    for triple in itertools.product(*pools):
        cost = sum(row["A_consumed_g"] for row in triple)
        samples = sum(len(row["sample_times_min"]) for row in triple)
        if cost <= MAX_A_G+1e-12 and samples <= MAX_SAMPLES:
            feasible.append((sum(row["screening_score"] for row in triple), -cost,
                             tuple(row["candidate_id"] for row in triple), triple))
    if not feasible:
        raise RuntimeError("No feasible three-contrast campaign on the disclosed grid.")
    # Score first; exact ties prefer less material, then stable candidate IDs.
    best = sorted(feasible, key=lambda row: (-row[0], -row[1], row[2]))[0]
    selected = [{**row, "run_id": f"E{index+1}"} for index, row in enumerate(best[-1])]
    return selected, {"n_screened_conditions": len(candidates), "n_feasible_triples": len(feasible),
                      "selected_screening_score": best[0],
                      "selection_rule": "highest summed marginal censored-normal Bhattacharyya score among feasible role-constrained triples"}


def experiment_records(selected):
    return [{key: row[key] for key in ["run_id", "temperature_C", "A0_M", "B0_M", "sample_times_min"]}
            | {"assay": list(SPECIES)} for row in selected]


def audit_budget(experiments):
    if not 3 <= len(experiments) <= 10:
        raise ValueError("The campaign must have 3-10 experiments.")
    if len({row["run_id"] for row in experiments}) != len(experiments):
        raise ValueError("Experiment IDs must be unique.")
    total_a, total_samples = 0.0, 0
    for experiment in experiments:
        t = float(experiment["temperature_C"])
        a0, b0 = float(experiment["A0_M"]), float(experiment["B0_M"])
        times = np.asarray(experiment["sample_times_min"], dtype=float)
        if not np.isfinite([t, a0, b0]).all() or not 50 <= t <= 120 or not 0.1 <= a0 <= 1.5 or not 0.1 <= b0 <= 1.5:
            raise ValueError("An experiment violates temperature or charge limits.")
        if times.ndim != 1 or len(times) == 0 or not np.isfinite(times).all() or (times < 0).any() or (times > 960).any():
            raise ValueError("Sample times must be finite and between 0 and 960 min.")
        if (np.diff(times) <= 0).any():
            raise ValueError("Sample times must increase strictly without duplicates.")
        if len(experiment["assay"]) != 4 or set(experiment["assay"]) != set(SPECIES):
            raise ValueError("Each paid timepoint must assay A, B, P and Q, and cannot assay I.")
        total_a += a_consumed(a0)
        total_samples += len(times)
    if total_a > MAX_A_G+1e-10 or total_samples > MAX_SAMPLES:
        raise ValueError("Campaign exceeds the material or analytical-sample budget.")
    return {"experiments": len(experiments), "A_consumed_g": total_a,
            "A_budget_g": MAX_A_G, "A_remaining_g": MAX_A_G-total_a,
            "analytical_samples": total_samples, "sample_budget": MAX_SAMPLES,
            "unused_samples": MAX_SAMPLES-total_samples, "vessel_mL": 5.0,
            "all_constraints_passed": True}


def as_condition(experiment):
    return Conditions(experiment["temperature_C"], experiment["A0_M"], experiment["B0_M"],
                      np.asarray(experiment["sample_times_min"], dtype=float))


def draw_coordinates(model, estimate, n, scale, rng):
    center, covariance = np.asarray(estimate["center"]), np.asarray(estimate["covariance"])
    chol = np.linalg.cholesky(covariance)*scale
    lower, upper = fitter.parameter_bounds(model)
    accepted, attempted = [], 0
    while sum(len(batch) for batch in accepted) < n:
        count = n-sum(len(batch) for batch in accepted)
        proposed = center+rng.normal(size=(count, len(center)))@chol.T
        accepted.append(proposed[((proposed > lower) & (proposed < upper)).all(axis=1)])
        attempted += count
        if attempted > 2*n:
            raise RuntimeError("Too many uncertainty draws reach parameter bounds; review this approximation.")
    return np.concatenate(accepted), attempted-n


def simulated_means(model, experiments, draws):
    # One vector per complete campaign; never redraw parameters per time/run.
    output = []
    for x in draws:
        params = fitter.parameters(model, x)
        blocks = [simulate(model, as_condition(experiment), params)[list(SPECIES)].to_numpy(float).ravel()
                  for experiment in experiments]
        output.append(np.concatenate(blocks))
    return np.asarray(output)


def observation_schema(experiments, q_noise_scale):
    species = np.array([s for row in experiments for _ in row["sample_times_min"] for s in SPECIES])
    relative = np.array([loader.NOISE[s][0] for s in species])
    absolute = np.array([loader.NOISE[s][1] for s in species])
    q = species == "Q"
    relative[q] *= q_noise_scale
    absolute[q] *= q_noise_scale
    return relative, absolute, q


def integrated_log_likelihood(y, censored, means, relative, absolute):
    # Each inner vector defines a joint likelihood over every sample in every
    # run. Average these JOINT likelihoods, not separately averaged marginals.
    sigma = np.hypot(relative*means, absolute)
    result = []
    for begin in range(0, len(y), 32):
        block = y[begin:begin+32]
        flags = censored[begin:begin+32]
        z = (block[:, None, :]-means[None, :, :])/sigma[None, :, :]
        log_numeric = -0.5*z**2-np.log(sigma)[None, :, :]-0.5*np.log(2*np.pi)
        log_censored = log_ndtr((loader.Q_LOQ_M-means)/sigma)
        joint = np.where(flags[:, None, :], log_censored[None, :, :], log_numeric).sum(axis=2)
        result.append(logsumexp(joint, axis=1)-np.log(len(means)))
    return np.concatenate(result)


def forecast_discrimination(experiments, estimates, label, parameter_scale=1.0, q_noise_scale=1.0, seed=SEED):
    rng = np.random.default_rng(seed)
    relative, absolute, is_q = observation_schema(experiments, q_noise_scale)
    inner, truth, rejections = {}, {}, {}
    for model in MODELS:
        draws, rejected = draw_coordinates(model, estimates[model], INNER_DRAWS+OUTER_DRAWS, parameter_scale, rng)
        means = simulated_means(model, experiments, draws)
        inner[model], truth[model] = means[:INNER_DRAWS], means[INNER_DRAWS:]
        rejections[model] = rejected
    summaries = []
    for true_model in MODELS:
        means = truth[true_model]
        y = means+rng.normal(size=means.shape)*np.hypot(relative*means, absolute)
        censored = (y < loader.Q_LOQ_M) & is_q
        # Hide the simulated sub-LOQ value. Only the categorical event survives.
        y[censored] = loader.Q_LOQ_M
        scores = np.column_stack([integrated_log_likelihood(y, censored, inner[model], relative, absolute)
                                  for model in MODELS])
        chosen = np.asarray(MODELS)[np.argmax(scores, axis=1)]
        index = MODELS.index(true_model)
        true_advantage = scores[:, index]-scores[:, 1-index]
        summaries.append({"true_model_in_simulation": true_model, "simulated_campaigns": len(y),
                          "correct_selections": int(np.sum(chosen == true_model)),
                          "fraction_correct": float(np.mean(chosen == true_model)),
                          "true_model_log_score_advantage_5th_percentile": float(np.quantile(true_advantage, .05)),
                          "true_model_log_score_advantage_median": float(np.median(true_advantage)),
                          "mean_number_of_censored_Q_results": float(censored.sum(axis=1).mean())})
    return {"scenario": label, "parameter_SD_multiplier": parameter_scale,
            "Q_assay_SD_multiplier": q_noise_scale, "integration_draws_per_model": INNER_DRAWS,
            "simulations_per_true_model": OUTER_DRAWS,
            "model_comparison_weights": {"M2": 0.5, "M4": 0.5},
            "parameter_bound_rejections": rejections,
            "mean_fraction_correct": float(np.mean([row["fraction_correct"] for row in summaries])),
            "by_true_model": summaries,
            "interpretation": "simulation forecast conditional on these two mechanisms and the stated local uncertainty/error assumptions; not observed validation accuracy"}


def reference_campaign():
    return [{"run_id": f"R{index+1}", "temperature_C": temperature, "A0_M": 1.0, "B0_M": 0.25,
             "sample_times_min": [0., 30., 90., 180., 360., 600., 960.], "assay": list(SPECIES)}
            for index, temperature in enumerate([65., 75., 85.])]


def source_manifest():
    names = ["suggest_campaign.py", "predict_system_b.py", "fit_system_b.py", "load_system_b.py",
             "src/candidates/models.py", "src/candidates/simulate.py", "data/system_B/measurements.csv",
             "results/system_B_fits.json"]
    return {name: predictor.digest(ROOT / name) for name in names}


def prediction_details(selected, estimates):
    tables = []
    for row in selected:
        # Dense curves for scientifically useful figures; samples remain the
        # chosen seven exact times and alone count against the campaign budget.
        times = np.unique(np.r_[np.linspace(0, max(row["sample_times_min"]), 161), row["sample_times_min"]])
        table = predictor.predict_condition(row["run_id"], Conditions(row["temperature_C"], row["A0_M"], row["B0_M"], times), estimates)
        table["role"] = row["role"]
        table["planned_sample"] = table.time_min.isin(row["sample_times_min"])
        tables.append(table)
    return pd.concat(tables, ignore_index=True)


def make_figures(summary, candidates, predictions):
    FIGURES.mkdir(parents=True, exist_ok=True)
    selected = summary["selected_candidates"]
    fig, axes = plt.subplots(2, 3, figsize=(13.5, 7.8))
    for column, experiment in enumerate(selected):
        group = predictions[predictions.run_id.eq(experiment["run_id"])]
        for model, color, style in [("M2", "#9C3F74", "--"), ("M4", "#176B87", "-")]:
            q = group[group.species.eq("Q")].sort_values("time_min")
            ax = axes[0, column]
            halfwidth = 1.6448536269514722*q[f"{model}_total_sd"]
            ax.fill_between(q.time_min, np.maximum(0, q[f"{model}_prediction"]-halfwidth),
                            q[f"{model}_prediction"]+halfwidth, color=color, alpha=.12)
            ax.plot(q.time_min, q[f"{model}_prediction"], color=color, ls=style, label=model)
            sampled = q[q.planned_sample]
            ax.scatter(sampled.time_min, sampled[f"{model}_prediction"], color=color, s=18, zorder=3)
            p = group[group.species.eq("P")].sort_values("time_min")
            ratio = np.divide(q[f"{model}_prediction"].to_numpy(), p[f"{model}_prediction"].to_numpy(),
                              out=np.full(len(p), np.nan), where=p[f"{model}_prediction"].to_numpy() > 1e-8)
            axes[1, column].plot(p.time_min, ratio*1000, color=color, ls=style)
        axes[0, column].axhline(loader.Q_LOQ_M, color="#D97941", ls=":", label="Q LOQ")
        axes[0, column].set_title(f"{experiment['run_id']}: {experiment['role']}\nA0={experiment['A0_M']:g} M; B0={experiment['B0_M']:g} M; 75 C", fontsize=10)
        axes[0, column].set_ylabel("Predicted Q [mol/L]")
        axes[0, column].ticklabel_format(axis="y", style="sci", scilimits=(0, 0))
        axes[1, column].set_ylabel("Predicted molar Q/P x 1000")
        for row in range(2):
            axes[row, column].set_xlabel("Time [min]")
            axes[row, column].set_ylim(bottom=0)
            axes[row, column].grid(alpha=.15)
    handles, labels = axes[0, 0].get_legend_handles_labels()
    handles.append(Line2D([], [], marker="o", ls="", color="#333333"))
    labels.append("Planned sample (all four species)")
    handles.append(Patch(facecolor="#777777", alpha=.15))
    labels.append("Approx. 90% assay interval")
    fig.legend(handles, labels, loc="upper center", bbox_to_anchor=(.5, .955), ncol=5, fontsize=8)
    fig.suptitle("Proposed campaign: changing A tests the impurity rate law", fontsize=14)
    fig.text(.5, .015, "Model forecasts, not observations. Q/P is a molar ratio; it does not establish the isolated-API mass specification. Dots are proposed samples.",
             ha="center", fontsize=8)
    fig.tight_layout(rect=(0, .04, 1, .91))
    fig.savefig(FIGURES / "system_B_campaign_predictions.png", dpi=160)
    plt.close(fig)

    fig, axes = plt.subplots(1, 2, figsize=(12, 4.8), gridspec_kw={"width_ratios": [1.15, 1]})
    for role, color, marker in [("A-depleted", "#176B87", "o"), ("balanced", "#9C3F74", "s"), ("A-rich", "#D97941", "^")]:
        group = candidates[candidates.role.eq(role)]
        axes[0].scatter(group.A_consumed_g, group.screening_score, color=color, marker=marker, label=role, alpha=.7)
    for chosen in selected:
        axes[0].scatter(chosen["A_consumed_g"], chosen["screening_score"], s=150, facecolors="none", edgecolors="#222222", linewidths=1.5)
        axes[0].annotate(chosen["run_id"], (chosen["A_consumed_g"], chosen["screening_score"]), xytext=(5, 6), textcoords="offset points")
    axes[0].set_xlabel("A consumed per 5 mL run [g]")
    axes[0].set_ylabel("Marginal screening score (heuristic)")
    axes[0].set_ylim(0, 1.18*candidates.screening_score.max())
    axes[0].set_title("Search across the three mechanistic contrasts")
    axes[0].legend(fontsize=8)
    forecasts = summary["simulation_forecasts"]
    labels = ["Proposed\nnominal", "Proposed\nstress", "Historical charge\nreference"]
    rates = [100*item["mean_fraction_correct"] for item in forecasts]
    axes[1].bar(labels, rates, color=["#176B87", "#8BAAB6", "#999999"], width=.6)
    for i, rate in enumerate(rates):
        axes[1].text(i, rate+2, f"{rate:.1f}%", ha="center")
    axes[1].axhline(50, color="#555555", ls=":", label="50% two-model guessing rate")
    axes[1].set_ylim(0, 112)
    axes[1].set_ylabel("Correct model selections in simulations [%]")
    axes[1].set_title("Joint-campaign simulation forecast")
    axes[1].legend(loc="lower left", fontsize=8)
    for ax in axes:
        ax.grid(axis="y", alpha=.15)
        ax.set_axisbelow(True)
    budget = summary["budget"]
    fig.suptitle(f"Campaign audit: {budget['experiments']} runs, {budget['A_consumed_g']:.3f}/5 g A, {budget['analytical_samples']}/60 samples", fontsize=13)
    fig.text(.5, .025, "Simulation results assume M2 or M4 is true. Stress: twice the parameter SD and 1.5x Q assay SD. This is not observed experimental accuracy.",
             ha="center", fontsize=8)
    fig.tight_layout(rect=(0, .065, 1, .94))
    fig.savefig(FIGURES / "system_B_campaign_design.png", dpi=160)
    plt.close(fig)


def try_replot():
    path = RESULTS / "system_B_campaign_summary.json"
    if not path.exists():
        return False
    summary = json.loads(path.read_text(encoding="utf-8"))
    for name, saved in summary["source_sha256"].items():
        if not (ROOT / name).exists() or not predictor.matches_source(ROOT / name, saved):
            print("Campaign inputs changed; recalculating the design.", flush=True)
            return False
    for name, saved in summary["artifact_sha256"].items():
        if not (ROOT / name).exists() or not predictor.matches_source(ROOT / name, saved):
            print("Campaign outputs changed or are missing; recalculating the design.", flush=True)
            return False
    with CAMPAIGN.open(encoding="utf-8") as handle:
        audit_budget(yaml.safe_load(handle)["experiments"])
    candidates = pd.read_csv(RESULTS / "system_B_campaign_candidates.csv")
    predictions = pd.read_csv(RESULTS / "system_B_campaign_predictions.csv")
    make_figures(summary, candidates, predictions)
    print("Regenerated both campaign figures from checked saved design results.")
    return True


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--figures-only", action="store_true")
    args = parser.parse_args()
    started = time.perf_counter()
    if args.figures_only and try_replot():
        return
    RESULTS.mkdir(exist_ok=True)
    FIGURES.mkdir(parents=True, exist_ok=True)
    runs = loader.load_runs()
    report = predictor.checked_report(RESULTS / "system_B_fits.json")
    print("System B campaign design: three complementary charges at 75 C.", flush=True)
    estimates = {model: predictor.uncertainty(model, runs, report["models"][model]) for model in MODELS}
    candidates = screen_candidates(estimates)
    selected, search = select_campaign(candidates)
    experiments = experiment_records(selected)
    budget = audit_budget(experiments)
    print(f"Screened {len(candidates)} conditions; compared {search['n_feasible_triples']} feasible three-run combinations.", flush=True)
    for row in selected:
        print(f"  {row['run_id']}: {row['role']}; A0={row['A0_M']:g} M, B0={row['B0_M']:g} M; {row['A_consumed_g']:.3f} g A", flush=True)
    forecasts = []
    print("Forecasting discrimination with independent simulated campaigns (not experimental outcomes).", flush=True)
    for label, design, parameter_scale, q_scale, seed in [
        ("proposed_nominal", experiments, 1., 1., SEED),
        ("proposed_stress", experiments, 2., 1.5, SEED+1),
        ("historical_charge_reference", reference_campaign(), 1., 1., SEED+2),
    ]:
        print(f"  Simulation audit: {label}", flush=True)
        forecasts.append(forecast_discrimination(design, estimates, label, parameter_scale, q_scale, seed))
    rationale = (
        "Precommitted single-block discrimination campaign: all three runs are at 75 C, "
        "with A-depleted, balanced, and concentrated A-rich charges. M2 predicts dQ/dP=k3/k2; "
        "M4 predicts dQ/dP=(k3/k2)*A(t). Changing A concentration and depletion therefore tests "
        "the distinguishing rate-law term while holding temperature fixed. Conditions were "
        "selected from a disclosed charge grid under the 5 g material cap using a marginal "
        "predictive-distance heuristic that includes assay/parameter uncertainty and Q censoring. "
        "Each run samples the initial charge, early and mid-course product formation, and four "
        "times selected for model separation; all four observable species are assayed. "
        "A separate joint-campaign simulation audit shares one parameter vector across all runs "
        "and includes a broader-uncertainty/Q-noise sensitivity check. These are forecasts, not "
        "new experimental evidence. Three runs meet the minimum campaign size; the unused "
        "material and sample allowance are retained rather than assigned to nearly redundant "
        "conditions. No run depends on another run's outcome. No measurement of I is proposed. "
        "This campaign does not establish isolated-API compliance with the 15 ppm w/w limit."
    )
    # Keep the baseline's documented top-level and experiment-key schema.
    with CAMPAIGN.open("w", encoding="utf-8", newline="\n") as handle:
        yaml.safe_dump({"rationale": rationale, "experiments": experiments}, handle, sort_keys=False)
    # Re-read the actual submission file to audit its serialized contents.
    with CAMPAIGN.open(encoding="utf-8") as handle:
        audit_budget(yaml.safe_load(handle)["experiments"])
    candidate_table = pd.DataFrame(candidates)
    candidate_table["selected"] = candidate_table.candidate_id.isin([row["candidate_id"] for row in selected])
    candidate_table["sample_times_min"] = candidate_table.sample_times_min.map(json.dumps)
    candidate_table.to_csv(RESULTS / "system_B_campaign_candidates.csv", index=False)
    details = prediction_details(selected, estimates)
    details.to_csv(RESULTS / "system_B_campaign_predictions.csv", index=False)
    paths = ["submission/campaign.yaml", "results/system_B_campaign_candidates.csv", "results/system_B_campaign_predictions.csv"]
    summary = {"design_stage": "proposed experiments; no new experimental outcomes available",
               "source_sha256": source_manifest(), "artifact_sha256": {name: predictor.digest(ROOT / name) for name in paths},
               "charge_grid_M": CHARGES.tolist(), "time_grid_min": TIMES.tolist(),
               "search_temperature_C": 75.0, "search": search, "budget": budget,
               "selected_candidates": selected, "parameter_uncertainty": estimates,
               "simulation_forecasts": forecasts, "reference_experiments": reference_campaign(),
               "reference_budget": audit_budget(reference_campaign()), "random_seed": SEED,
               "limitations": [
                   "A role-constrained finite-grid heuristic, not a globally optimal continuous or Bayesian design.",
                   "Marginal screening distances are not additive joint information when parameters are shared; the separate joint simulation audit addresses this dependence approximately.",
                   "Simulation forecasts are conditional on M2/M4 and local normal parameter uncertainty, not observed success rates or guarantees.",
                   "Q's released-run numeric interval coverage was 15/18; the stress scenario is sensitivity analysis, not a proven recalibration.",
                   "Gaussian independent assay errors are assumed before Q censoring; additional zero clipping and correlated assay error are not modelled.",
                   "Exactly three runs at 75 C target stoichiometric discrimination; temperature dependence and independent run-to-run replication are not tested by this campaign.",
                   "No future outcomes feed back into predictions, ranking or the committed block of experiments.",
                   "No inference about isolated-API mass impurity compliance follows from molar concentration ratios alone.",
               ]}
    predictor.save_json(RESULTS / "system_B_campaign_summary.json", summary)
    make_figures(summary, candidate_table, details)
    print("\nSystem B campaign design complete.")
    print(f"Experiments: {budget['experiments']} (allowed 3-10).")
    print(f"A consumed: {budget['A_consumed_g']:.3f} / 5.000 g; remaining {budget['A_remaining_g']:.3f} g.")
    print(f"Analytical samples: {budget['analytical_samples']} / 60; all assay A, B, P, Q.")
    print("All temperature, concentration, time and single-block checks passed.")
    print("\nSelected experiments:")
    columns = ["run_id", "role", "temperature_C", "A0_M", "B0_M", "sample_times_min"]
    print(pd.DataFrame(selected)[columns].to_string(index=False))
    print("\nSimulation forecasts only (not observed validation accuracy):")
    for item in forecasts:
        print(f"  {item['scenario']}: {100*item['mean_fraction_correct']:.1f}% model-selection accuracy over {2*OUTER_DRAWS} simulated campaigns.")
    print("No new experimental measurements were obtained or added to the training data.")
    print("Saved submission/campaign.yaml, results/system_B_campaign_summary.json and two diagnostic CSV files.")
    print("Saved system_B_campaign_predictions.png and system_B_campaign_design.png in submission/figures.")
    print(f"Campaign design and audit took {time.perf_counter()-started:.1f} seconds on this computer.")


if __name__ == "__main__":
    main()
