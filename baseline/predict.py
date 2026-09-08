"""just predict -> submission/predictions.csv (naive baseline).

Run as `python baseline/predict.py`, so this file's own directory is on sys.path and the
sibling `reference` module imports directly -- no path manipulation needed here.
"""
import os

from reference import BASELINE_MODEL, SUBMISSION, ensure_submission_dir, fit_baseline, predict


def main():
    ensure_submission_dir()
    params = fit_baseline(BASELINE_MODEL)
    df = predict(params, BASELINE_MODEL)
    out = os.path.join(SUBMISSION, "predictions.csv")
    df.to_csv(out, index=False)
    print(f"[baseline] fitted {BASELINE_MODEL}, wrote {len(df)} predictions -> {out}")

if __name__ == "__main__":
    main()
