"""just replicate -> regenerate presentation figures (naive baseline)."""
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd
from reference import DATA_B, SUBMISSION, ensure_submission_dir


def main():
    ensure_submission_dir()
    df = pd.read_csv(DATA_B)
    fig, axes = plt.subplots(1, 2, figsize=(11, 4))
    for sp, ax in zip(["P", "Q"], axes):
        g = df[(df["species"] == sp) & (df["value"] != "<LOQ")].copy()
        g["value"] = g["value"].astype(float)
        for rid, gg in g.groupby("run_id"):
            ax.plot(gg["time_min"], gg["value"], "o-", ms=3, label=rid)
        ax.set_title(f"System B: {sp}"); ax.set_xlabel("time [min]"); ax.set_ylabel(f"[{sp}] mol/L")
    axes[0].legend(fontsize=7)
    fig.tight_layout()
    out = os.path.join(SUBMISSION, "figures", "system_B_overview.png")
    fig.savefig(out, dpi=110)
    print(f"[baseline] wrote figure -> {out}")

if __name__ == "__main__":
    main()
