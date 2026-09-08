"""just suggest -> submission/campaign.yaml (naive baseline: 3 runs on the released conditions)."""
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import yaml
from reference import SUBMISSION, ensure_submission_dir


def main():
    ensure_submission_dir()
    # Deliberately weak but LEGAL: 3 runs (the minimum) at the released charge and ratio --
    # 1.0 M A, 4:1 (B0 = 0.25) -> 1.25 g each, 3.75 g total < 5.0 g, 18 samples < 60. They sit
    # right on top of the released conditions, so they add nothing you don't already have.
    # As the handout warns, an experiment close to data you have tells you very little.
    # Replace me with experiments that actually discriminate.
    times = [0, 120, 240, 480, 720, 960]
    campaign = {
        "rationale": "BASELINE placeholder: 3 runs at the released 1.0 M / 4:1 conditions -- "
                     "duplicates existing data, so it is legal but uninformative. Replace me.",
        "experiments": [
            {"run_id": f"E{i+1}", "temperature_C": T, "A0_M": 1.0, "B0_M": 0.25,
             "sample_times_min": times, "assay": ["A", "B", "P", "Q"]}
            for i, T in enumerate([65, 75, 85])
        ],
    }
    out = os.path.join(SUBMISSION, "campaign.yaml")
    with open(out, "w") as fh:
        yaml.safe_dump(campaign, fh, sort_keys=False)
    print(f"[baseline] wrote campaign -> {out}")

if __name__ == "__main__":
    main()
