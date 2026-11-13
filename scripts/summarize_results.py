"""Group ablation results by configuration and report mean ± std of the deltas.

Verdict rule (paired seeds): "effective" if mean Δ mAP@50:95 exceeds the noise
floor and at least 2/3 of seeds are positive; "harmful" for the mirror case;
otherwise "no effect" (n>=3) or "pending".

Usage: python scripts/summarize_results.py [results_dir] [--noise 0.006]
"""

import argparse
import glob
import json
import os
import re
import statistics as st
from collections import defaultdict


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("results_dir", nargs="?", default="results")
    ap.add_argument("--noise", type=float, default=0.006, help="baseline run-to-run spread")
    args = ap.parse_args()

    groups = defaultdict(list)
    ssl = defaultdict(dict)
    for path in sorted(glob.glob(os.path.join(args.results_dir, "*_seed*.json"))):
        d = json.load(open(path))
        if d.get("valid") is False:
            continue
        key = re.sub(r"_seed\d+$", "", os.path.splitext(os.path.basename(path))[0])
        if d.get("experiment") == "ssl":
            pair = (d["dataset"], d["train_fraction"], d["student_epochs"], d["seed"])
            ssl[pair + (d.get("split_seed", 0),)][d["ssl_arm"]] = d
            continue
        if "arms" in d:
            # Multi-arm result: one row per non-baseline arm, delta vs baseline.
            for arm in d["arms"]:
                if arm == "baseline":
                    continue
                groups[f"{key} [{arm}]"].append(
                    {
                        "delta_mAP50_95": d[f"delta_{arm}_mAP50_95"],
                        "delta_mAP50": d[f"delta_{arm}_mAP50"],
                    }
                )
            continue
        groups[key].append(d)

    # Pseudo-label arms: paired (same training seed and label subset) delta vs the supervised arm.
    for (ds, frac, ep, _seed, _split), arms in ssl.items():
        base = arms.get("supervised")
        if base is None:
            continue
        for arm, d in arms.items():
            if arm == "supervised":
                continue
            groups[f"{ds}_ssl_frac{frac}_ep{ep} [{arm} vs supervised]"].append(
                {
                    "delta_mAP50_95": d["mAP50_95"] - base["mAP50_95"],
                    "delta_mAP50": d["mAP50"] - base["mAP50"],
                }
            )

    print(f"{'config':<58} {'n':>2}  {'Δ mAP50:95 mean ± std':>24}  {'Δ mAP50 mean':>12}  seeds(+)  verdict")
    for key, runs in sorted(groups.items()):
        d95 = [r["delta_mAP50_95"] for r in runs]
        d50 = [r["delta_mAP50"] for r in runs]
        mean = st.mean(d95)
        std = st.stdev(d95) if len(d95) > 1 else float("nan")
        pos = sum(x > 0 for x in d95)
        if len(runs) < 3:
            verdict = "pending"
        elif mean > args.noise and pos >= 2 * len(runs) / 3:
            verdict = "EFFECTIVE"
        elif mean < -args.noise and pos <= len(runs) / 3:
            verdict = "harmful"
        else:
            verdict = "no effect"
        print(f"{key:<58} {len(runs):>2}  {mean:+.4f} ± {std:.4f}{'':>8}  {st.mean(d50):+.4f}      {pos}/{len(runs)}     {verdict}")


if __name__ == "__main__":
    main()
