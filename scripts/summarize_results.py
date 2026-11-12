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
    for path in sorted(glob.glob(os.path.join(args.results_dir, "*_seed*.json"))):
        d = json.load(open(path))
        if d.get("valid") is False:
            continue
        key = re.sub(r"_seed\d+$", "", os.path.splitext(os.path.basename(path))[0])
        groups[key].append(d)

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
