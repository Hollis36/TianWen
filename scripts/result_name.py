"""Print a descriptive filename stem for a Kaggle ablation_result.json."""

import json
import sys

d = json.load(open(sys.argv[1]))
parts = [
    d.get("dataset", "run"),
    d.get("detector", "det"),
    d.get("distill_mode", "feature"),
    f"ep{d['max_epochs']}" if d.get("max_epochs") else f"{d.get('max_steps', 'x')}steps",
]
if d.get("arms"):
    parts.append("arms-" + "-".join(d["arms"]))
if d.get("distill_weight", 1.0) != 1.0:
    parts.append(f"w{d['distill_weight']}")
if d.get("train_fraction", 1.0) != 1.0:
    parts.append(f"frac{d['train_fraction']}")
vlm = d.get("vlm", "")
if vlm and "clip-vit-base-patch32" not in vlm:
    parts.append(vlm.split("/")[-1])
parts.append(f"seed{d.get('seed', 0)}")
print("_".join(str(p) for p in parts))
