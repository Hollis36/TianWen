#!/usr/bin/env python3
"""Run a controlled distillation ablation and print the mAP delta.

Trains the same detector twice on the same data — VLM teacher off (baseline) vs
on (distilled) — and reports whether the VLM improves the detector.

Usage:
    # COCO is auto-discovered (Kaggle/Colab) unless dataset paths are configured
    python tools/ablation.py detector=yolov8 vlm=clip dataset=coco

    # Quick slice for a first number
    python tools/ablation.py detector=yolov8 vlm=clip dataset=coco \
        ablation.max_steps=500 ablation.limit_val_batches=50
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

import hydra
from omegaconf import DictConfig, OmegaConf

from tianwen.datasets import build_datamodule, discover_coco
from tianwen.utils.ablation import run_distillation_ablation


@hydra.main(config_path="../configs", config_name="config", version_base=None)
def main(cfg: DictConfig) -> None:
    dataset_cfg = OmegaConf.to_container(cfg.dataset, resolve=True)

    # Auto-discover COCO paths if they aren't present on disk.
    if dataset_cfg.get("name") == "coco" and not Path(str(dataset_cfg.get("val_ann", ""))).exists():
        found = discover_coco()
        if found:
            print(f"Discovered COCO at {found['root']}")
            dataset_cfg.update(found)
        else:
            print("WARNING: COCO not found on disk; set dataset paths or use dataset=dummy.")

    datamodule = build_datamodule(dataset_cfg)

    ablation_cfg = cfg.get("ablation", {})
    result = run_distillation_ablation(
        detector_cfg=OmegaConf.to_container(cfg.detector, resolve=True),
        vlm_cfg=OmegaConf.to_container(cfg.vlm, resolve=True),
        datamodule=datamodule,
        distill_mode=cfg.fusion.get("distill_mode", "feature"),
        max_steps=ablation_cfg.get("max_steps", 500),
        limit_val_batches=ablation_cfg.get("limit_val_batches", 50),
        accelerator=cfg.trainer.get("accelerator", "auto"),
        precision=str(cfg.trainer.get("precision", "32")),
    )

    print("\n=== Distillation ablation (does the VLM teacher help?) ===")
    print(f"  detector={cfg.detector.get('type')}  vlm={cfg.vlm.get('type')}  mode=feature")
    print(
        f"  baseline  mAP@50={result['baseline_mAP50']:.4f}  mAP@50:95={result['baseline_mAP50_95']:.4f}"
    )
    print(
        f"  distilled mAP@50={result['distilled_mAP50']:.4f}  mAP@50:95={result['distilled_mAP50_95']:.4f}"
    )
    print(
        f"  delta     mAP@50={result['delta_mAP50']:+.4f}  mAP@50:95={result['delta_mAP50_95']:+.4f}"
    )


if __name__ == "__main__":
    main()
