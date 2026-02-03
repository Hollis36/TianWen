#!/usr/bin/env python3
"""
Complete VLM Knowledge Distillation Pipeline

This script runs the entire pipeline:
1. Generate VLM soft labels
2. Train confidence calibrator
3. Evaluate distilled model

Usage:
    python run_distillation_pipeline.py --samples 500 --vlm qwen2-vl-7b --epochs 20
"""

import sys
from pathlib import Path
project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))

import argparse
import time
from datetime import timedelta


def run_pipeline(
    num_samples: int = 500,
    vlm_type: str = "qwen2-vl-7b",
    num_epochs: int = 20,
    batch_size: int = 64,
    learning_rate: float = 1e-3,
    verify_threshold: float = 0.7,
    skip_label_generation: bool = False,
    skip_training: bool = False,
):
    """Run the complete distillation pipeline."""

    print("=" * 70)
    print("VLM KNOWLEDGE DISTILLATION PIPELINE")
    print("=" * 70)
    print(f"\nConfiguration:")
    print(f"  Samples: {num_samples}")
    print(f"  VLM: {vlm_type}")
    print(f"  Epochs: {num_epochs}")
    print(f"  Batch size: {batch_size}")
    print(f"  Learning rate: {learning_rate}")
    print(f"  Verify threshold: {verify_threshold}")
    print()

    soft_labels_dir = Path("d:/TianWen/soft_labels")
    checkpoint_dir = Path("d:/TianWen/distillation_checkpoints")
    soft_labels_file = soft_labels_dir / f"soft_labels_val_{vlm_type.replace('/', '_')}.pkl"

    total_start = time.time()

    # Step 1: Generate soft labels
    if not skip_label_generation:
        print("\n" + "=" * 70)
        print("STEP 1: Generating VLM Soft Labels")
        print("=" * 70)

        step1_start = time.time()

        from generate_vlm_soft_labels import generate_soft_labels
        generate_soft_labels(
            split="val",
            num_samples=num_samples,
            vlm_type=vlm_type,
            verify_threshold=verify_threshold,
        )

        step1_time = time.time() - step1_start
        print(f"\n✓ Step 1 completed in {timedelta(seconds=int(step1_time))}")
    else:
        print("\n[Skipping Step 1: Using existing soft labels]")
        if not soft_labels_file.exists():
            print(f"ERROR: Soft labels file not found: {soft_labels_file}")
            print("Please run without --skip_labels first")
            return

    # Step 2: Train calibrator
    if not skip_training:
        print("\n" + "=" * 70)
        print("STEP 2: Training Confidence Calibrator")
        print("=" * 70)

        step2_start = time.time()

        from train_vlm_distillation import train_calibrator
        model, history = train_calibrator(
            soft_labels_file=soft_labels_file,
            num_epochs=num_epochs,
            batch_size=batch_size,
            learning_rate=learning_rate,
        )

        step2_time = time.time() - step2_start
        print(f"\n✓ Step 2 completed in {timedelta(seconds=int(step2_time))}")
    else:
        print("\n[Skipping Step 2: Using existing checkpoint]")
        if not (checkpoint_dir / "calibrator_best.pt").exists():
            print(f"ERROR: Checkpoint not found: {checkpoint_dir / 'calibrator_best.pt'}")
            print("Please run without --skip_training first")
            return

    # Step 3: Evaluate
    print("\n" + "=" * 70)
    print("STEP 3: Evaluating Distilled Model")
    print("=" * 70)

    step3_start = time.time()

    from evaluate_distilled_model import run_benchmark
    results = run_benchmark(
        num_samples=min(num_samples, 100),  # Use subset for faster evaluation
        calibrator_checkpoint=str(checkpoint_dir / "calibrator_best.pt"),
    )

    step3_time = time.time() - step3_start
    print(f"\n✓ Step 3 completed in {timedelta(seconds=int(step3_time))}")

    # Summary
    total_time = time.time() - total_start

    print("\n" + "=" * 70)
    print("PIPELINE COMPLETE")
    print("=" * 70)
    print(f"\nTotal time: {timedelta(seconds=int(total_time))}")
    print(f"\nResults:")
    print(f"  Baseline mAP@50: {results['baseline']['mAP@50']:.4f}")
    print(f"  Distilled mAP@50: {results['distilled']['mAP@50']:.4f}")
    print(f"  Improvement: {results['improvement']['mAP@50']:+.4f} ({results['improvement']['mAP@50_percent']:+.2f}%)")

    print("\nFiles generated:")
    print(f"  Soft labels: {soft_labels_file}")
    print(f"  Checkpoint: {checkpoint_dir / 'calibrator_best.pt'}")
    print(f"  Results: d:/TianWen/distillation_results/benchmark_results.json")

    print("\n" + "=" * 70)

    return results


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run VLM Knowledge Distillation Pipeline")
    parser.add_argument("--samples", type=int, default=500,
                       help="Number of images for soft label generation")
    parser.add_argument("--vlm", type=str, default="qwen2-vl-7b",
                       choices=["qwen2-vl-2b", "qwen2-vl-7b", "qwen2.5-vl-7b"],
                       help="VLM model for generating soft labels")
    parser.add_argument("--epochs", type=int, default=20,
                       help="Number of training epochs")
    parser.add_argument("--batch_size", type=int, default=64,
                       help="Training batch size")
    parser.add_argument("--lr", type=float, default=1e-3,
                       help="Learning rate")
    parser.add_argument("--threshold", type=float, default=0.7,
                       help="Confidence threshold for VLM verification")
    parser.add_argument("--skip_labels", action="store_true",
                       help="Skip soft label generation (use existing)")
    parser.add_argument("--skip_training", action="store_true",
                       help="Skip training (use existing checkpoint)")
    args = parser.parse_args()

    run_pipeline(
        num_samples=args.samples,
        vlm_type=args.vlm,
        num_epochs=args.epochs,
        batch_size=args.batch_size,
        learning_rate=args.lr,
        verify_threshold=args.threshold,
        skip_label_generation=args.skip_labels,
        skip_training=args.skip_training,
    )
