#!/usr/bin/env python3
"""Export a standalone detector from a trained TianWen fusion checkpoint.

After distilling VLM knowledge into the detector, ship just the detector — the
exported checkpoint has no VLM weights or dependencies.

Usage:
    python tools/export.py --checkpoint runs/exp/last.ckpt --output detector.pt
"""

import argparse
import sys
from pathlib import Path

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from tianwen.utils.export import export_detector_from_training_checkpoint


def main() -> None:
    parser = argparse.ArgumentParser(description="Export a standalone detector.")
    parser.add_argument(
        "--checkpoint", "-c", required=True, help="Trained Lightning checkpoint (.ckpt)"
    )
    parser.add_argument(
        "--output", "-o", required=True, help="Output path for the standalone detector (.pt)"
    )
    parser.add_argument("--map-location", default="cpu", help="Device to map tensors to")
    args = parser.parse_args()

    payload = export_detector_from_training_checkpoint(
        args.checkpoint, args.output, map_location=args.map_location
    )
    num_params = sum(t.numel() for t in payload["state_dict"].values())
    print(
        f"Exported detector '{payload['detector_cfg'].get('type')}' "
        f"({num_params:,} params, {len(payload['state_dict'])} tensors) to {args.output}"
    )


if __name__ == "__main__":
    main()
