#!/usr/bin/env python
"""CLI: run YOLO detection over collected imagery metadata.

Usage:
    python scripts/detect_objects.py --metadata data/metadata/seoul_gangnam_metadata.parquet
    python scripts/detect_objects.py --metadata-dir data/metadata
"""
import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import pandas as pd

from src.config import PROJECT_ROOT, load_config
from src.detect import detect_images
from src.utils import setup_logging


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--metadata", help="Path to a single metadata CSV/Parquet file")
    parser.add_argument(
        "--metadata-dir",
        help="Directory of *_metadata.parquet files to merge and run detection on "
        "(default: config paths.metadata)",
    )
    args = parser.parse_args()

    setup_logging()
    config = load_config()

    if args.metadata:
        detect_images(args.metadata, config)
        return

    metadata_dir = Path(args.metadata_dir) if args.metadata_dir else PROJECT_ROOT / config["paths"]["metadata"]
    parquet_files = sorted(metadata_dir.glob("*_metadata.parquet"))
    if not parquet_files:
        raise SystemExit(f"No metadata files found in {metadata_dir}. Run collect_imagery.py first.")

    merged = pd.concat([pd.read_parquet(p) for p in parquet_files], ignore_index=True)
    merged_path = metadata_dir / "_merged_metadata.parquet"
    merged.to_parquet(merged_path, index=False)
    detect_images(merged_path, config)


if __name__ == "__main__":
    main()
