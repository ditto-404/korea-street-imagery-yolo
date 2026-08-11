#!/usr/bin/env python
"""CLI: run the full pipeline - collect imagery, then run YOLO detection.

Usage:
    python scripts/run_pipeline.py --all
    python scripts/run_pipeline.py --region seoul_gangnam
"""
import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.collect import collect_all_regions, collect_region
from src.config import PROJECT_ROOT, load_config
from src.detect import detect_images
from src.utils import setup_logging


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--region", help="Region name defined in config.yaml")
    parser.add_argument("--all", action="store_true", help="Run for all regions in config.yaml")
    args = parser.parse_args()

    setup_logging()
    config = load_config()

    if args.all or not args.region:
        meta_df = collect_all_regions(config)
    else:
        meta_df = collect_region(args.region, config)

    if meta_df.empty:
        raise SystemExit("No images were collected. Check your Mapillary access token and region bbox.")

    tmp_path = PROJECT_ROOT / config["paths"]["metadata"] / "_pipeline_run.parquet"
    meta_df.to_parquet(tmp_path, index=False)
    detect_images(tmp_path, config)


if __name__ == "__main__":
    main()
