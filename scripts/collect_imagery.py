#!/usr/bin/env python
"""CLI: collect Mapillary imagery + metadata for one or all configured regions.

Usage:
    python scripts/collect_imagery.py --region seoul_gangnam
    python scripts/collect_imagery.py --all
"""
import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.collect import collect_all_regions, collect_region
from src.config import load_config
from src.utils import setup_logging


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--region", help="Region name defined in config.yaml")
    parser.add_argument("--all", action="store_true", help="Collect all regions in config.yaml")
    args = parser.parse_args()

    setup_logging()
    config = load_config()

    if args.all or not args.region:
        collect_all_regions(config)
    else:
        collect_region(args.region, config)


if __name__ == "__main__":
    main()
