"""Collect Mapillary street-level imagery and metadata for configured regions."""
from __future__ import annotations

import logging
from typing import Optional

import pandas as pd
from tqdm import tqdm

from .config import PROJECT_ROOT, get_access_token, load_config
from .mapillary_client import MapillaryClient
from .utils import generate_bbox_tiles

logger = logging.getLogger(__name__)


def collect_region(region_name: str, config: Optional[dict] = None) -> pd.DataFrame:
    config = config or load_config()
    region = config["regions"].get(region_name)
    if region is None:
        raise KeyError(f"Unknown region '{region_name}'. Check config.yaml.")

    client = MapillaryClient(get_access_token())
    tile_size = config["mapillary"]["bbox_tile_size"]
    per_tile_limit = config["mapillary"]["images_per_tile"]
    image_field = config["mapillary"].get("image_field", "thumb_1024_url")
    fields = f"id,captured_at,geometry,compass_angle,sequence,{image_field}"

    raw_dir = PROJECT_ROOT / config["paths"]["raw_images"] / region_name
    raw_dir.mkdir(parents=True, exist_ok=True)

    seen_ids = set()
    records = []

    tiles = list(generate_bbox_tiles(tuple(region["bbox"]), tile_size))
    for tile in tqdm(tiles, desc=f"[{region_name}] scanning tiles"):
        for item in client.search_images(tile, limit=per_tile_limit, fields=fields):
            image_id = item["id"]
            if image_id in seen_ids:
                continue
            seen_ids.add(image_id)

            coords = item.get("geometry", {}).get("coordinates", [None, None])
            lon, lat = coords[0], coords[1]
            image_url = item.get(image_field)
            local_path = raw_dir / f"{image_id}.jpg"

            if not image_url:
                continue
            try:
                client.download_image(image_url, str(local_path))
            except Exception as exc:
                logger.warning("Failed to download image %s: %s", image_id, exc)
                continue

            records.append(
                {
                    "image_id": image_id,
                    "region": region_name,
                    "captured_at": item.get("captured_at"),
                    "lon": lon,
                    "lat": lat,
                    "compass_angle": item.get("compass_angle"),
                    "sequence_id": item.get("sequence"),
                    "local_path": str(local_path.relative_to(PROJECT_ROOT)),
                }
            )

    df = pd.DataFrame(records)
    if not df.empty and "captured_at" in df.columns:
        df["captured_at"] = pd.to_datetime(df["captured_at"], unit="ms", utc=True, errors="coerce")

    metadata_dir = PROJECT_ROOT / config["paths"]["metadata"]
    metadata_dir.mkdir(parents=True, exist_ok=True)
    df.to_csv(metadata_dir / f"{region_name}_metadata.csv", index=False, encoding="utf-8-sig")
    df.to_parquet(metadata_dir / f"{region_name}_metadata.parquet", index=False)

    logger.info("Collected %d images for region '%s'", len(df), region_name)
    return df


def collect_all_regions(config: Optional[dict] = None) -> pd.DataFrame:
    config = config or load_config()
    frames = [collect_region(name, config) for name in config["regions"]]
    return pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()
