"""Shared helpers: logging setup and bounding-box tiling."""
from __future__ import annotations

import logging
from typing import Iterator, Tuple

BBox = Tuple[float, float, float, float]


def setup_logging(level: int = logging.INFO) -> None:
    logging.basicConfig(
        level=level,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    )


def generate_bbox_tiles(bbox: BBox, tile_size_deg: float) -> Iterator[BBox]:
    """Split a large bounding box into smaller (min_lon, min_lat, max_lon, max_lat) tiles.

    Mapillary's /images endpoint caps results per request, so large areas
    are tiled into smaller bboxes to get fuller coverage.
    """
    min_lon, min_lat, max_lon, max_lat = bbox
    lon = min_lon
    while lon < max_lon:
        next_lon = min(lon + tile_size_deg, max_lon)
        lat = min_lat
        while lat < max_lat:
            next_lat = min(lat + tile_size_deg, max_lat)
            yield (lon, lat, next_lon, next_lat)
            lat = next_lat
        lon = next_lon
