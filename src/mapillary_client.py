"""Thin client for the free Mapillary Graph API v4.

Docs: https://www.mapillary.com/developer/api-documentation
Only public, free endpoints are used. A free access token is required -
see .env.example for how to obtain one.
"""
from __future__ import annotations

import time
from typing import Dict, Iterator, Optional

import requests

from .utils import BBox

BASE_URL = "https://graph.mapillary.com"
DEFAULT_FIELDS = "id,captured_at,geometry,compass_angle,sequence,thumb_1024_url"


class MapillaryClient:
    def __init__(self, access_token: str, timeout: int = 30, retries: int = 3):
        self.access_token = access_token
        self.timeout = timeout
        self.retries = retries
        self.session = requests.Session()

    def _get(self, url: str, params: Optional[Dict] = None) -> dict:
        params = dict(params or {})
        params["access_token"] = self.access_token

        last_error: Optional[Exception] = None
        for attempt in range(self.retries):
            try:
                resp = self.session.get(url, params=params, timeout=self.timeout)
                resp.raise_for_status()
                return resp.json()
            except requests.RequestException as exc:
                last_error = exc
                time.sleep(2**attempt)
        raise RuntimeError(f"Mapillary API request failed after {self.retries} attempts: {last_error}")

    def search_images(
        self, bbox: BBox, limit: int = 100, fields: str = DEFAULT_FIELDS
    ) -> Iterator[dict]:
        """Yield image metadata records within a bounding box.

        bbox: (min_lon, min_lat, max_lon, max_lat)
        """
        url = f"{BASE_URL}/images"
        params: Optional[Dict] = {
            "fields": fields,
            "bbox": ",".join(str(v) for v in bbox),
            "limit": limit,
        }

        while url:
            payload = self._get(url, params)
            for record in payload.get("data", []):
                yield record

            url = payload.get("paging", {}).get("next")
            # The "next" URL already carries the full query string.
            params = None

    def download_image(self, image_url: str, dest_path: str) -> None:
        resp = self.session.get(image_url, timeout=self.timeout)
        resp.raise_for_status()
        with open(dest_path, "wb") as f:
            f.write(resp.content)
