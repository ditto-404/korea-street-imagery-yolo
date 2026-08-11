"""Load pipeline configuration from config.yaml and secrets from .env."""
from __future__ import annotations

import os
from pathlib import Path
from typing import Optional

import yaml
from dotenv import load_dotenv

PROJECT_ROOT = Path(__file__).resolve().parent.parent
CONFIG_PATH = PROJECT_ROOT / "config.yaml"

load_dotenv(PROJECT_ROOT / ".env")


def load_config(path: Path = CONFIG_PATH) -> dict:
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def get_access_token() -> str:
    token: Optional[str] = os.getenv("MAPILLARY_ACCESS_TOKEN")
    if not token:
        raise RuntimeError(
            "MAPILLARY_ACCESS_TOKEN is not set. Copy .env.example to .env "
            "and add a free Mapillary access token from "
            "https://www.mapillary.com/dashboard/developers"
        )
    return token
