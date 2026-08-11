"""Run YOLO object detection on collected imagery and save results.

Saves per-detection rows (image_id, class, confidence, bbox, ...) to
CSV/Parquet, and writes annotated copies of images that have detections.
"""
from __future__ import annotations

import logging
from pathlib import Path
from typing import List, Optional

import cv2
import pandas as pd
from tqdm import tqdm
from ultralytics import YOLO

from .config import PROJECT_ROOT, load_config

logger = logging.getLogger(__name__)

BOX_COLOR = (0, 200, 0)
TEXT_COLOR = (255, 255, 255)


def _draw_boxes(image_path: Path, detections: List[dict], out_path: Path) -> None:
    img = cv2.imread(str(image_path))
    if img is None:
        logger.warning("Could not read image for annotation: %s", image_path)
        return

    for det in detections:
        x1, y1, x2, y2 = int(det["x1"]), int(det["y1"]), int(det["x2"]), int(det["y2"])
        label = f'{det["class_name"]} {det["confidence"]:.2f}'
        cv2.rectangle(img, (x1, y1), (x2, y2), BOX_COLOR, 2)
        (tw, th), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)
        cv2.rectangle(img, (x1, max(y1 - th - 6, 0)), (x1 + tw + 4, y1), BOX_COLOR, -1)
        cv2.putText(img, label, (x1 + 2, max(y1 - 4, th)), cv2.FONT_HERSHEY_SIMPLEX, 0.5, TEXT_COLOR, 1, cv2.LINE_AA)

    out_path.parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(out_path), img)


def detect_images(metadata_path, config: Optional[dict] = None) -> pd.DataFrame:
    config = config or load_config()
    det_cfg = config["detection"]

    metadata_path = Path(metadata_path)
    meta_df = pd.read_parquet(metadata_path) if metadata_path.suffix == ".parquet" else pd.read_csv(metadata_path)

    model = YOLO(det_cfg["model"])
    target_classes = set(det_cfg.get("target_classes", []))
    conf_threshold = det_cfg.get("confidence", 0.25)
    iou_threshold = det_cfg.get("iou", 0.45)

    annotated_dir = PROJECT_ROOT / config["paths"]["annotated_images"]
    detections_dir = PROJECT_ROOT / config["paths"]["detections"]
    annotated_dir.mkdir(parents=True, exist_ok=True)
    detections_dir.mkdir(parents=True, exist_ok=True)

    all_rows = []

    for _, row in tqdm(meta_df.iterrows(), total=len(meta_df), desc="Running YOLO detection"):
        image_path = PROJECT_ROOT / row["local_path"]
        if not image_path.exists():
            continue

        result = model.predict(source=str(image_path), conf=conf_threshold, iou=iou_threshold, verbose=False)[0]

        image_detections = []
        for box in result.boxes:
            class_name = model.names[int(box.cls[0])]
            if target_classes and class_name not in target_classes:
                continue

            x1, y1, x2, y2 = (float(v) for v in box.xyxy[0])
            det = {
                "image_id": row["image_id"],
                "region": row.get("region"),
                "lon": row.get("lon"),
                "lat": row.get("lat"),
                "captured_at": row.get("captured_at"),
                "class_name": class_name,
                "confidence": float(box.conf[0]),
                "x1": x1,
                "y1": y1,
                "x2": x2,
                "y2": y2,
                "image_width": result.orig_shape[1],
                "image_height": result.orig_shape[0],
            }
            image_detections.append(det)
            all_rows.append(det)

        if image_detections:
            _draw_boxes(image_path, image_detections, annotated_dir / f"{row['image_id']}.jpg")

    det_df = pd.DataFrame(all_rows)
    det_df.to_csv(detections_dir / "detections.csv", index=False, encoding="utf-8-sig")
    det_df.to_parquet(detections_dir / "detections.parquet", index=False)

    logger.info("Saved %d detections to %s", len(det_df), detections_dir)
    return det_df
