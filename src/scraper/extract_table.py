from __future__ import annotations

import logging
import re
from dataclasses import dataclass

import numpy as np

from scraper.util import apply_replacements, best_name_match, clean_number, is_number

logger = logging.getLogger(__name__)


@dataclass
class OcrCell:
    text: str
    x1: int
    y1: int
    x2: int
    y2: int


def ocr_to_rows(
    ocr_result,
    min_confidence: float,
    frame_h: int,
    margin: int,
    row_y_tolerance: int,
) -> list[list[OcrCell]]:
    """
    Convert flat PaddleOCR output into rows of OcrCell, sorted top-to-bottom
    with cells in each row sorted left-to-right.  Cells whose bounding box
    touches the top/bottom margin are dropped to avoid partial rows.
    """
    if not ocr_result:
        return []

    page = ocr_result[0]

    # Log the result structure once per process so format issues are visible without --debug.
    if not hasattr(ocr_to_rows, "_format_logged"):
        keys = list(page.keys()) if isinstance(page, dict) else "n/a"
        logger.info("OCR result format — type: %s, keys: %s", type(page).__name__, keys)
        ocr_to_rows._format_logged = True  # type: ignore[attr-defined]

    # PaddleOCR 3.x predict() wraps detections under a 'res' key, each entry
    # being a dict with 'text', 'score', and 'text_region' (or 'bbox').
    # Older / alternative builds use flat parallel arrays keyed 'dt_polys',
    # 'rec_text', 'rec_score'. Fall back to the 2.x list-of-pairs format last.
    detections: list[tuple] = []
    if isinstance(page, dict):
        if "rec_texts" in page:
            # PaddleOCR 3.x OCRResult (rec_polys are 4-point polygons)
            polys  = page.get("rec_polys",  [])
            texts  = page.get("rec_texts",  [])
            scores = page.get("rec_scores", [])
            detections = list(zip(polys, texts, scores))
        elif "res" in page:
            for item in page["res"]:
                pts  = item.get("text_region") or item.get("bbox") or []
                text = item.get("text", "")
                conf = item.get("score", 0.0)
                detections.append((pts, text, conf))
        else:
            polys  = page.get("dt_polys",  [])
            texts  = page.get("rec_text",  [])
            scores = page.get("rec_score", [])
            detections = list(zip(polys, texts, scores))
    elif isinstance(page, list):
        detections = [(l[0], l[1][0], l[1][1]) for l in page if l]

    if not detections:
        logger.debug("OCR returned no detections for this frame")
        return []

    # (y_center, x_center, text, x1, y1, x2, y2)
    cells: list[tuple] = []
    conf_dropped = 0
    margin_dropped = 0
    for pts, text, conf in detections:
        if conf < min_confidence or not str(text).strip():
            conf_dropped += 1
            logger.debug("Detection dropped (conf=%.3f < %.2f): %r", conf, min_confidence, text)
            continue
        if pts is None or len(pts) == 0:
            logger.debug("Skipping detection with empty bbox: '%s'", text)
            continue
        ys = [p[1] for p in pts]
        xs = [p[0] for p in pts]
        y_top, y_bot = min(ys), max(ys)
        if y_top < margin or y_bot > frame_h - margin:
            margin_dropped += 1
            logger.debug("Detection dropped by margin (y=%d-%d, margin=%d, frame_h=%d): %r",
                         y_top, y_bot, margin, frame_h, text)
            continue
        x_left, x_right = min(xs), max(xs)
        cells.append((
            (y_top + y_bot) / 2,
            (x_left + x_right) / 2,
            str(text).strip(),
            int(x_left), int(y_top), int(x_right), int(y_bot),
        ))

    if not cells:
        if conf_dropped or margin_dropped:
            logger.info(
                "0 cells kept — conf_dropped=%d (min=%.2f), margin_dropped=%d (margin=%d, frame_h=%d)",
                conf_dropped, min_confidence, margin_dropped, margin, frame_h,
            )
        return []

    cells.sort(key=lambda c: c[0])

    rows: list[list[tuple]] = []
    current: list[tuple] = [cells[0]]
    for cell in cells[1:]:
        mean_y = float(np.mean([c[0] for c in current]))
        if abs(cell[0] - mean_y) <= row_y_tolerance:
            current.append(cell)
        else:
            rows.append(sorted(current, key=lambda c: c[1]))
            current = [cell]
    rows.append(sorted(current, key=lambda c: c[1]))

    return [
        [OcrCell(text=c[2], x1=c[3], y1=c[4], x2=c[5], y2=c[6]) for c in row]
        for row in rows
    ]


def validate_row(
    raw: list[str],
    names: list[str] | None,
    name_match_cutoff: float,
    replacements: list[list[str]] | None = None,
) -> tuple[str, str, str] | None:
    """
    Validate and normalise a raw text row into (name, num1, num2).
    Returns None if the row doesn't match the expected schema.
    """
    if len(raw) < 2:
        return None

    name_cell = apply_replacements(raw[0], replacements)

    if names:
        matched = best_name_match(name_cell, names, name_match_cutoff)
        if matched is None:
            logger.debug("Row rejected: '%s' not in predetermined names", name_cell)
            return None
        name_cell = matched
    elif not re.search(r"[A-Za-z]", name_cell):
        return None

    nums = [clean_number(c) for c in raw[1:] if is_number(c)]
    if not nums:
        logger.debug("Row rejected: no numeric cells after '%s'", name_cell)
        return None

    while len(nums) < 2:
        nums.append("")

    return (name_cell, nums[0], nums[1])
