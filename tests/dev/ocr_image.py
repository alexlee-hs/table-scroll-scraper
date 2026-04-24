"""
Dev script: run PaddleOCR on a single image and print the raw result
structure alongside the parsed rows.

Usage:
    python tests/dev/ocr_image.py
    python tests/dev/ocr_image.py --no-invert
    python tests/dev/ocr_image.py path/to/other.png
"""
from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

import cv2
from paddleocr import PaddleOCR

from scraper.extract_table import ocr_to_rows
from scraper.main import load_config

RESOURCES = Path(__file__).parents[1] / "resources"
DEFAULT_IMAGE = RESOURCES / "test_image01.png"


def main() -> None:
    parser = argparse.ArgumentParser(description="Run OCR on a test image.")
    parser.add_argument("image", nargs="?", default=str(DEFAULT_IMAGE),
                        help="Path to image file (default: tests/resources/test_image01.png)")
    parser.add_argument("--no-invert", action="store_true",
                        help="Skip frame inversion")
    args = parser.parse_args()

    logging.basicConfig(
        level=logging.DEBUG,
        format="%(asctime)s [%(levelname)-8s] %(name)s: %(message)s",
        datefmt="%H:%M:%S",
    )
    logging.getLogger("ppocr").setLevel(logging.WARNING)

    image_path = Path(args.image)
    if not image_path.exists():
        sys.exit(f"Image not found: {image_path}")

    frame = cv2.imread(str(image_path))
    if frame is None:
        sys.exit(f"cv2 could not read image: {image_path}")

    print(f"Image loaded: {image_path.name}  shape={frame.shape}")

    cfg = load_config()
    invert = cfg.get("invert_frame", False) and not args.no_invert
    if invert:
        frame = cv2.bitwise_not(frame)
        print("Frame inverted (white-on-dark → dark-on-light)")

    ocr = PaddleOCR(use_angle_cls=True, lang=cfg.get("lang", "ch"), enable_mkldnn=False)
    ocr_result = ocr.predict(frame)

    # ── raw result ────────────────────────────────────────────────────────────
    print("\n── RAW OCR RESULT ───────────────────────────────────────────────")
    print(f"Type: {type(ocr_result).__name__}, length: {len(ocr_result)}")
    if ocr_result:
        page = ocr_result[0]
        print(f"ocr_result[0] type: {type(page).__name__}")
        if isinstance(page, dict):
            print(f"Keys: {list(page.keys())}")
            for key, val in page.items():
                preview = repr(val)[:120] if not isinstance(val, list) else f"[{len(val)} items]"
                print(f"  {key}: {preview}")
                if isinstance(val, list) and val:
                    print(f"    first item: {repr(val[0])[:200]}")
        else:
            print(repr(page)[:400])

    # ── parsed rows ───────────────────────────────────────────────────────────
    print("\n── PARSED ROWS ─────────────────────────────────────────────────")
    rows = ocr_to_rows(
        ocr_result,
        min_confidence=0.0,
        frame_h=frame.shape[0],
        margin=0,
        row_y_tolerance=cfg.get("row_y_tolerance", 15),
    )
    print(f"Rows found: {len(rows)}")
    for i, row in enumerate(rows):
        cells = [(c.text, f"({c.x1},{c.y1},{c.x2},{c.y2})") for c in row]
        print(f"  Row {i + 1}: {cells}")


if __name__ == "__main__":
    main()
