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
from scraper.util import binarize

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

    if cfg.get("binarize", False):
        frame = binarize(frame)
        print("Frame binarized (Otsu's threshold)")

    ocr_pad = cfg.get("ocr_padding", 0)
    if ocr_pad:
        frame = cv2.copyMakeBorder(
            frame, ocr_pad, ocr_pad, ocr_pad, ocr_pad,
            cv2.BORDER_CONSTANT, value=(255, 255, 255),
        )
        print(f"OCR padding applied: {ocr_pad}px white border on all sides")

    # Save the preprocessed frame so you can inspect what OCR actually sees.
    debug_path = image_path.parent / f"{image_path.stem}_preprocessed.png"
    cv2.imwrite(str(debug_path), frame)
    print(f"Preprocessed frame saved to: {debug_path}")

    # PaddleOCR expects RGB; OpenCV loads BGR.
    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    ocr = PaddleOCR(use_angle_cls=True, lang=cfg.get("lang", "ch"), enable_mkldnn=False)
    ocr_result = ocr.predict(frame_rgb)

    # ── raw detections ────────────────────────────────────────────────────────
    print("\n── RAW DETECTIONS ───────────────────────────────────────────────")
    if not ocr_result:
        print("  (empty result)")
    else:
        page = ocr_result[0]
        if not isinstance(page, dict) or "rec_texts" not in page:
            print(f"  Unexpected format — type={type(page).__name__}, "
                  f"keys={list(page.keys()) if isinstance(page, dict) else 'n/a'}")
        else:
            texts  = page.get("rec_texts",  [])
            scores = page.get("rec_scores", [])
            polys  = page.get("rec_polys",  [])
            print(f"  {len(texts)} detection(s)\n")
            print(f"  {'#':<4}  {'score':>6}  {'x_left':>7}  {'x_right':>8}  {'y_top':>6}  {'y_bot':>6}  text")
            print(f"  {'-'*4}  {'-'*6}  {'-'*7}  {'-'*8}  {'-'*6}  {'-'*6}  ----")
            for i, (text, score, pts) in enumerate(zip(texts, scores, polys)):
                if pts is not None and len(pts):
                    xs = [p[0] for p in pts]
                    ys = [p[1] for p in pts]
                    x1, x2 = int(min(xs)), int(max(xs))
                    y1, y2 = int(min(ys)), int(max(ys))
                else:
                    x1 = x2 = y1 = y2 = -1
                print(f"  {i:<4}  {score:>6.3f}  {x1:>7}  {x2:>8}  {y1:>6}  {y2:>6}  {text!r}")

    # ── parsed rows ───────────────────────────────────────────────────────────
    print("\n── PARSED ROWS ─────────────────────────────────────────────────")
    rows = ocr_to_rows(
        ocr_result,
        min_confidence=0.0,
        frame_h=frame.shape[0],
        margin=ocr_pad,
        row_y_tolerance=cfg.get("row_y_tolerance", 25),
    )
    print(f"Rows found: {len(rows)}")
    for i, row in enumerate(rows):
        print(f"\n  Row {i + 1}:")
        for j, cell in enumerate(row):
            print(f"    Cell {j + 1}: '{cell.text}'  bbox=({cell.x1},{cell.y1},{cell.x2},{cell.y2})")


if __name__ == "__main__":
    main()
