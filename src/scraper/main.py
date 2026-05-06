from __future__ import annotations

import argparse
import csv
import json
import logging
import sys
from collections import Counter, defaultdict
from datetime import date
from pathlib import Path

import cv2
from paddleocr import PaddleOCR
from tqdm import tqdm

from scraper.digit_recognition import DigitRecogniser
from scraper.extract_table import ocr_to_rows, validate_row
from scraper.read_video import FrameStats, iter_frames, video_info
from scraper.util import binarize

logger = logging.getLogger(__name__)

_CONFIG_PATH = Path(__file__).parent / "config" / "config.json"


def load_config() -> dict:
    with _CONFIG_PATH.open(encoding="utf-8") as f:
        return json.load(f)


def _deduplicate(results: list[tuple]) -> list[tuple]:
    groups = defaultdict(list)
    for row in results:
        groups[row[0]].append(row)

    deduped = []
    for name, rows in groups.items():
        v1 = Counter(r[1] for r in rows if r[1]).most_common(1)
        v2 = Counter(r[2] for r in rows if r[2]).most_common(1)
        deduped.append((name, v1[0][0] if v1 else "", v2[0][0] if v2 else ""))
    return deduped


def resolve_date(cfg: dict, override: str | None) -> str:
    """Return YYYYmmdd: CLI override > config value > today."""
    raw = override or cfg.get("video_date")
    if raw:
        return str(raw)
    return date.today().strftime("%Y%m%d")


def run(
    video_path: str,
    output_csv: str,
    names: list[str] | None,
    replacements: list[list[str]] | None,
    digit_rec: DigitRecogniser | None,
    crop: tuple[int, int, int, int] | None,
    blur_threshold: float,
    motion_threshold: float,
    hash_diff: int,
    frame_skip: int,
    min_confidence: float,
    row_y_tolerance: int,
    name_match_cutoff: float,
    invert_frame: bool = False,
    binarize_frame: bool = False,
    ocr_padding: int = 0,
    use_angle_cls: bool = False,
) -> None:
    info = video_info(video_path)
    logger.info("Video: %.1f fps, %d frames total", info["fps"], info["total_frames"])
    if names:
        logger.info("Predetermined names loaded: %d", len(names))
    if replacements:
        logger.info("Name replacements loaded: %d", len(replacements))
    if digit_rec and digit_rec.is_ready():
        logger.info("Digit recogniser active")
    if crop:
        logger.info("Crop region: %s", crop)

    use_digit_rec = digit_rec is not None and digit_rec.is_ready()
    logging.getLogger("ppocr").setLevel(logging.WARNING)
    ocr = PaddleOCR(use_angle_cls=use_angle_cls, lang="ch", enable_mkldnn=False)
    stats = FrameStats()
    seen_rows: set[tuple] = set()
    results: list[tuple] = []
    total_detections = 0
    total_rejected = 0

    with tqdm(total=info["total_frames"], unit="frame", desc="Scanning") as bar:
        for _frame_idx, frame in iter_frames(
            video_path=video_path,
            frame_skip=frame_skip,
            blur_threshold=blur_threshold,
            motion_threshold=motion_threshold,
            hash_diff=hash_diff,
            crop=crop,
            stats=stats,
            on_frame=bar.update,
        ):
            ocr_input = cv2.bitwise_not(frame) if invert_frame else frame
            if binarize_frame:
                ocr_input = binarize(ocr_input)
            if ocr_padding:
                ocr_input = cv2.copyMakeBorder(
                    ocr_input, ocr_padding, ocr_padding, ocr_padding, ocr_padding,
                    cv2.BORDER_CONSTANT, value=(255, 255, 255),
                )
            frame_h = ocr_input.shape[0]
            ocr_result = ocr.predict(cv2.cvtColor(ocr_input, cv2.COLOR_BGR2RGB))
            rows = ocr_to_rows(ocr_result, min_confidence, frame_h, ocr_padding, row_y_tolerance)
            total_detections += len(rows)

            for row in rows:
                text_row = []
                for i, cell in enumerate(row):
                    if i > 0 and use_digit_rec:
                        h, w = frame.shape[:2]
                        y1 = max(0, cell.y1 - ocr_padding)
                        y2 = min(h, cell.y2 - ocr_padding)
                        x1 = max(0, cell.x1 - ocr_padding)
                        x2 = min(w, cell.x2 - ocr_padding)
                        crop_img = frame[y1:y2, x1:x2]
                        recognised = digit_rec.read_number(crop_img) if crop_img.size else None
                        text_row.append(recognised if recognised is not None else cell.text)
                    else:
                        text_row.append(cell.text)

                normalized = validate_row(text_row, names, name_match_cutoff, replacements)
                if normalized and normalized not in seen_rows:
                    seen_rows.add(normalized)
                    results.append(normalized)
                    logger.debug("New row: %s", normalized)
                    bar.set_postfix(rows=len(results))
                else:
                    total_rejected += 1
                    logger.debug("Row rejected: %s", text_row)

    logger.info(
        "Frames  — total: %d, blurry: %d, in_motion: %d, similar: %d, OCR'd: %d",
        stats.total, stats.blurry, stats.in_motion, stats.similar, stats.processed,
    )
    logger.info(
        "Rows    — detected: %d, rejected: %d, unique kept: %d",
        total_detections, total_rejected, len(results),
    )

    results = _deduplicate(results)
    logger.info("After deduplication: %d unique rows", len(results))

    out = Path(output_csv)
    out.parent.mkdir(parents=True, exist_ok=True)
    with out.open("w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["Name", "Value1", "Value2"])
        writer.writerows(results)

    logger.info("CSV written to: %s", out.resolve())


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Extract table rows from a scrolling-table video.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--date", metavar="YYYYmmdd",
        help="Date to use in the video filename (overrides config; default: today)",
    )
    parser.add_argument("--debug", action="store_true", help="Enable DEBUG-level logging")
    args = parser.parse_args()

    logging.basicConfig(
        level=logging.DEBUG if args.debug else logging.INFO,
        format="%(asctime)s [%(levelname)-8s] %(name)s: %(message)s",
        datefmt="%H:%M:%S",
        stream=sys.stderr,
    )

    cfg = load_config()

    date_str = resolve_date(cfg, args.date)
    video_path = cfg["video_prefix"] + date_str + cfg.get("video_extension", ".mp4")
    logger.info("Video path: %s", video_path)

    names = None
    replacements = None
    if cfg.get("names_file"):
        members_path = _CONFIG_PATH.parent / cfg["names_file"]
        members = json.loads(members_path.read_text(encoding="utf-8"))
        names = members.get("names")
        replacements = members.get("name_replacements")

    digit_rec = None
    if cfg.get("digits_dir"):
        digit_rec = DigitRecogniser(_CONFIG_PATH.parent / cfg["digits_dir"])

    crop = tuple(cfg["crop"]) if cfg.get("crop") else None

    run(
        video_path=video_path,
        output_csv=cfg["output_csv"],
        names=names,
        replacements=replacements,
        digit_rec=digit_rec,
        crop=crop,
        blur_threshold=cfg["blur_threshold"],
        motion_threshold=cfg["motion_threshold"],
        hash_diff=cfg["hash_diff"],
        frame_skip=cfg["frame_skip"],
        min_confidence=cfg["min_confidence"],
        row_y_tolerance=cfg["row_y_tolerance"],
        name_match_cutoff=cfg["name_match_cutoff"],
        invert_frame=cfg.get("invert_frame", False),
        binarize_frame=cfg.get("binarize", False),
        ocr_padding=cfg.get("ocr_padding", 0),
        use_angle_cls=cfg.get("use_angle_cls", False),
    )
