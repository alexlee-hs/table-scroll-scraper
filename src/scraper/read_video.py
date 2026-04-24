from __future__ import annotations

import logging
from collections.abc import Callable, Generator
from dataclasses import dataclass, field

import cv2
import imagehash
import numpy as np

from scraper.util import is_blurry, phash_of

logger = logging.getLogger(__name__)


@dataclass
class FrameStats:
    total: int = field(default=0)
    blurry: int = field(default=0)
    in_motion: int = field(default=0)
    similar: int = field(default=0)
    processed: int = field(default=0)


def video_info(video_path: str) -> dict:
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise IOError(f"Cannot open video: {video_path}")
    info = {
        "fps": cap.get(cv2.CAP_PROP_FPS),
        "total_frames": int(cap.get(cv2.CAP_PROP_FRAME_COUNT)),
    }
    cap.release()
    return info


_MOTION_SCALE = (160, 90)  # downscale size for fast motion scoring


def _motion_score(a: np.ndarray, b: np.ndarray) -> float:
    """Mean absolute pixel difference between two grayscale frames, downscaled for speed."""
    sa = cv2.resize(a, _MOTION_SCALE)
    sb = cv2.resize(b, _MOTION_SCALE)
    return float(np.mean(np.abs(sa.astype(np.int16) - sb.astype(np.int16))))


def iter_frames(
    video_path: str,
    frame_skip: int,
    blur_threshold: float,
    motion_threshold: float,
    hash_diff: int,
    crop: tuple[int, int, int, int] | None,
    stats: FrameStats,
    on_frame: Callable[[], None] | None = None,
) -> Generator[tuple[int, np.ndarray], None, None]:
    """
    Yield (frame_idx, frame_bgr) for every frame that passes all filters:
      1. frame_skip  — coarse rate limiter
      2. blur        — discard frames too blurry for OCR
      3. motion      — discard frames mid-scroll (high inter-frame diff)
      4. hash        — discard frames identical to the last OCR'd frame

    on_frame() is called for every frame read from disk so tqdm stays accurate.
    """
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise IOError(f"Cannot open video: {video_path}")

    prev_hash: imagehash.ImageHash | None = None
    prev_gray: np.ndarray | None = None
    frame_idx = 0

    try:
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            frame_idx += 1
            stats.total += 1
            if on_frame is not None:
                on_frame()

            if frame_idx % frame_skip != 0:
                continue

            if crop:
                x1, y1, x2, y2 = crop
                frame = frame[y1:y2, x1:x2]

            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

            if is_blurry(gray, blur_threshold):
                stats.blurry += 1
                logger.debug("Frame %d skipped: blurry", frame_idx)
                prev_gray = gray
                continue

            if prev_gray is not None:
                score = _motion_score(prev_gray, gray)
                if score > motion_threshold:
                    stats.in_motion += 1
                    logger.debug("Frame %d skipped: in motion (score=%.1f)", frame_idx, score)
                    prev_gray = gray
                    continue
            prev_gray = gray

            h = phash_of(frame)
            if prev_hash is not None and (h - prev_hash) <= hash_diff:
                stats.similar += 1
                logger.debug("Frame %d skipped: near-duplicate", frame_idx)
                continue
            prev_hash = h

            stats.processed += 1
            yield frame_idx, frame

    finally:
        cap.release()
        logger.info(
            "Video read complete — total: %d, blurry: %d, in_motion: %d, similar: %d, OCR'd: %d",
            stats.total,
            stats.blurry,
            stats.in_motion,
            stats.similar,
            stats.processed,
        )
