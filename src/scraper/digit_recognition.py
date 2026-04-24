from __future__ import annotations

import logging
from pathlib import Path

import cv2
import numpy as np

logger = logging.getLogger(__name__)

_DIGIT_SIZE = (32, 32)
_MIN_CONTOUR_AREA = 30   # px² — smaller blobs are noise
_MATCH_THRESHOLD = 0.50  # normalised cross-correlation minimum to accept a digit
_TEMPLATE_EXTS = ("png", "jpg", "jpeg", "bmp")


def _to_binary(gray: np.ndarray) -> np.ndarray:
    """Otsu threshold → white foreground on black background."""
    _, binary = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
    return binary


def _normalise(img: np.ndarray) -> np.ndarray:
    return cv2.resize(img, _DIGIT_SIZE, interpolation=cv2.INTER_AREA)


class DigitRecogniser:
    def __init__(self, digits_dir: Path):
        self.templates: dict[int, np.ndarray] = {}
        for d in range(10):
            for ext in _TEMPLATE_EXTS:
                p = digits_dir / f"{d}.{ext}"
                if p.exists():
                    raw = cv2.imread(str(p), cv2.IMREAD_GRAYSCALE)
                    if raw is not None:
                        self.templates[d] = _normalise(_to_binary(raw)).astype(np.float32)
                        break
        if self.templates:
            logger.info("Digit recogniser: loaded templates for %s", sorted(self.templates))
        else:
            logger.warning("Digit recogniser: no templates found in %s", digits_dir)

    def is_ready(self) -> bool:
        return bool(self.templates)

    def read_number(self, cell_bgr: np.ndarray) -> str | None:
        """
        Segment digit contours from a number-cell crop and match each against
        templates.  Returns the number string, or None if any digit is
        ambiguous or no digits are found (caller should fall back to OCR text).
        """
        gray = cv2.cvtColor(cell_bgr, cv2.COLOR_BGR2GRAY)
        binary = _to_binary(gray)

        contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        if not contours:
            return None

        digit_crops: list[tuple[int, np.ndarray]] = []
        for cnt in contours:
            x, y, w, h = cv2.boundingRect(cnt)
            if w * h < _MIN_CONTOUR_AREA:
                continue
            digit_crops.append((x, binary[y:y + h, x:x + w]))

        if not digit_crops:
            return None

        digit_crops.sort(key=lambda c: c[0])

        result = []
        for _, crop in digit_crops:
            d = self._match(crop)
            if d is None:
                logger.debug("Digit recognition: ambiguous crop, falling back to OCR")
                return None
            result.append(str(d))

        return "".join(result)

    def _match(self, crop: np.ndarray) -> int | None:
        normalised = _normalise(crop).astype(np.float32)
        best_score, best_digit = -1.0, None
        for digit, template in self.templates.items():
            score = float(
                cv2.matchTemplate(normalised, template, cv2.TM_CCOEFF_NORMED)[0, 0]
            )
            if score > best_score:
                best_score, best_digit = score, digit
        if best_score < _MATCH_THRESHOLD:
            logger.debug("Digit recognition: best score %.2f below threshold", best_score)
            return None
        return best_digit
