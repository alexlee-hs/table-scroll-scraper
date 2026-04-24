from __future__ import annotations

import difflib
import re

import cv2
import imagehash
import numpy as np
from PIL import Image

_NUMBER_RE = re.compile(r"^-?[\d,.\s]+$")


def is_blurry(gray: np.ndarray, threshold: float) -> bool:
    return cv2.Laplacian(gray, cv2.CV_64F).var() < threshold


def phash_of(bgr: np.ndarray) -> imagehash.ImageHash:
    return imagehash.phash(Image.fromarray(cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)))


def is_number(text: str) -> bool:
    return bool(_NUMBER_RE.match(text))


def clean_number(text: str) -> str:
    return re.sub(r"[,\s]", "", text)


def best_name_match(text: str, names: list[str], cutoff: float) -> str | None:
    matches = difflib.get_close_matches(text, names, n=1, cutoff=cutoff)
    return matches[0] if matches else None


def apply_replacements(text: str, replacements: list[list[str]] | None) -> str:
    """Exact-match replace known OCR misreads before fuzzy name matching."""
    if not replacements:
        return text
    for from_text, to_text in replacements:
        if text == from_text:
            return to_text
    return text
