"""Clef detection - classify treble vs bass clef using template matching.
Based largely on the content below, but optimized for treble and bass clef
https://docs.opencv.org/3.4/d4/dc6/tutorial_py_template_matching.html
"""

from pathlib import Path

import cv2 as cv
import numpy as np
from cv2.typing import MatLike

from constants import (
    CLEF_MATCH_SCALES,
    CLEF_MIN_CONFIDENCE,
    CLEF_ROI_WIDTH_FRAC,
    CLEF_TIE_MARGIN,
    CLEF_TRIM_WHITE_THRESH,
)
from schema import ClefDetection, ClefKind
from symbol_templates import CLEF_BASS, CLEF_TREBLE
from template_geometry import fit_to_roi, resize_to_height, to_gray

_treble_template = None
_bass_template = None


def detect_clef(clef_key_crop: MatLike) -> ClefDetection:
    treble_template, bass_template = _load_templates()

    roi = _prepare_roi(clef_key_crop)
    if roi is None:
        return _empty_detection()

    treble_score, treble_rect = _letterbox_match(roi, treble_template)
    bass_score, bass_rect = _letterbox_match(roi, bass_template)

    slide_treble, _ = _multi_scale_match(roi, treble_template)
    slide_bass, _ = _multi_scale_match(roi, bass_template)

    clef, confidence = _select_clef(treble_score, bass_score)

    tx, ty, tw, th = treble_rect
    bx, by, bw, bh = bass_rect

    return ClefDetection(
        clef=clef,
        confidence=confidence,
        letter_score_treble=treble_score,
        letter_score_bass=bass_score,
        slide_score_treble=slide_treble,
        slide_score_bass=slide_bass,
        treble_match_top_left=(tx, ty),
        treble_match_size=(tw, th),
        bass_match_top_left=(bx, by),
        bass_match_size=(bw, bh),
    )


def _load_templates() -> tuple[MatLike, MatLike]:
    global _treble_template, _bass_template
    if _treble_template is None:
        _treble_template = _load_and_trim(CLEF_TREBLE)
    if _bass_template is None:
        _bass_template = _load_and_trim(CLEF_BASS)
    return _treble_template, _bass_template


def _load_and_trim(path: Path) -> MatLike:
    img = cv.imread(str(path), cv.IMREAD_COLOR)
    if img is None:
        raise FileNotFoundError(f"Cannot load clef template: {path}")
    return _trim_white_border(to_gray(img))


def _trim_white_border(gray: MatLike, thresh: int = CLEF_TRIM_WHITE_THRESH) -> MatLike:
    _, inv = cv.threshold(gray, thresh, 255, cv.THRESH_BINARY_INV)
    pts = cv.findNonZero(inv)
    if pts is None:
        return gray
    x, y, w, h = cv.boundingRect(pts)
    if w < 2 or h < 2:
        return gray
    return gray[y : y + h, x : x + w]


def _prepare_roi(clef_key_crop: MatLike) -> MatLike | None:
    gray = to_gray(clef_key_crop)
    gray = cv.bitwise_not(gray)
    width = gray.shape[1]
    roi = gray[:, : max(1, int(width * CLEF_ROI_WIDTH_FRAC))]
    if roi.shape[0] < 8 or roi.shape[1] < 8:
        return None
    return roi


def _empty_detection() -> ClefDetection:
    return ClefDetection(
        clef=None,
        confidence=0.0,
        letter_score_treble=0.0,
        letter_score_bass=0.0,
        slide_score_treble=0.0,
        slide_score_bass=0.0,
        treble_match_top_left=(0, 0),
        treble_match_size=(0, 0),
        bass_match_top_left=(0, 0),
        bass_match_size=(0, 0),
    )


def _select_clef(
    treble_score: float, bass_score: float
) -> tuple[ClefKind | None, float]:
    if treble_score + CLEF_TIE_MARGIN >= bass_score:
        winner, confidence = "treble", treble_score
    else:
        winner, confidence = "bass", bass_score

    if confidence < CLEF_MIN_CONFIDENCE:
        return None, confidence
    return winner, confidence


def _letterbox_match(roi: MatLike, template: MatLike) -> tuple[float, tuple]:
    roi_h, roi_w = roi.shape[:2]
    th, tw = template.shape[:2]

    if th < 1 or tw < 1:
        return 0.0, (0, 0, 0, 0)

    scale = max(min((roi_w - 1) / tw, (roi_h - 1) / th) * 0.99, 1e-6)
    new_w = max(1, int(round(tw * scale)))
    new_h = max(1, int(round(th * scale)))

    interp = cv.INTER_AREA if scale < 1.0 else cv.INTER_CUBIC
    resized = cv.resize(template, (new_w, new_h), interpolation=interp)

    canvas = np.full((roi_h, roi_w), 255, dtype=np.uint8)
    y0, x0 = max(0, (roi_h - new_h) // 2), 0
    y1, x1 = min(roi_h, y0 + new_h), min(roi_w, x0 + new_w)
    canvas[y0:y1, x0:x1] = resized[: y1 - y0, : x1 - x0]

    result = cv.matchTemplate(roi, canvas, cv.TM_CCOEFF_NORMED)
    return float(result[0, 0]), (x0, y0, x1 - x0, y1 - y0)


def _multi_scale_match(roi: MatLike, template: MatLike) -> tuple[float, tuple]:
    roi_h, roi_w = roi.shape[:2]
    best_score = 0.0
    best_rect = (0, 0, 0, 0)

    for scale_frac in CLEF_MATCH_SCALES:
        target_h = max(12, min(roi_h - 1, int(round(roi_h * scale_frac))))
        scaled = resize_to_height(template, target_h)
        scaled = fit_to_roi(scaled, roi_h, roi_w)
        sh, sw = scaled.shape[:2]

        if sh < 4 or sw < 4 or sh > roi_h or sw > roi_w:
            continue

        result = cv.matchTemplate(roi, scaled, cv.TM_CCOEFF_NORMED)
        _, max_val, _, max_loc = cv.minMaxLoc(result)

        if max_val > best_score:
            best_score = float(max_val)
            best_rect = (int(max_loc[0]), int(max_loc[1]), sw, sh)

    return best_score, best_rect
