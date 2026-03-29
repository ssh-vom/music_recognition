"""Clef detection (treble vs bass) via template matching.

Uses OpenCV matchTemplate with letterbox and multi-scale approaches.
Based on: https://docs.opencv.org/4.x/d4/dc6/tutorial_py_template_matching.html
"""

from pathlib import Path

import cv2 as cv
import numpy as np
from cv2.typing import MatLike

from schema import ClefDetection, ClefKind
from symbol_templates import CLEF_BASS, CLEF_TREBLE

# Template constants - loaded once at module level
_TREBLE_TEMPLATE = None
_BASS_TEMPLATE = None

# Detection thresholds (as ratios for scale invariance)
CLEF_HORIZONTAL_FRACTION = 0.42  # Left portion of crop to search (clef region)
MIN_CONFIDENCE = 0.15  # Minimum match score to accept a clef detection
LETTERBOX_TIE_MARGIN = 0.06  # Margin for preferring treble when scores are close


def _load_templates() -> tuple[MatLike, MatLike]:
    """Load and cache clef templates, trimming white margins."""
    global _TREBLE_TEMPLATE, _BASS_TEMPLATE

    if _TREBLE_TEMPLATE is None:
        _TREBLE_TEMPLATE = _load_and_trim(CLEF_TREBLE)
    if _BASS_TEMPLATE is None:
        _BASS_TEMPLATE = _load_and_trim(CLEF_BASS)

    return _TREBLE_TEMPLATE, _BASS_TEMPLATE


def _load_and_trim(path: Path) -> MatLike:
    """Load template image and trim white margins."""
    img = cv.imread(str(path), cv.IMREAD_COLOR)
    if img is None:
        raise FileNotFoundError(f"Cannot read clef template: {path}")
    gray = _to_gray(img)
    return _trim_white(gray)


def _to_gray(image: MatLike) -> MatLike:
    """Convert image to grayscale if needed."""
    if len(image.shape) == 2:
        return image
    return cv.cvtColor(image, cv.COLOR_BGR2GRAY)


def _trim_white(gray: MatLike, white_thresh: int = 248) -> MatLike:
    """Trim white margins from template image.

    Removes padding so matching targets the glyph, not empty space.
    """
    _, inv = cv.threshold(gray, white_thresh, 255, cv.THRESH_BINARY_INV)
    pts = cv.findNonZero(inv)
    if pts is None:
        return gray
    x, y, w, h = cv.boundingRect(pts)
    if w < 2 or h < 2:
        return gray
    return gray[y : y + h, x : x + w]


def detect_clef(clef_key_crop: MatLike) -> ClefDetection:
    """Detect treble or bass clef in a clef+key signature crop.

    Uses two matching strategies:
    1. Letterbox match: Template scaled to fit full crop height (whole glyph)
    2. Multi-scale match: Sliding window at multiple scales (catches partial matches)

    Returns ClefDetection with scores for both strategies.
    """
    treble_template, bass_template = _load_templates()

    # Prepare ROI - left portion of crop where clef appears
    left = _prepare_left_roi(clef_key_crop)
    if left is None:
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

    # Letterbox matching - template fits full crop height
    treble_score, treble_rect = _letterbox_match(left, treble_template)
    bass_score, bass_rect = _letterbox_match(left, bass_template)

    # Multi-scale sliding window matching
    slide_treble, _ = _multi_scale_match(left, treble_template)
    slide_bass, _ = _multi_scale_match(left, bass_template)

    # Decision: treble wins if within tie margin, otherwise higher score wins
    kind, confidence = _decide_clef(treble_score, bass_score)

    tx, ty, tw, th = treble_rect
    bx, by, bw, bh = bass_rect

    return ClefDetection(
        clef=kind,
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


def _prepare_left_roi(clef_key_crop: MatLike) -> MatLike | None:
    """Extract left portion of crop where clef appears.

    Uses left 42% of crop width - clef appears in left portion,
    key signature appears to the right.
    """
    gray = _to_gray(clef_key_crop)
    # Invert staff-erased binary (black ink on white → white ink on black for matching)
    gray = cv.bitwise_not(gray)

    width = gray.shape[1]
    left = gray[:, : max(1, int(width * CLEF_HORIZONTAL_FRACTION))]

    if left.shape[0] < 8 or left.shape[1] < 8:
        return None
    return left


def _decide_clef(
    treble_score: float, bass_score: float
) -> tuple[ClefKind | None, float]:
    """Decide clef type based on match scores.

    Treble wins if: treble_score >= bass_score - tie_margin
    This slight bias prevents flip-flopping when scores are similar.
    """
    if treble_score + LETTERBOX_TIE_MARGIN >= bass_score:
        winner = "treble"
        confidence = treble_score
    else:
        winner = "bass"
        confidence = bass_score

    if confidence < MIN_CONFIDENCE:
        return None, confidence
    return winner, confidence


def _letterbox_match(roi_gray: MatLike, template_gray: MatLike) -> tuple[float, tuple]:
    """Match template to ROI using letterbox approach.

    Scales template to fit full ROI height while maintaining aspect ratio.
    Template is centered vertically, placed at left edge horizontally.
    Returns match score and rectangle (x, y, w, h) of template position.
    """
    roi_h, roi_w = roi_gray.shape[:2]
    th, tw = template_gray.shape[:2]

    if th < 1 or tw < 1:
        return 0.0, (0, 0, 0, 0)

    # Scale to fit height with small margin
    scale = min((roi_w - 1) / tw, (roi_h - 1) / th) * 0.99
    scale = max(scale, 1e-6)

    new_w = max(1, int(round(tw * scale)))
    new_h = max(1, int(round(th * scale)))

    resized = cv.resize(
        template_gray,
        (new_w, new_h),
        interpolation=cv.INTER_AREA if scale < 1.0 else cv.INTER_CUBIC,
    )

    # Create canvas and center template vertically, align left
    canvas = np.full((roi_h, roi_w), 255, dtype=np.uint8)
    y0 = max(0, (roi_h - new_h) // 2)
    x0 = 0
    y1 = min(roi_h, y0 + new_h)
    x1 = min(roi_w, x0 + new_w)
    rw = x1 - x0
    rh = y1 - y0
    canvas[y0:y1, x0:x1] = resized[:rh, :rw]

    # Template matching with normalized correlation coefficient
    result = cv.matchTemplate(roi_gray, canvas, cv.TM_CCOEFF_NORMED)
    score = float(result[0, 0])
    rect = (x0, y0, rw, rh)

    return score, rect


def _multi_scale_match(
    roi_gray: MatLike, template_gray: MatLike
) -> tuple[float, tuple]:
    """Match template using sliding window at multiple scales.

    Tests scales: 78%, 88%, 95%, 100% of ROI height
    Returns best score and rectangle of best match.
    """
    roi_h, roi_w = roi_gray.shape[:2]
    best_score = 0.0
    best_rect = (0, 0, 0, 0)

    for frac in (0.78, 0.88, 0.95, 1.0):
        target_h = max(12, min(roi_h - 1, int(round(roi_h * frac))))
        scaled = _resize_to_height(template_gray, target_h)
        scaled = _fit_to_roi(scaled, roi_h, roi_w)
        sh, sw = scaled.shape[:2]

        if sh < 4 or sw < 4 or sh > roi_h or sw > roi_w:
            continue

        result = cv.matchTemplate(roi_gray, scaled, cv.TM_CCOEFF_NORMED)
        _, max_val, _, max_loc = cv.minMaxLoc(result)
        score = float(max_val)

        if score > best_score:
            best_score = score
            x, y = int(max_loc[0]), int(max_loc[1])
            best_rect = (x, y, sw, sh)

    return best_score, best_rect


def _resize_to_height(template_gray: MatLike, target_h: int) -> MatLike:
    """Resize template to target height, maintaining aspect ratio."""
    th, tw = template_gray.shape[:2]
    if th < 1 or target_h < 1:
        return template_gray

    scale = target_h / th
    new_w = max(1, int(round(tw * scale)))
    new_h = max(1, int(round(th * scale)))

    interp = cv.INTER_AREA if scale < 1.0 else cv.INTER_CUBIC
    return cv.resize(template_gray, (new_w, new_h), interpolation=interp)


def _fit_to_roi(template_gray: MatLike, roi_h: int, roi_w: int) -> MatLike:
    """Ensure template fits within ROI dimensions."""
    th, tw = template_gray.shape[:2]
    if th <= 0 or tw <= 0:
        return template_gray

    if th <= roi_h and tw <= roi_w:
        return template_gray

    scale = min((roi_h - 1) / th, (roi_w - 1) / tw) * 0.99
    scale = max(scale, 1e-3)
    new_w = max(1, int(round(tw * scale)))
    new_h = max(1, int(round(th * scale)))

    return cv.resize(template_gray, (new_w, new_h), interpolation=cv.INTER_AREA)
