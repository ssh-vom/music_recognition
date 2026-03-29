"""Accidental detection (sharp/flat) via template matching."""

from pathlib import Path

import cv2 as cv
from cv2.typing import MatLike

from schema import Accidental
from symbol_templates import ACCIDENTAL_FLAT, ACCIDENTAL_SHARP

_SHARP_TEMPLATE = None
_FLAT_TEMPLATE = None

MATCH_THRESHOLD = 0.5
SCALE_FRACTIONS = (0.35, 0.5, 0.65, 0.8)
MIN_PEAK_DISTANCE_FRAC = 0.55
NOTEHEAD_CLEARANCE_FRAC = 0.22


def _load_templates() -> tuple[MatLike, MatLike]:
    """Load and cache sharp/flat templates."""
    global _SHARP_TEMPLATE, _FLAT_TEMPLATE

    if _SHARP_TEMPLATE is None:
        _SHARP_TEMPLATE = _load_template(ACCIDENTAL_SHARP)
    if _FLAT_TEMPLATE is None:
        _FLAT_TEMPLATE = _load_template(ACCIDENTAL_FLAT)

    return _SHARP_TEMPLATE, _FLAT_TEMPLATE


def _load_template(path: Path) -> MatLike:
    """Load template image and convert to grayscale."""
    image = cv.imread(str(path), cv.IMREAD_COLOR)
    if image is None:
        raise FileNotFoundError(f"Cannot read accidental template: {path}")
    return _to_gray(image)


def _to_gray(image: MatLike) -> MatLike:
    """Convert image to grayscale if needed."""
    if len(image.shape) == 2:
        return image
    return cv.cvtColor(image, cv.COLOR_BGR2GRAY)


def detect_measure_accidentals(
    mask: MatLike,
    staff,
    measure,
    measure_index: int,
    detected_notes: list,
) -> list[Accidental]:
    """Detect sharps and flats in a measure (left of first notehead)."""
    first_note_x = _exclusive_x_before_first_note(staff, detected_notes)
    if first_note_x is None or first_note_x < 4:
        return []

    roi = cv.bitwise_not(mask)
    if roi.size == 0:
        return []

    x_end = min(first_note_x, roi.shape[1])
    if x_end < 4:
        return []
    roi = roi[:, :x_end]

    matches = _match_templates_in_roi(roi, staff.spacing)

    accidentals = []
    for score, cx, cy, kind in matches:
        accidentals.append(
            Accidental(
                kind=kind,
                staff_index=measure.staff_index,
                measure_index=measure_index,
                center_x=cx,
                center_y=cy,
                confidence=score,
                region="measure",
            )
        )

    accidentals.sort(key=lambda a: (a.center_x, a.kind))
    return accidentals


def detect_key_signature_accidentals(
    clef_key_crop: MatLike,
    staff,
    staff_index: int,
    x_start: int = 0,
    x_end: int | None = None,
) -> list[Accidental]:
    """Detect sharps and flats in the key signature."""
    roi = cv.bitwise_not(clef_key_crop)
    if roi.size == 0:
        return []

    width = roi.shape[1]
    if x_end is None:
        x_end = width
    x_start = max(0, min(width, x_start))
    x_end = max(x_start, min(width, x_end))
    key_roi = roi[:, x_start:x_end]

    if key_roi.shape[1] < 4:
        return []

    matches = _match_templates_in_roi(key_roi, staff.spacing)

    accidentals = []
    for score, cx, cy, kind in matches:
        accidentals.append(
            Accidental(
                kind=kind,
                staff_index=staff_index,
                measure_index=-1,
                center_x=x_start + cx,
                center_y=cy,
                confidence=score,
                region="header",
            )
        )

    accidentals.sort(key=lambda a: (a.center_x, a.kind))
    return accidentals


def _exclusive_x_before_first_note(staff, detected_notes) -> int | None:
    """Calculate x-coordinate before first notehead (with margin).

    Returns right boundary for accidental search area.
    """
    if not detected_notes:
        return None

    first_x = min(n.center_x for n in detected_notes)
    margin = max(1, int(round(staff.spacing * NOTEHEAD_CLEARANCE_FRAC)))
    end = first_x - margin

    if end < 4:
        return None
    return end


def _match_templates_in_roi(roi: MatLike, spacing: float) -> list[tuple]:
    """Run template matching for both sharp and flat in ROI.

    Tests multiple template scales and applies non-maximum suppression.
    Returns list of (score, cx, cy, kind) tuples.
    """
    if roi.shape[0] < 4 or roi.shape[1] < 4:
        return []

    sharp_template, flat_template = _load_templates()

    min_dist = max(4, int(round(spacing * MIN_PEAK_DISTANCE_FRAC)))
    candidates = []

    for kind, template in [("sharp", sharp_template), ("flat", flat_template)]:
        for frac in SCALE_FRACTIONS:
            target_h = max(4, int(round(spacing * frac)))
            scaled = _resize_to_height(template, target_h)
            scaled = _fit_to_roi(scaled, roi.shape[0], roi.shape[1])
            th, tw = scaled.shape[:2]

            if th < 3 or tw < 3:
                continue

            result = cv.matchTemplate(roi, scaled, cv.TM_CCOEFF_NORMED)
            peaks = _gather_peaks(result, MATCH_THRESHOLD, min_dist, tw, th)

            for score, cx, cy in peaks:
                candidates.append((score, cx, cy, kind))

    return _nms(candidates, min_dist)


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


def _gather_peaks(
    result: MatLike, threshold: float, min_dist: int, tw: int, th: int
) -> list[tuple]:
    """Gather peaks from matchTemplate result using greedy suppression.

    Finds local maxima above threshold, suppresses nearby points.
    Returns list of (score, cx, cy) tuples.
    """
    work = result.copy()
    peaks = []

    while True:
        _, max_val, _, max_loc = cv.minMaxLoc(work)
        if max_val < threshold:
            break

        x, y = int(max_loc[0]), int(max_loc[1])
        cx = x + tw // 2  # Center of template match
        cy = y + th // 2

        peaks.append((float(max_val), cx, cy))

        # Suppress nearby points
        cv.circle(work, (cx, cy), min_dist, 0, thickness=-1)

    return peaks


def _nms(candidates: list, min_dist: int) -> list:
    """Non-maximum suppression: keep strongest peaks that don't overlap.

    Candidates are sorted by score (descending), then spatially filtered.
    """
    candidates = sorted(candidates, key=lambda t: -t[0])
    kept = []

    for score, cx, cy, kind in candidates:
        # Check if this peak is too close to any kept peak
        overlaps = any(
            (cx - ox) ** 2 + (cy - oy) ** 2 < (min_dist * min_dist)
            for _, ox, oy, _ in kept
        )

        if overlaps:
            continue

        kept.append((score, cx, cy, kind))

    return kept
