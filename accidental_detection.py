"""Accidental detection (sharp/flat) via template matching."""

from pathlib import Path

import cv2 as cv
import numpy as np
from cv2.typing import MatLike

from image_utils import to_gray
from schema import Accidental
from symbol_templates import ACCIDENTAL_FLAT, ACCIDENTAL_SHARP
from template_geometry import fit_to_roi, resize_to_height

_SHARP_TEMPLATE = None
_FLAT_TEMPLATE = None

MATCH_THRESHOLD = 0.5
SCALE_FRACTIONS = (0.35, 0.5, 0.65, 0.8)
MIN_PEAK_DISTANCE_FRAC = 0.55
NOTEHEAD_CLEARANCE_FRAC = 0.22


def _load_templates() -> tuple[MatLike, MatLike]:
    global _SHARP_TEMPLATE, _FLAT_TEMPLATE
    if _SHARP_TEMPLATE is None:
        _SHARP_TEMPLATE = _load_template(ACCIDENTAL_SHARP)
    if _FLAT_TEMPLATE is None:
        _FLAT_TEMPLATE = _load_template(ACCIDENTAL_FLAT)
    return _SHARP_TEMPLATE, _FLAT_TEMPLATE


def _load_template(path: Path) -> MatLike:
    image = cv.imread(str(path), cv.IMREAD_COLOR)
    if image is None:
        raise FileNotFoundError(f"Cannot read accidental template: {path}")
    return to_gray(image)


def detect_key_signature_accidentals(
    clef_key_crop: MatLike,
    staff,
    staff_index: int,
    x_start: int,
    x_end: int,
) -> list[Accidental]:
    """Detect sharps and flats in the key signature."""
    width = clef_key_crop.shape[1]
    x_start = max(0, min(width, x_start))
    x_end = max(x_start, min(width, x_end))
    key_roi_mask = clef_key_crop[:, x_start:x_end]

    if key_roi_mask.shape[1] < 4:
        return []

    # For tiny crops, template matching is unstable; use geometry instead
    use_geometric = staff.spacing <= 8.5 or key_roi_mask.shape[1] <= 24
    if use_geometric:
        matches = _detect_header_accidentals_geometric(key_roi_mask, staff.spacing)
    else:
        matches = _match_templates_in_roi(cv.bitwise_not(key_roi_mask), staff.spacing)

    accidentals = [
        Accidental(
            kind=kind,
            staff_index=staff_index,
            measure_index=-1,
            center_x=x_start + cx,
            center_y=cy,
            confidence=score,
            region="header",
        )
        for score, cx, cy, kind in matches
    ]
    accidentals.sort(key=lambda a: (a.center_x, a.kind))
    return accidentals


def _exclusive_x_before_first_note(staff, detected_notes) -> int | None:
    if not detected_notes:
        return None
    first_x = min(n.center_x for n in detected_notes)
    end = first_x - max(1, int(round(staff.spacing * NOTEHEAD_CLEARANCE_FRAC)))
    return end if end >= 4 else None


def _match_templates_in_roi(roi: MatLike, spacing: float) -> list[tuple]:
    if roi.shape[0] < 4 or roi.shape[1] < 4:
        return []

    sharp_template, flat_template = _load_templates()
    min_dist = max(4, int(round(spacing * MIN_PEAK_DISTANCE_FRAC)))
    candidates = []

    for kind, template in [("sharp", sharp_template), ("flat", flat_template)]:
        for frac in SCALE_FRACTIONS:
            target_h = max(4, int(round(spacing * frac)))
            scaled = resize_to_height(template, target_h)
            scaled = fit_to_roi(scaled, roi.shape[0], roi.shape[1])
            th, tw = scaled.shape[:2]

            if th < 3 or tw < 3:
                continue

            result = cv.matchTemplate(roi, scaled, cv.TM_CCOEFF_NORMED)
            for score, cx, cy in _gather_peaks(
                result, MATCH_THRESHOLD, min_dist, tw, th
            ):
                candidates.append((score, cx, cy, kind))

    return _nms(candidates, min_dist)


def _detect_header_accidentals_geometric(
    roi: MatLike, spacing: float
) -> list[tuple[float, int, int, str]]:
    """Detect header accidentals by counting vertical stroke clusters.

    Sharps have two tall stroke clusters; flats have one.
    """
    if roi.size == 0:
        return []

    count, labels, stats, _ = cv.connectedComponentsWithStats(roi, connectivity=8)
    min_area = max(8, int(round(spacing * spacing * 0.18)))
    min_height = max(6, int(round(spacing * 1.8)))
    out: list[tuple[float, int, int, str]] = []

    for i in range(1, count):
        area = int(stats[i, cv.CC_STAT_AREA])
        if area < min_area:
            continue
        x = int(stats[i, cv.CC_STAT_LEFT])
        y = int(stats[i, cv.CC_STAT_TOP])
        w = int(stats[i, cv.CC_STAT_WIDTH])
        h = int(stats[i, cv.CC_STAT_HEIGHT])
        if h < min_height or w < 2:
            continue

        comp = (labels[y : y + h, x : x + w] == i).astype(np.uint8)
        tall_threshold = max(3, int(round(h * 0.75)))
        tall_cols = [
            idx for idx, v in enumerate(np.sum(comp, axis=0)) if v >= tall_threshold
        ]
        tall_clusters = _count_index_clusters(tall_cols)

        if tall_clusters >= 2:
            out.append((0.9, x + w // 2, y + h // 2, "sharp"))
        elif tall_clusters == 1:
            out.append((0.85, x + w // 2, y + h // 2, "flat"))

    return out


def _count_index_clusters(indices: list[int], max_gap: int = 1) -> int:
    if not indices:
        return 0
    clusters = 1
    prev = indices[0]
    for value in indices[1:]:
        if value - prev > max_gap:
            clusters += 1
        prev = value
    return clusters


def _gather_peaks(
    result: MatLike, threshold: float, min_dist: int, tw: int, th: int
) -> list[tuple]:
    work = result.copy()
    peaks = []

    while True:
        _, max_val, _, max_loc = cv.minMaxLoc(work)
        if max_val < threshold:
            break
        x, y = int(max_loc[0]), int(max_loc[1])
        cx, cy = x + tw // 2, y + th // 2
        peaks.append((float(max_val), cx, cy))
        cv.circle(work, (cx, cy), min_dist, 0, thickness=-1)

    return peaks


def _nms(candidates: list, min_dist: int) -> list:
    candidates = sorted(candidates, key=lambda t: -t[0])
    kept = []

    for score, cx, cy, kind in candidates:
        if any(
            (cx - ox) ** 2 + (cy - oy) ** 2 < min_dist * min_dist
            for _, ox, oy, _ in kept
        ):
            continue
        kept.append((score, cx, cy, kind))

    return kept
