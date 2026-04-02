from pathlib import Path

import cv2 as cv
import numpy as np
from cv2.typing import MatLike

from schema import Accidental
from utils import (
    ACCIDENTAL_FLAT,
    ACCIDENTAL_SHARP,
    fit_to_roi,
    resize_to_height,
    to_gray,
)

_SHARP_TEMPLATE = None
_FLAT_TEMPLATE = None

MATCH_THRESHOLD = 0.5
SCALE_FRACTIONS = (0.35, 0.5, 0.65, 0.8)
MIN_PEAK_DISTANCE_FRAC = 0.55


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


def detect_header_key_signature(
    clef_key_crop: MatLike,
    staff,
    staff_index: int,
    x_start: int,
    x_end: int,
) -> list[Accidental]:
    accidentals = detect_key_signature_accidentals(
        clef_key_crop=clef_key_crop,
        staff=staff,
        staff_index=staff_index,
        x_start=x_start,
        x_end=x_end,
    )
    return _clean_header_accidentals(accidentals, clef_key_crop, staff.spacing)


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


def _clean_header_accidentals(
    accidentals: list[Accidental], crop: MatLike, spacing: float
) -> list[Accidental]:
    if not accidentals or crop.size == 0:
        return []

    deduped = _dedup_by_x(accidentals, max(2, int(round(spacing * 0.60))))
    if len(deduped) < 2:
        return deduped

    count, labels, stats, _ = cv.connectedComponentsWithStats(crop, connectivity=8)
    min_area = max(8, int(round(spacing * spacing * 0.10)))
    best_by_label: dict[int, Accidental] = {}
    unlabeled: list[Accidental] = []

    for glyph in deduped:
        x = max(0, min(crop.shape[1] - 1, glyph.center_x))
        y = max(0, min(crop.shape[0] - 1, glyph.center_y))
        label = int(labels[y, x])
        if label <= 0 or int(stats[label, cv.CC_STAT_AREA]) < min_area:
            unlabeled.append(glyph)
            continue

        cleaned = Accidental(
            kind=_accidental_kind_from_component(label, labels, stats) or glyph.kind,
            staff_index=glyph.staff_index,
            measure_index=glyph.measure_index,
            center_x=glyph.center_x,
            center_y=glyph.center_y,
            confidence=glyph.confidence,
            region=glyph.region,
        )
        current = best_by_label.get(label)
        if current is None or cleaned.confidence > current.confidence:
            best_by_label[label] = cleaned

    cleaned = sorted(
        [*best_by_label.values(), *unlabeled], key=lambda glyph: glyph.center_x
    )
    return _keep_dominant_kind(cleaned)


def _dedup_by_x(accidentals: list[Accidental], x_tol: int) -> list[Accidental]:
    kept: list[Accidental] = []
    for glyph in sorted(accidentals, key=lambda g: g.confidence, reverse=True):
        if any(abs(glyph.center_x - other.center_x) <= x_tol for other in kept):
            continue
        kept.append(glyph)
    kept.sort(key=lambda glyph: glyph.center_x)
    return kept


def _accidental_kind_from_component(label: int, labels, stats) -> str | None:
    left = int(stats[label, cv.CC_STAT_LEFT])
    top = int(stats[label, cv.CC_STAT_TOP])
    width = int(stats[label, cv.CC_STAT_WIDTH])
    height = int(stats[label, cv.CC_STAT_HEIGHT])
    component = (labels[top : top + height, left : left + width] == label).astype(
        np.uint8
    )
    tall_threshold = max(3, int(round(height * 0.75)))
    tall_cols = [
        idx
        for idx, value in enumerate(np.sum(component, axis=0))
        if value >= tall_threshold
    ]
    tall_clusters = _count_index_clusters(tall_cols)
    if tall_clusters >= 2:
        return "sharp"
    if tall_clusters == 1:
        return "flat"
    return None


def _keep_dominant_kind(accidentals: list[Accidental]) -> list[Accidental]:
    if len(accidentals) < 2:
        return accidentals

    counts: dict[str, list[float]] = {"sharp": [0, 0.0], "flat": [0, 0.0]}
    for glyph in accidentals:
        counts[glyph.kind][0] += 1
        counts[glyph.kind][1] += glyph.confidence

    if counts["sharp"][0] > counts["flat"][0]:
        dominant = "sharp"
    elif counts["flat"][0] > counts["sharp"][0]:
        dominant = "flat"
    else:
        dominant = "sharp" if counts["sharp"][1] >= counts["flat"][1] else "flat"

    return [glyph for glyph in accidentals if glyph.kind == dominant]


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
