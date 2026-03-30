"""Bar line detection - find vertical lines separating measures."""

import cv2 as cv
from cv2.typing import MatLike

from constants import (
    BAR_CLOSE_KERNEL_HEIGHT_FRAC,
    BAR_CLOSE_KERNEL_MIN,
    BAR_DOUBLE_MIN_WIDTH_FRAC,
    BAR_LEFT_MARGIN_FRAC,
    BAR_LEFT_RELAXED_EXTRA_FRAC,
    BAR_MAX_WIDTH_FRAC,
    BAR_MERGE_DISTANCE_FRAC,
    BAR_MIN_DENSITY,
    BAR_MIN_HEIGHT_FRAC,
    BAR_PAIR_GAP_FRAC,
    BAR_RIGHT_MARGIN_FRAC,
    BAR_SEARCH_LEFT_SKIP_FRAC,
    BAR_SEARCH_LEFT_SKIP_OTHER_FRAC,
)
from schema import BarLine, Staff


def find_bars(image: MatLike, staffs: list[Staff]) -> list[BarLine]:
    all_bars = []
    for staff_idx, staff in enumerate(staffs):
        all_bars.extend(_find_staff_bars(image, staff, staff_idx))
    return sorted(all_bars, key=lambda b: (b.staff_index, b.x))


def _find_staff_bars(image: MatLike, staff: Staff, staff_idx: int) -> list[BarLine]:
    y0, y1 = staff.top, staff.bottom + 1
    staff_h = y1 - y0
    roi = image[y0:y1, :]

    skip_mult = BAR_SEARCH_LEFT_SKIP_FRAC if staff_idx == 0 else BAR_SEARCH_LEFT_SKIP_OTHER_FRAC
    left_skip = int(round(skip_mult * staff.spacing))
    work = roi[:, left_skip:]

    if work.size == 0:
        return []

    kernel_h = max(BAR_CLOSE_KERNEL_MIN, int(round(BAR_CLOSE_KERNEL_HEIGHT_FRAC * staff.spacing)))
    kernel = cv.getStructuringElement(cv.MORPH_RECT, (1, kernel_h))
    joined = cv.morphologyEx(work, cv.MORPH_CLOSE, kernel)

    contours, _ = cv.findContours(joined, cv.RETR_EXTERNAL, cv.CHAIN_APPROX_SIMPLE)

    bars = _contours_to_bars(contours, staff, left_skip, staff_h, y0, y1, staff_idx)
    return _merge_and_classify_pairs(bars, staff, staff_idx)


def _contours_to_bars(
    contours, staff: Staff, left_skip: int, staff_h: int, y0: int, y1: int, staff_idx: int
) -> list[BarLine]:
    staff_right = max(line.x_end for line in staff.lines)

    max_width = int(round(BAR_MAX_WIDTH_FRAC * staff.spacing))
    min_double_width = int(round(BAR_DOUBLE_MIN_WIDTH_FRAC * staff.spacing))
    right_margin = int(round(BAR_RIGHT_MARGIN_FRAC * staff.spacing))
    left_margin = int(round(BAR_LEFT_MARGIN_FRAC * staff.spacing))
    left_relaxed_max = max_width + int(round(BAR_LEFT_RELAXED_EXTRA_FRAC * staff.spacing))
    min_height = int(round(BAR_MIN_HEIGHT_FRAC * staff_h))

    bars = []
    for contour in contours:
        x, _, w, h = cv.boundingRect(contour)

        if h < min_height:
            continue

        area = w * h
        if area == 0:
            continue
        density = cv.contourArea(contour) / float(area)

        abs_left = left_skip + x
        abs_center = abs_left + w // 2
        abs_right = abs_left + w - 1

        near_right = abs_center >= staff_right - right_margin
        near_left = abs_center <= left_skip + left_margin
        left_relaxed = near_left and w <= left_relaxed_max and density >= 0.50

        if density < BAR_MIN_DENSITY and not left_relaxed:
            continue
        if w > max_width and density < 0.75 and not left_relaxed:
            continue

        if w >= min_double_width and near_right:
            bars.append(BarLine(x=abs_left, y_top=y0, y_bottom=y1-1, kind="double_left", repeat="none", staff_index=staff_idx))
            bars.append(BarLine(x=abs_right, y_top=y0, y_bottom=y1-1, kind="double_right", repeat="none", staff_index=staff_idx))
        else:
            bars.append(BarLine(x=abs_center, y_top=y0, y_bottom=y1-1, kind="single", repeat="none", staff_index=staff_idx))

    return bars


def _merge_and_classify_pairs(bars: list[BarLine], staff: Staff, staff_idx: int) -> list[BarLine]:
    if len(bars) < 2:
        return bars

    bars = sorted(bars, key=lambda b: b.x)
    merge_dist = max(3, int(round(BAR_MERGE_DISTANCE_FRAC * staff.spacing)))

    merged = [bars[0]]
    for bar in bars[1:]:
        prev = merged[-1]
        if bar.x - prev.x <= merge_dist:
            if bar.kind == "single" and prev.kind == "single":
                prev.x = (prev.x + bar.x) // 2
            else:
                merged.append(bar)
        else:
            merged.append(bar)

    staff_right = max(line.x_end for line in staff.lines)
    left_skip = int(round(
        BAR_SEARCH_LEFT_SKIP_FRAC * staff.spacing if staff_idx == 0
        else BAR_SEARCH_LEFT_SKIP_OTHER_FRAC * staff.spacing
    ))

    edge_margin = int(round(BAR_LEFT_MARGIN_FRAC * staff.spacing))
    left_edge = left_skip + edge_margin
    right_edge = staff_right - edge_margin
    pair_gap = max(2, int(round(BAR_PAIR_GAP_FRAC * staff.spacing)))

    result = []
    i = 0
    while i < len(merged):
        if i + 1 < len(merged):
            left, right = merged[i], merged[i + 1]
            if (
                left.kind == "single" and right.kind == "single"
                and right.x - left.x <= pair_gap
                and (right.x <= left_edge or left.x >= right_edge)
            ):
                left.kind = "double_left"
                right.kind = "double_right"
                result.extend([left, right])
                i += 2
                continue

        result.append(merged[i])
        i += 1

    return result
