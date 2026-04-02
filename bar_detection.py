"""Bar line detection - find vertical lines separating measures."""

import cv2 as cv
import numpy as np
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
    REPEAT_DOT_BAR_GAP_FRAC,
    REPEAT_DOT_MAX_AREA_FRAC,
    REPEAT_DOT_MAX_SIZE_FRAC,
    REPEAT_DOT_MIN_AREA_FRAC,
    REPEAT_DOT_SEARCH_WIDTH_FRAC,
    REPEAT_DOT_X_ALIGNMENT_FRAC,
    REPEAT_DOT_Y_TOLERANCE_FRAC,
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

    # the first staff has clef + key + time signature on the left; later staves only repeat the clef
    skip_mult = (
        BAR_SEARCH_LEFT_SKIP_FRAC if staff_idx == 0 else BAR_SEARCH_LEFT_SKIP_OTHER_FRAC
    )
    left_skip = int(round(skip_mult * staff.spacing))
    work = roi[:, left_skip:]

    if work.size == 0:
        return []

    # vertical close kernel reconnects bar line segments that were broken by staff line erasure
    kernel_h = max(
        BAR_CLOSE_KERNEL_MIN, int(round(BAR_CLOSE_KERNEL_HEIGHT_FRAC * staff.spacing))
    )
    kernel = cv.getStructuringElement(cv.MORPH_RECT, (1, kernel_h))
    joined = cv.morphologyEx(work, cv.MORPH_CLOSE, kernel)

    contours, _ = cv.findContours(joined, cv.RETR_EXTERNAL, cv.CHAIN_APPROX_SIMPLE)

    bars = _contours_to_bars(contours, staff, left_skip, staff_h, y0, y1, staff_idx)
    bars = _merge_and_classify_pairs(bars, staff, staff_idx)
    _classify_repeat_markers(roi=roi, bars=bars, staff=staff, y0=y0)
    return bars


def _contours_to_bars(
    contours,
    staff: Staff,
    left_skip: int,
    staff_h: int,
    y0: int,
    y1: int,
    staff_idx: int,
) -> list[BarLine]:
    staff_right = max(line.x_end for line in staff.lines)

    max_width = int(round(BAR_MAX_WIDTH_FRAC * staff.spacing))
    min_double_width = int(round(BAR_DOUBLE_MIN_WIDTH_FRAC * staff.spacing))
    right_margin = int(round(BAR_RIGHT_MARGIN_FRAC * staff.spacing))
    left_margin = int(round(BAR_LEFT_MARGIN_FRAC * staff.spacing))
    left_relaxed_max = max_width + int(
        round(BAR_LEFT_RELAXED_EXTRA_FRAC * staff.spacing)
    )
    min_height = int(round(BAR_MIN_HEIGHT_FRAC * staff_h))

    bars = []
    for contour in contours:
        x, _, w, h = cv.boundingRect(contour)

        if h < min_height:
            continue

        area = w * h
        if area == 0:
            continue
        # density filters out sparse blobs — real bar lines fill almost their entire bounding box
        density = cv.contourArea(contour) / float(area)

        abs_left = left_skip + x
        abs_center = abs_left + w // 2
        abs_right = abs_left + w - 1

        near_right = abs_center >= staff_right - right_margin
        near_left = abs_center <= left_skip + left_margin
        # blobs near the left edge can be thicker due to clef artifacts, so we relax the width limit there
        left_relaxed = near_left and w <= left_relaxed_max and density >= 0.50
        if not left_relaxed and (
            density < BAR_MIN_DENSITY or (w > max_width and density < 0.75)
        ):
            continue

        # a wide blob near the right edge is a double bar; split it into left and right strokes
        kind_xs = (
            [("double_left", abs_left), ("double_right", abs_right)]
            if w >= min_double_width and near_right
            else [("single", abs_center)]
        )

        for kind, x_pos in kind_xs:
            bars.append(
                BarLine(
                    x=x_pos,
                    y_top=y0,
                    y_bottom=y1 - 1,
                    kind=kind,
                    repeat="none",
                    staff_index=staff_idx,
                )
            )

    return bars


def _merge_and_classify_pairs(
    bars: list[BarLine], staff: Staff, staff_idx: int
) -> list[BarLine]:
    if len(bars) < 2:
        return bars

    bars = sorted(bars, key=lambda b: b.x)
    merge_dist = max(3, int(round(BAR_MERGE_DISTANCE_FRAC * staff.spacing)))

    # average together any two single bars that landed very close to each other
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
    left_skip = int(
        round(
            BAR_SEARCH_LEFT_SKIP_FRAC * staff.spacing
            if staff_idx == 0
            else BAR_SEARCH_LEFT_SKIP_OTHER_FRAC * staff.spacing
        )
    )

    edge_margin = int(round(BAR_LEFT_MARGIN_FRAC * staff.spacing))
    left_edge = left_skip + edge_margin
    right_edge = staff_right - edge_margin
    pair_gap = max(2, int(round(BAR_PAIR_GAP_FRAC * staff.spacing)))

    result = []
    i = 0
    while i < len(merged):
        if i + 1 < len(merged):
            left, right = merged[i], merged[i + 1]
            # two singles sitting close together near a staff edge are the two strokes of a double bar
            if (
                left.kind == "single"
                and right.kind == "single"
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


def _classify_repeat_markers(
    *, roi: MatLike, bars: list[BarLine], staff: Staff, y0: int
) -> None:
    if len(bars) < 2:
        return

    pair_gap = max(2, int(round(BAR_PAIR_GAP_FRAC * staff.spacing)))
    i = 0
    while i + 1 < len(bars):
        left, right = bars[i], bars[i + 1]
        if (
            left.kind == "double_left"
            and right.kind == "double_right"
            and right.x - left.x <= pair_gap
        ):
            # dots to the right of the bar = begin repeat; dots to the left = end repeat
            has_left = _has_repeat_dots_on_side(
                roi=roi,
                staff=staff,
                y0=y0,
                bar_x=left.x,
                side="left",
            )
            has_right = _has_repeat_dots_on_side(
                roi=roi,
                staff=staff,
                y0=y0,
                bar_x=right.x,
                side="right",
            )
            repeat = (
                "begin"
                if has_right and not has_left
                else "end"
                if has_left and not has_right
                else None
            )
            if repeat:
                left.repeat = right.repeat = repeat
            i += 2
        else:
            i += 1


def _has_repeat_dots_on_side(
    *,
    roi: MatLike,
    staff: Staff,
    y0: int,
    bar_x: int,
    side: str,
) -> bool:
    h, w = roi.shape[:2]
    if h == 0 or w == 0:
        return False

    search_w = max(4, int(round(staff.spacing * REPEAT_DOT_SEARCH_WIDTH_FRAC)))
    gap = max(1, int(round(staff.spacing * REPEAT_DOT_BAR_GAP_FRAC)))

    if side == "left":
        x0 = max(0, bar_x - gap - search_w)
        x1 = max(x0 + 1, min(w, bar_x - gap))
    else:
        x0 = max(0, bar_x + gap)
        x1 = max(x0 + 1, min(w, bar_x + gap + search_w))

    if x1 - x0 < 2:
        return False

    window = roi[:, x0:x1]
    binary = ((window > 0).astype(np.uint8)) * 255
    count, _, stats, centroids = cv.connectedComponentsWithStats(binary, connectivity=8)

    min_area = max(
        1, int(round(staff.spacing * staff.spacing * REPEAT_DOT_MIN_AREA_FRAC))
    )
    max_area = max(
        2, int(round(staff.spacing * staff.spacing * REPEAT_DOT_MAX_AREA_FRAC))
    )
    max_size = max(2, int(round(staff.spacing * REPEAT_DOT_MAX_SIZE_FRAC)))

    line_mid = int(round(staff.lines[2].y - y0))
    expected_top = line_mid - int(round(staff.spacing * 0.5))
    expected_bottom = line_mid + int(round(staff.spacing * 0.5))
    y_tol = max(2, int(round(staff.spacing * REPEAT_DOT_Y_TOLERANCE_FRAC)))
    x_align_tol = max(2, int(round(staff.spacing * REPEAT_DOT_X_ALIGNMENT_FRAC)))

    candidates: list[tuple[int, int]] = []
    for i in range(1, count):
        area = int(stats[i, cv.CC_STAT_AREA])
        comp_w = int(stats[i, cv.CC_STAT_WIDTH])
        comp_h = int(stats[i, cv.CC_STAT_HEIGHT])
        if area < min_area or area > max_area:
            continue
        if comp_w > max_size or comp_h > max_size:
            continue
        cx = int(round(float(centroids[i][0])))
        cy = int(round(float(centroids[i][1])))
        candidates.append((cx, cy))

    if len(candidates) < 2:
        return False

    # repeat dots sit in the two spaces around the middle line of the staff — one above, one below
    top = [p for p in candidates if abs(p[1] - expected_top) <= y_tol]
    bottom = [p for p in candidates if abs(p[1] - expected_bottom) <= y_tol]
    if not top or not bottom:
        return False

    # the two dots must also be vertically aligned with each other
    for tx, _ in top:
        for bx, _ in bottom:
            if abs(tx - bx) <= x_align_tol:
                return True
    return False
