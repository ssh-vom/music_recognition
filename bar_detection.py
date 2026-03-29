"""Bar line detection - pure functions, no classes, minimal state."""

import cv2 as cv
from cv2.typing import MatLike

from schema import BarLine, RepeatKind, Staff


def find_bars(image: MatLike, staffs: list[Staff]) -> list[BarLine]:
    """Find all bar lines in the sheet music.

    Pure function - takes image and staff list, returns detected bars.
    No side effects, no internal state to manage.
    """
    all_bars: list[BarLine] = []

    for staff_idx, staff in enumerate(staffs):
        staff_bars = _find_staff_bars(image, staff, staff_idx)
        all_bars.extend(staff_bars)

    return sorted(all_bars, key=lambda b: (b.staff_index, b.x))


def _find_staff_bars(image: MatLike, staff: Staff, staff_idx: int) -> list[BarLine]:
    """Find bars in a single staff region."""
    y0, y1 = staff.top, staff.bottom + 1
    staff_height = y1 - y0
    roi = image[y0:y1, :]

    # Skip left header area (clef/key signature)
    left_skip = int(round(5.0 * staff.spacing))
    work = roi[:, left_skip:]
    if work.size == 0:
        return []

    # Close vertical gaps in bar lines (kernel height = 2x spacing)
    kernel_h = max(5, int(round(2.0 * staff.spacing)))
    close_kernel = cv.getStructuringElement(cv.MORPH_RECT, (1, kernel_h))
    joined = cv.morphologyEx(work, cv.MORPH_CLOSE, close_kernel)

    # Find vertical contours
    contours, _ = cv.findContours(joined, cv.RETR_EXTERNAL, cv.CHAIN_APPROX_SIMPLE)

    bars = _contours_to_bars(
        contours, staff, left_skip, staff_height, y0, y1, staff_idx
    )
    return _merge_and_classify_pairs(bars, staff, staff_idx)


def _contours_to_bars(
    contours,
    staff: Staff,
    left_skip: int,
    staff_height: int,
    y0: int,
    y1: int,
    staff_idx: int,
) -> list[BarLine]:
    """Convert contours to bar line candidates."""
    staff_right = max(line.x_end for line in staff.lines)
    max_width = int(round(0.6 * staff.spacing))
    min_double_width = int(round(1.0 * staff.spacing))
    right_margin = int(round(2.0 * staff.spacing))
    left_margin = int(round(4.0 * staff.spacing))

    bars = []
    for contour in contours:
        x, _, width, height = cv.boundingRect(contour)

        # Filter by height (must be at least 40% of staff height)
        if height < int(round(0.4 * staff_height)):
            continue

        # Filter by density (must be at least 55% solid)
        area = width * height
        if area == 0:
            continue
        density = cv.contourArea(contour) / float(area)

        abs_left = left_skip + x
        abs_center = abs_left + width // 2
        abs_right = abs_left + width - 1

        near_right_edge = abs_center >= staff_right - right_margin
        near_left_edge = abs_center <= left_skip + left_margin
        left_relaxed_max = max_width + int(round(0.7 * staff.spacing))
        is_left_relaxed = (
            near_left_edge and width <= left_relaxed_max and density >= 0.50
        )

        # Density filter
        if density < 0.55 and not is_left_relaxed:
            continue

        # Thick bar filter
        if width > max_width and density < 0.75 and not is_left_relaxed:
            continue

        # Determine bar type
        if width >= min_double_width and near_right_edge:
            # Double bar at staff end
            bars.append(
                BarLine(
                    x=abs_left,
                    y_top=y0,
                    y_bottom=y1 - 1,
                    kind="double_left",
                    repeat="none",
                    staff_index=staff_idx,
                )
            )
            bars.append(
                BarLine(
                    x=abs_right,
                    y_top=y0,
                    y_bottom=y1 - 1,
                    kind="double_right",
                    repeat="none",
                    staff_index=staff_idx,
                )
            )
        else:
            bars.append(
                BarLine(
                    x=abs_center,
                    y_top=y0,
                    y_bottom=y1 - 1,
                    kind="single",
                    repeat="none",
                    staff_index=staff_idx,
                )
            )

    return bars


def _merge_and_classify_pairs(
    bars: list[BarLine], staff: Staff, staff_idx: int
) -> list[BarLine]:
    """Merge nearby bars and classify close pairs as double bars."""
    if len(bars) < 2:
        return bars

    bars.sort(key=lambda b: b.x)

    # Merge nearby singles
    merge_dist = max(3, int(round(0.5 * staff.spacing)))
    merged: list[BarLine] = [bars[0]]

    for bar in bars[1:]:
        prev = merged[-1]
        if bar.x - prev.x <= merge_dist:
            if bar.kind == "single" and prev.kind == "single":
                prev.x = (prev.x + bar.x) // 2
            else:
                merged.append(bar)
        else:
            merged.append(bar)

    # Convert edge pairs to double bars
    staff_right = max(line.x_end for line in staff.lines)
    # First staff: use larger left skip for key/time signature
    # Subsequent staffs: only skip clef area since time signature already decided
    if staff_idx == 0:
        left_skip = int(round(5.0 * staff.spacing))
    else:
        left_skip = int(round(2.0 * staff.spacing))
    edge_margin = int(round(4.0 * staff.spacing))
    left_edge = left_skip + edge_margin
    right_edge = staff_right - edge_margin
    pair_gap = max(2, int(round(1.5 * staff.spacing)))

    result: list[BarLine] = []
    i = 0
    while i < len(merged):
        if i + 1 < len(merged):
            left, right = merged[i], merged[i + 1]
            is_close = right.x - left.x <= pair_gap
            on_edge = right.x <= left_edge or left.x >= right_edge

            if (
                left.kind == "single"
                and right.kind == "single"
                and is_close
                and on_edge
            ):
                left.kind = "double_left"
                right.kind = "double_right"
                result.extend([left, right])
                i += 2
                continue

        result.append(merged[i])
        i += 1

    # Detect repeats on double bar pairs
    _mark_repeats(result, staff)

    return result


def _mark_repeats(bars: list[BarLine], staff: Staff) -> None:
    """Mark repeat signs on double bar pairs."""
    y0 = staff.top
    spacing = staff.spacing

    # Repeat dot positions (between lines 2-3 and 3-4)
    dot_y_top = int(round((staff.lines[1].y + staff.lines[2].y) / 2.0)) - y0
    dot_y_bottom = int(round((staff.lines[2].y + staff.lines[3].y) / 2.0)) - y0
    dot_window = int(round(1.0 * spacing))
    dot_tol = max(1, int(round(0.45 * spacing)))
    dot_max_size = max(1, int(round(1.0 * spacing)))

    for i in range(len(bars) - 1):
        left, right = bars[i], bars[i + 1]
        if left.kind != "double_left" or right.kind != "double_right":
            continue

        # Check for dots
        repeat = _classify_repeat_dots(
            left.x,
            right.x,
            dot_y_top,
            dot_y_bottom,
            dot_window,
            dot_tol,
            dot_max_size,
            spacing,
            staff,
        )
        left.repeat = repeat
        right.repeat = repeat


def _classify_repeat_dots(
    left_x: int,
    right_x: int,
    dot_y_top: int,
    dot_y_bottom: int,
    dot_window: int,
    dot_tol: int,
    dot_max_size: int,
    spacing: float,
    staff: Staff,
) -> RepeatKind:
    """Check for repeat dots near a double bar."""
    # NOTE: This is a placeholder - full implementation would check actual image
    # For now, use position-based heuristics

    staff_right = max(line.x_end for line in staff.lines)
    pair_center = (left_x + right_x) // 2
    left_skip = int(round(5.0 * spacing))
    edge_margin = int(round(4.0 * spacing))
    left_edge = left_skip + edge_margin
    right_edge = staff_right - edge_margin

    near_left = pair_center <= left_edge
    near_right = pair_center >= right_edge

    # Position-based classification (would be image-based in full impl)
    if near_left:
        return "begin"
    if near_right:
        return "end"

    return "none"
