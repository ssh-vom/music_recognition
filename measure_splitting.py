"""Split staves into individual measures using bar line positions."""

import cv2 as cv
from cv2.typing import MatLike

from schema import Clef, KeySignature, Measure, TimeSignature
from staff_detection import erase_staff_for_notes

LEFT_HEADER_SPACINGS = 7.0
BAR_TRIM_RATIO = 0.25
MIN_WIDTH_PX = 4


def split_measures(
    bars: list,
    staffs: list,
    *,
    left_header_spacings: float = LEFT_HEADER_SPACINGS,
    first_staff_conservative_spacings: float | None = None,
) -> dict[int, list[Measure]]:
    barlines_by_staff = _group_barlines_by_staff(bars, len(staffs))
    measures_map = {}

    for staff_index, staff in enumerate(staffs):
        measures_map[staff_index] = _split_staff(
            staff=staff,
            staff_index=staff_index,
            staff_bars=barlines_by_staff[staff_index],
            left_header_spacings=left_header_spacings,
        )

    if first_staff_conservative_spacings is not None and 0 in measures_map:
        _apply_first_staff_start_policy(measures_map, staffs, first_staff_conservative_spacings)

    return measures_map


def _group_barlines_by_staff(bars: list, num_staffs: int) -> dict[int, list]:
    grouped = {i: [] for i in range(num_staffs)}
    for bar in bars:
        if 0 <= bar.staff_index < num_staffs:
            grouped[bar.staff_index].append(bar)
    for staff_bars in grouped.values():
        staff_bars.sort(key=lambda b: b.x)
    return grouped


def _staff_left(staff) -> int:
    return min(line.x_start for line in staff.lines)


def _staff_right(staff) -> int:
    return max(line.x_end for line in staff.lines) + 1


def _content_start_x(staff, left_header_spacings: float = LEFT_HEADER_SPACINGS) -> int:
    staff_left = _staff_left(staff)
    staff_right = _staff_right(staff)
    content_start = staff_left + int(round(left_header_spacings * staff.spacing))
    return max(staff_left, min(staff_right, content_start))


def _bar_trim_px(staff) -> int:
    return max(1, int(round(BAR_TRIM_RATIO * staff.spacing)))


def _usable_bars(staff_bars: list, content_start_x: int, staff_right: int) -> list:
    return [b for b in staff_bars if content_start_x < b.x < staff_right]


def _build_measure(x_start: int, x_end: int, staff, staff_index: int) -> Measure | None:
    if x_end - x_start < MIN_WIDTH_PX:
        return None
    return Measure(x_start=x_start, x_end=x_end, y_top=staff.top, y_bottom=staff.bottom, staff_index=staff_index)


def _split_staff(
    staff, staff_index: int, staff_bars: list, left_header_spacings: float = LEFT_HEADER_SPACINGS
) -> list[Measure]:
    if not staff.lines:
        return []

    staff_right = _staff_right(staff)
    content_start_x = _content_start_x(staff, left_header_spacings)

    if content_start_x >= staff_right:
        return []

    usable_bars = _usable_bars(staff_bars, content_start_x, staff_right)

    if not usable_bars:
        measure = _build_measure(content_start_x, staff_right, staff, staff_index)
        return [measure] if measure else []

    measures = []
    trim = _bar_trim_px(staff)
    current_start = content_start_x

    for index, bar in enumerate(usable_bars):
        if bar.kind == "double_left":
            if index + 1 < len(usable_bars) and usable_bars[index + 1].kind == "double_right":
                continue
        elif bar.kind == "double_right":
            pass
        elif bar.kind != "single":
            raise ValueError(f"Unknown bar kind: {bar.kind}")

        measure = _build_measure(current_start, bar.x - trim, staff, staff_index)
        if measure is not None:
            measures.append(measure)

        next_start = bar.x + trim
        if next_start > current_start:
            current_start = next_start

    final_measure = _build_measure(current_start, staff_right, staff, staff_index)
    if final_measure is not None:
        measures.append(final_measure)

    return measures


def _apply_first_staff_start_policy(
    measures_map: dict[int, list[Measure]], staffs: list, spacings: float
) -> None:
    if 0 not in measures_map or not measures_map[0]:
        return
    first_staff = staffs[0]
    conservative_start = _staff_left(first_staff) + int(round(first_staff.spacing * spacings))
    measures_map[0][0].x_start = max(measures_map[0][0].x_start, conservative_start)


def crop_measures(
    measures_map: dict[int, list[Measure]],
    image: MatLike,
    staffs: list,
    notes_image: MatLike | None = None,
) -> dict[int, list[MatLike]]:
    crops = {}

    for staff_index, measures in measures_map.items():
        staff_crops = []
        for measure in measures:
            src = notes_image if notes_image is not None else image
            crop = src[measure.y_top : measure.y_bottom + 1, measure.x_start : measure.x_end]

            if notes_image is not None:
                cleaned = crop
            else:
                gray = crop if len(crop.shape) == 2 else cv.cvtColor(crop, cv.COLOR_BGR2GRAY)
                cleaned = erase_staff_for_notes(gray, staffs=staffs)

            staff_crops.append(cleaned)
        crops[staff_index] = staff_crops

    return crops


def extract_clef_regions(staffs: list) -> dict[int, Clef]:
    clef_by_staff = {}
    for staff_index, staff in enumerate(staffs):
        assert staff.lines
        x_start = _staff_left(staff)
        x_end = _content_start_x(staff)
        assert x_end > x_start
        clef_by_staff[staff_index] = Clef(
            staff_index=staff_index,
            kind=None,
            x_start=x_start,
            x_end=x_end,
            y_top=staff.top,
            y_bottom=staff.bottom,
            key_signature=KeySignature(),
            time_signature=TimeSignature(),
        )
    return clef_by_staff


def crop_clef_regions(
    clefs: dict[int, Clef], image: MatLike, notes_image: MatLike | None = None
) -> dict[int, MatLike]:
    src = notes_image if notes_image is not None else image
    return {
        staff_index: src[clef.y_top : clef.y_bottom + 1, clef.x_start : clef.x_end]
        for staff_index, clef in clefs.items()
    }
