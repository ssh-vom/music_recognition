"""Measure splitting - split staves into measures using barline positions.

Pure functions for dividing sheet music into individual measures and cropping regions.
"""

import cv2 as cv
from cv2.typing import MatLike

from schema import Clef, KeySignature, Measure, TimeSignature
from staff_detection import erase_staff_for_notes


# Configuration constants (as ratios of staff spacing for scale invariance)
LEFT_HEADER_SPACINGS = 7.0  # Width of clef+key signature header region
BAR_TRIM_RATIO = 0.25  # Trim this much on each side of bar line to get measure boundary
MIN_WIDTH_PX = 4  # Minimum width for a valid measure
FIRST_STAFF_CONSERVATIVE_SPACINGS = None  # Optional: extra margin for first staff


def split_measures(
    bars: list, staffs: list, *, first_staff_conservative_spacings: float | None = None
) -> dict[int, list[Measure]]:
    """Split all staves into measures using detected bar lines.

    Returns dict mapping staff_index -> list of Measure objects.
    """
    barlines_by_staff = _group_barlines_by_staff(bars, len(staffs))
    measures_map: dict[int, list[Measure]] = {}

    for staff_index, staff in enumerate(staffs):
        staff_bars = barlines_by_staff[staff_index]
        measures_map[staff_index] = _split_staff(
            staff=staff,
            staff_index=staff_index,
            staff_bars=staff_bars,
        )

    # Apply first staff start policy if configured
    if first_staff_conservative_spacings is not None and 0 in measures_map:
        _apply_first_staff_start_policy(
            measures_map, staffs, first_staff_conservative_spacings
        )

    return measures_map


def _group_barlines_by_staff(bars: list, num_staffs: int) -> dict[int, list]:
    """Group bar lines by their staff index."""
    grouped: dict[int, list] = {i: [] for i in range(num_staffs)}

    for bar in bars:
        if 0 <= bar.staff_index < num_staffs:
            grouped[bar.staff_index].append(bar)

    for staff_bars in grouped.values():
        staff_bars.sort(key=lambda b: b.x)

    return grouped


def _staff_left(staff) -> int:
    """Get leftmost x-coordinate of staff."""
    return min(line.x_start for line in staff.lines)


def _staff_right(staff) -> int:
    """Get rightmost x-coordinate of staff (exclusive)."""
    return max(line.x_end for line in staff.lines) + 1


def _content_start_x(staff) -> int:
    """Calculate where actual note content starts (after clef/key header).

    Left header width = 7.0 * staff.spacing (clef + key signature)
    """
    staff_left = _staff_left(staff)
    staff_right = _staff_right(staff)
    header_width = int(round(LEFT_HEADER_SPACINGS * staff.spacing))
    content_start = staff_left + header_width

    return max(staff_left, min(staff_right, content_start))


def _bar_trim_px(staff) -> int:
    """Calculate trim amount on each side of bar line."""
    trim = int(round(BAR_TRIM_RATIO * staff.spacing))
    return max(1, trim)


def _usable_bars(staff_bars: list, content_start_x: int, staff_right: int) -> list:
    """Filter bar lines to those within the usable content region."""
    usable = []

    for bar in staff_bars:
        if bar.x <= content_start_x:
            continue
        if bar.x >= staff_right:
            continue
        usable.append(bar)

    return usable


def _build_measure(x_start: int, x_end: int, staff, staff_index: int) -> Measure | None:
    """Create a Measure object if dimensions are valid."""
    if x_end - x_start < MIN_WIDTH_PX:
        return None

    return Measure(
        x_start=x_start,
        x_end=x_end,
        y_top=staff.top,
        y_bottom=staff.bottom,
        staff_index=staff_index,
    )


def _split_staff(staff, staff_index: int, staff_bars: list) -> list[Measure]:
    """Split a single staff into measures using its bar lines."""
    if not staff.lines:
        return []

    staff_right = _staff_right(staff)
    content_start_x = _content_start_x(staff)

    if content_start_x >= staff_right:
        return []

    usable_bars = _usable_bars(staff_bars, content_start_x, staff_right)

    # No bars found - create single measure spanning entire staff
    if not usable_bars:
        measure = _build_measure(content_start_x, staff_right, staff, staff_index)
        return [measure] if measure else []

    measures: list[Measure] = []
    trim = _bar_trim_px(staff)
    current_start = content_start_x

    for index, bar in enumerate(usable_bars):
        # Handle double bars (pairs of left/right bar lines)
        if bar.kind == "double_left":
            if (
                index + 1 < len(usable_bars)
                and usable_bars[index + 1].kind == "double_right"
            ):
                continue  # Skip the left half of double bar pair
        elif bar.kind == "double_right":
            pass  # Process the right half
        elif bar.kind != "single":
            raise ValueError(f"Unknown bar kind: {bar.kind}")

        # Measure ends just before the bar line
        current_end = bar.x - trim
        measure = _build_measure(current_start, current_end, staff, staff_index)

        if measure is not None:
            measures.append(measure)

        # Next measure starts just after the bar line
        next_start = bar.x + trim
        if next_start > current_start:
            current_start = next_start

    # Final measure from last bar to staff end
    final_measure = _build_measure(current_start, staff_right, staff, staff_index)
    if final_measure is not None:
        measures.append(final_measure)

    return measures


def _apply_first_staff_start_policy(
    measures_map: dict[int, list[Measure]], staffs: list, spacings: float
) -> None:
    """Apply conservative start policy to first staff.

    Ensures first measure doesn't start too early (avoids clef/key area).
    """
    if 0 not in measures_map or not measures_map[0]:
        return

    first_staff = staffs[0]
    first_staff_left = _staff_left(first_staff)
    conservative_start = first_staff_left + int(round(first_staff.spacing * spacings))
    first_measure = measures_map[0][0]
    first_measure.x_start = max(first_measure.x_start, conservative_start)


def crop_measures(
    measures_map: dict[int, list[Measure]],
    image: MatLike,
    staffs: list,
    notes_image: MatLike | None = None,
) -> dict[int, list[MatLike]]:
    """Crop measure regions from the sheet image.

    If notes_image provided, uses that for cropping (staff-erased).
    Otherwise erases staff lines from the crop.
    """
    crops: dict[int, list[MatLike]] = {}

    for staff_index, measures in measures_map.items():
        staff_crops: list[MatLike] = []

        for measure in measures:
            # Prefer staff-erased notes image if available
            src = notes_image if notes_image is not None else image
            crop = src[
                measure.y_top : measure.y_bottom + 1,
                measure.x_start : measure.x_end,
            ]

            if notes_image is not None:
                # Already staff-erased
                cleaned = crop
            else:
                # Need to erase staff lines from this crop
                gray = (
                    crop
                    if len(crop.shape) == 2
                    else cv.cvtColor(crop, cv.COLOR_BGR2GRAY)
                )
                cleaned = erase_staff_for_notes(gray, staffs=staffs)

            staff_crops.append(cleaned)

        crops[staff_index] = staff_crops

    return crops


def extract_clef_regions(staffs: list) -> dict[int, Clef]:
    """Extract clef+key signature header regions for each staff.

    Returns dict mapping staff_index -> Clef object with bounding box.
    Kind/key/time filled later by clef detection.
    """
    clef_by_staff: dict[int, Clef] = {}

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
    """Crop clef+key header regions from the image.

    Prefers staff-erased notes_image if provided for cleaner detection.
    """
    src = notes_image if notes_image is not None else image
    crops: dict[int, MatLike] = {}

    for staff_index, clef in clefs.items():
        crop = src[clef.y_top : clef.y_bottom + 1, clef.x_start : clef.x_end]
        crops[staff_index] = crop

    return crops
