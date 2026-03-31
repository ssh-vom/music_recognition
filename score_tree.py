"""
The data structure used to represent a score.
Lends itself to a tree, but flattened for ease of use/parsing.
We take the detection results and turn it into this tree of objects dictated by our schema:
A Score consists of:
    Score: [
        staffs,
        bars,
        measures,
        clefs,
        notes,
    ]
Each of these branch, which lends them to this structure, we use the staff_index to identify which staff they belong to,
after we have cropped the measures separately.
"""

from cv2.typing import MatLike

from schema import BarLine, Clef, ClefDetection, Measure, Note, Score, Staff


def build_score(
    *,
    image_path: str,
    sheet_image: MatLike,
    staffs: list[Staff],
    bars: list[BarLine],
    clefs_by_staff: dict[int, Clef],
    clef_detections: dict[int, ClefDetection],
    clef_key_crops: dict[int, MatLike],
    measures_map: dict[int, list[Measure]],
    measure_crops: dict[int, list[MatLike]],
    notes_mask: MatLike,
    bars_mask: MatLike,
) -> Score:
    bars_by_staff: dict[int, list[BarLine]] = {i: [] for i in range(len(staffs))}
    for bar in bars:
        bars_by_staff[bar.staff_index].append(bar)
    for staff_bars in bars_by_staff.values():
        staff_bars.sort(key=lambda bar: bar.x)

    all_measures: list[Measure] = []

    for staff_index, staff in enumerate(staffs):
        staff_measures = measures_map.get(staff_index, [])
        staff_crops = measure_crops.get(staff_index, [])
        staff_bars = bars_by_staff.get(staff_index, [])

        for measure_index, measure in enumerate(staff_measures):
            if measure_index < len(staff_crops):
                measure.crop = staff_crops[measure_index]
            all_measures.append(measure)

        closing_bars = _closing_bars_for_measures(staff_bars, staff_measures)
        staff_measure_list = [m for m in all_measures if m.staff_index == staff_index]
        for measure_index, measure in enumerate(staff_measure_list):
            if measure_index < len(closing_bars):
                measure.closing_bar = closing_bars[measure_index]

    all_notes: list[Note] = []
    for measure in all_measures:
        all_notes.extend(measure.notes)

    return Score(
        image_path=image_path,
        sheet_image=sheet_image,
        staffs=staffs,
        measures=all_measures,
        bars=bars,
        notes=all_notes,
        clefs=clefs_by_staff,
        clef_detections=clef_detections,
        notes_mask=notes_mask,
        bars_mask=bars_mask,
    )


def _closing_bars_for_measures(
    staff_bars: list[BarLine], staff_measures: list[Measure]
) -> list[BarLine]:
    if not staff_bars or not staff_measures:
        return []

    first_start = staff_measures[0].x_start
    last_end = staff_measures[-1].x_end
    usable = [bar for bar in staff_bars if first_start < bar.x < (last_end + 12)]

    closers: list[BarLine] = []
    for index, bar in enumerate(usable):
        if bar.kind == "double_left":
            if index + 1 < len(usable) and usable[index + 1].kind == "double_right":
                continue
        closers.append(bar)

    return closers[: max(0, len(staff_measures) - 1)]
