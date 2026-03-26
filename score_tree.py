from dataclasses import dataclass, field

from cv2.typing import MatLike

from schema import BarLine, Clef, ClefDetection, Measure, Note, Staff


@dataclass
class MeasureNode:
    index: int
    measure: Measure
    crop: MatLike | None = None
    notes: list[Note] = field(default_factory=list)
    # Bar that closes this measure boundary to the next measure (if any).
    closing_bar: BarLine | None = None


@dataclass
class StaffNode:
    index: int
    staff: Staff
    clef: Clef | None = None
    clef_detection: ClefDetection | None = None
    bars: list[BarLine] = field(default_factory=list)
    measures: list[MeasureNode] = field(default_factory=list)
    clef_key_crop: MatLike | None = None
    # Bar nearest the right edge of the staff (used for final repeat/final bar).
    end_bar: BarLine | None = None


@dataclass
class ScoreTree:
    image_path: str
    sheet_image: MatLike
    staff_nodes: list[StaffNode]
    notes_mask: MatLike | None = None
    bars_mask: MatLike | None = None

    def staff_node_map(self) -> dict[int, StaffNode]:
        return {node.index: node for node in self.staff_nodes}


def build_score_tree(
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
) -> ScoreTree:
    bars_by_staff: dict[int, list[BarLine]] = {i: [] for i in range(len(staffs))}
    for bar in bars:
        if 0 <= bar.staff_index < len(staffs):
            bars_by_staff[bar.staff_index].append(bar)
    for staff_bars in bars_by_staff.values():
        staff_bars.sort(key=lambda bar: bar.x)

    staff_nodes: list[StaffNode] = []
    for staff_index, staff in enumerate(staffs):
        staff_measures = measures_map.get(staff_index, [])
        staff_crops = measure_crops.get(staff_index, [])
        measure_nodes: list[MeasureNode] = []
        for measure_index, measure in enumerate(staff_measures):
            crop = staff_crops[measure_index] if measure_index < len(staff_crops) else None
            measure_nodes.append(
                MeasureNode(
                    index=measure_index,
                    measure=measure,
                    crop=crop,
                )
            )

        staff_bars = bars_by_staff.get(staff_index, [])
        closing_bars = _closing_bars_for_measures(staff_bars, staff_measures)
        for measure_index, measure_node in enumerate(measure_nodes):
            if measure_index < len(closing_bars):
                measure_node.closing_bar = closing_bars[measure_index]

        end_bar = _staff_end_bar(
            staff_bars=staff_bars,
            last_measure=staff_measures[-1] if staff_measures else None,
        )

        staff_nodes.append(
            StaffNode(
                index=staff_index,
                staff=staff,
                clef=clefs_by_staff.get(staff_index),
                clef_detection=clef_detections.get(staff_index),
                bars=staff_bars,
                measures=measure_nodes,
                clef_key_crop=clef_key_crops.get(staff_index),
                end_bar=end_bar,
            )
        )

    return ScoreTree(
        image_path=image_path,
        sheet_image=sheet_image,
        staff_nodes=staff_nodes,
        notes_mask=notes_mask,
        bars_mask=bars_mask,
    )


def _closing_bars_for_measures(
    staff_bars: list[BarLine],
    staff_measures: list[Measure],
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

    needed = max(0, len(staff_measures) - 1)
    return closers[:needed]


def _staff_end_bar(
    staff_bars: list[BarLine],
    last_measure: Measure | None,
) -> BarLine | None:
    if last_measure is None:
        return None

    tol = 10
    right_candidates = [
        bar for bar in staff_bars if bar.x >= (last_measure.x_end - tol)
    ]
    if not right_candidates:
        return None
    return max(right_candidates, key=lambda bar: bar.x)
