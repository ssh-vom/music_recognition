"""Split staves into measures using barline positions; crop regions from the page."""

from dataclasses import dataclass

import cv2 as cv
from cv2.typing import MatLike

from schema import BarLine, Measure, Staff, ClefAndKeySignature
from staff_detection import erase_staff_for_notes


@dataclass
class MeasureDetectionConfig:
    left_header_spacings: float = 7.0
    bar_trim_ratio: float = 0.25
    min_width_px: float = 4


class MeasureSplitter:
    def __init__(
        self,
        bars: list[BarLine],
        staffs: list[Staff],
        sheet_img: MatLike,
        config: MeasureDetectionConfig | None = None,
        notes_image: MatLike | None = None,
    ):
        self.bars = bars
        self.staffs = staffs
        self.image = sheet_img
        self.config = config or MeasureDetectionConfig()
        # Full page after erase_staff_for_notes; measure crops are slices of this.
        self.notes_image = notes_image

    def _group_barlines_by_staff(self) -> dict[int, list[BarLine]]:
        grouped: dict[int, list[BarLine]] = {i: [] for i in range(len(self.staffs))}

        for bar in self.bars:
            if 0 <= bar.staff_index < len(self.staffs):
                grouped[bar.staff_index].append(bar)

        for staff_bars in grouped.values():
            staff_bars.sort(key=lambda bar: bar.x)

        return grouped

    def _staff_left(self, staff: Staff) -> int:
        return min(line.x_start for line in staff.lines)

    def _staff_right(self, staff: Staff) -> int:
        return max(line.x_end for line in staff.lines) + 1

    def _content_start_x(self, staff: Staff) -> int:
        staff_left = self._staff_left(staff)
        staff_right = self._staff_right(staff)
        header_width = int(round(self.config.left_header_spacings * staff.spacing))
        content_start = staff_left + header_width

        if content_start < staff_left:
            return staff_left

        if content_start > staff_right:
            return staff_right

        return content_start

    def _bar_trim_px(self, staff: Staff) -> int:
        trim = int(round(self.config.bar_trim_ratio * staff.spacing))

        if trim < 1:
            return 1

        return trim

    def _usable_bars(
        self,
        staff_bars: list[BarLine],
        content_start_x: int,
        staff_right: int,
    ) -> list[BarLine]:
        usable_bars: list[BarLine] = []

        for bar in staff_bars:
            if bar.x <= content_start_x:
                continue

            if bar.x >= staff_right:
                continue

            usable_bars.append(bar)

        return usable_bars

    def _build_measure(
        self,
        x_start: int,
        x_end: int,
        staff: Staff,
        staff_index: int,
    ) -> Measure | None:
        if x_end - x_start < self.config.min_width_px:
            return None

        return Measure(
            x_start=x_start,
            x_end=x_end,
            y_top=staff.top,
            y_bottom=staff.bottom,
            staff_index=staff_index,
        )

    def _split_staff(
        self,
        staff: Staff,
        staff_index: int,
        staff_bars: list[BarLine],
    ) -> list[Measure]:
        if not staff.lines:
            return []

        staff_right = self._staff_right(staff)
        content_start_x = self._content_start_x(staff)

        if content_start_x >= staff_right:
            return []

        usable_bars = self._usable_bars(staff_bars, content_start_x, staff_right)

        if not usable_bars:
            measure = self._build_measure(
                content_start_x, staff_right, staff, staff_index
            )
            if measure is None:
                return []
            return [measure]

        measures: list[Measure] = []
        trim = self._bar_trim_px(staff)
        current_start = content_start_x

        for index, bar in enumerate(usable_bars):
            match bar.kind:
                case "single":
                    pass
                case "double_left":
                    if (
                        index + 1 < len(usable_bars)
                        and usable_bars[index + 1].kind == "double_right"
                    ):
                        continue
                case "double_right":
                    pass
                case _:
                    assert False, f"Unknown bar kind: {bar.kind}"

            current_end = bar.x - trim
            measure = self._build_measure(
                current_start, current_end, staff, staff_index
            )

            if measure is not None:
                measures.append(measure)

            next_start = bar.x + trim
            if next_start > current_start:
                current_start = next_start

        final_measure = self._build_measure(
            current_start, staff_right, staff, staff_index
        )
        if final_measure is not None:
            measures.append(final_measure)

        return measures

    def split_measures(self) -> dict[int, list[Measure]]:
        barlines_by_staff = self._group_barlines_by_staff()
        measures_map: dict[int, list[Measure]] = {}

        for staff_index, staff in enumerate(self.staffs):
            staff_bars = barlines_by_staff[staff_index]
            measures_map[staff_index] = self._split_staff(
                staff=staff,
                staff_index=staff_index,
                staff_bars=staff_bars,
            )

        return measures_map

    def crop_measures(self) -> dict[int, list[MatLike]]:
        measures_map = self.split_measures()
        crops: dict[int, list[MatLike]] = {}

        for staff_index, measures in measures_map.items():
            staff_crops: list[MatLike] = []

            for measure in measures:
                src = self.notes_image if self.notes_image is not None else self.image
                crop = src[
                    measure.y_top : measure.y_bottom + 1,
                    measure.x_start : measure.x_end,
                ]
                if self.notes_image is not None:
                    cleaned = crop
                else:
                    gray = (
                        crop
                        if len(crop.shape) == 2
                        else cv.cvtColor(crop, cv.COLOR_BGR2GRAY)
                    )
                    cleaned = erase_staff_for_notes(gray)
                staff_crops.append(cleaned)

            crops[staff_index] = staff_crops

        return crops

    def extract_clef_and_key_signatures(self) -> dict[int, ClefAndKeySignature]:

        clef_key_map: dict[int, ClefAndKeySignature] = {}

        for staff_index, staff in enumerate(self.staffs):
            assert staff.lines
            x_start = self._staff_left(staff)
            x_end = self._content_start_x(staff)
            assert x_end > x_start

            clef_key_map[staff_index] = ClefAndKeySignature(
                x_start=x_start,
                x_end=x_end,
                y_top=staff.top,
                y_bottom=staff.bottom,
                staff_index=staff_index,
            )

        return clef_key_map

    def crop_clef_and_key_signatures(self) -> dict[int, MatLike]:
        clef_key_map = self.extract_clef_and_key_signatures()

        crops: dict[int, MatLike] = {}

        for staff_index, region in clef_key_map.items():
            crop = self.image[
                region.y_top : region.y_bottom + 1,
                region.x_start : region.x_end,
            ]
            crops[staff_index] = crop
        return crops
