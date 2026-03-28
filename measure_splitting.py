"""Split staves into measures using barline positions; crop regions from the page."""

from dataclasses import dataclass

import cv2 as cv

from schema import Clef, KeySignature, Measure, TimeSignature
from staff_detection import StaffDetectionConfig, erase_staff_for_notes


class MeasureDetectionConfig:
    left_header_spacings = 7.0
    bar_trim_ratio = 0.25
    min_width_px = 4
    first_staff_conservative_spacings = None


class MeasureSplitter:
    def __init__(
        self, bars, staffs, sheet_img, config=None, notes_image=None, staff_config=None
    ):
        self.bars = bars
        self.staffs = staffs
        self.image = sheet_img
        self.config = config or MeasureDetectionConfig()
        self.staff_config = staff_config or StaffDetectionConfig()
        self.notes_image = notes_image

    def _group_barlines_by_staff(self):
        grouped = {i: [] for i in range(len(self.staffs))}

        for bar in self.bars:
            if 0 <= bar.staff_index < len(self.staffs):
                grouped[bar.staff_index].append(bar)

        for staff_bars in grouped.values():
            staff_bars.sort(key=lambda b: b.x)

        return grouped

    def _staff_left(self, staff):
        return min(line.x_start for line in staff.lines)

    def _staff_right(self, staff):
        return max(line.x_end for line in staff.lines) + 1

    def _content_start_x(self, staff):
        staff_left = self._staff_left(staff)
        staff_right = self._staff_right(staff)
        header_width = int(round(self.config.left_header_spacings * staff.spacing))
        content_start = staff_left + header_width

        if content_start < staff_left:
            return staff_left

        if content_start > staff_right:
            return staff_right

        return content_start

    def _bar_trim_px(self, staff):
        trim = int(round(self.config.bar_trim_ratio * staff.spacing))
        return max(1, trim)

    def _usable_bars(self, staff_bars, content_start_x, staff_right):
        usable = []

        for bar in staff_bars:
            if bar.x <= content_start_x:
                continue
            if bar.x >= staff_right:
                continue
            usable.append(bar)

        return usable

    def _build_measure(self, x_start, x_end, staff, staff_index):
        if x_end - x_start < self.config.min_width_px:
            return None

        return Measure(
            x_start=x_start,
            x_end=x_end,
            y_top=staff.top,
            y_bottom=staff.bottom,
            staff_index=staff_index,
        )

    def _split_staff(self, staff, staff_index, staff_bars):
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

        measures = []
        trim = self._bar_trim_px(staff)
        current_start = content_start_x

        for index, bar in enumerate(usable_bars):
            if bar.kind == "double_left":
                if (
                    index + 1 < len(usable_bars)
                    and usable_bars[index + 1].kind == "double_right"
                ):
                    continue
            elif bar.kind == "double_right":
                pass
            elif bar.kind != "single":
                raise ValueError(f"Unknown bar kind: {bar.kind}")

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

    def split_measures(self):
        barlines_by_staff = self._group_barlines_by_staff()
        measures_map = {}

        for staff_index, staff in enumerate(self.staffs):
            staff_bars = barlines_by_staff[staff_index]
            measures_map[staff_index] = self._split_staff(
                staff=staff,
                staff_index=staff_index,
                staff_bars=staff_bars,
            )

        self._apply_first_staff_start_policy(measures_map)
        return measures_map

    def _apply_first_staff_start_policy(self, measures_map):
        spacings = self.config.first_staff_conservative_spacings
        if spacings is None:
            return

        if 0 not in measures_map or not measures_map[0]:
            return

        first_staff = self.staffs[0]
        first_staff_left = self._staff_left(first_staff)
        conservative_start = first_staff_left + int(
            round(first_staff.spacing * spacings)
        )
        first_measure = measures_map[0][0]
        first_measure.x_start = max(first_measure.x_start, conservative_start)

    def crop_measures(self):
        measures_map = self.split_measures()
        crops = {}

        for staff_index, measures in measures_map.items():
            staff_crops = []

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
                    cleaned = erase_staff_for_notes(
                        gray, staffs=self.staffs, config=self.staff_config
                    )
                staff_crops.append(cleaned)

            crops[staff_index] = staff_crops

        return crops

    def extract_clef_and_key_signatures(self):
        """Left header region per staff (clef + key area); kind/key/time filled by detection."""
        clef_by_staff = {}

        for staff_index, staff in enumerate(self.staffs):
            assert staff.lines
            x_start = self._staff_left(staff)
            x_end = self._content_start_x(staff)
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

    def crop_clef_and_key_signatures(self, clefs=None, *, source_image=None):
        """Crop clef + key header. Prefer source_image (e.g. staff-erased notes_image)."""
        clef_by_staff = (
            clefs if clefs is not None else self.extract_clef_and_key_signatures()
        )

        src = source_image
        if src is None:
            src = self.notes_image if self.notes_image is not None else self.image

        crops = {}

        for staff_index, clef in clef_by_staff.items():
            crop = src[clef.y_top : clef.y_bottom + 1, clef.x_start : clef.x_end]
            crops[staff_index] = crop
        return crops
