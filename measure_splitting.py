from cv2.typing import MatLike

from schema import BarLine, Measure, Staff


class MeasureSplitter:
    def __init__(self, bars: list[BarLine], staffs: list[Staff], sheet_img: MatLike):
        self.bars = bars
        self.staffs = staffs
        self.image = sheet_img

    def _group_barlines_by_staff(self) -> dict[int, list[BarLine]]:
        grouped: dict[int, list[BarLine]] = {i: [] for i in range(len(self.staffs))}

        for bar in self.bars:
            if 0 <= bar.staff_index < len(self.staffs):
                grouped[bar.staff_index].append(bar)

        for staff_bars in grouped.values():
            staff_bars.sort(key=lambda bar: bar.x)

        return grouped

    def split_measures(self) -> dict[int, list[Measure]]:
        barlines_by_staff = self._group_barlines_by_staff()
        measures_map: dict[int, list[Measure]] = {}

        for staff_index, staff in enumerate(self.staffs):
            staff_bars = barlines_by_staff[staff_index]
            staff_left = min(line.x_start for line in staff.lines)

            boundaries = [staff_left] + [bar.x for bar in staff_bars]
            measures: list[Measure] = []

            for x_start, x_end in zip(boundaries, boundaries[1:]):
                if x_end <= x_start:
                    continue

                measures.append(
                    Measure(
                        x_start=x_start,
                        x_end=x_end,
                        y_top=staff.top,
                        y_bottom=staff.bottom,
                        staff_index=staff_index,
                    )
                )

            measures_map[staff_index] = measures

        return measures_map

    def crop_measures(self) -> dict[int, list[MatLike]]:
        measures_map = self.split_measures()
        crops: dict[int, list[MatLike]] = {}

        for staff_index, measures in measures_map.items():
            crops[staff_index] = [
                self.image[
                    measure.y_top : measure.y_bottom + 1,
                    measure.x_start : measure.x_end,
                ]
                for measure in measures
            ]

        return crops
