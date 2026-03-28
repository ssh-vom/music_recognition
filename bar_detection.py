from dataclasses import dataclass

import cv2 as cv

from constants import MASK_OFF, MASK_ON
from schema import BarKind, BarLine, RepeatKind, Staff


@dataclass(frozen=True)
class BarOverlayConfig:
    line_color = (MASK_OFF, MASK_OFF, MASK_ON)
    line_thickness = 1


@dataclass(frozen=True)
class BarDetectionConfig:
    left_skip_spacings = 5.0
    vertical_close_height_ratio = 2.0
    min_height_ratio = 0.4
    min_density = 0.55
    max_width_ratio = 0.6
    thick_bar_min_density = 0.75
    merge_distance_ratio = 0.5
    double_bar_min_width_spacings = 1.0
    double_bar_right_margin_spacings = 2.0
    repeat_dot_window_spacings = 1.0
    repeat_dot_max_size_ratio = 1.0
    repeat_dot_y_tolerance_ratio = 0.45
    repeat_dot_min_area = 2.0


@dataclass(frozen=True)
class BarCandidate:
    x: int
    kind: BarKind


class BarDetector:
    """
    Detects measure bar lines using staff geometry and vertical blob detection.

    Process:
    1. For each staff, crop the image to staff vertical bounds
    2. Skip left area to ignore clef/key signature
    3. Apply vertical closing to join broken bar fragments
    4. Find contours of vertical components
    5. Filter contours by height, width, and density to isolate barlines
    6. Extract x positions and merge nearby detections
    7. Return sorted barline positions
    """

    def __init__(self, binary_img, original_img, staffs):
        self.original = original_img
        self.image = binary_img
        self.staffs = staffs
        self.config = BarDetectionConfig()
        self.overlay_config = BarOverlayConfig()
        self.bars = []

    def _classify_repeat(self, roi, staff, y0, left_x, right_x):
        spacing = staff.spacing
        dot_window = int(round(self.config.repeat_dot_window_spacings * spacing))
        dot_max_size = max(
            1, int(round(self.config.repeat_dot_max_size_ratio * spacing))
        )
        dot_tol = max(1, int(round(self.config.repeat_dot_y_tolerance_ratio * spacing)))
        dot_min_area = self.config.repeat_dot_min_area

        dot_y_top = int(round((staff.lines[1].y + staff.lines[2].y) / 2.0)) - y0
        dot_y_bottom = int(round((staff.lines[2].y + staff.lines[3].y) / 2.0)) - y0

        def side_has_dots(x_start, x_end):
            if x_end <= x_start:
                return False

            window = roi[:, x_start:x_end]
            if window.size == 0:
                return False

            contours, _ = cv.findContours(
                window, cv.RETR_EXTERNAL, cv.CHAIN_APPROX_SIMPLE
            )
            top_xs = []
            bottom_xs = []

            for contour in contours:
                area = cv.contourArea(contour)
                if area < dot_min_area:
                    continue

                x, y, width, height = cv.boundingRect(contour)
                if width > dot_max_size or height > dot_max_size:
                    continue

                center_x = x + width // 2
                center_y = y + height // 2
                if abs(center_y - dot_y_top) <= dot_tol:
                    top_xs.append(center_x)
                if abs(center_y - dot_y_bottom) <= dot_tol:
                    bottom_xs.append(center_x)

            for top_x in top_xs:
                for bottom_x in bottom_xs:
                    if abs(top_x - bottom_x) <= dot_max_size:
                        return True

            return False

        left_start = max(0, left_x - dot_window)
        left_end = max(0, left_x - 1)
        right_start = min(roi.shape[1], right_x + 1)
        right_end = min(roi.shape[1], right_x + dot_window + 1)

        has_left = side_has_dots(left_start, left_end)
        has_right = side_has_dots(right_start, right_end)

        pair_center = (left_x + right_x) // 2
        left_skip = min(
            roi.shape[1], int(round(self.config.left_skip_spacings * spacing))
        )
        edge_margin = int(round(4.0 * spacing))
        staff_right = max(line.x_end for line in staff.lines)
        near_left_edge = pair_center <= left_skip + edge_margin
        near_right_edge = pair_center >= staff_right - edge_margin

        if near_left_edge:
            if has_right:
                return "begin"
            return "none"

        if near_right_edge:
            if has_left:
                return "end"
            return "none"

        if has_right and not has_left:
            return "begin"
        if has_left and not has_right:
            return "end"
        return "none"

    def detect(self):
        """Detect barline positions for each staff using vertical blob analysis."""
        bars = []

        for staff_index, staff in enumerate(self.staffs):
            y0 = staff.top
            y1 = staff.bottom + 1
            staff_height = y1 - y0
            roi = self.image[y0:y1, :]

            left_skip = min(
                roi.shape[1], int(round(self.config.left_skip_spacings * staff.spacing))
            )
            work = roi[:, left_skip:]
            if work.size == 0:
                continue

            close_kernel = cv.getStructuringElement(
                cv.MORPH_RECT,
                (
                    1,
                    max(
                        5,
                        int(
                            round(
                                self.config.vertical_close_height_ratio * staff.spacing
                            )
                        ),
                    ),
                ),
            )
            joined = cv.morphologyEx(work, cv.MORPH_CLOSE, close_kernel)

            contours, _ = cv.findContours(
                joined, cv.RETR_EXTERNAL, cv.CHAIN_APPROX_SIMPLE
            )

            candidates = []
            staff_right = max(line.x_end for line in staff.lines)
            max_width = max(3, int(round(self.config.max_width_ratio * staff.spacing)))
            min_double_width = max(
                6, int(round(self.config.double_bar_min_width_spacings * staff.spacing))
            )
            right_margin = int(
                round(self.config.double_bar_right_margin_spacings * staff.spacing)
            )
            left_margin = int(round(4.0 * staff.spacing))
            left_relaxed_max_width = max_width + int(round(0.7 * staff.spacing))

            for contour in contours:
                x, _, width, height = cv.boundingRect(contour)

                if height < int(round(self.config.min_height_ratio * staff_height)):
                    continue

                density = cv.contourArea(contour) / float(width * height)

                abs_left = left_skip + x
                abs_right = left_skip + x + width - 1
                abs_center = left_skip + x + width // 2
                near_right_edge = abs_center >= staff_right - right_margin
                near_left_edge = abs_center <= left_skip + left_margin
                is_left_relaxed = (
                    near_left_edge
                    and width <= left_relaxed_max_width
                    and density >= 0.50
                )

                if density < self.config.min_density and not is_left_relaxed:
                    continue

                if (
                    width > max_width
                    and density < self.config.thick_bar_min_density
                    and not is_left_relaxed
                ):
                    continue

                if width >= min_double_width and near_right_edge:
                    candidates.append(BarCandidate(x=abs_left, kind="double_left"))
                    candidates.append(BarCandidate(x=abs_right, kind="double_right"))
                    continue

                candidates.append(BarCandidate(x=abs_center, kind="single"))

            if not candidates:
                continue

            candidates.sort(key=lambda c: c.x)
            merge_distance = max(
                3, int(round(self.config.merge_distance_ratio * staff.spacing))
            )

            merged = [candidates[0]]
            for c in candidates[1:]:
                if c.x - merged[-1].x > merge_distance:
                    merged.append(c)
                    continue

                if c.kind != "single" or merged[-1].kind != "single":
                    merged.append(c)
                    continue

                merged[-1] = BarCandidate(x=(merged[-1].x + c.x) // 2, kind="single")

            pair_gap = max(2, int(round(1.5 * staff.spacing)))
            edge_margin = int(round(4.0 * staff.spacing))
            left_edge = left_skip + edge_margin
            right_edge = staff_right - edge_margin
            typed = []
            i = 0

            while i < len(merged):
                if i + 1 < len(merged):
                    left = merged[i]
                    right = merged[i + 1]
                    is_close_pair = right.x - left.x <= pair_gap
                    on_edge = right.x <= left_edge or left.x >= right_edge

                    if (
                        left.kind == "single"
                        and right.kind == "single"
                        and is_close_pair
                        and on_edge
                    ):
                        typed.append(BarCandidate(x=left.x, kind="double_left"))
                        typed.append(BarCandidate(x=right.x, kind="double_right"))
                        i += 2
                        continue

                typed.append(merged[i])
                i += 1

            repeats: list[RepeatKind] = ["none"] * len(typed)
            i = 0
            while i + 1 < len(typed):
                left = typed[i]
                right = typed[i + 1]

                if left.kind == "double_left" and right.kind == "double_right":
                    repeat = self._classify_repeat(roi, staff, y0, left.x, right.x)
                    repeats[i] = repeat
                    repeats[i + 1] = repeat
                    i += 2
                    continue

                i += 1

            for index, c in enumerate(typed):
                bars.append(
                    BarLine(
                        x=c.x,
                        y_top=y0,
                        y_bottom=y1 - 1,
                        kind=c.kind,
                        repeat=repeats[index],
                        staff_index=staff_index,
                    )
                )

        bars.sort(key=lambda b: (b.staff_index, b.x))
        self.bars = bars
        return bars

    def draw_overlay(self):
        """Draw detected barlines on the original image for visualization."""
        overlay = self.original.copy()

        for bar in self.bars:
            cv.line(
                overlay,
                (bar.x, bar.y_top),
                (bar.x, bar.y_bottom),
                self.overlay_config.line_color,
                self.overlay_config.line_thickness,
            )

        return overlay
