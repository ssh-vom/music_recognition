from dataclasses import dataclass

import cv2 as cv
from cv2.typing import MatLike

from constants import MASK_OFF, MASK_ON
from schema import BarKind, BarLine, RepeatKind, Staff

# Configuration and detection logic for vertical barline detection


@dataclass(frozen=True)
class BarOverlayConfig:
    line_color: tuple[int, int, int] = (MASK_OFF, MASK_OFF, MASK_ON)
    line_thickness: int = 1


@dataclass(frozen=True)
class BarDetectionConfig:
    left_skip_spacings: float = (
        5.0  # staff spacings to skip from the left to ignore clef/key signature
    )
    # Multiplier for staff spacing to determine closing kernel height for joining broken bars
    vertical_close_height_ratio: float = 2.0
    # Minimum bar height as ratio of staff height (filters out short stems)
    min_height_ratio: float = 0.4
    # Minimum contour density (area/width*height) to accept as a bar (filters out noise)
    min_density: float = 0.55
    # Maximum bar width as ratio of staff spacing (thicker bars need higher density)
    max_width_ratio: float = 0.6
    # Minimum density for bars wider than max_width_ratio (allows thick final bars)
    thick_bar_min_density: float = 0.75
    # Maximum distance between bar centers to merge into single bar (in spacings)
    merge_distance_ratio: float = 0.5
    # Wide contour threshold used to split final double bars into left/right bars
    double_bar_min_width_spacings: float = 1.0
    # How close to the right edge a wide contour must be to be treated as final double bar
    double_bar_right_margin_spacings: float = 2.0
    # Horizontal search window for repeat dots around a double bar pair
    repeat_dot_window_spacings: float = 1.0
    # Maximum repeat-dot blob size as ratio of staff spacing
    repeat_dot_max_size_ratio: float = 1.0
    # Vertical tolerance around expected repeat-dot centers
    repeat_dot_y_tolerance_ratio: float = 0.45
    # Minimum contour area to accept a repeat dot blob
    repeat_dot_min_area_px: float = 2.0


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

    def __init__(
        self,
        binary_img: MatLike,
        original_img: MatLike,
        staffs: list[Staff],
    ):
        """Initialize bar detector with processed images and detected staffs.

        Args:
            binary_img: Binarized image (staff lines removed preferred)
            original_img: Original grayscale/color image for drawing overlays
            staffs: List of detected staff objects with geometry information
        """
        self.original = original_img
        self.image = binary_img
        self.staffs = staffs
        self.config = BarDetectionConfig()
        self.overlay_config = BarOverlayConfig()
        self.bars: list[BarLine] = []

    def _classify_repeat(
        self,
        roi: MatLike,
        staff: Staff,
        y0: int,
        left_x: int,
        right_x: int,
    ) -> RepeatKind:
        assert len(staff.lines) == 5

        spacing = staff.spacing
        dot_window = int(round(self.config.repeat_dot_window_spacings * spacing))
        dot_max_size = max(
            1, int(round(self.config.repeat_dot_max_size_ratio * spacing))
        )
        dot_tol = max(1, int(round(self.config.repeat_dot_y_tolerance_ratio * spacing)))
        dot_min_area = self.config.repeat_dot_min_area_px

        dot_y_top = int(round((staff.lines[1].y + staff.lines[2].y) / 2.0)) - y0
        dot_y_bottom = int(round((staff.lines[2].y + staff.lines[3].y) / 2.0)) - y0

        def side_has_dots(x_start: int, x_end: int) -> bool:
            if x_end <= x_start:
                return False

            window = roi[:, x_start:x_end]
            if window.size == 0:
                return False

            contours, _ = cv.findContours(
                window, cv.RETR_EXTERNAL, cv.CHAIN_APPROX_SIMPLE
            )
            top_xs: list[int] = []
            bottom_xs: list[int] = []

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

    def detect(self) -> list[BarLine]:
        """Detect barline positions for each staff using vertical blob analysis.

        Returns:
            List of BarLine objects sorted by staff index then x position.
        """
        bars: list[BarLine] = []

        for staff_index, staff in enumerate(self.staffs):
            # Get staff vertical bounds and height
            y0 = staff.top
            y1 = staff.bottom + 1
            staff_height = y1 - y0
            roi = self.image[y0:y1, :]

            # Skip clef/key signature area on the left
            left_skip = min(
                roi.shape[1], int(round(self.config.left_skip_spacings * staff.spacing))
            )
            work = roi[:, left_skip:]
            if work.size == 0:
                continue

            # Create vertical closing kernel to join broken bar fragments
            # Kernel height is based on staff spacing to adapt to different scales
            close_kernel = cv.getStructuringElement(
                cv.MORPH_RECT,
                (
                    1,  # width: keep vertical orientation
                    max(
                        5,  # minimum kernel size
                        int(
                            round(
                                self.config.vertical_close_height_ratio * staff.spacing
                            )
                        ),
                    ),
                ),
            )
            # Apply closing to connect vertically aligned components
            joined = cv.morphologyEx(work, cv.MORPH_CLOSE, close_kernel)

            # Find all connected components in the processed image
            contours, _ = cv.findContours(
                joined, cv.RETR_EXTERNAL, cv.CHAIN_APPROX_SIMPLE
            )

            # Collect x positions of valid barline candidates
            candidates: list[BarCandidate] = []
            staff_right = max(line.x_end for line in staff.lines)
            max_width = max(
                3,  # minimum width in pixels
                int(round(self.config.max_width_ratio * staff.spacing)),
            )
            min_double_width = max(
                6,
                int(round(self.config.double_bar_min_width_spacings * staff.spacing)),
            )
            right_margin = int(
                round(self.config.double_bar_right_margin_spacings * staff.spacing)
            )
            left_margin = int(round(4.0 * staff.spacing))
            left_relaxed_max_width = max_width + int(round(0.7 * staff.spacing))

            for contour in contours:
                # Get bounding box of component
                x, _, width, height = cv.boundingRect(contour)

                # Filter by minimum height (remove short stems and noise)
                if height < int(round(self.config.min_height_ratio * staff_height)):
                    continue

                # Calculate density (area/width*height) to distinguish solid bars from noise
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

                # For wider components, require higher density to avoid false positives
                # from wide noise blobs.
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

                # Convert to absolute x coordinate and store
                candidates.append(BarCandidate(x=abs_center, kind="single"))

            # Skip staff if no valid barlines found
            if not candidates:
                continue

            # Sort and merge nearby detections to handle slight variations
            candidates.sort(key=lambda candidate: candidate.x)
            merge_distance = max(
                3,  # minimum merge distance in pixels
                int(round(self.config.merge_distance_ratio * staff.spacing)),
            )

            merged_candidates: list[BarCandidate] = [candidates[0]]
            for candidate in candidates[1:]:
                # If current detection is far from last merged one, start new group
                if candidate.x - merged_candidates[-1].x > merge_distance:
                    merged_candidates.append(candidate)
                    continue

                # Never merge explicit double bars with anything.
                if candidate.kind != "single" or merged_candidates[-1].kind != "single":
                    merged_candidates.append(candidate)
                    continue

                # Otherwise, update position to average of merged detections
                merged_candidates[-1] = BarCandidate(
                    x=(merged_candidates[-1].x + candidate.x) // 2,
                    kind="single",
                )

            pair_gap = max(2, int(round(1.5 * staff.spacing)))
            edge_margin = int(round(4.0 * staff.spacing))
            left_edge = left_skip + edge_margin
            right_edge = staff_right - edge_margin
            typed_candidates: list[BarCandidate] = []
            i = 0

            while i < len(merged_candidates):
                if i + 1 < len(merged_candidates):
                    left = merged_candidates[i]
                    right = merged_candidates[i + 1]
                    is_close_pair = right.x - left.x <= pair_gap
                    on_edge = right.x <= left_edge or left.x >= right_edge

                    if (
                        left.kind == "single"
                        and right.kind == "single"
                        and is_close_pair
                        and on_edge
                    ):
                        typed_candidates.append(
                            BarCandidate(x=left.x, kind="double_left")
                        )
                        typed_candidates.append(
                            BarCandidate(x=right.x, kind="double_right")
                        )
                        i += 2
                        continue

                typed_candidates.append(merged_candidates[i])
                i += 1

            repeats: list[RepeatKind] = ["none"] * len(typed_candidates)
            i = 0
            while i + 1 < len(typed_candidates):
                left = typed_candidates[i]
                right = typed_candidates[i + 1]

                if left.kind == "double_left" and right.kind == "double_right":
                    repeat = self._classify_repeat(roi, staff, y0, left.x, right.x)
                    repeats[i] = repeat
                    repeats[i + 1] = repeat
                    i += 2
                    continue

                i += 1

            # Create BarLine objects for each merged detection
            for index, candidate in enumerate(typed_candidates):
                bars.append(
                    BarLine(
                        x=candidate.x,
                        y_top=y0,
                        y_bottom=y1 - 1,
                        kind=candidate.kind,
                        repeat=repeats[index],
                        staff_index=staff_index,
                    )
                )

        # Sort all detected barlines by staff then x position
        bars.sort(key=lambda bar: (bar.staff_index, bar.x))
        self.bars = bars
        return bars

    def draw_overlay(self) -> MatLike:
        """Draw detected barlines on the original image for visualization.

        Returns:
            Copy of original image with barlines drawn as red vertical lines.
        """
        overlay = self.original.copy()

        for bar in self.bars:
            # Draw a vertical line at the detected barline position
            cv.line(
                overlay,
                (bar.x, bar.y_top),  # Start point (top of staff)
                (bar.x, bar.y_bottom),  # End point (bottom of staff)
                self.overlay_config.line_color,  # Red color for visibility
                self.overlay_config.line_thickness,
            )

        return overlay
