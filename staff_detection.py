"""
Musical staff detection from sheet music images.

This module provides algorithms to detect musical staff lines in scanned or
digital sheet music. It uses computer vision techniques including morphological
operations and peak detection to identify the characteristic 5-line staff patterns
used in standard music notation.
"""

from dataclasses import dataclass

import cv2 as cv
import numpy as np
from cv2.typing import MatLike
from constants import MASK_ON, MASK_OFF

from schema import Staff, StaffLine


@dataclass(frozen=True)
class StaffDetectionConfig:
    """Configuration parameters for the staff detection algorithm.
    These control the sensitivity and tolerance of line detection.
    Default values work well for standard sheet music scans at 300 DPI.
    """

    gaussian_blur_kernel: tuple[int, int] = (5, 5)
    horizontal_kernel_min_width: int = 25
    horizontal_kernel_width_divisor: int = 12
    row_cluster_max_gap_px: int = 1
    min_row_strength_ratio: float = 0.35
    line_extent_half_window_px: int = 1
    staff_line_count: int = 5
    staff_gap_tolerance_ratio: float = 0.35
    min_staff_gap_tolerance_px: float = 2.0
    staff_vertical_margin_in_spacings: float = 2.0
    removal_band_half_height_ratio: float = 0.2


@dataclass(frozen=True)
class StaffOverlayConfig:
    """Configuration for visualizing detected staff lines.
    Controls colors, thickness, and positioning of overlays when drawing
    staff detection results on the original image.
    """

    line_color: tuple[int, int, int] = (0, 255, 0)
    line_thickness: int = 1
    box_color: tuple[int, int, int] = (255, 0, 0)
    box_thickness: int = 1
    label_x: int = 10
    label_min_y: int = 15
    label_offset_y: int = 5
    label_font_scale: float = 0.5
    label_color: tuple[int, int, int] = (0, 0, 255)
    label_thickness: int = 1


class StaffDetector:
    """Detects musical staff lines in sheet music images.
    The overlying class object that stores all methods for 5-line staff detection,
    we binarizae and morphologically filter to isolate horizontal lines, then pattern match
    to group lines into a valid staff system.
    """

    def __init__(
        self,
        sheet_img: MatLike,
        config: StaffDetectionConfig | None = None,
        overlay_config: StaffOverlayConfig | None = None,
    ):
        """Initialize the staff detector.

        Args:
            sheet_img: Input image (grayscale or color) of sheet music
            config: Detection parameters (uses defaults if not provided)
            overlay_config: Visualization parameters (uses defaults if not provided)
        """
        self.image = sheet_img
        self.config = config or StaffDetectionConfig()
        self.overlay_config = overlay_config or StaffOverlayConfig()

    def to_gray(self) -> MatLike:
        """
        Converts the input image to grayscale if needed.

        If the image is already grayscale (2D array), returns a copy.
        Otherwise converts from BGR color to grayscale.
        """
        if len(self.image.shape) == 2:
            return self.image.copy()
        return cv.cvtColor(self.image, cv.COLOR_BGR2GRAY)

    def binarize(self, gray_image: MatLike) -> MatLike:
        """
        Applies Gaussian blur and Otsu thresholding to create a binary image.

        Gaussian blur smooths out noise and small variations. Otsu's method
        automatically determines the optimal threshold to separate foreground
        (dark) from background (light). The result is inverted so staff lines
        are white (255) on black (0) background.
        """

        blurred = cv.GaussianBlur(gray_image, self.config.gaussian_blur_kernel, 0)
        _, binary = cv.threshold(
            blurred,
            MASK_OFF,
            MASK_ON,
            cv.THRESH_BINARY_INV + cv.THRESH_OTSU,
        )
        return binary

    def extract_horizontal_lines(self, binary_image: MatLike) -> MatLike:
        """
        Isolates horizontal lines using morphological opening.

        This uses a wide rectangular kernel (much wider than tall) to detect
        long horizontal structures like musical staff lines while filtering out
        notes, symbols, and other non-horizontal elements.
        """
        # Calculate kernel width based on image width (shape[1]) - wider images need wider kernels
        # to detect staff lines that span most of the page width

        kernel_width = max(
            self.config.horizontal_kernel_min_width,
            binary_image.shape[1] // self.config.horizontal_kernel_width_divisor,
        )
        kernel = cv.getStructuringElement(cv.MORPH_RECT, (kernel_width, 1))
        return cv.morphologyEx(binary_image, cv.MORPH_OPEN, kernel)

    def _cluster_adjacent_rows(self, rows: np.ndarray) -> list[int]:
        """
        Groups consecutive row indices and returns their center points.
        When staff lines are thick or slightly blurred, they may span multiple
        pixel rows. This merges adjacent rows into single center values to
        represent one staff line rather than a thick band.
        """
        max_gap = self.config.row_cluster_max_gap_px

        if rows.size == 0:
            return []

        centers: list[int] = []
        start = int(rows[0])
        prev = start

        for value in rows[1:]:  # starting from the adjacent rows
            y = int(value)
            if y - prev <= max_gap:
                prev = y
                continue
            # add to the centers based on the previous and start
            centers.append((start + prev) // 2)
            start = y
            prev = y

        centers.append((start + prev) // 2)
        return centers

    def _row_centers_from_mask(self, line_mask: MatLike) -> list[int]:
        """
        Finds the center y-coordinates of horizontal lines from the line mask.
        Calculates pixel density per row, finds peaks (rows with many white pixels),
        and filters to only keep strong peaks that likely represent staff lines.
        """
        min_ratio = self.config.min_row_strength_ratio

        row_strength = np.sum(line_mask > MASK_OFF, axis=1).astype(np.float32)
        if row_strength.size == 0:
            return []

        peak = float(np.max(row_strength))
        if peak == 0.0:
            return []

        threshold = peak * min_ratio
        candidate_rows = np.flatnonzero(row_strength >= threshold)
        return self._cluster_adjacent_rows(candidate_rows)

    def _line_extent(self, line_mask: MatLike, y: int) -> tuple[int, int]:
        """
        Finds the horizontal start and end coordinates of a staff line at row y.

        Searches a small vertical window around the line to find all connected
        pixels, then returns the leftmost and rightmost x coordinates.
        """
        half_window = self.config.line_extent_half_window_px

        top = max(0, y - half_window)
        bottom = min(line_mask.shape[0], y + half_window + 1)
        cols = np.flatnonzero(np.any(line_mask[top:bottom, :] > MASK_OFF, axis=0))

        if cols.size == 0:
            return 0, line_mask.shape[1] - 1

        return int(cols[0]), int(cols[-1])

    def _group_staffs(self, line_centers: list[int], line_mask: MatLike) -> list[Staff]:
        """
        Groups detected line centers into Staff objects (sets of 5 lines).

        A musical staff has exactly 5 lines with consistent spacing. This method:
        1. Takes groups of 5 consecutive lines
        2. Calculates the spacing between each pair
        3. Checks if all spacings are similar (within tolerance)
        4. Creates Staff objects for valid groups
        """
        staffs: list[Staff] = []
        staff_line_count = self.config.staff_line_count
        gap_count = staff_line_count - 1
        i = 0

        while i + staff_line_count <= len(line_centers):
            # Take the next staff_line_count (default = 5) lines as a candidate staff
            candidate = line_centers[i : i + staff_line_count]
            gaps = [candidate[j + 1] - candidate[j] for j in range(gap_count)]
            mean_gap = sum(gaps) / gap_count

            if mean_gap <= 0:
                i += 1
                continue

            # Calculate tolerance: how much spacing variation is acceptable
            # Uses the larger of: absolute minimum or ratio-based tolerance
            tolerance = max(
                self.config.min_staff_gap_tolerance_px,
                mean_gap * self.config.staff_gap_tolerance_ratio,
            )

            # Check if all gaps are similar enough to be a valid staff
            is_staff = all(abs(gap - mean_gap) <= tolerance for gap in gaps)

            if not is_staff:
                i += 1
                continue

            # Build StaffLine objects with horizontal extents
            lines: list[StaffLine] = []
            for y in candidate:
                x_start, x_end = self._line_extent(line_mask, y)
                lines.append(StaffLine(y=y, x_start=x_start, x_end=x_end))

            # Calculate vertical bounds with padding (for visualization/processing)
            padding = self.config.staff_vertical_margin_in_spacings * mean_gap
            top = max(0, int(candidate[0] - padding))
            bottom = min(line_mask.shape[0] - 1, int(candidate[-1] + padding))

            staffs.append(Staff(lines=lines, spacing=mean_gap, top=top, bottom=bottom))
            i += staff_line_count

        return staffs

    def detect(self) -> tuple[list[Staff], MatLike, MatLike]:
        """
        Run the complete staff detection pipeline.

        Pipeline steps:
        1. Convert to grayscale for simpler processing
        2. Binarize using Otsu's method to separate lines from background
        3. Extract horizontal lines using morphological opening
        4. Find line centers by analyzing row pixel densities
        5. Group lines into staff systems (sets of 5 lines)

        Returns:
            staffs: List of detected Staff objects, each containing 5 lines
            binary: Binary image after thresholding
            line_mask: Binary mask showing only horizontal line pixels
        """
        gray = self.to_gray()
        binary = self.binarize(gray)
        line_mask = self.extract_horizontal_lines(binary)
        line_centers = self._row_centers_from_mask(line_mask)
        staffs = self._group_staffs(line_centers, line_mask)
        return staffs, binary, line_mask

    def draw_overlay(self, staffs: list[Staff]) -> MatLike:
        """Draw detected staff lines and bounding boxes on the image."""
        overlay_config = self.overlay_config

        if len(self.image.shape) == 2:
            overlay = cv.cvtColor(self.image, cv.COLOR_GRAY2BGR)
        else:
            overlay = self.image.copy()

        for idx, staff in enumerate(staffs):
            for line in staff.lines:
                cv.line(
                    overlay,
                    (line.x_start, line.y),
                    (line.x_end, line.y),
                    overlay_config.line_color,
                    overlay_config.line_thickness,
                )

            cv.rectangle(
                overlay,
                (0, staff.top),
                (overlay.shape[1] - 1, staff.bottom),
                overlay_config.box_color,
                overlay_config.box_thickness,
            )

            cv.putText(
                overlay,
                f"staff {idx}  spacing={staff.spacing:.1f}px",
                (
                    overlay_config.label_x,
                    max(
                        overlay_config.label_min_y,
                        staff.top - overlay_config.label_offset_y,
                    ),
                ),
                cv.FONT_HERSHEY_SIMPLEX,
                overlay_config.label_font_scale,
                overlay_config.label_color,
                overlay_config.label_thickness,
                cv.LINE_AA,
            )

        return overlay

    def remove_staffs(self, staffs: list[Staff]) -> MatLike:
        gray = self.to_gray()
        binary = self.binarize(gray)
        horizontal = self.extract_horizontal_lines(binary)

        allowed = np.zeros_like(horizontal)

        for staff in staffs:
            band_half = max(
                1,
                int(round(staff.spacing * self.config.removal_band_half_height_ratio)),
            )

            for line in staff.lines:
                y0 = max(0, line.y - band_half)
                y1 = min(horizontal.shape[0], line.y + band_half + 1)
                x0 = max(0, line.x_start)
                x1 = min(horizontal.shape[1], line.x_end + 1)

                allowed[y0:y1, x0:x1] = MASK_ON

        staff_line_mask = cv.bitwise_and(horizontal, allowed)
        cleaned = cv.subtract(binary, staff_line_mask)
        return cleaned
