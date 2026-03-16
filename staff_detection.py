from dataclasses import dataclass

import cv2 as cv
import numpy as np
from cv2.typing import MatLike


@dataclass
class StaffLine:
    y: int
    x_start: int
    x_end: int


@dataclass
class Staff:
    lines: list[StaffLine]
    spacing: float
    top: int
    bottom: int


class StaffDetector:
    def __init__(self, sheet_img: MatLike):
        self.I = sheet_img
        pass

    def to_gray(self) -> MatLike:
        if len(self.I.shape) == 2:
            return self.I.copy()
        return cv.cvtColor(self.I, cv.COLOR_BGR2GRAY)

    def binarize(self, gray_image: MatLike) -> MatLike:

        blurred = cv.GaussianBlur(gray_image, (5, 5), 0)
        _, binary = cv.threshold(
            blurred,
            0,
            255,
            cv.THRESH_BINARY_INV + cv.THRESH_OTSU,
        )
        return binary

    def extract_horizontal_lines(self, binary_image: MatLike) -> MatLike:
        """Use a wide horizontal kernel to isolate staff lines."""
        kernel_width = max(25, binary_image.shape[1] // 12)
        kernel = cv.getStructuringElement(cv.MORPH_RECT, (kernel_width, 1))
        return cv.morphologyEx(binary_image, cv.MORPH_OPEN, kernel)

    def _cluster_adjacent_rows(self, rows: np.ndarray, max_gap: int = 1) -> list[int]:
        """Merge adjacent row indices into single center values."""
        if rows.size == 0:
            return []

        centers: list[int] = []
        start = int(rows[0])
        prev = start

        for value in rows[1:]:
            y = int(value)
            if y - prev <= max_gap:
                prev = y
                continue
            centers.append((start + prev) // 2)
            start = y
            prev = y

        centers.append((start + prev) // 2)
        return centers

    def _row_centers_from_mask(
        self, line_mask: MatLike, min_ratio: float = 0.35
    ) -> list[int]:
        row_strength = np.sum(line_mask > 0, axis=1).astype(np.float32)
        if row_strength.size == 0:
            return []

        peak = float(np.max(row_strength))
        if peak == 0.0:
            return []

        threshold = peak * min_ratio
        candidate_rows = np.flatnonzero(row_strength >= threshold)
        return self._cluster_adjacent_rows(candidate_rows)

    # ------------------------------------------------------------------
    # Line extent: find x-range of a detected staff line
    # ------------------------------------------------------------------

    def _line_extent(self, line_mask: MatLike, y: int) -> tuple[int, int]:
        """Return (x_start, x_end) for the line at row y."""
        top = max(0, y - 1)
        bottom = min(line_mask.shape[0], y + 2)
        cols = np.flatnonzero(np.any(line_mask[top:bottom, :] > 0, axis=0))

        if cols.size == 0:
            return 0, line_mask.shape[1] - 1

        return int(cols[0]), int(cols[-1])

    # ------------------------------------------------------------------
    # Staff grouping: cluster line centers into groups of 5
    # ------------------------------------------------------------------

    def _group_staffs(
        self,
        line_centers: list[int],
        line_mask: MatLike,
        gap_tolerance_ratio: float = 0.35,
    ) -> list[Staff]:
        """Group detected line centers into Staff objects (5 lines each)."""
        staffs: list[Staff] = []
        i = 0

        while i + 4 < len(line_centers):
            candidate = line_centers[i : i + 5]
            gaps = [candidate[j + 1] - candidate[j] for j in range(4)]
            mean_gap = sum(gaps) / 4.0

            if mean_gap <= 0:
                i += 1
                continue

            tolerance = max(2.0, mean_gap * gap_tolerance_ratio)
            is_staff = all(abs(gap - mean_gap) <= tolerance for gap in gaps)

            if not is_staff:
                i += 1
                continue

            lines: list[StaffLine] = []
            for y in candidate:
                x_start, x_end = self._line_extent(line_mask, y)
                lines.append(StaffLine(y=y, x_start=x_start, x_end=x_end))

            top = max(0, int(candidate[0] - 2 * mean_gap))
            bottom = min(line_mask.shape[0] - 1, int(candidate[-1] + 2 * mean_gap))

            staffs.append(Staff(lines=lines, spacing=mean_gap, top=top, bottom=bottom))
            i += 5

        return staffs

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def detect(self) -> tuple[list[Staff], MatLike, MatLike]:
        """Run the full detection pipeline.

        Returns:
            staffs:    list of detected Staff objects
            binary:    the binarized image
            line_mask: morphological mask of horizontal lines
        """
        gray = self.to_gray()
        binary = self.binarize(gray)
        line_mask = self.extract_horizontal_lines(binary)
        line_centers = self._row_centers_from_mask(line_mask)
        staffs = self._group_staffs(line_centers, line_mask)
        return staffs, binary, line_mask

    def draw_overlay(self, staffs: list[Staff]) -> MatLike:
        """Draw detected staff lines and bounding boxes on the image."""
        if len(self.I.shape) == 2:
            overlay = cv.cvtColor(self.I, cv.COLOR_GRAY2BGR)
        else:
            overlay = self.I.copy()

        for idx, staff in enumerate(staffs):
            for line in staff.lines:
                cv.line(
                    overlay,
                    (line.x_start, line.y),
                    (line.x_end, line.y),
                    (0, 255, 0),
                    1,
                )

            cv.rectangle(
                overlay,
                (0, staff.top),
                (overlay.shape[1] - 1, staff.bottom),
                (255, 0, 0),
                1,
            )

            cv.putText(
                overlay,
                f"staff {idx}  spacing={staff.spacing:.1f}px",
                (10, max(15, staff.top - 5)),
                cv.FONT_HERSHEY_SIMPLEX,
                0.5,
                (0, 0, 255),
                1,
                cv.LINE_AA,
            )

        return overlay
