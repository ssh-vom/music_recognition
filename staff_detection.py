"""
Staff detection and two ways to erase staff lines from the page:

  detect()     — find each staff (5 lines, spacing, bounds) using Otsu + morphology.
  erase_staff_for_bars() — same Otsu binary, erase only near detected lines (barlines stay).
  erase_staff_for_notes() — adaptive binarization + horizontal morphology; subtract only
  inside bands around detected staff lines (same as bars), so beams and other short
  horizontals between lines are kept.

After subtraction, an optional vertical MORPH_CLOSE can seal slits from staff removal.
That repair is applied only inside thin bands around each *detected* staff line (not
globally), so hollow note heads and other holes away from staff heights stay intact.
"""

from dataclasses import dataclass

import cv2 as cv
import numpy as np

from constants import MASK_OFF, MASK_ON
from schema import Staff, StaffLine


@dataclass(frozen=True)
class StaffDetectionConfig:
    gaussian_blur_kernel = (5, 5)
    horizontal_kernel_min_width = 25
    horizontal_kernel_width_divisor = 12
    row_cluster_max_gap = 1
    min_row_strength_ratio = 0.35
    line_extent_half_window = 1
    staff_line_count = 5
    staff_gap_tolerance_ratio = 0.35
    min_staff_gap_tolerance = 2.0
    staff_vertical_margin_in_spacings = 2.0
    removal_band_half_height_ratio = 0.2
    slit_repair_vertical = 3
    slit_repair_band_half_height_ratio = 0.1


@dataclass(frozen=True)
class StaffOverlayConfig:
    line_color = (0, 255, 0)
    line_thickness = 1
    box_color = (255, 0, 0)
    box_thickness = 1
    label_x = 10
    label_min_y = 15
    label_offset_y = 5
    label_font_scale = 0.5
    label_color = (0, 0, 255)
    label_thickness = 1


def _repair_slits(ink, vertical_extent):
    """Close thin horizontal gaps in foreground using vertical MORPH_CLOSE."""
    if vertical_extent <= 0:
        return ink
    k = max(3, min(7, vertical_extent))
    kernel = cv.getStructuringElement(cv.MORPH_RECT, (1, k))
    return cv.morphologyEx(ink, cv.MORPH_CLOSE, kernel)


def _slit_repair_band_mask(shape, staffs, config):
    """255 only near detected staff line y positions (tight band)."""
    h, w = int(shape[0]), int(shape[1])
    mask = np.zeros((h, w), dtype=np.uint8)
    ratio = config.slit_repair_band_half_height_ratio
    for staff in staffs:
        band_half = max(1, int(round(staff.spacing * ratio)))
        for line in staff.lines:
            y0 = max(0, line.y - band_half)
            y1 = min(h, line.y + band_half + 1)
            x0 = max(0, line.x_start)
            x1 = min(w, line.x_end + 1)
            mask[y0:y1, x0:x1] = MASK_ON
    return mask


def _blend_slit_repair(original, repaired, staffs, config):
    """Use repaired pixels only where band mask is set."""
    mask = _slit_repair_band_mask(original.shape, staffs, config)
    return np.where(mask > 0, repaired, original).astype(original.dtype)


def _otsu_binary(gray, config):
    blurred = cv.GaussianBlur(gray, config.gaussian_blur_kernel, 0)
    _, binary = cv.threshold(
        blurred, MASK_OFF, MASK_ON, cv.THRESH_BINARY_INV + cv.THRESH_OTSU
    )
    return binary


def _horizontal_kernel_width(image_width, config):
    w = max(
        config.horizontal_kernel_min_width,
        image_width // config.horizontal_kernel_width_divisor,
    )
    return max(1, min(w, image_width))


def _horizontal_line_mask(binary, config):
    kw = _horizontal_kernel_width(binary.shape[1], config)
    kernel = cv.getStructuringElement(cv.MORPH_RECT, (kw, 1))
    return cv.morphologyEx(binary, cv.MORPH_OPEN, kernel)


def _staff_removal_band_mask(shape, staffs, config):
    """Where horizontal staff ink may be removed (around each detected staff line)."""
    h, w = int(shape[0]), int(shape[1])
    allowed = np.zeros((h, w), dtype=np.uint8)
    for staff in staffs:
        band = max(1, int(round(staff.spacing * config.removal_band_half_height_ratio)))
        for line in staff.lines:
            y0 = max(0, line.y - band)
            y1 = min(h, line.y + band + 1)
            x0 = max(0, line.x_start)
            x1 = min(w, line.x_end + 1)
            allowed[y0:y1, x0:x1] = MASK_ON
    return allowed


def erase_staff_for_bars(binary, staffs, config):
    """Remove staff ink using line positions from detect(); keeps vertical bar strokes."""
    horizontal = _horizontal_line_mask(binary, config)
    allowed = _staff_removal_band_mask(binary.shape, staffs, config)
    out = cv.subtract(binary, cv.bitwise_and(horizontal, allowed))
    if config.slit_repair_vertical <= 0:
        return out
    repaired = _repair_slits(out, config.slit_repair_vertical)
    return _blend_slit_repair(out, repaired, staffs, config)


def erase_staff_for_notes(gray, staffs=None, config=None):
    """Adaptive threshold, horizontal open, subtract staff lines only near detected y positions."""
    inverted = cv.bitwise_not(gray)
    bw = cv.adaptiveThreshold(
        inverted, MASK_ON, cv.ADAPTIVE_THRESH_MEAN_C, cv.THRESH_BINARY, 15, -2
    )
    h = np.copy(bw)
    k = max(1, h.shape[1] // 30)
    structure = cv.getStructuringElement(cv.MORPH_RECT, (k, 1))
    h = cv.dilate(cv.erode(h, structure), structure)
    if staffs is not None and config is not None:
        allowed = _staff_removal_band_mask(bw.shape, staffs, config)
        out = cv.subtract(bw, cv.bitwise_and(h, allowed))
    else:
        out = cv.subtract(bw, h)
    if staffs is None or config is None or config.slit_repair_vertical <= 0:
        return out
    repaired = _repair_slits(out, config.slit_repair_vertical)
    return _blend_slit_repair(out, repaired, staffs, config)


class StaffDetector:
    """Find staves via Otsu binarization, horizontal line mask, and 5-line grouping."""

    def __init__(self, sheet_img, config=None, overlay_config=None):
        self.image = sheet_img
        self.config = config or StaffDetectionConfig()
        self.overlay_config = overlay_config or StaffOverlayConfig()

    def to_gray(self):
        if len(self.image.shape) == 2:
            return self.image.copy()
        return cv.cvtColor(self.image, cv.COLOR_BGR2GRAY)

    def binarize(self, gray_image):
        return _otsu_binary(gray_image, self.config)

    def extract_horizontal_lines(self, binary_image):
        return _horizontal_line_mask(binary_image, self.config)

    def _cluster_adjacent_rows(self, rows):
        max_gap = self.config.row_cluster_max_gap
        if rows.size == 0:
            return []
        centers = []
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

    def _row_centers_from_mask(self, line_mask):
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

    def _line_extent(self, line_mask, y):
        half = self.config.line_extent_half_window
        top = max(0, y - half)
        bottom = min(line_mask.shape[0], y + half + 1)
        cols = np.flatnonzero(np.any(line_mask[top:bottom, :] > MASK_OFF, axis=0))
        if cols.size == 0:
            return 0, line_mask.shape[1] - 1
        return int(cols[0]), int(cols[-1])

    def _group_staffs(self, line_centers, line_mask):
        staffs = []
        n = self.config.staff_line_count
        gap_count = n - 1
        i = 0
        while i + n <= len(line_centers):
            candidate = line_centers[i : i + n]
            gaps = [candidate[j + 1] - candidate[j] for j in range(gap_count)]
            mean_gap = sum(gaps) / gap_count
            if mean_gap <= 0:
                i += 1
                continue
            tolerance = max(
                self.config.min_staff_gap_tolerance,
                mean_gap * self.config.staff_gap_tolerance_ratio,
            )
            if not all(abs(g - mean_gap) <= tolerance for g in gaps):
                i += 1
                continue
            lines = []
            for y in candidate:
                x0, x1 = self._line_extent(line_mask, y)
                lines.append(StaffLine(y=y, x_start=x0, x_end=x1))
            pad = self.config.staff_vertical_margin_in_spacings * mean_gap
            top = max(0, int(candidate[0] - pad))
            bottom = min(line_mask.shape[0] - 1, int(candidate[-1] + pad))
            staffs.append(Staff(lines=lines, spacing=mean_gap, top=top, bottom=bottom))
            i += n
        return staffs

    def detect(self):
        gray = self.to_gray()
        binary = self.binarize(gray)
        line_mask = self.extract_horizontal_lines(binary)
        centers = self._row_centers_from_mask(line_mask)
        staffs = self._group_staffs(centers, line_mask)
        return staffs, binary, line_mask

    def draw_overlay(self, staffs):
        oc = self.overlay_config
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
                    oc.line_color,
                    oc.line_thickness,
                )
            cv.rectangle(
                overlay,
                (0, staff.top),
                (overlay.shape[1] - 1, staff.bottom),
                oc.box_color,
                oc.box_thickness,
            )
            cv.putText(
                overlay,
                f"staff {idx}  spacing={staff.spacing:.1f}px",
                (oc.label_x, max(oc.label_min_y, staff.top - oc.label_offset_y)),
                cv.FONT_HERSHEY_SIMPLEX,
                oc.label_font_scale,
                oc.label_color,
                oc.label_thickness,
                cv.LINE_AA,
            )
        return overlay
