"""Staff detection - locate the 5-line staves in sheet music."""

import cv2 as cv
import numpy as np
from cv2.typing import MatLike

from constants import (
    BLUR_KERNEL_SIZE,
    LINE_CLUSTER_MAX_GAP,
    LINE_DETECTION_MIN_RATIO,
    LINES_PER_STAFF,
    MASK_BACKGROUND,
    MASK_FOREGROUND,
    SLIT_REPAIR_BAND_FRAC,
    SLIT_REPAIR_KERNEL_MAX,
    SLIT_REPAIR_KERNEL_MIN,
    STAFF_ERASE_BAND_FRAC,
    STAFF_LINE_KERNEL_MIN_WIDTH,
    STAFF_LINE_KERNEL_WIDTH_FRAC,
    STAFF_SPACING_TOLERANCE_FRAC,
    STAFF_SPACING_TOLERANCE_MIN,
    STAFF_VERTICAL_PADDING_FRAC,
)
from image_utils import to_gray
from schema import Staff, StaffLine


def find_staffs(image: MatLike) -> tuple[list[Staff], MatLike, MatLike]:
    """
    Pipeline for finding the staff lines in sheet music.
    We return the staff lines, the binary image of extraction and the mask used
    for ease of visualiation/debugging in the report.
    """
    gray = to_gray(image)  # convert to grayscale
    binary = binarize(gray)  # binarize the grayscale image
    line_mask = extract_horizontal_lines(binary)  # extract horizontal lines
    centers = find_line_centers(line_mask)  # find the centers of the staff lines
    staffs = group_into_staffs(centers, line_mask, binary.shape)  # group into staffs
    return staffs, binary, line_mask


def binarize(gray: MatLike) -> MatLike:
    blurred = cv.GaussianBlur(gray, BLUR_KERNEL_SIZE, 0)
    _, binary = cv.threshold(
        blurred, MASK_BACKGROUND, MASK_FOREGROUND, cv.THRESH_BINARY_INV + cv.THRESH_OTSU
    )
    return binary


def extract_horizontal_lines(binary: MatLike) -> MatLike:
    image_width = binary.shape[1]
    kernel_width = max(
        STAFF_LINE_KERNEL_MIN_WIDTH, int(image_width * STAFF_LINE_KERNEL_WIDTH_FRAC)
    )
    kernel_width = max(1, min(kernel_width, image_width))
    kernel = cv.getStructuringElement(cv.MORPH_RECT, (kernel_width, 1))
    return cv.morphologyEx(binary, cv.MORPH_OPEN, kernel)


def find_line_centers(line_mask: MatLike) -> list[int]:
    row_strength = np.sum(line_mask > MASK_BACKGROUND, axis=1).astype(np.float32)

    if row_strength.size == 0:
        return []

    peak = float(np.max(row_strength))
    if peak == 0.0:
        return []

    candidate_rows = np.flatnonzero(row_strength >= peak * LINE_DETECTION_MIN_RATIO)
    return _cluster_rows(candidate_rows)


def _cluster_rows(rows: np.ndarray, max_gap: int = LINE_CLUSTER_MAX_GAP) -> list[int]:
    if rows.size == 0:
        return []

    centers = []
    start = int(rows[0])
    prev = start

    for value in rows[1:]:
        y = int(value)
        if y - prev > max_gap:
            centers.append((start + prev) // 2)
            start = y
        prev = y

    centers.append((start + prev) // 2)
    return centers


def group_into_staffs(
    line_centers: list[int], line_mask: MatLike, shape: tuple
) -> list[Staff]:
    staffs = []
    gap_count = LINES_PER_STAFF - 1
    i = 0

    while i + LINES_PER_STAFF <= len(line_centers):
        candidate = line_centers[i : i + LINES_PER_STAFF]
        gaps = [candidate[j + 1] - candidate[j] for j in range(gap_count)]
        mean_gap = sum(gaps) / gap_count

        if mean_gap <= 0:
            i += 1
            continue

        tolerance = max(
            STAFF_SPACING_TOLERANCE_MIN, mean_gap * STAFF_SPACING_TOLERANCE_FRAC
        )
        if not all(abs(g - mean_gap) <= tolerance for g in gaps):
            i += 1
            continue

        lines = []
        for y in candidate:
            x0, x1 = _line_extent(line_mask, y)
            lines.append(StaffLine(y=y, x_start=x0, x_end=x1))

        pad = STAFF_VERTICAL_PADDING_FRAC * mean_gap
        top = max(0, int(candidate[0] - pad))
        bottom = min(shape[0] - 1, int(candidate[-1] + pad))

        staffs.append(Staff(lines=lines, spacing=mean_gap, top=top, bottom=bottom))
        i += LINES_PER_STAFF

    return staffs


def _line_extent(line_mask: MatLike, y: int, half_window: int = 1) -> tuple[int, int]:
    y0 = max(0, y - half_window)
    y1 = min(line_mask.shape[0], y + half_window + 1)
    cols = np.flatnonzero(np.any(line_mask[y0:y1, :] > MASK_BACKGROUND, axis=0))

    if cols.size == 0:
        return 0, line_mask.shape[1] - 1

    return int(cols[0]), int(cols[-1])


def erase_staff_for_bars(binary: MatLike, staffs: list[Staff]) -> MatLike:
    horizontal = extract_horizontal_lines(binary)
    allowed = _staff_removal_band_mask(binary.shape, staffs)
    result = cv.subtract(binary, cv.bitwise_and(horizontal, allowed))
    return _repair_slits(result, staffs)


def erase_staff_for_notes(gray: MatLike, staffs: list[Staff]) -> MatLike:
    inverted = cv.bitwise_not(gray)
    bw = cv.adaptiveThreshold(
        inverted, MASK_FOREGROUND, cv.ADAPTIVE_THRESH_MEAN_C, cv.THRESH_BINARY, 15, -2
    )

    kernel_width = max(1, bw.shape[1] // 30)
    structure = cv.getStructuringElement(cv.MORPH_RECT, (kernel_width, 1))
    staff_reconstruction = cv.dilate(cv.erode(bw, structure), structure)

    allowed = _staff_removal_band_mask(bw.shape, staffs)
    result = cv.subtract(bw, cv.bitwise_and(staff_reconstruction, allowed))
    return _repair_slits(result, staffs)


def _staff_removal_band_mask(shape: tuple, staffs: list[Staff]) -> MatLike:
    h, w = shape[:2]
    mask = np.zeros((h, w), dtype=np.uint8)

    for staff in staffs:
        band = max(1, int(round(staff.spacing * STAFF_ERASE_BAND_FRAC)))
        for line in staff.lines:
            y0 = max(0, line.y - band)
            y1 = min(h, line.y + band + 1)
            x0 = max(0, line.x_start)
            x1 = min(w, line.x_end + 1)
            mask[y0:y1, x0:x1] = MASK_FOREGROUND

    return mask


def _repair_slits(ink: MatLike, staffs: list[Staff]) -> MatLike:
    """Heal gaps where stems crossed staff lines, so filled noteheads stay solid."""
    h, w = ink.shape[:2]
    repair_mask = np.zeros((h, w), dtype=np.uint8)

    for staff in staffs:
        band = max(1, int(round(staff.spacing * SLIT_REPAIR_BAND_FRAC)))
        for line in staff.lines:
            y0 = max(0, line.y - band)
            y1 = min(h, line.y + band + 1)
            x0 = max(0, line.x_start)
            x1 = min(w, line.x_end + 1)
            repair_mask[y0:y1, x0:x1] = MASK_FOREGROUND

    kernel_height = max(SLIT_REPAIR_KERNEL_MIN, min(SLIT_REPAIR_KERNEL_MAX, 3))
    kernel = cv.getStructuringElement(cv.MORPH_RECT, (1, kernel_height))
    repaired = cv.morphologyEx(ink, cv.MORPH_CLOSE, kernel)

    return np.where(repair_mask > 0, repaired, ink).astype(ink.dtype)
