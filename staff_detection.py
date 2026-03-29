"""Staff detection - find the 5-line staves."""

import cv2 as cv
import numpy as np
from cv2.typing import MatLike

from constants import MASK_OFF, MASK_ON
from schema import Staff, StaffLine


def find_staves(image: MatLike) -> tuple[list[Staff], MatLike, MatLike]:
    gray = to_gray(image)
    binary = binarize(gray)
    line_mask = extract_horizontal_lines(binary)
    centers = find_line_centers(line_mask)
    staffs = group_into_staves(centers, line_mask, binary.shape)
    return staffs, binary, line_mask


def to_gray(image: MatLike) -> MatLike:
    if len(image.shape) == 2:
        return image.copy()
    return cv.cvtColor(image, cv.COLOR_BGR2GRAY)


def binarize(gray: MatLike) -> MatLike:
    blurred = cv.GaussianBlur(gray, (5, 5), 0)
    _, binary = cv.threshold(
        blurred, MASK_OFF, MASK_ON, cv.THRESH_BINARY_INV + cv.THRESH_OTSU
    )
    return binary


def extract_horizontal_lines(binary: MatLike) -> MatLike:
    image_width = binary.shape[1]
    kernel_width = max(25, image_width // 12)
    kernel_width = max(1, min(kernel_width, image_width))
    kernel = cv.getStructuringElement(cv.MORPH_RECT, (kernel_width, 1))
    return cv.morphologyEx(binary, cv.MORPH_OPEN, kernel)


def find_line_centers(line_mask: MatLike) -> list[int]:
    min_ratio = 0.35
    row_strength = np.sum(line_mask > MASK_OFF, axis=1).astype(np.float32)

    if row_strength.size == 0:
        return []

    peak = float(np.max(row_strength))
    if peak == 0.0:
        return []

    threshold = peak * min_ratio
    candidate_rows = np.flatnonzero(row_strength >= threshold)
    return _cluster_rows(candidate_rows)


def _cluster_rows(rows: np.ndarray, max_gap: int = 1) -> list[int]:
    if rows.size == 0:
        return []

    centers = []
    start = int(rows[0])
    prev = start

    for value in rows[1:]:
        y = int(value)
        if y - prev <= max_gap:
            prev = y
        else:
            centers.append((start + prev) // 2)
            start = y
            prev = y

    centers.append((start + prev) // 2)
    return centers


def group_into_staves(
    line_centers: list[int], line_mask: MatLike, shape: tuple
) -> list[Staff]:
    staffs = []
    n = 5  # lines per staff
    gap_count = n - 1
    i = 0

    while i + n <= len(line_centers):
        candidate = line_centers[i : i + n]
        gaps = [candidate[j + 1] - candidate[j] for j in range(gap_count)]
        mean_gap = sum(gaps) / gap_count

        if mean_gap <= 0:
            i += 1
            continue

        tolerance = max(2.0, mean_gap * 0.35)
        if not all(abs(g - mean_gap) <= tolerance for g in gaps):
            i += 1
            continue

        lines = []
        for y in candidate:
            x0, x1 = _line_extent(line_mask, y)
            lines.append(StaffLine(y=y, x_start=x0, x_end=x1))

        pad = 2.0 * mean_gap
        top = max(0, int(candidate[0] - pad))
        bottom = min(shape[0] - 1, int(candidate[-1] + pad))

        staffs.append(Staff(lines=lines, spacing=mean_gap, top=top, bottom=bottom))
        i += n

    return staffs


def _line_extent(line_mask: MatLike, y: int, half_window: int = 1) -> tuple[int, int]:
    top = max(0, y - half_window)
    bottom = min(line_mask.shape[0], y + half_window + 1)
    cols = np.flatnonzero(np.any(line_mask[top:bottom, :] > MASK_OFF, axis=0))

    if cols.size == 0:
        return 0, line_mask.shape[1] - 1

    return int(cols[0]), int(cols[-1])


def erase_staff_for_bars(binary: MatLike, staffs: list[Staff]) -> MatLike:
    horizontal = extract_horizontal_lines(binary)
    allowed = _staff_removal_band_mask(binary.shape, staffs)
    out = cv.subtract(binary, cv.bitwise_and(horizontal, allowed))

    repaired = _repair_slits(out, 3)
    return _blend_slit_repair(out, repaired, staffs)


def erase_staff_for_notes(gray: MatLike, staffs: list[Staff] | None = None) -> MatLike:
    inverted = cv.bitwise_not(gray)
    bw = cv.adaptiveThreshold(
        inverted, MASK_ON, cv.ADAPTIVE_THRESH_MEAN_C, cv.THRESH_BINARY, 15, -2
    )

    k = max(1, bw.shape[1] // 30)
    structure = cv.getStructuringElement(cv.MORPH_RECT, (k, 1))
    h = cv.dilate(cv.erode(bw, structure), structure)

    if staffs is not None:
        allowed = _staff_removal_band_mask(bw.shape, staffs)
        out = cv.subtract(bw, cv.bitwise_and(h, allowed))
        repaired = _repair_slits(out, 3)
        return _blend_slit_repair(out, repaired, staffs)
    else:
        return cv.subtract(bw, h)


def _staff_removal_band_mask(shape: tuple, staffs: list[Staff]) -> MatLike:
    h, w = int(shape[0]), int(shape[1])
    allowed = np.zeros((h, w), dtype=np.uint8)

    for staff in staffs:
        band = max(1, int(round(staff.spacing * 0.2)))
        for line in staff.lines:
            y0 = max(0, line.y - band)
            y1 = min(h, line.y + band + 1)
            x0 = max(0, line.x_start)
            x1 = min(w, line.x_end + 1)
            allowed[y0:y1, x0:x1] = MASK_ON

    return allowed


def _repair_slits(ink: MatLike, vertical_extent: int) -> MatLike:
    if vertical_extent <= 0:
        return ink
    k = max(3, min(7, vertical_extent))
    kernel = cv.getStructuringElement(cv.MORPH_RECT, (1, k))
    return cv.morphologyEx(ink, cv.MORPH_CLOSE, kernel)


def _blend_slit_repair(
    original: MatLike, repaired: MatLike, staffs: list[Staff]
) -> MatLike:
    h, w = original.shape[:2]
    mask = np.zeros((h, w), dtype=np.uint8)

    for staff in staffs:
        band_half = max(1, int(round(staff.spacing * 0.1)))
        for line in staff.lines:
            y0 = max(0, line.y - band_half)
            y1 = min(h, line.y + band_half + 1)
            x0 = max(0, line.x_start)
            x1 = min(w, line.x_end + 1)
            mask[y0:y1, x0:x1] = MASK_ON

    return np.where(mask > 0, repaired, original).astype(original.dtype)
