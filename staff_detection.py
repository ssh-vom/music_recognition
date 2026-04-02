import cv2 as cv
import numpy as np
from cv2.typing import MatLike

from constants import Constants as const
from schema import Staff, StaffLine
from utils import to_gray


def find_staffs(image: MatLike) -> tuple[list[Staff], MatLike, MatLike]:
    """
    Pipeline for finding the staff lines in sheet music.
    We return the staff lines, the binary image of extraction and the mask used
    for ease of visualiation/debugging in the report.
    """
    gray = to_gray(image)
    binary = binarize(gray)
    line_mask = extract_horizontal_lines(binary)
    centers = find_line_centers(line_mask)
    staffs = group_into_staffs(centers, line_mask, binary.shape)
    return staffs, binary, line_mask


def binarize(gray: MatLike) -> MatLike:
    blurred = cv.GaussianBlur(gray, const.BLUR_KERNEL_SIZE, 0)
    # THRESH_BINARY_INV so ink becomes foreground (255); Otsu picks the threshold automatically
    _, binary = cv.threshold(
        blurred,
        const.MASK_BACKGROUND,
        const.MASK_FOREGROUND,
        cv.THRESH_BINARY_INV + cv.THRESH_OTSU,
    )
    return binary


def extract_horizontal_lines(binary: MatLike) -> MatLike:
    image_width = binary.shape[1]
    kernel_width = max(
        const.STAFF_LINE_KERNEL_MIN_WIDTH,
        int(image_width * const.STAFF_LINE_KERNEL_WIDTH_FRAC),
    )
    kernel_width = max(1, min(kernel_width, image_width))
    # a wide horizontal open kernel removes anything shorter than kernel_width, leaving only long horizontal strokes
    kernel = cv.getStructuringElement(cv.MORPH_RECT, (kernel_width, 1))
    return cv.morphologyEx(binary, cv.MORPH_OPEN, kernel)


def find_line_centers(line_mask: MatLike) -> list[int]:
    # count how many pixels are lit in each row — staff lines will have much higher counts than gaps
    row_strength = np.sum(line_mask > const.MASK_BACKGROUND, axis=1).astype(np.float32)

    if row_strength.size == 0:
        return []

    peak = float(np.max(row_strength))
    if peak == 0.0:
        return []

    candidate_rows = np.flatnonzero(
        row_strength >= peak * const.LINE_DETECTION_MIN_RATIO
    )
    return _cluster_rows(candidate_rows)


def _cluster_rows(
    rows: np.ndarray, max_gap: int = const.LINE_CLUSTER_MAX_GAP
) -> list[int]:
    if rows.size == 0:
        return []

    centers = []
    start = int(rows[0])
    prev = start

    # consecutive rows within max_gap of each other belong to the same line; take their midpoint
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
    gap_count = const.LINES_PER_STAFF - 1
    i = 0

    while i + const.LINES_PER_STAFF <= len(line_centers):
        candidate = line_centers[i : i + const.LINES_PER_STAFF]
        gaps = [candidate[j + 1] - candidate[j] for j in range(gap_count)]
        mean_gap = sum(gaps) / gap_count

        if mean_gap <= 0:
            i += 1
            continue

        # real staff lines are evenly spaced; reject candidates where any gap deviates too much from the mean
        tolerance = max(
            const.STAFF_SPACING_TOLERANCE_MIN,
            mean_gap * const.STAFF_SPACING_TOLERANCE_FRAC,
        )
        if not all(abs(g - mean_gap) <= tolerance for g in gaps):
            i += 1
            continue

        lines = []
        for y in candidate:
            x0, x1 = _line_extent(line_mask, y)
            lines.append(StaffLine(y=y, x_start=x0, x_end=x1))

        # pad vertically so stems above/below the outermost lines are inside the staff bounding box
        pad = const.STAFF_VERTICAL_PADDING_FRAC * mean_gap
        top = max(0, int(candidate[0] - pad))
        bottom = min(shape[0] - 1, int(candidate[-1] + pad))

        staffs.append(Staff(lines=lines, spacing=mean_gap, top=top, bottom=bottom))
        i += const.LINES_PER_STAFF

    return staffs


def _line_extent(line_mask: MatLike, y: int, half_window: int = 1) -> tuple[int, int]:
    y0 = max(0, y - half_window)
    y1 = min(line_mask.shape[0], y + half_window + 1)
    cols = np.flatnonzero(np.any(line_mask[y0:y1, :] > const.MASK_BACKGROUND, axis=0))

    if cols.size == 0:
        return 0, line_mask.shape[1] - 1

    return int(cols[0]), int(cols[-1])


def erase_staff_for_bars(binary: MatLike, staffs: list[Staff]) -> MatLike:
    # use the global binary here because bar lines need hard vertical edges to survive
    horizontal = extract_horizontal_lines(binary)
    allowed = _staff_removal_band_mask(binary.shape, staffs)
    result = cv.subtract(binary, cv.bitwise_and(horizontal, allowed))
    return _repair_slits(result, staffs)


def erase_staff_for_notes(
    gray: MatLike, staffs: list[Staff]
) -> tuple[MatLike, MatLike]:
    """
    Erase staff lines for note detection.

    Returns:
        Tuple of (raw_adaptive_mask, processed_mask)
        - raw_adaptive_mask: Binary image after adaptive thresholding (staff lines intact)
        - processed_mask: After staff removal and slit repair
    """
    inverted = cv.bitwise_not(gray)
    # adaptive threshold handles uneven lighting across the page better than a global threshold
    bw = cv.adaptiveThreshold(
        inverted,
        const.MASK_FOREGROUND,
        cv.ADAPTIVE_THRESH_MEAN_C,
        cv.THRESH_BINARY,
        15,
        -2,
    )

    # reconstruct staff lines using open/close so we only erase actual line pixels, not nearby noteheads
    kernel_width = max(1, bw.shape[1] // 30)
    structure = cv.getStructuringElement(cv.MORPH_RECT, (kernel_width, 1))
    staff_reconstruction = cv.dilate(cv.erode(bw, structure), structure)

    allowed = _staff_removal_band_mask(bw.shape, staffs)
    result = cv.subtract(bw, cv.bitwise_and(staff_reconstruction, allowed))
    processed = _repair_slits(result, staffs)

    return bw, processed


def _staff_removal_band_mask(shape: tuple, staffs: list[Staff]) -> MatLike:
    # only allow erasure within a narrow band around each line so noteheads sitting near a line aren't wiped out
    h, w = shape[:2]
    mask = np.zeros((h, w), dtype=np.uint8)

    for staff in staffs:
        band = max(1, int(round(staff.spacing * const.STAFF_ERASE_BAND_FRAC)))
        for line in staff.lines:
            y0 = max(0, line.y - band)
            y1 = min(h, line.y + band + 1)
            x0 = max(0, line.x_start)
            x1 = min(w, line.x_end + 1)
            mask[y0:y1, x0:x1] = const.MASK_FOREGROUND

    return mask


def _repair_slits(ink: MatLike, staffs: list[Staff]) -> MatLike:
    """When a stem crosses a staff line, erasure leaves a horizontal gap through the notehead.
    A vertical close kernel inside the line band closes that gap so filled noteheads stay solid."""
    h, w = ink.shape[:2]
    repair_mask = np.zeros((h, w), dtype=np.uint8)

    for staff in staffs:
        band = max(1, int(round(staff.spacing * const.SLIT_REPAIR_BAND_FRAC)))
        for line in staff.lines:
            y0 = max(0, line.y - band)
            y1 = min(h, line.y + band + 1)
            x0 = max(0, line.x_start)
            x1 = min(w, line.x_end + 1)
            repair_mask[y0:y1, x0:x1] = const.MASK_FOREGROUND

    # 1-pixel-wide vertical kernel so we only close vertical gaps, not horizontal ones
    kernel_height = max(
        const.SLIT_REPAIR_KERNEL_MIN, min(const.SLIT_REPAIR_KERNEL_MAX, 3)
    )
    kernel = cv.getStructuringElement(cv.MORPH_RECT, (1, kernel_height))
    repaired = cv.morphologyEx(ink, cv.MORPH_CLOSE, kernel)

    # only apply the repair inside the band; leave everything else untouched
    return np.where(repair_mask > 0, repaired, ink).astype(ink.dtype)
