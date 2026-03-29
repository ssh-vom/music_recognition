"""Visualization functions for all detection modules.

Centralized location for creating debug visualizations and saving artifacts.
All functions take detection results and produce visual outputs.
"""

import cv2 as cv
import numpy as np
from cv2.typing import MatLike

from schema import BarLine, Note, Staff


# =============================================================================
# Staff Detection Visualization
# =============================================================================


def draw_staff_overlay(image: MatLike, staffs: list[Staff]) -> MatLike:
    """Draw detected staff lines and bounding boxes on image."""
    if len(image.shape) == 2:
        overlay = cv.cvtColor(image, cv.COLOR_GRAY2BGR)
    else:
        overlay = image.copy()

    line_color = (0, 255, 0)  # Green for staff lines
    box_color = (255, 0, 0)  # Blue for staff boundaries
    label_color = (0, 0, 255)  # Red for labels

    for idx, staff in enumerate(staffs):
        # Draw lines
        for line in staff.lines:
            cv.line(
                overlay,
                (line.x_start, line.y),
                (line.x_end, line.y),
                line_color,
                1,
            )

        # Draw bounding box
        cv.rectangle(
            overlay,
            (0, staff.top),
            (overlay.shape[1] - 1, staff.bottom),
            box_color,
            1,
        )

        # Label
        label = f"staff {idx}  spacing={staff.spacing:.1f}px"
        label_y = max(15, staff.top - 5)
        cv.putText(
            overlay,
            label,
            (10, label_y),
            cv.FONT_HERSHEY_SIMPLEX,
            0.5,
            label_color,
            1,
            cv.LINE_AA,
        )

    return overlay


def save_staff_detection(
    image: MatLike,
    gray: MatLike,
    binary: MatLike,
    line_mask: MatLike,
    staffs: list[Staff],
    artifacts,
) -> dict:
    """Save all staff detection visualization artifacts."""
    paths = {}

    # 01: Grayscale
    paths["01_grayscale"] = artifacts.write_image(
        artifacts.sections.staff, "01_grayscale.jpg", gray
    )

    # 02: Otsu binary (inverted so staff is white)
    binary_display = cv.bitwise_not(binary)
    paths["02_otsu_binary"] = artifacts.write_image(
        artifacts.sections.staff, "02_otsu_binary.jpg", binary_display
    )

    # 03: Horizontal line mask (inverted)
    line_mask_display = cv.bitwise_not(line_mask)
    paths["03_horizontal_lines"] = artifacts.write_image(
        artifacts.sections.staff, "03_horizontal_lines.jpg", line_mask_display
    )

    # 04: Final overlay
    overlay = draw_staff_overlay(image, staffs)
    paths["04_staff_overlay"] = artifacts.write_image(
        artifacts.sections.staff, "04_staff_overlay.jpg", overlay
    )

    return paths


# =============================================================================
# Bar Detection Visualization
# =============================================================================


def draw_bars_overlay(image: MatLike, bars: list[BarLine]) -> MatLike:
    """Draw detected bar lines on the original image.

    Draws vertical lines at detected bar positions across staff height.
    """
    overlay = image.copy()
    bar_color = (0, 0, 255)  # Red in BGR for visibility

    for bar in bars:
        cv.line(
            overlay,
            (bar.x, bar.y_top),
            (bar.x, bar.y_bottom),
            bar_color,
            2,  # Thickness 2 for visibility
        )

    return overlay


def _get_bar_processing_intermediates(
    image: MatLike,
    first_staff: Staff,
) -> tuple[MatLike | None, MatLike | None]:
    """Get intermediate processing steps for bar detection visualization.

    Returns:
        (joined_mask, contours_viz) - morphological result and contour overlay
    """
    # Extract first staff ROI to show processing
    y0, y1 = first_staff.top, first_staff.bottom + 1
    roi = image[y0:y1, :]

    # Skip left header area
    left_skip = int(round(5.0 * first_staff.spacing))
    work = roi[:, left_skip:]

    if work.size == 0:
        return None, None

    # Vertical close (same as in detection)
    kernel_h = max(5, int(round(2.0 * first_staff.spacing)))
    close_kernel = cv.getStructuringElement(cv.MORPH_RECT, (1, kernel_h))
    joined = cv.morphologyEx(work, cv.MORPH_CLOSE, close_kernel)

    # Find contours
    contours, _ = cv.findContours(joined, cv.RETR_EXTERNAL, cv.CHAIN_APPROX_SIMPLE)

    # Create contour overlay on original
    contour_viz = (
        cv.cvtColor(work, cv.COLOR_GRAY2BGR) if len(work.shape) == 2 else work.copy()
    )
    cv.drawContours(contour_viz, contours, -1, (0, 255, 0), 1)

    return joined, contour_viz


def save_bar_visualization(
    image: MatLike,
    bars_mask: MatLike,
    staffs: list[Staff],
    bars: list[BarLine],
    artifacts,
) -> dict:
    """Save all bar detection visualization artifacts with intermediate steps.

    Creates:
    01_input_mask - The binary mask used for detection (inverted for display)
    02_vertical_close - Morphological closing result on first staff
    03_contours - Detected contours on first staff
    04_bar_overlay - Final bar lines overlaid on original image
    """
    paths = {}

    # 01: Input mask (the binary mask used for detection, inverted so black ink on white)
    if len(bars_mask.shape) == 2:
        input_display = cv.bitwise_not(bars_mask)
    else:
        input_display = bars_mask.copy()
    paths["01_input_mask"] = artifacts.write_image(
        artifacts.sections.bars, "01_input_mask.jpg", input_display
    )

    # 02 & 03: Processing intermediates on first staff
    if staffs:
        first_staff = staffs[0]
        joined, contour_viz = _get_bar_processing_intermediates(bars_mask, first_staff)

        if joined is not None:
            # 02: Vertical close result (inverted for display)
            joined_display = cv.bitwise_not(joined)
            paths["02_vertical_close"] = artifacts.write_image(
                artifacts.sections.bars, "02_vertical_close.jpg", joined_display
            )

            # 03: Contours visualization
            paths["03_contours"] = artifacts.write_image(
                artifacts.sections.bars, "03_contours.jpg", contour_viz
            )

    # 04: Final overlay on original image (not the mask)
    overlay = draw_bars_overlay(image, bars)
    paths["04_bar_overlay"] = artifacts.write_image(
        artifacts.sections.bars, "04_bar_overlay.jpg", overlay
    )

    return paths


# =============================================================================
# Note Detection Visualization
# =============================================================================


def draw_notes_on_mask(mask: MatLike, notes: list[Note]) -> MatLike:
    """Draw notes on a measure mask for visualization."""
    overlay = cv.cvtColor(cv.bitwise_not(mask), cv.COLOR_GRAY2BGR)

    for note in notes:
        center = (note.center_x, note.center_y)
        cv.circle(overlay, center, 3, (0, 0, 255), 1)

        conf = note.step_confidence if note.step_confidence else "?"
        pitch = (
            f"{note.pitch_letter}{note.octave}"
            if note.pitch_letter and note.octave
            else "?"
        )
        dur = note.duration_class if note.duration_class else "?"
        label = f"{note.step} {conf} {pitch} {dur}"

        cv.putText(
            overlay,
            label,
            (note.center_x + 4, note.center_y - 4),
            cv.FONT_HERSHEY_SIMPLEX,
            0.35,
            (0, 255, 0),
            1,
            cv.LINE_AA,
        )

    return overlay


def save_notes_visualization(
    notes_mask: MatLike,
    score_tree,
    artifacts,
) -> dict:
    """Save note detection visualization artifacts.

    Creates overlays for each measure and full sheet.
    """
    paths = {}

    # 01: Input mask (notes mask - staff erased, inverted for display)
    if notes_mask is not None:
        notes_mask_display = cv.bitwise_not(notes_mask)
        paths["01_notes_mask"] = artifacts.write_image(
            artifacts.sections.masks, "01_notes_mask.jpg", notes_mask_display
        )

    # 02: Individual measure overlays
    measure_dir = artifacts.ensure_subdir(
        artifacts.sections.notes, "02_measure_overlays"
    )
    for staff_node in score_tree.staff_nodes:
        staff_dir = measure_dir / f"staff_{staff_node.index}"
        staff_dir.mkdir(exist_ok=True)
        for measure_node in staff_node.measures:
            if measure_node.crop is not None:
                overlay = draw_notes_on_mask(measure_node.crop, measure_node.notes)
                overlay_path = staff_dir / f"measure_{measure_node.index}.jpg"
                cv.imwrite(str(overlay_path), overlay)
    paths["02_measure_overlays"] = measure_dir

    # 03: Full sheet notes overlay
    full_overlay = _draw_full_notes_overlay(score_tree)
    paths["03_full_notes_overlay"] = artifacts.write_image(
        artifacts.sections.notes, "03_full_notes_overlay.jpg", full_overlay
    )

    return paths


def _draw_full_notes_overlay(score_tree) -> MatLike:
    """Draw notes overlay on the full sheet image."""
    out = score_tree.sheet_image.copy()
    font = cv.FONT_HERSHEY_SIMPLEX

    confidence_color = {
        "high": (0, 180, 0),
        "medium": (0, 180, 220),
        "low": (0, 80, 255),
    }

    for staff_node in score_tree.staff_nodes:
        for measure_node in staff_node.measures:
            measure = measure_node.measure
            cv.rectangle(
                out,
                (measure.x_start, measure.y_top),
                (measure.x_end - 1, measure.y_bottom),
                (120, 120, 120),
                1,
            )
            for note in measure_node.notes:
                abs_x = measure.x_start + note.center_x
                abs_y = measure.y_top + note.center_y
                color = confidence_color.get(
                    note.step_confidence or "unknown", (160, 160, 160)
                )
                cv.circle(out, (abs_x, abs_y), 4, color, 2)
                pitch_label = (
                    f"{note.pitch_letter}{note.octave}"
                    if note.pitch_letter is not None and note.octave is not None
                    else "?"
                )
                duration_label = note.duration_class if note.duration_class else "?"
                label = f"{note.step} {pitch_label} {duration_label}"
                cv.putText(
                    out,
                    label,
                    (abs_x + 5, abs_y - 5),
                    font,
                    0.35,
                    color,
                    1,
                    cv.LINE_AA,
                )

    return out
