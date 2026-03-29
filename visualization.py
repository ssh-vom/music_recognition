"""Visualization functions for all detection modules.

Centralized location for creating debug visualizations and saving artifacts.
All functions take detection results and produce visual outputs.
"""

import cv2 as cv
from cv2.typing import MatLike

from schema import BarLine, Staff


def draw_staff_overlay(image: MatLike, staffs: list[Staff]) -> MatLike:
    """Draw detected staff lines and bounding boxes on image."""
    if len(image.shape) == 2:
        overlay = cv.cvtColor(image, cv.COLOR_GRAY2BGR)
    else:
        overlay = image.copy()

    line_color = (0, 255, 0)
    box_color = (255, 0, 0)
    label_color = (0, 0, 255)

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

    # 02: Otsu binary (inverted)
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


def draw_bars_overlay(image: MatLike, bars: list[BarLine]) -> MatLike:
    """Draw detected bar lines on image."""
    overlay = image.copy()
    color = (0, 0, 255)  # Blue

    for bar in bars:
        cv.line(overlay, (bar.x, bar.y_top), (bar.x, bar.y_bottom), color, 1)

    return overlay


def save_bar_visualization(
    image: MatLike,
    bars: list[BarLine],
    artifacts,
) -> dict:
    """Save bar detection visualization artifacts."""
    paths = {}

    # Input mask (inverted)
    paths["01_input_mask"] = artifacts.write_image(
        artifacts.sections.bars, "01_input_mask.jpg", cv.bitwise_not(image)
    )

    # Final overlay
    overlay = draw_bars_overlay(image, bars)
    paths["02_bar_overlay"] = artifacts.write_image(
        artifacts.sections.bars, "02_bar_overlay.jpg", overlay
    )

    return paths
