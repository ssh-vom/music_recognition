"""Visualization functions for all detection modules.

Centralized location for creating debug visualizations and saving artifacts.
All functions take detection results and produce visual outputs.
"""

import cv2 as cv
import numpy as np
from cv2.typing import MatLike

from schema import BarLine, ClefDetection, Note, Score, Staff


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


def _draw_connected_components(mask: MatLike, count: int, stats, centroids) -> MatLike:
    """Draw all connected components with unique colors.

    Returns color-coded visualization showing all detected blobs before filtering.
    """
    overlay = cv.cvtColor(cv.bitwise_not(mask), cv.COLOR_GRAY2BGR)

    colors = [
        (255, 0, 0),
        (0, 255, 0),
        (0, 0, 255),
        (255, 255, 0),
        (255, 0, 255),
        (0, 255, 255),
        (128, 255, 0),
        (255, 128, 0),
        (128, 0, 255),
        (0, 128, 255),
    ]

    for i in range(1, count):
        x = int(stats[i, cv.CC_STAT_LEFT])
        y = int(stats[i, cv.CC_STAT_TOP])
        w = int(stats[i, cv.CC_STAT_WIDTH])
        h = int(stats[i, cv.CC_STAT_HEIGHT])
        cx = int(round(centroids[i][0]))
        cy = int(round(centroids[i][1]))

        color = colors[i % len(colors)]
        cv.rectangle(overlay, (x, y), (x + w, y + h), color, 1)
        cv.circle(overlay, (cx, cy), 3, color, -1)
        cv.putText(
            overlay,
            str(i),
            (x, y - 2),
            cv.FONT_HERSHEY_SIMPLEX,
            0.4,
            color,
            1,
            cv.LINE_AA,
        )

    return overlay


def _draw_filtered_components(mask: MatLike, filtered_info: list[dict]) -> MatLike:
    """Draw components colored by whether they passed geometric filtering.

    Green = passed all filters
    Red = failed (with reason in label)
    """
    overlay = cv.cvtColor(cv.bitwise_not(mask), cv.COLOR_GRAY2BGR)

    for info in filtered_info:
        x = info["x"] - info["w"] // 2
        y = info["y"] - info["h"] // 2
        w = info["w"]
        h = info["h"]

        if info["passed"]:
            color = (0, 200, 0)  # Green
            label = f"{info['id']}: OK"
        else:
            color = (0, 0, 200)  # Red
            reasons = []
            area_ratio = info["area"] / (w * h) if w * h > 0 else 0
            if info["area"] < 100:  # Approximate min_area check
                reasons.append("area")
            if info["w"] < 10 or info["h"] < 10:  # Approximate size check
                reasons.append("size")
            if info["aspect"] < 0.45 or info["aspect"] > 2.2:
                reasons.append("aspect")
            label = f"{info['id']}: {','.join(reasons) if reasons else 'fail'}"

        cv.rectangle(overlay, (x, y), (x + w, y + h), color, 1)
        cv.circle(overlay, (info["x"], info["y"]), 3, color, -1)
        cv.putText(
            overlay,
            label,
            (x, y - 2),
            cv.FONT_HERSHEY_SIMPLEX,
            0.35,
            color,
            1,
            cv.LINE_AA,
        )

    return overlay


def _draw_merge_comparison(
    mask: MatLike,
    centers_before: list[tuple[int, int]],
    centers_after: list[list],
) -> MatLike:
    """Draw centers before and after merging.

    Shows original detections (blue) and merged result (green).
    """
    overlay = cv.cvtColor(cv.bitwise_not(mask), cv.COLOR_GRAY2BGR)

    # Draw original centers (before merge)
    for cx, cy in centers_before:
        cv.circle(overlay, (cx, cy), 4, (255, 0, 0), 1)  # Blue outline
        cv.circle(overlay, (cx, cy), 2, (255, 0, 0), -1)  # Blue fill

    # Draw merged centers (after merge)
    for cx, cy, count in centers_after:
        cv.circle(overlay, (cx, cy), 5, (0, 200, 0), 2)  # Green outline
        cv.circle(overlay, (cx, cy), 3, (0, 200, 0), -1)  # Green fill
        if count > 1:
            cv.putText(
                overlay,
                f"x{count}",
                (cx + 6, cy - 6),
                cv.FONT_HERSHEY_SIMPLEX,
                0.4,
                (0, 200, 0),
                1,
                cv.LINE_AA,
            )

    return overlay


def _draw_stem_augmentation(
    mask: MatLike,
    stem_info: dict,
    centers_before: list[list],
    centers_after: list[list],
) -> MatLike:
    """Draw stem detection and augmentation results.

    Shows:
    - All detected stems (red/yellow based on rejection reason)
    - Added noteheads (green)
    - Original detections (blue)
    """
    overlay = cv.cvtColor(cv.bitwise_not(mask), cv.COLOR_GRAY2BGR)

    # Draw original centers
    for cx, cy, _ in centers_before:
        cv.circle(overlay, (cx, cy), 3, (255, 0, 0), -1)  # Blue

    # Draw all stems found
    if "all_stems" in stem_info and stem_info["all_stems"]:
        for stem in stem_info["all_stems"]:
            x, y, w, h = stem["x"], stem["y"], stem["w"], stem["h"]

            if stem.get("added"):
                color = (0, 200, 0)  # Green for added
                thickness = 2
            elif "rejected" in stem:
                color = (0, 200, 255)  # Yellow for rejected
                thickness = 1
            else:
                color = (0, 0, 200)  # Red for other
                thickness = 1

            cv.rectangle(overlay, (x, y), (x + w, y + h), color, thickness)

            if stem.get("added"):
                note_x = stem.get("note_x", x + w // 2)
                note_y = stem.get("note_y", y + h - 5)
                cv.circle(overlay, (note_x, note_y), 4, (0, 200, 0), -1)
                cv.line(overlay, (x + w // 2, y), (note_x, note_y), (0, 200, 0), 1)

    # Draw final centers after augmentation
    for cx, cy, _ in centers_after:
        cv.circle(overlay, (cx, cy), 5, (0, 200, 0), 2)  # Green outline

    return overlay


def save_notes_visualization(
    notes_mask: MatLike,
    score: Score,
    artifacts,
    intermediates_by_measure: dict[tuple[int, int], dict] | None = None,
) -> dict:
    """Save note detection visualization artifacts with intermediate steps.

    Creates:
    01: Input mask (notes mask - staff erased)
    02: Morphological processing intermediates (per measure)
    03: Connected components (per measure)
    04: Geometric filtering (per measure)
    05: Merge comparison (per measure)
    06: Stem augmentation (per measure)
    07: Final notes overlay (per measure)
    08: Full sheet notes overlay
    """
    paths = {}

    # 01: Input mask (notes mask - staff erased, inverted for display)
    if notes_mask is not None:
        notes_mask_display = cv.bitwise_not(notes_mask)
        paths["01_notes_mask"] = artifacts.write_image(
            artifacts.sections.masks, "01_notes_mask.jpg", notes_mask_display
        )

    # Create directories for intermediate visualizations
    morph_dir = artifacts.ensure_subdir(artifacts.sections.notes, "02_morphological")
    cc_dir = artifacts.ensure_subdir(
        artifacts.sections.notes, "03_connected_components"
    )
    filter_dir = artifacts.ensure_subdir(
        artifacts.sections.notes, "04_geometric_filtering"
    )
    merge_dir = artifacts.ensure_subdir(artifacts.sections.notes, "05_merge_comparison")
    stem_dir = artifacts.ensure_subdir(artifacts.sections.notes, "06_stem_augmentation")
    final_dir = artifacts.ensure_subdir(artifacts.sections.notes, "07_final_notes")

    # Process each staff and its measures
    for staff_idx, staff in enumerate(score.staffs):
        staff_measures = score.get_measures_for_staff(staff_idx)

        for measure_idx, measure in enumerate(staff_measures):
            key = (staff_idx, measure_idx)

            if measure.crop is None:
                continue

            mask = measure.crop

            # Check if we have intermediates for this measure
            intermediates = (
                intermediates_by_measure.get(key) if intermediates_by_measure else None
            )

            # 02: Morphological processing
            if intermediates and "opened_mask" in intermediates:
                opened_display = cv.bitwise_not(intermediates["opened_mask"])
                cv.imwrite(
                    str(
                        morph_dir
                        / f"staff_{staff_idx}_measure_{measure_idx}_opened.jpg"
                    ),
                    opened_display,
                )
            if intermediates and "notehead_mask" in intermediates:
                notehead_display = cv.bitwise_not(intermediates["notehead_mask"])
                cv.imwrite(
                    str(
                        morph_dir
                        / f"staff_{staff_idx}_measure_{measure_idx}_notehead.jpg"
                    ),
                    notehead_display,
                )

            # 03: Connected components
            if intermediates and "connected_components" in intermediates:
                cc_data = intermediates["connected_components"]
                cc_overlay = _draw_connected_components(
                    mask,
                    cc_data["count"],
                    cc_data["stats"],
                    cc_data["centroids"],
                )
                cv.imwrite(
                    str(cc_dir / f"staff_{staff_idx}_measure_{measure_idx}.jpg"),
                    cc_overlay,
                )

            # 04: Geometric filtering
            if intermediates and "filtered_components" in intermediates:
                filter_overlay = _draw_filtered_components(
                    mask, intermediates["filtered_components"]
                )
                cv.imwrite(
                    str(filter_dir / f"staff_{staff_idx}_measure_{measure_idx}.jpg"),
                    filter_overlay,
                )

            # 05: Merge comparison
            if (
                intermediates
                and "raw_centers_before_merge" in intermediates
                and "centers_after_merge" in intermediates
            ):
                merge_overlay = _draw_merge_comparison(
                    mask,
                    intermediates["raw_centers_before_merge"],
                    intermediates["centers_after_merge"],
                )
                cv.imwrite(
                    str(merge_dir / f"staff_{staff_idx}_measure_{measure_idx}.jpg"),
                    merge_overlay,
                )

            # 06: Stem augmentation
            if intermediates and "stem_augmentation" in intermediates:
                stem_overlay = _draw_stem_augmentation(
                    mask,
                    intermediates["stem_augmentation"],
                    intermediates.get("centers_after_merge", []),
                    intermediates.get("centers_after_stems", []),
                )
                cv.imwrite(
                    str(stem_dir / f"staff_{staff_idx}_measure_{measure_idx}.jpg"),
                    stem_overlay,
                )

            # 07: Final notes overlay
            measure_notes = score.get_notes_for_measure(staff_idx, measure_idx)
            final_overlay = draw_notes_on_mask(mask, measure_notes)
            cv.imwrite(
                str(final_dir / f"staff_{staff_idx}_measure_{measure_idx}.jpg"),
                final_overlay,
            )

    # Store directory paths
    paths["02_morphological"] = morph_dir
    paths["03_connected_components"] = cc_dir
    paths["04_geometric_filtering"] = filter_dir
    paths["05_merge_comparison"] = merge_dir
    paths["06_stem_augmentation"] = stem_dir
    paths["07_final_notes"] = final_dir

    # 08: Full sheet notes overlay
    full_overlay = _draw_full_notes_overlay(score)
    paths["08_full_notes_overlay"] = artifacts.write_image(
        artifacts.sections.notes, "08_full_notes_overlay.jpg", full_overlay
    )

    return paths


def _draw_full_notes_overlay(score: Score) -> MatLike:
    """Draw notes overlay on the full sheet image."""
    out = score.sheet_image.copy()
    font = cv.FONT_HERSHEY_SIMPLEX

    confidence_color = {
        "high": (0, 180, 0),
        "medium": (0, 180, 220),
        "low": (0, 80, 255),
    }

    for staff_idx, staff in enumerate(score.staffs):
        staff_measures = score.get_measures_for_staff(staff_idx)
        for measure_idx, measure in enumerate(staff_measures):
            cv.rectangle(
                out,
                (measure.x_start, measure.y_top),
                (measure.x_end - 1, measure.y_bottom),
                (120, 120, 120),
                1,
            )
            measure_notes = score.get_notes_for_measure(staff_idx, measure_idx)
            for note in measure_notes:
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


# =============================================================================
# Clef Detection Visualization
# =============================================================================


def draw_clef_overlay(
    clef_crop: MatLike, detection: ClefDetection, clef_kind: str | None
) -> MatLike:
    """Draw clef detection overlay on the clef crop image.

    Shows template matching boxes and detection scores.
    """
    if len(clef_crop.shape) == 2:
        overlay = cv.cvtColor(cv.bitwise_not(clef_crop), cv.COLOR_GRAY2BGR)
    else:
        overlay = clef_crop.copy()

    if clef_kind is None or detection is None:
        return overlay

    # Draw crop boundary
    x1, y1 = 0, 0
    x2, y2 = overlay.shape[1], overlay.shape[0]
    cv.rectangle(overlay, (x1, y1), (x2, y2), (100, 100, 100), 1)

    # Draw template matching boxes
    choice = _choose_clef_overlay_rect(clef_kind, detection)
    if choice is not None:
        rect, color = choice
        _draw_overlay_box(overlay, rect, color, thickness=3)

    # Add label with clef type and scores
    name = clef_kind if clef_kind else "?"
    label = f"{name}  T={detection.letter_score_treble:.2f} B={detection.letter_score_bass:.2f}"
    cv.putText(
        overlay,
        label,
        (6, 22),
        cv.FONT_HERSHEY_SIMPLEX,
        0.55,
        (30, 30, 30),
        2,
        cv.LINE_AA,
    )

    return overlay


def _choose_clef_overlay_rect(clef_kind: str, detection: ClefDetection):
    """Choose which template match box to draw for clef overlay."""
    if (
        clef_kind == "treble"
        and detection.treble_match_top_left is not None
        and detection.treble_match_size is not None
    ):
        x, y = detection.treble_match_top_left
        w, h = detection.treble_match_size
        return (x, y, w, h), (0, 200, 100)  # Green for treble

    if (
        clef_kind == "bass"
        and detection.bass_match_top_left is not None
        and detection.bass_match_size is not None
    ):
        x, y = detection.bass_match_top_left
        w, h = detection.bass_match_size
        return (x, y, w, h), (0, 120, 255)  # Orange for bass

    if (
        detection.treble_match_top_left is not None
        and detection.treble_match_size is not None
    ):
        x, y = detection.treble_match_top_left
        w, h = detection.treble_match_size
        return (x, y, w, h), (180, 180, 180)  # Gray for undetermined

    return None


def _draw_overlay_box(image: MatLike, rect: tuple, color: tuple, *, thickness: int = 3):
    """Draw a rectangle on the image."""
    x, y, w, h = rect
    if w < 2 or h < 2:
        return
    cv.rectangle(
        image,
        (x, y),
        (x + w - 1, y + h - 1),
        color,
        thickness,
    )


# =============================================================================
# Measure Splitting Visualization
# =============================================================================


def draw_measure_boundaries(image: MatLike, measures_map: dict[int, list]) -> MatLike:
    """Draw measure boundaries on the sheet image.

    Draws blue rectangles around each detected measure region.
    """
    overlay = image.copy()

    for staff_index, measures in measures_map.items():
        for measure in measures:
            cv.rectangle(
                overlay,
                (measure.x_start, measure.y_top),
                (measure.x_end - 1, measure.y_bottom),
                (255, 0, 0),  # Blue
                1,
            )

    return overlay


def save_measure_visualization(
    sheet_image: MatLike,
    measures_map: dict[int, list],
    measure_crops: dict[int, list[MatLike]],
    artifacts,
) -> dict:
    """Save measure splitting visualization artifacts.

    Creates:
    01_measure_boundaries - Full sheet with measure rectangles
    02_measure_crops/staff_X/measure_Y.jpg - Individual cropped measures
    """
    paths = {}

    # 01: Measure boundaries overlay on full sheet
    overlay = draw_measure_boundaries(sheet_image, measures_map)
    paths["01_measure_boundaries"] = artifacts.write_image(
        artifacts.sections.pipeline, "01_measure_boundaries.jpg", overlay
    )

    # 02: Individual measure crops organized by staff
    crops_dir = artifacts.ensure_subdir(artifacts.sections.pipeline, "02_measure_crops")
    for staff_index, crops in measure_crops.items():
        staff_dir = crops_dir / f"staff_{staff_index}"
        staff_dir.mkdir(exist_ok=True)
        for measure_index, crop in enumerate(crops):
            if len(crop.shape) == 2:
                crop_display = cv.bitwise_not(crop)
            else:
                crop_display = crop
            crop_path = staff_dir / f"measure_{measure_index}.jpg"
            cv.imwrite(str(crop_path), crop_display)
    paths["02_measure_crops"] = crops_dir

    return paths


def save_clef_visualization(
    clef_key_crops: dict[int, MatLike],
    clefs_by_staff: dict,
    clef_detections: dict[int, ClefDetection],
    artifacts,
) -> dict:
    """Save clef detection visualization artifacts.

    Creates:
    01_clef_header_crops - Individual clef+key crops per staff
    02_detection_staff_X - Detection overlay for each staff
    """
    paths = {}

    # 01: Save individual clef header crops
    clef_crops_dir = artifacts.ensure_subdir(
        artifacts.sections.clef, "01_clef_header_crops"
    )
    for staff_index, crop in clef_key_crops.items():
        crop_path = clef_crops_dir / f"staff_{staff_index}.jpg"
        if len(crop.shape) == 2:
            display_crop = cv.bitwise_not(crop)
        else:
            display_crop = crop
        cv.imwrite(str(crop_path), display_crop)
    paths["01_clef_header_crops"] = clef_crops_dir

    # 02: Detection overlays for each staff
    for staff_index, crop in clef_key_crops.items():
        clef = clefs_by_staff.get(staff_index)
        detection = clef_detections.get(staff_index)
        clef_kind = clef.kind if clef else None

        if detection is not None:
            overlay = draw_clef_overlay(crop, detection, clef_kind)
            overlay_path = artifacts.write_image(
                artifacts.sections.clef,
                f"02_detection_staff_{staff_index}.jpg",
                overlay,
            )
            paths[f"02_detection_staff_{staff_index}"] = overlay_path

    return paths


# =============================================================================
# Accidental Detection Visualization
# =============================================================================


def draw_accidentals_overlay(image: MatLike, accidentals: list) -> MatLike:
    """Draw detected accidentals (sharps/flats) on image.

    Uses cross markers for visibility:
    - Magenta (#FF00FF) for sharps
    - Orange (#FF8000) for flats
    """
    out = image.copy()

    for acc in accidentals:
        # Color based on accidental type
        color = (255, 0, 255) if acc.kind == "sharp" else (255, 128, 0)

        # Draw cross marker at accidental center
        cv.drawMarker(
            out,
            (acc.center_x, acc.center_y),
            color,
            markerType=cv.MARKER_CROSS,
            markerSize=10,
            thickness=1,
            line_type=cv.LINE_AA,
        )

        # Add text label (S for sharp, F for flat)
        label = acc.kind[0].upper()
        cv.putText(
            out,
            label,
            (acc.center_x + 6, acc.center_y + 4),
            cv.FONT_HERSHEY_SIMPLEX,
            0.4,
            color,
            1,
            cv.LINE_AA,
        )

    return out


def save_accidental_visualization(
    image: MatLike,
    accidentals: list,
    artifacts,
) -> dict:
    """Save accidental detection visualization.

    Creates overlay showing detected sharps and flats.
    """
    paths = {}

    overlay = draw_accidentals_overlay(image, accidentals)
    paths["accidentals_overlay"] = artifacts.write_image(
        artifacts.sections.notes, "accidentals_overlay.jpg", overlay
    )

    return paths
