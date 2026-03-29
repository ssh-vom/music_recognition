"""Visualization functions for all detection modules."""

import cv2 as cv
import numpy as np
from cv2.typing import MatLike

from schema import BarLine, ClefDetection, Note, Score, Staff


# Staff Detection Visualization


def draw_staff_overlay(image: MatLike, staffs: list[Staff]) -> MatLike:
    if len(image.shape) == 2:
        overlay = cv.cvtColor(image, cv.COLOR_GRAY2BGR)
    else:
        overlay = image.copy()

    line_color = (0, 255, 0)
    box_color = (255, 0, 0)
    label_color = (0, 0, 255)

    for idx, staff in enumerate(staffs):
        for line in staff.lines:
            cv.line(
                overlay,
                (line.x_start, line.y),
                (line.x_end, line.y),
                line_color,
                1,
            )

        cv.rectangle(
            overlay,
            (0, staff.top),
            (overlay.shape[1] - 1, staff.bottom),
            box_color,
            1,
        )

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
    paths = {}

    paths["01_grayscale"] = artifacts.write_image(
        artifacts.sections.staff, "01_grayscale.jpg", gray
    )

    binary_display = cv.bitwise_not(binary)
    paths["02_otsu_binary"] = artifacts.write_image(
        artifacts.sections.staff, "02_otsu_binary.jpg", binary_display
    )

    line_mask_display = cv.bitwise_not(line_mask)
    paths["03_horizontal_lines"] = artifacts.write_image(
        artifacts.sections.staff, "03_horizontal_lines.jpg", line_mask_display
    )

    overlay = draw_staff_overlay(image, staffs)
    paths["04_staff_overlay"] = artifacts.write_image(
        artifacts.sections.staff, "04_staff_overlay.jpg", overlay
    )

    return paths


# Bar Detection Visualization


def draw_bars_overlay(image: MatLike, bars: list[BarLine]) -> MatLike:
    overlay = image.copy()
    bar_color = (0, 0, 255)

    for bar in bars:
        cv.line(
            overlay,
            (bar.x, bar.y_top),
            (bar.x, bar.y_bottom),
            bar_color,
            2,
        )

    return overlay


def _get_bar_processing_intermediates(
    image: MatLike,
    first_staff: Staff,
) -> tuple[MatLike | None, MatLike | None]:
    y0, y1 = first_staff.top, first_staff.bottom + 1
    roi = image[y0:y1, :]

    left_skip = int(round(5.0 * first_staff.spacing))
    work = roi[:, left_skip:]

    if work.size == 0:
        return None, None

    kernel_h = max(5, int(round(2.0 * first_staff.spacing)))
    close_kernel = cv.getStructuringElement(cv.MORPH_RECT, (1, kernel_h))
    joined = cv.morphologyEx(work, cv.MORPH_CLOSE, close_kernel)

    contours, _ = cv.findContours(joined, cv.RETR_EXTERNAL, cv.CHAIN_APPROX_SIMPLE)

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
    paths = {}

    if len(bars_mask.shape) == 2:
        input_display = cv.bitwise_not(bars_mask)
    else:
        input_display = bars_mask.copy()
    paths["01_input_mask"] = artifacts.write_image(
        artifacts.sections.bars, "01_input_mask.jpg", input_display
    )

    if staffs:
        first_staff = staffs[0]
        joined, contour_viz = _get_bar_processing_intermediates(bars_mask, first_staff)

        if joined is not None:
            joined_display = cv.bitwise_not(joined)
            paths["02_vertical_close"] = artifacts.write_image(
                artifacts.sections.bars, "02_vertical_close.jpg", joined_display
            )

            paths["03_contours"] = artifacts.write_image(
                artifacts.sections.bars, "03_contours.jpg", contour_viz
            )

    overlay = draw_bars_overlay(image, bars)
    paths["04_bar_overlay"] = artifacts.write_image(
        artifacts.sections.bars, "04_bar_overlay.jpg", overlay
    )

    return paths


# Note Detection Visualization


def draw_notes_on_mask(mask: MatLike, notes: list[Note]) -> MatLike:
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
    overlay = cv.cvtColor(cv.bitwise_not(mask), cv.COLOR_GRAY2BGR)

    for info in filtered_info:
        x = info["x"] - info["w"] // 2
        y = info["y"] - info["h"] // 2
        w = info["w"]
        h = info["h"]

        if info["passed"]:
            color = (0, 200, 0)
            label = f"{info['id']}: OK"
        else:
            color = (0, 0, 200)
            reasons = []
            area_ratio = info["area"] / (w * h) if w * h > 0 else 0
            if info["area"] < 100:
                reasons.append("area")
            if info["w"] < 10 or info["h"] < 10:
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
    overlay = cv.cvtColor(cv.bitwise_not(mask), cv.COLOR_GRAY2BGR)

    for cx, cy in centers_before:
        cv.circle(overlay, (cx, cy), 4, (255, 0, 0), 1)
        cv.circle(overlay, (cx, cy), 2, (255, 0, 0), -1)

    for cx, cy, count in centers_after:
        cv.circle(overlay, (cx, cy), 5, (0, 200, 0), 2)
        cv.circle(overlay, (cx, cy), 3, (0, 200, 0), -1)
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
    overlay = cv.cvtColor(cv.bitwise_not(mask), cv.COLOR_GRAY2BGR)

    for cx, cy, _ in centers_before:
        cv.circle(overlay, (cx, cy), 3, (255, 0, 0), -1)

    if "all_stems" in stem_info and stem_info["all_stems"]:
        for stem in stem_info["all_stems"]:
            x, y, w, h = stem["x"], stem["y"], stem["w"], stem["h"]

            if stem.get("added"):
                color = (0, 200, 0)
                thickness = 2
            elif "rejected" in stem:
                color = (0, 200, 255)
                thickness = 1
            else:
                color = (0, 0, 200)
                thickness = 1

            cv.rectangle(overlay, (x, y), (x + w, y + h), color, thickness)

            if stem.get("added"):
                note_x = stem.get("note_x", x + w // 2)
                note_y = stem.get("note_y", y + h - 5)
                cv.circle(overlay, (note_x, note_y), 4, (0, 200, 0), -1)
                cv.line(overlay, (x + w // 2, y), (note_x, note_y), (0, 200, 0), 1)

    for cx, cy, _ in centers_after:
        cv.circle(overlay, (cx, cy), 5, (0, 200, 0), 2)

    return overlay


def save_notes_visualization(
    notes_mask: MatLike,
    score: Score,
    artifacts,
    intermediates_by_measure: dict[tuple[int, int], dict] | None = None,
) -> dict:
    paths = {}

    if notes_mask is not None:
        notes_mask_display = cv.bitwise_not(notes_mask)
        paths["01_notes_mask"] = artifacts.write_image(
            artifacts.sections.masks, "01_notes_mask.jpg", notes_mask_display
        )

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

    for staff_idx, staff in enumerate(score.staffs):
        staff_measures = score.get_measures_for_staff(staff_idx)

        for measure_idx, measure in enumerate(staff_measures):
            key = (staff_idx, measure_idx)

            if measure.crop is None:
                continue

            mask = measure.crop
            intermediates = (
                intermediates_by_measure.get(key) if intermediates_by_measure else None
            )

            if intermediates and "opened_mask" in intermediates:
                opened_display = cv.bitwise_not(intermediates["opened_mask"])
                cv.imwrite(
                    str(morph_dir / f"staff_{staff_idx}_measure_{measure_idx}_opened.jpg"),
                    opened_display,
                )
            if intermediates and "notehead_mask" in intermediates:
                notehead_display = cv.bitwise_not(intermediates["notehead_mask"])
                cv.imwrite(
                    str(morph_dir / f"staff_{staff_idx}_measure_{measure_idx}_notehead.jpg"),
                    notehead_display,
                )

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

            if intermediates and "filtered_components" in intermediates:
                filter_overlay = _draw_filtered_components(
                    mask, intermediates["filtered_components"]
                )
                cv.imwrite(
                    str(filter_dir / f"staff_{staff_idx}_measure_{measure_idx}.jpg"),
                    filter_overlay,
                )

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

            measure_notes = score.get_notes_for_measure(staff_idx, measure_idx)
            final_overlay = draw_notes_on_mask(mask, measure_notes)
            cv.imwrite(
                str(final_dir / f"staff_{staff_idx}_measure_{measure_idx}.jpg"),
                final_overlay,
            )

    paths["02_morphological"] = morph_dir
    paths["03_connected_components"] = cc_dir
    paths["04_geometric_filtering"] = filter_dir
    paths["05_merge_comparison"] = merge_dir
    paths["06_stem_augmentation"] = stem_dir
    paths["07_final_notes"] = final_dir

    full_overlay = _draw_full_notes_overlay(score)
    paths["08_full_notes_overlay"] = artifacts.write_image(
        artifacts.sections.notes, "08_full_notes_overlay.jpg", full_overlay
    )

    return paths


def _draw_full_notes_overlay(score: Score) -> MatLike:
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
                duration_short = {
                    "whole": "w",
                    "half": "h", 
                    "quarter": "q",
                    "eighth": "8",
                    "sixteenth": "16",
                }.get(duration_label, duration_label)
                label = f"{note.step} {pitch_label} {duration_short}"
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


# Clef Detection Visualization


def draw_clef_overlay(
    clef_crop: MatLike, detection: ClefDetection, clef_kind: str | None
) -> MatLike:
    if len(clef_crop.shape) == 2:
        overlay = cv.cvtColor(cv.bitwise_not(clef_crop), cv.COLOR_GRAY2BGR)
    else:
        overlay = clef_crop.copy()

    if clef_kind is None or detection is None:
        return overlay

    x1, y1 = 0, 0
    x2, y2 = overlay.shape[1], overlay.shape[0]
    cv.rectangle(overlay, (x1, y1), (x2, y2), (100, 100, 100), 1)

    choice = _choose_clef_overlay_rect(clef_kind, detection)
    if choice is not None:
        rect, color = choice
        _draw_overlay_box(overlay, rect, color, thickness=3)

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
    if (
        clef_kind == "treble"
        and detection.treble_match_top_left is not None
        and detection.treble_match_size is not None
    ):
        x, y = detection.treble_match_top_left
        w, h = detection.treble_match_size
        return (x, y, w, h), (0, 200, 100)

    if (
        clef_kind == "bass"
        and detection.bass_match_top_left is not None
        and detection.bass_match_size is not None
    ):
        x, y = detection.bass_match_top_left
        w, h = detection.bass_match_size
        return (x, y, w, h), (0, 120, 255)

    if (
        detection.treble_match_top_left is not None
        and detection.treble_match_size is not None
    ):
        x, y = detection.treble_match_top_left
        w, h = detection.treble_match_size
        return (x, y, w, h), (180, 180, 180)

    return None


def _draw_overlay_box(image: MatLike, rect: tuple, color: tuple, *, thickness: int = 3):
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


# Measure Splitting Visualization


def draw_measure_boundaries(image: MatLike, measures_map: dict[int, list]) -> MatLike:
    overlay = image.copy()

    for staff_index, measures in measures_map.items():
        for measure in measures:
            cv.rectangle(
                overlay,
                (measure.x_start, measure.y_top),
                (measure.x_end - 1, measure.y_bottom),
                (255, 0, 0),
                1,
            )

    return overlay


def save_measure_visualization(
    sheet_image: MatLike,
    measures_map: dict[int, list],
    measure_crops: dict[int, list[MatLike]],
    artifacts,
) -> dict:
    paths = {}

    overlay = draw_measure_boundaries(sheet_image, measures_map)
    paths["01_measure_boundaries"] = artifacts.write_image(
        artifacts.sections.pipeline, "01_measure_boundaries.jpg", overlay
    )

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
    paths = {}

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


# Accidental Detection Visualization


def draw_accidentals_overlay(image: MatLike, accidentals: list) -> MatLike:
    out = image.copy()

    for acc in accidentals:
        color = (255, 0, 255) if acc.kind == "sharp" else (255, 128, 0)

        cv.drawMarker(
            out,
            (acc.center_x, acc.center_y),
            color,
            markerType=cv.MARKER_CROSS,
            markerSize=10,
            thickness=1,
            line_type=cv.LINE_AA,
        )

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
    paths = {}

    overlay = draw_accidentals_overlay(image, accidentals)
    paths["accidentals_overlay"] = artifacts.write_image(
        artifacts.sections.notes, "accidentals_overlay.jpg", overlay
    )

    return paths
