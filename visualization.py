import cv2 as cv
from cv2.typing import MatLike

from schema import Accidental, BarLine, Clef, ClefDetection, Note, Score, Staff


def draw_staff_overlay(image: MatLike, staffs: list[Staff]) -> MatLike:
    overlay = (
        cv.cvtColor(image, cv.COLOR_GRAY2BGR) if len(image.shape) == 2 else image.copy()
    )

    for idx, staff in enumerate(staffs):
        for line in staff.lines:
            cv.line(
                overlay, (line.x_start, line.y), (line.x_end, line.y), (0, 255, 0), 1
            )
        cv.rectangle(
            overlay,
            (0, staff.top),
            (overlay.shape[1] - 1, staff.bottom),
            (255, 0, 0),
            1,
        )
        label_y = max(15, staff.top - 5)
        cv.putText(
            overlay,
            f"staff {idx}  spacing={staff.spacing:.1f}px",
            (10, label_y),
            cv.FONT_HERSHEY_SIMPLEX,
            0.5,
            (0, 0, 255),
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
    paths["02_otsu_binary"] = artifacts.write_image(
        artifacts.sections.staff, "02_otsu_binary.jpg", cv.bitwise_not(binary)
    )
    paths["03_horizontal_lines"] = artifacts.write_image(
        artifacts.sections.staff, "03_horizontal_lines.jpg", cv.bitwise_not(line_mask)
    )
    paths["04_staff_overlay"] = artifacts.write_image(
        artifacts.sections.staff,
        "04_staff_overlay.jpg",
        draw_staff_overlay(image, staffs),
    )
    return paths


def draw_bars_overlay(image: MatLike, bars: list[BarLine]) -> MatLike:
    overlay = image.copy()
    for bar in bars:
        cv.line(overlay, (bar.x, bar.y_top), (bar.x, bar.y_bottom), (0, 0, 255), 2)
    return overlay


def _get_bar_processing_intermediates(
    image: MatLike, first_staff: Staff
) -> tuple[MatLike | None, MatLike | None]:
    y0, y1 = first_staff.top, first_staff.bottom + 1
    work = image[y0:y1, :][:, int(round(5.0 * first_staff.spacing)) :]

    if work.size == 0:
        return None, None

    kernel_h = max(5, int(round(2.0 * first_staff.spacing)))
    joined = cv.morphologyEx(
        work, cv.MORPH_CLOSE, cv.getStructuringElement(cv.MORPH_RECT, (1, kernel_h))
    )

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

    input_display = (
        cv.bitwise_not(bars_mask) if len(bars_mask.shape) == 2 else bars_mask.copy()
    )
    paths["01_input_mask"] = artifacts.write_image(
        artifacts.sections.bars, "01_input_mask.jpg", input_display
    )

    if staffs:
        joined, contour_viz = _get_bar_processing_intermediates(bars_mask, staffs[0])
        if joined is not None:
            paths["02_vertical_close"] = artifacts.write_image(
                artifacts.sections.bars, "02_vertical_close.jpg", cv.bitwise_not(joined)
            )
            paths["03_contours"] = artifacts.write_image(
                artifacts.sections.bars, "03_contours.jpg", contour_viz
            )

    paths["04_bar_overlay"] = artifacts.write_image(
        artifacts.sections.bars, "04_bar_overlay.jpg", draw_bars_overlay(image, bars)
    )
    return paths


def draw_notes_on_mask(mask: MatLike, notes: list[Note]) -> MatLike:
    overlay = cv.cvtColor(cv.bitwise_not(mask), cv.COLOR_GRAY2BGR)

    for note in notes:
        cv.circle(overlay, (note.center_x, note.center_y), 3, (0, 0, 255), 1)
        conf = note.step_confidence if note.step_confidence else "?"
        pitch = (
            f"{note.pitch_letter}{note.octave}"
            if note.pitch_letter and note.octave
            else "?"
        )
        dur = note.duration_class if note.duration_class else "?"
        cv.putText(
            overlay,
            f"{note.step} {conf} {pitch} {dur}",
            (note.center_x + 4, note.center_y - 4),
            cv.FONT_HERSHEY_SIMPLEX,
            0.35,
            (0, 255, 0),
            1,
            cv.LINE_AA,
        )

    return overlay


def _draw_filtered_components(mask: MatLike, filtered_info: list[dict]) -> MatLike:
    overlay = cv.cvtColor(cv.bitwise_not(mask), cv.COLOR_GRAY2BGR)

    for info in filtered_info:
        x = info["x"] - info["w"] // 2
        y = info["y"] - info["h"] // 2
        w, h = info["w"], info["h"]

        if info["passed"]:
            color = (0, 200, 0)
            label = f"{info['id']}: OK"
        else:
            color = (0, 0, 200)
            reasons = []
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


def _draw_stem_augmentation(
    mask: MatLike, stem_info: dict, centers_before: list, centers_after: list
) -> MatLike:
    overlay = cv.cvtColor(cv.bitwise_not(mask), cv.COLOR_GRAY2BGR)

    for cx, cy, _ in centers_before:
        cv.circle(overlay, (cx, cy), 3, (255, 0, 0), -1)

    if "all_stems" in stem_info and stem_info["all_stems"]:
        for stem in stem_info["all_stems"]:
            x, y, w, h = stem["x"], stem["y"], stem["w"], stem["h"]
            if stem.get("added"):
                color, thickness = (0, 200, 0), 2
            elif "rejected" in stem:
                color, thickness = (0, 200, 255), 1
            else:
                color, thickness = (0, 0, 200), 1
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
    raw_notes_mask: MatLike | None,
    notes_mask: MatLike,
    score: Score,
    artifacts,
    intermediates_by_measure: dict[tuple[int, int], dict] | None = None,
) -> dict:
    """Save note detection visualizations and intermediates."""
    paths = {}

    if raw_notes_mask is not None:
        paths["01_notes_mask"] = artifacts.write_image(
            artifacts.sections.masks,
            "01_notes_mask.jpg",
            cv.bitwise_not(raw_notes_mask),
        )

    morph_dir = artifacts.ensure_subdir(artifacts.sections.notes, "02_morphological")
    filter_dir = artifacts.ensure_subdir(
        artifacts.sections.notes, "03_geometric_filtering"
    )
    stem_dir = artifacts.ensure_subdir(artifacts.sections.notes, "04_stem_augmentation")
    final_dir = artifacts.ensure_subdir(artifacts.sections.notes, "05_final_notes")

    for staff_idx, _ in enumerate(score.staffs):
        for measure_idx, measure in enumerate(score.get_measures_for_staff(staff_idx)):
            if measure.crop is None:
                continue

            mask = measure.crop
            key = (staff_idx, measure_idx)
            intermediates = (
                intermediates_by_measure.get(key) if intermediates_by_measure else None
            )
            prefix = f"staff_{staff_idx}_measure_{measure_idx}"

            if intermediates and "notehead_mask" in intermediates:
                cv.imwrite(
                    str(morph_dir / f"{prefix}_notehead.jpg"),
                    cv.bitwise_not(intermediates["notehead_mask"]),
                )

            if intermediates and "filtered_components" in intermediates:
                cv.imwrite(
                    str(filter_dir / f"{prefix}.jpg"),
                    _draw_filtered_components(
                        mask, intermediates["filtered_components"]
                    ),
                )

            if intermediates and "stem_augmentation" in intermediates:
                cv.imwrite(
                    str(stem_dir / f"{prefix}.jpg"),
                    _draw_stem_augmentation(
                        mask,
                        intermediates["stem_augmentation"],
                        intermediates.get("centers_after_merge", []),
                        intermediates.get("centers_after_stems", []),
                    ),
                )

            cv.imwrite(
                str(final_dir / f"{prefix}.jpg"),
                draw_notes_on_mask(
                    mask, score.get_notes_for_measure(staff_idx, measure_idx)
                ),
            )

    paths["02_morphological"] = morph_dir
    paths["03_geometric_filtering"] = filter_dir
    paths["04_stem_augmentation"] = stem_dir
    paths["05_final_notes"] = final_dir
    paths["06_full_notes_overlay"] = artifacts.write_image(
        artifacts.sections.notes,
        "06_full_notes_overlay.jpg",
        _draw_full_notes_overlay(score),
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
    duration_short = {
        "whole": "w",
        "half": "h",
        "quarter": "q",
        "eighth": "8",
        "sixteenth": "16",
    }

    for staff_idx in range(len(score.staffs)):
        for measure_idx, measure in enumerate(score.get_measures_for_staff(staff_idx)):
            cv.rectangle(
                out,
                (measure.x_start, measure.y_top),
                (measure.x_end - 1, measure.y_bottom),
                (120, 120, 120),
                1,
            )
            for note in score.get_notes_for_measure(staff_idx, measure_idx):
                abs_x = measure.x_start + note.center_x
                abs_y = measure.y_top + note.center_y
                color = confidence_color.get(
                    note.step_confidence or "unknown", (160, 160, 160)
                )
                cv.circle(out, (abs_x, abs_y), 4, color, 2)
                pitch = (
                    f"{note.pitch_letter}{note.octave}"
                    if note.pitch_letter and note.octave
                    else "?"
                )
                dur = duration_short.get(
                    note.duration_class or "", note.duration_class or "?"
                )
                cv.putText(
                    out,
                    f"{note.step} {pitch} {dur}",
                    (abs_x + 5, abs_y - 5),
                    font,
                    0.35,
                    color,
                    1,
                    cv.LINE_AA,
                )

    return out


def draw_clef_overlay(
    clef_crop: MatLike, detection: ClefDetection, clef_kind: str | None
) -> MatLike:
    overlay = (
        cv.cvtColor(cv.bitwise_not(clef_crop), cv.COLOR_GRAY2BGR)
        if len(clef_crop.shape) == 2
        else clef_crop.copy()
    )

    if clef_kind is None or detection is None:
        return overlay

    cv.rectangle(
        overlay, (0, 0), (overlay.shape[1], overlay.shape[0]), (100, 100, 100), 1
    )

    choice = choose_clef_overlay_rect(clef_kind, detection)
    if choice is not None:
        rect, color = choice
        draw_clef_match_box(overlay, rect, color, thickness=3)

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


def choose_clef_overlay_rect(
    clef_kind: str | None, detection: ClefDetection
) -> tuple[tuple[int, int, int, int], tuple[int, int, int]] | None:
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


def draw_clef_match_box(
    image: MatLike,
    rect: tuple,
    color: tuple,
    *,
    origin_x: int = 0,
    origin_y: int = 0,
    thickness: int = 3,
) -> None:
    x, y, w, h = rect
    if w < 2 or h < 2:
        return
    cv.rectangle(
        image,
        (origin_x + x, origin_y + y),
        (origin_x + x + w - 1, origin_y + y + h - 1),
        color,
        thickness,
    )


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

    paths["01_measure_boundaries"] = artifacts.write_image(
        artifacts.sections.pipeline,
        "01_measure_boundaries.jpg",
        draw_measure_boundaries(sheet_image, measures_map),
    )

    crops_dir = artifacts.ensure_subdir(artifacts.sections.pipeline, "02_measure_crops")
    for staff_index, crops in measure_crops.items():
        staff_dir = crops_dir / f"staff_{staff_index}"
        staff_dir.mkdir(exist_ok=True)
        for measure_index, crop in enumerate(crops):
            crop_display = cv.bitwise_not(crop) if len(crop.shape) == 2 else crop
            cv.imwrite(str(staff_dir / f"measure_{measure_index}.jpg"), crop_display)
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
        display_crop = cv.bitwise_not(crop) if len(crop.shape) == 2 else crop
        cv.imwrite(str(clef_crops_dir / f"staff_{staff_index}.jpg"), display_crop)
    paths["01_clef_header_crops"] = clef_crops_dir

    for staff_index, crop in clef_key_crops.items():
        clef = clefs_by_staff.get(staff_index)
        detection = clef_detections.get(staff_index)
        if detection is not None:
            overlay = draw_clef_overlay(crop, detection, clef.kind if clef else None)
            paths[f"02_detection_staff_{staff_index}"] = artifacts.write_image(
                artifacts.sections.clef,
                f"02_detection_staff_{staff_index}.jpg",
                overlay,
            )

    return paths


def save_full_clef_overlay(
    score: Score,
    clefs_by_staff: dict[int, Clef],
    clef_detections: dict[int, ClefDetection],
    artifacts,
) -> None:
    clef_overlay = score.sheet_image.copy()
    font = cv.FONT_HERSHEY_SIMPLEX

    for staff_index in range(len(score.staffs)):
        clef = clefs_by_staff.get(staff_index)
        det = clef_detections.get(staff_index)
        if clef is None or det is None:
            continue

        x1, y1 = clef.x_start, clef.y_top
        x2, y2 = clef.x_end, clef.y_bottom
        cv.rectangle(clef_overlay, (x1, y1), (x2, y2), (100, 100, 100), 1)

        choice = choose_clef_overlay_rect(clef.kind, det)
        if choice is not None:
            rect, color = choice
            draw_clef_match_box(clef_overlay, rect, color, origin_x=x1, origin_y=y1)

        name = clef.kind if clef.kind else "?"
        label = f"Staff {staff_index}: {name}  T={det.letter_score_treble:.2f} B={det.letter_score_bass:.2f}"
        (tw, th), baseline = cv.getTextSize(label, font, 0.55, 2)
        pad = 5
        tx, ty = x1, y2 + th + pad + 4
        if ty + 8 > clef_overlay.shape[0]:
            ty = y1 - 8
        cv.rectangle(
            clef_overlay,
            (tx, ty - th - pad),
            (tx + tw + 2 * pad, ty + baseline + pad),
            (250, 250, 250),
            -1,
        )
        cv.rectangle(
            clef_overlay,
            (tx, ty - th - pad),
            (tx + tw + 2 * pad, ty + baseline + pad),
            (80, 80, 80),
            1,
        )
        cv.putText(
            clef_overlay, label, (tx + pad, ty), font, 0.55, (25, 25, 25), 2, cv.LINE_AA
        )

    artifacts.write_image(
        artifacts.sections.clef, "03_full_clef_overlay.jpg", clef_overlay
    )


def _draw_header_accidental_boxes(
    overlay: MatLike, detection_crop: MatLike, accidentals: list[Accidental]
) -> None:
    if detection_crop.size == 0 or not accidentals:
        return

    count, labels, stats, _ = cv.connectedComponentsWithStats(
        detection_crop, connectivity=8
    )
    h, w = detection_crop.shape[:2]

    for glyph in accidentals:
        x = max(0, min(w - 1, glyph.center_x))
        y = max(0, min(h - 1, glyph.center_y))
        label = int(labels[y, x])
        if label <= 0:
            continue
        left = int(stats[label, cv.CC_STAT_LEFT])
        top = int(stats[label, cv.CC_STAT_TOP])
        box_w = int(stats[label, cv.CC_STAT_WIDTH])
        box_h = int(stats[label, cv.CC_STAT_HEIGHT])
        color = (255, 0, 255) if glyph.kind == "sharp" else (255, 128, 0)
        cv.rectangle(
            overlay,
            (left, top),
            (left + box_w - 1, top + box_h - 1),
            color,
            1,
            cv.LINE_AA,
        )


def save_first_staff_accidental_visualization(
    raw_crop: MatLike,
    detection_crop: MatLike,
    header_accidentals: list[Accidental],
    min_x: int,
    max_x: int,
    artifacts,
) -> dict:
    overlay = raw_crop.copy()
    cv.line(
        overlay,
        (min_x, 0),
        (min_x, max(0, overlay.shape[0] - 1)),
        (80, 220, 80),
        1,
        cv.LINE_AA,
    )
    cv.line(
        overlay,
        (max_x, 0),
        (max_x, max(0, overlay.shape[0] - 1)),
        (0, 200, 255),
        1,
        cv.LINE_AA,
    )
    _draw_header_accidental_boxes(overlay, detection_crop, header_accidentals)
    return save_accidental_visualization(overlay, header_accidentals, artifacts)


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
        cv.putText(
            out,
            acc.kind[0].upper(),
            (acc.center_x + 6, acc.center_y + 4),
            cv.FONT_HERSHEY_SIMPLEX,
            0.4,
            color,
            1,
            cv.LINE_AA,
        )
    return out


def save_accidental_visualization(image: MatLike, accidentals: list, artifacts) -> dict:
    overlay = draw_accidentals_overlay(image, accidentals)
    return {
        "accidentals_overlay": artifacts.write_image(
            artifacts.sections.notes, "accidentals_overlay.jpg", overlay
        )
    }
