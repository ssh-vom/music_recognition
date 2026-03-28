"""Staff -> bars -> clef -> measures -> notes pipeline with tree output."""

from pathlib import Path
from statistics import median

import cv2 as cv
from cv2.typing import MatLike

from abc_export import write_abc_file
from artifact_writer import ArtifactWriter
from bar_detection import BarDetector
from clef_detection import ClefDetector
from measure_splitting import MeasureDetectionConfig, MeasureSplitter
from note_detection import NoteDetector, resolve_note_pitches
from schema import BarLine, Clef, ClefDetection, Measure, Note, Staff
from score_tree import ScoreTree, build_score_tree
from staff_detection import StaffDetector, erase_staff_for_bars, erase_staff_for_notes

OverlayRect = tuple[int, int, int, int]
OverlayChoice = tuple[OverlayRect, tuple[int, int, int]] | None

SHOW_WINDOWS = True
DEFAULT_TITLE = "Sheet Music"
DEFAULT_METER = "4/4"
DEFAULT_UNIT_NOTE_LENGTH = "1/4"
DEFAULT_KEY = "C"
DEFAULT_TEMPO_QPM = 120


def main() -> None:
    image_path = "./music_sheets/twinkle_twinkle_little_star.png"
    raw_bgr = cv.imread(filename=image_path)
    if raw_bgr is None:
        raise FileNotFoundError(f"Could not load image: {image_path}")
    sheet_bgr = crop_sheet_vertical(raw_bgr)

    staff_detector, staffs = detect_staffs(sheet_bgr)
    staff_overlay = staff_detector.draw_overlay(staffs)

    bars_mask = build_staff_erased_bars_mask(staff_detector, staffs)
    bars, bar_overlay = detect_bars(bars_mask, sheet_bgr, staffs)

    notes_mask = build_staff_erased_notes_mask(staff_detector, staffs)
    clefs_by_staff, clef_key_crops = extract_clef_header_crops(
        sheet_bgr, notes_mask, staffs, bars, staff_detector
    )
    clef_detections = detect_clefs(clef_key_crops, clefs_by_staff)

    measures_map, measure_crops = extract_measure_crops(
        sheet_bgr, notes_mask, staffs, bars, staff_detector
    )

    score_tree = build_score_tree(
        image_path=image_path,
        sheet_image=sheet_bgr,
        staffs=staffs,
        bars=bars,
        clefs_by_staff=clefs_by_staff,
        clef_detections=clef_detections,
        clef_key_crops=clef_key_crops,
        measures_map=measures_map,
        measure_crops=measure_crops,
        notes_mask=notes_mask,
        bars_mask=bars_mask,
    )
    populate_tree_with_notes(score_tree)

    clef_overlay = draw_clef_overlay(score_tree)
    notes_overlay = draw_notes_overlay(score_tree)

    clef_log = build_clef_log(score_tree)
    note_log = build_note_log(score_tree)
    print(clef_log, end="")
    print(note_log, end="")

    artifacts = ArtifactWriter(image_path=image_path)
    artifacts.write_image(artifacts.sections.staff, "staff_overlay.jpg", staff_overlay)
    artifacts.write_image(artifacts.sections.bars, "bar_overlay.jpg", bar_overlay)
    artifacts.write_image(
        artifacts.sections.masks, "notes_mask_inverted.jpg", cv.bitwise_not(notes_mask)
    )
    artifacts.write_image(artifacts.sections.clef, "clef_overlay.jpg", clef_overlay)
    artifacts.write_image(artifacts.sections.notes, "notes_overlay.jpg", notes_overlay)
    save_clef_debug_crops(score_tree, artifacts.section_dir(artifacts.sections.clef))
    save_note_debug_crops(score_tree, artifacts.section_dir(artifacts.sections.notes))

    abc_path = artifacts.text_path(artifacts.sections.export, "output.abc")
    write_abc_file(
        score_tree=score_tree,
        output_path=abc_path,
        title=DEFAULT_TITLE,
        meter=DEFAULT_METER,
        unit_note_length=DEFAULT_UNIT_NOTE_LENGTH,
        key=DEFAULT_KEY,
        tempo_qpm=DEFAULT_TEMPO_QPM,
    )

    artifacts.write_text(
        artifacts.sections.logs,
        "detections.txt",
        clef_log + note_log,
    )

    print(f"Artifacts written to: {artifacts.root}")
    print(f"Wrote ABC: {abc_path}")

    if SHOW_WINDOWS:
        cv.imshow("Staff lines", staff_overlay)
        cv.imshow("Bar detection", bar_overlay)
        cv.imshow("Staff erased (notes)", cv.bitwise_not(notes_mask))
        cv.imshow("Clef detection", clef_overlay)
        cv.imshow("Note detection", notes_overlay)
        cv.waitKey(0)
        cv.destroyAllWindows()


def crop_sheet_vertical(sheet_bgr: MatLike) -> MatLike:
    return sheet_bgr


def detect_staffs(sheet_bgr: MatLike) -> tuple[StaffDetector, list]:
    detector = StaffDetector(sheet_bgr)
    staffs, _, _ = detector.detect()
    return detector, staffs


def build_staff_erased_notes_mask(
    staff_detector: StaffDetector,
    staffs: list,
) -> MatLike:
    return erase_staff_for_notes(
        staff_detector.to_gray(),
        staffs=staffs,
        config=staff_detector.config,
    )


def build_staff_erased_bars_mask(
    staff_detector: StaffDetector,
    staffs: list[Staff],
) -> MatLike:
    gray = staff_detector.to_gray()
    binary = staff_detector.binarize(gray)
    return erase_staff_for_bars(
        binary=binary,
        staffs=staffs,
        config=staff_detector.config,
    )


def detect_bars(
    bars_mask: MatLike,
    sheet_bgr: MatLike,
    staffs: list[Staff],
) -> tuple[list[BarLine], MatLike]:
    detector = BarDetector(
        binary_img=bars_mask,
        original_img=sheet_bgr,
        staffs=staffs,
    )
    bars = detector.detect()
    return bars, detector.draw_overlay()


def extract_clef_header_crops(
    sheet_bgr: MatLike,
    notes_mask: MatLike,
    staffs: list,
    bars: list[BarLine],
    staff_detector: StaffDetector,
) -> tuple[dict[int, Clef], dict[int, MatLike]]:
    splitter = MeasureSplitter(
        bars=bars,
        staffs=staffs,
        sheet_img=sheet_bgr,
        notes_image=notes_mask,
        staff_config=staff_detector.config,
    )
    clefs_by_staff = splitter.extract_clef_and_key_signatures()
    crops = splitter.crop_clef_and_key_signatures(clefs=clefs_by_staff)
    return clefs_by_staff, crops


def detect_clefs(
    clef_key_crops: dict[int, MatLike],
    clefs_by_staff: dict[int, Clef],
) -> dict[int, ClefDetection]:
    detector = ClefDetector()
    detections: dict[int, ClefDetection] = {}
    for staff_index, crop in clef_key_crops.items():
        detection = detector.detect(crop)
        detections[staff_index] = detection
        clefs_by_staff[staff_index].kind = detection.clef
    return detections


def extract_measure_crops(
    sheet_bgr: MatLike,
    notes_mask: MatLike,
    staffs: list[Staff],
    bars: list[BarLine],
    staff_detector: StaffDetector,
) -> tuple[dict[int, list[Measure]], dict[int, list[MatLike]]]:
    measure_config = MeasureDetectionConfig()
    measure_config.left_header_spacings = 5.2
    measure_config.first_staff_conservative_spacings = 7.0
    splitter = MeasureSplitter(
        bars=bars,
        staffs=staffs,
        sheet_img=sheet_bgr,
        config=measure_config,
        notes_image=notes_mask,
        staff_config=staff_detector.config,
    )
    measures_map = splitter.split_measures()
    measure_crops = splitter.crop_measures()
    return measures_map, measure_crops


def populate_tree_with_notes(score_tree: ScoreTree) -> None:
    note_detector = NoteDetector()
    for staff_node in score_tree.staff_nodes:
        clef = staff_node.clef
        for measure_node in staff_node.measures:
            if measure_node.crop is None:
                continue
            detected_notes = note_detector.detect(
                mask=measure_node.crop,
                staff=staff_node.staff,
                measure=measure_node.measure,
                measure_index=measure_node.index,
            )
            detected_notes = remove_left_edge_header_bleed(
                notes=detected_notes,
                staff=staff_node.staff,
                measure_index=measure_node.index,
            )
            resolve_note_pitches(detected_notes, clef)
            measure_node.notes = detected_notes


def remove_left_edge_header_bleed(
    notes: list[Note],
    staff: Staff,
    measure_index: int,
) -> list[Note]:
    if measure_index != 0:
        return notes
    if len(notes) < 3:
        return notes

    first_note = notes[0]
    left_edge_threshold = max(2, int(round(staff.spacing * 0.40)))
    if first_note.center_x > left_edge_threshold:
        return notes

    remaining_steps = [note.step for note in notes[1:]]
    typical_step = median(remaining_steps)
    if first_note.step >= typical_step + 2:
        return notes[1:]
    return notes


def build_clef_log(score_tree: ScoreTree) -> str:
    lines: list[str] = []
    for staff_node in score_tree.staff_nodes:
        detection = staff_node.clef_detection
        if detection is None:
            continue
        lines.append(
            f"staff {staff_node.index}: {detection.clef!r}  "
            f"letter T/B={detection.letter_score_treble:.3f}/{detection.letter_score_bass:.3f}  "
            f"slide T/B={detection.slide_score_treble:.3f}/{detection.slide_score_bass:.3f}\n"
        )
    return "".join(lines)


def build_note_log(score_tree: ScoreTree) -> str:
    lines: list[str] = []
    for staff_node in score_tree.staff_nodes:
        for measure_node in staff_node.measures:
            notes = measure_node.notes
            lines.append(
                f"staff {staff_node.index}, measure {measure_node.index}: {len(notes)} noteheads detected\n"
            )
            for note in measure_node.notes:
                pitch_label = (
                    f"{note.pitch_letter}{note.octave}"
                    if note.pitch_letter is not None and note.octave is not None
                    else "?"
                )
                duration_label = note.duration_class if note.duration_class else "?"
                confidence_label = note.step_confidence if note.step_confidence else "?"
                lines.append(
                    f"  x={note.center_x:>4}, y={note.center_y:>4}, step={note.step:>3}, "
                    f"conf={confidence_label:<6}, pitch={pitch_label:<4}, duration={duration_label}\n"
                )
    return "".join(lines)


def choose_overlay_rect(clef: Clef, detection: ClefDetection) -> OverlayChoice:
    if (
        clef.kind == "treble"
        and detection.treble_match_top_left is not None
        and detection.treble_match_size is not None
    ):
        x, y = detection.treble_match_top_left
        w, h = detection.treble_match_size
        return (x, y, w, h), (0, 200, 100)
    if (
        clef.kind == "bass"
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


def draw_box(
    image: MatLike,
    rect: OverlayRect,
    color: tuple[int, int, int],
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


def draw_clef_overlay(score_tree: ScoreTree) -> MatLike:
    out = score_tree.sheet_image.copy()
    font = cv.FONT_HERSHEY_SIMPLEX

    for staff_node in score_tree.staff_nodes:
        clef = staff_node.clef
        det = staff_node.clef_detection
        if clef is None or det is None:
            continue

        staff_index = staff_node.index
        x1, y1 = clef.x_start, clef.y_top
        x2, y2 = clef.x_end, clef.y_bottom

        cv.rectangle(out, (x1, y1), (x2, y2), (100, 100, 100), 1)

        choice = choose_overlay_rect(clef, det)
        if choice is not None:
            rect, color = choice
            draw_box(out, rect, color, origin_x=x1, origin_y=y1, thickness=3)

        name = clef.kind if clef.kind else "?"
        label = (
            f"Staff {staff_index}: {name}   "
            f"letter T={det.letter_score_treble:.2f}  B={det.letter_score_bass:.2f}"
        )
        (tw, th), baseline = cv.getTextSize(label, font, 0.55, 2)
        pad = 5
        tx, ty = x1, y2 + th + pad + 4
        if ty + 8 > out.shape[0]:
            ty = y1 - 8
        cv.rectangle(
            out,
            (tx, ty - th - pad),
            (tx + tw + 2 * pad, ty + baseline + pad),
            (250, 250, 250),
            -1,
        )
        cv.rectangle(
            out,
            (tx, ty - th - pad),
            (tx + tw + 2 * pad, ty + baseline + pad),
            (80, 80, 80),
            1,
        )
        cv.putText(out, label, (tx + pad, ty), font, 0.55, (25, 25, 25), 2, cv.LINE_AA)

    return out


def draw_notes_overlay(
    score_tree: ScoreTree,
) -> MatLike:
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


def save_clef_debug_crops(
    score_tree: ScoreTree,
    output_dir: Path,
) -> None:
    for staff_node in score_tree.staff_nodes:
        staff_index = staff_node.index
        crop = staff_node.clef_key_crop
        clef = staff_node.clef
        det = staff_node.clef_detection
        if crop is None or clef is None or det is None:
            continue

        if len(crop.shape) == 2:
            tile = cv.cvtColor(cv.bitwise_not(crop), cv.COLOR_GRAY2BGR)
        else:
            tile = crop.copy()

        choice = choose_overlay_rect(clef, det)
        if choice is not None:
            rect, color = choice
            draw_box(tile, rect, color, thickness=3)

        cv.putText(
            tile,
            f"{clef.kind or '?'}  T={det.letter_score_treble:.2f} B={det.letter_score_bass:.2f}",
            (6, 22),
            cv.FONT_HERSHEY_SIMPLEX,
            0.55,
            (30, 30, 30),
            2,
            cv.LINE_AA,
        )
        cv.imwrite(str(output_dir / f"debug_clef_staff{staff_index}.jpg"), tile)


def save_note_debug_crops(
    score_tree: ScoreTree,
    output_dir: Path,
) -> None:
    detector = NoteDetector()
    for staff_node in score_tree.staff_nodes:
        for measure_node in staff_node.measures:
            if measure_node.crop is None:
                continue
            tile = detector.draw_overlay(measure_node.crop, measure_node.notes)
            cv.imwrite(
                str(
                    output_dir
                    / f"debug_measure_staff{staff_node.index}_m{measure_node.index}.jpg"
                ),
                tile,
            )


if __name__ == "__main__":
    main()
