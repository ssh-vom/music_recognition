"""Staff + clef + simple note detection baseline."""

import cv2 as cv
from cv2.typing import MatLike

from schema import Clef, ClefDetection, Measure, Note, Staff
from staff_detection import StaffDetector, erase_staff_for_bars, erase_staff_for_notes
from bar_detection import BarDetector, BarLine
from measure_splitting import MeasureDetectionConfig, MeasureSplitter
from clef_detection import ClefDetector, ClefDetectorConfig
from note_detection import NoteDetector, resolve_note_pitches
from abc_export import write_abc_file

OverlayRect = tuple[int, int, int, int]
OverlayChoice = tuple[OverlayRect, tuple[int, int, int]] | None


def main() -> None:
    image_path = "./twinkle_twinkle_little_star.png"
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
    log_clef_detections(clef_detections)
    measures_map, measure_crops = extract_measure_crops(
        sheet_bgr, notes_mask, staffs, bars, staff_detector
    )
    notes_by_staff = detect_notes(
        measure_crops=measure_crops,
        measures_map=measures_map,
        staffs=staffs,
        clefs_by_staff=clefs_by_staff,
    )
    log_note_detections(notes_by_staff)
    write_abc_file(
        notes_by_staff=notes_by_staff,
        measures_map=measures_map,
        bars=bars,
        output_path="output.abc",
        title="Twinkle Twinkle Little Star",
        meter="4/4",
        unit_note_length="1/4",
        key="C",
        tempo_qpm=120,
    )
    print("Wrote ABC: output.abc")

    clef_overlay = draw_clef_overlay_simple(sheet_bgr, clefs_by_staff, clef_detections)
    notes_overlay = draw_notes_overlay(sheet_bgr, measures_map, notes_by_staff)
    save_clef_debug_crops(clef_key_crops, clefs_by_staff, clef_detections)
    save_note_debug_crops(measure_crops, notes_by_staff)

    cv.imwrite("staff_overlay.jpg", staff_overlay)
    cv.imwrite("bar_overlay.jpg", bar_overlay)
    cv.imwrite("clef_overlay.jpg", clef_overlay)
    cv.imwrite("notes_overlay.jpg", notes_overlay)

    cv.imshow("Staff lines", staff_overlay)
    cv.imshow("Bar detection", bar_overlay)
    cv.imshow("Staff erased (notes)", cv.bitwise_not(notes_mask))
    cv.imshow("Clef detection", clef_overlay)
    cv.imshow("Note detection", notes_overlay)
    cv.waitKey(0)
    cv.destroyAllWindows()


def crop_sheet_vertical(sheet_bgr: MatLike) -> MatLike:
    # Keep full page by default. Earlier hard-coded cropping could clip
    # top/bottom systems on differently-framed uploads.
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
    detector = ClefDetector(ClefDetectorConfig())
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
    # Notes often begin slightly earlier than the conservative header crop used for
    # clef/key extraction, so use a smaller left header skip here.
    measure_config = MeasureDetectionConfig(left_header_spacings=5.2)
    splitter = MeasureSplitter(
        bars=bars,
        staffs=staffs,
        sheet_img=sheet_bgr,
        config=measure_config,
        notes_image=notes_mask,
        staff_config=staff_detector.config,
    )
    measures_map = splitter.split_measures()

    # Keep first staff conservative so time-signature glyphs stay in header,
    # while lower staffs can start earlier to keep pickup notes.
    first_staff_index = 0
    if first_staff_index in measures_map and measures_map[first_staff_index]:
        first_staff = staffs[first_staff_index]
        first_staff_left = min(line.x_start for line in first_staff.lines)
        conservative_start = first_staff_left + int(round(first_staff.spacing * 7.0))
        first_measure = measures_map[first_staff_index][0]
        first_measure.x_start = max(first_measure.x_start, conservative_start)

    measure_crops: dict[int, list[MatLike]] = {}
    for staff_index, measures in measures_map.items():
        staff_crops: list[MatLike] = []
        for measure in measures:
            crop = notes_mask[
                measure.y_top : measure.y_bottom + 1,
                measure.x_start : measure.x_end,
            ]
            staff_crops.append(crop)
        measure_crops[staff_index] = staff_crops

    return measures_map, measure_crops


def detect_notes(
    measure_crops: dict[int, list[MatLike]],
    measures_map: dict[int, list[Measure]],
    staffs: list[Staff],
    clefs_by_staff: dict[int, Clef],
) -> dict[int, list[list[Note]]]:
    note_detector = NoteDetector()
    notes_by_staff: dict[int, list[list[Note]]] = {}

    for staff_index, crops in measure_crops.items():
        measures = measures_map.get(staff_index, [])
        if not measures:
            notes_by_staff[staff_index] = []
            continue

        staff_notes: list[list[Note]] = []
        staff = staffs[staff_index]
        clef = clefs_by_staff.get(staff_index)

        for measure_index, (measure, crop) in enumerate(zip(measures, crops)):
            detected_notes = note_detector.detect(
                cleaned_measure_mask=crop,
                staff=staff,
                measure=measure,
                measure_index=measure_index,
            )
            resolve_note_pitches(detected_notes, clef)
            staff_notes.append(detected_notes)

        notes_by_staff[staff_index] = staff_notes

    return notes_by_staff


def log_clef_detections(detections: dict[int, ClefDetection]) -> None:
    for staff_index, detection in detections.items():
        print(
            f"staff {staff_index}: {detection.clef!r}  "
            f"letter T/B={detection.letter_score_treble:.3f}/{detection.letter_score_bass:.3f}  "
            f"slide T/B={detection.slide_score_treble:.3f}/{detection.slide_score_bass:.3f}"
        )


def log_note_detections(notes_by_staff: dict[int, list[list[Note]]]) -> None:
    for staff_index, staff_measures in notes_by_staff.items():
        for measure_index, notes in enumerate(staff_measures):
            print(
                f"staff {staff_index}, measure {measure_index}: {len(notes)} noteheads detected"
            )
            for note in notes:
                pitch_label = (
                    f"{note.pitch_letter}{note.octave}"
                    if note.pitch_letter is not None and note.octave is not None
                    else "?"
                )
                duration_label = note.duration_class if note.duration_class else "?"
                confidence_label = note.step_confidence if note.step_confidence else "?"
                print(
                    f"  x={note.center_x:>4}, y={note.center_y:>4}, step={note.step:>3}, "
                    f"conf={confidence_label:<6}, pitch={pitch_label:<4}, duration={duration_label}"
                )


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


def draw_clef_overlay_simple(
    sheet_bgr: MatLike,
    clefs_by_staff: dict[int, Clef],
    detections: dict[int, ClefDetection],
) -> MatLike:
    """
    One header outline per staff, one box for the chosen clef template match, one label line.
    Letter scores (T vs B) are in the label so you can see the decision without extra clutter.
    """
    out = sheet_bgr.copy()
    font = cv.FONT_HERSHEY_SIMPLEX

    for staff_index, clef in clefs_by_staff.items():
        det = detections[staff_index]
        x1, y1 = clef.x_start, clef.y_top
        x2, y2 = clef.x_end, clef.y_bottom

        # Header search region (single, calm outline)
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
    sheet_bgr: MatLike,
    measures_map: dict[int, list[Measure]],
    notes_by_staff: dict[int, list[list[Note]]],
) -> MatLike:
    out = sheet_bgr.copy()
    font = cv.FONT_HERSHEY_SIMPLEX
    confidence_color = {
        "high": (0, 180, 0),
        "medium": (0, 180, 220),
        "low": (0, 80, 255),
    }

    for staff_index, measure_notes in notes_by_staff.items():
        measures = measures_map.get(staff_index, [])
        for measure_index, notes in enumerate(measure_notes):
            if measure_index >= len(measures):
                continue
            measure = measures[measure_index]
            cv.rectangle(
                out,
                (measure.x_start, measure.y_top),
                (measure.x_end - 1, measure.y_bottom),
                (120, 120, 120),
                1,
            )
            for note in notes:
                abs_x = measure.x_start + note.center_x
                abs_y = measure.y_top + note.center_y
                color = confidence_color.get(note.step_confidence, (160, 160, 160))
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
    clef_key_crops: dict[int, MatLike],
    clefs_by_staff: dict[int, Clef],
    detections: dict[int, ClefDetection],
) -> None:
    """Inverted crops with only the winning match rectangle (same colors as full-page overlay)."""
    for staff_index, crop in clef_key_crops.items():
        if len(crop.shape) == 2:
            tile = cv.cvtColor(cv.bitwise_not(crop), cv.COLOR_GRAY2BGR)
        else:
            tile = crop.copy()

        clef = clefs_by_staff[staff_index]
        det = detections[staff_index]
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
        cv.imwrite(f"debug_clef_staff{staff_index}.jpg", tile)


def save_note_debug_crops(
    measure_crops: dict[int, list[MatLike]],
    notes_by_staff: dict[int, list[list[Note]]],
) -> None:
    detector = NoteDetector()
    for staff_index, crops in measure_crops.items():
        note_lists = notes_by_staff.get(staff_index, [])
        for measure_index, crop in enumerate(crops):
            notes = note_lists[measure_index] if measure_index < len(note_lists) else []
            tile = detector.draw_overlay(crop, notes)
            cv.imwrite(f"debug_measure_staff{staff_index}_m{measure_index}.jpg", tile)


if __name__ == "__main__":
    main()
