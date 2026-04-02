import re
from statistics import median

import cv2 as cv
import pytesseract
from cv2.typing import MatLike

from abc_export import write_abc_file
from accidental_detection import detect_header_key_signature
from artifact_writer import ArtifactWriter
from bar_detection import find_bars
from clef_detection import detect_clef
from detection_logs import abc_key_from_score, detection_logs_text, meter_from_score
from constants import DEFAULT_TITLE, DEFAULT_UNIT_NOTE_LENGTH
from utils import to_gray
from measure_splitting import (
    crop_clef_regions,
    crop_measures,
    extract_clef_regions,
    split_measures,
)
from note_detection import find_notes, resolve_pitches
from rhythm_detection import refine_beamed_durations
from schema import (
    Accidental,
    BarLine,
    Clef,
    KeySignature,
    Score,
    Staff,
    TimeSignature,
    build_score,
    HeaderAnalysis,
)
from staff_detection import (
    erase_staff_for_bars,
    erase_staff_for_notes,
    find_staffs,
)
from visualization import (
    draw_bars_overlay,
    save_bar_visualization,
    save_clef_visualization,
    save_first_staff_accidental_visualization,
    save_full_clef_overlay,
    save_measure_visualization,
    save_notes_visualization,
    save_staff_detection,
)


def run_pipeline(image_path: str, show_windows: bool = False) -> Score:
    print(f"Processing: {image_path}")
    artifacts = ArtifactWriter(image_path=image_path)

    raw_bgr = cv.imread(filename=image_path)
    if raw_bgr is None:
        raise FileNotFoundError(f"Could not load image: {image_path}")

    print("Detecting staff lines...")
    staffs, binary, line_mask = find_staffs(raw_bgr)
    gray = to_gray(raw_bgr)
    staff_intermediates = save_staff_detection(
        image=raw_bgr,
        gray=gray,
        binary=binary,
        line_mask=line_mask,
        staffs=staffs,
        artifacts=artifacts,
    )
    print(f"  {len(staffs)} staff(s)")

    notes_mask_raw, notes_mask = erase_staff_for_notes(gray, staffs)
    bars_mask = erase_staff_for_bars(binary, staffs)
    artifacts.write_image(
        artifacts.sections.masks, "03_bars_mask_erased.jpg", cv.bitwise_not(bars_mask)
    )

    print("Detecting bar lines...")
    bars = find_bars(image=bars_mask, staffs=staffs)
    save_bar_visualization(
        image=raw_bgr,
        bars_mask=bars_mask,
        staffs=staffs,
        bars=bars,
        artifacts=artifacts,
    )
    print(f"  {len(bars)} bar line(s)")

    print("Detecting clefs...")
    clefs_by_staff = extract_clef_regions(staffs)
    clef_key_crops = crop_clef_regions(clefs_by_staff, raw_bgr, notes_mask)
    raw_clef_key_crops = crop_clef_regions(clefs_by_staff, raw_bgr)
    clef_detections = {}
    for staff_index, crop in clef_key_crops.items():
        detection = detect_clef(crop)
        clef_detections[staff_index] = detection
        clefs_by_staff[staff_index].kind = detection.clef
    save_clef_visualization(clef_key_crops, clefs_by_staff, clef_detections, artifacts)
    print(f"  {[clef.kind for clef in clefs_by_staff.values()]}")

    header = _analyze_first_staff_header(
        clef_key_crops=clef_key_crops,
        raw_clef_key_crops=raw_clef_key_crops,
        clefs_by_staff=clefs_by_staff,
        staffs=staffs,
        bars=bars,
    )
    for clef in clefs_by_staff.values():
        clef.key_signature = header.key_signature
    if header.time_signature is not None and 0 in clefs_by_staff:
        clefs_by_staff[0].time_signature = header.time_signature

    print("Splitting into measures...")
    content_start_overrides = (
        {0: header.content_start_x} if header.content_start_x is not None else None
    )
    measures_map = split_measures(
        bars=bars,
        staffs=staffs,
        left_header_spacings=5.2,
        content_start_overrides=content_start_overrides,
    )
    measure_crops = crop_measures(
        measures_map=measures_map,
        notes_image=notes_mask,
    )
    _refine_first_measure_start(
        measures_map=measures_map,
        measure_crops=measure_crops,
        notes_mask=notes_mask,
        staffs=staffs,
    )
    save_measure_visualization(
        sheet_image=raw_bgr,
        measures_map=measures_map,
        measure_crops=measure_crops,
        artifacts=artifacts,
    )
    total_measures = sum(len(m) for m in measures_map.values())
    print(f"  {total_measures} measure(s)")

    print("Building score...")
    score = build_score(
        image_path=image_path,
        sheet_image=raw_bgr,
        staffs=staffs,
        bars=bars,
        clefs_by_staff=clefs_by_staff,
        clef_detections=clef_detections,
        measures_map=measures_map,
        measure_crops=measure_crops,
    )
    if header.header_accidentals and 0 in score.clefs:
        score.clefs[0].key_header_glyphs = header.header_accidentals

    print("Detecting notes...")
    note_intermediates = _populate_notes(score)
    print(f"  {len(score.notes)} note(s)")

    print("Creating visualizations...")
    save_full_clef_overlay(score, clefs_by_staff, clef_detections, artifacts)
    save_notes_visualization(
        raw_notes_mask=notes_mask_raw,
        notes_mask=notes_mask,
        score=score,
        artifacts=artifacts,
        intermediates_by_measure=note_intermediates,
    )
    if staffs and header.search_min_x is not None and header.search_max_x is not None:
        crop = raw_clef_key_crops.get(0)
        detection_crop = clef_key_crops.get(0)
        if crop is not None and detection_crop is not None:
            save_first_staff_accidental_visualization(
                raw_crop=crop,
                detection_crop=detection_crop,
                header_accidentals=header.header_accidentals,
                min_x=header.search_min_x,
                max_x=header.search_max_x,
                artifacts=artifacts,
            )

    artifacts.write_text(
        artifacts.sections.logs,
        "detections.txt",
        detection_logs_text(score),
    )

    meter = meter_from_score(score)
    key = abc_key_from_score(score)
    abc_path = artifacts.path(artifacts.sections.export, "output.abc")
    write_abc_file(
        score_tree=score,
        output_path=abc_path,
        title=DEFAULT_TITLE,
        meter=meter,
        unit_note_length=DEFAULT_UNIT_NOTE_LENGTH,
        key=key,
        tempo_qpm=None,
    )

    print(
        f"Done — {len(staffs)} staves, {total_measures} measures, {len(score.notes)} notes → {artifacts.root}"
    )

    if show_windows:
        cv.imshow(
            "Staff Detection", staff_intermediates.get("04_staff_overlay", raw_bgr)
        )
        cv.imshow("Bar Detection", draw_bars_overlay(raw_bgr, bars))
        cv.waitKey(0)
        cv.destroyAllWindows()

    return score


def _analyze_first_staff_header(
    clef_key_crops: dict[int, MatLike],
    raw_clef_key_crops: dict[int, MatLike],
    clefs_by_staff: dict[int, Clef],
    staffs: list[Staff],
    bars: list[BarLine],
) -> HeaderAnalysis:
    clef = clefs_by_staff.get(0)
    default_content_start = clef.x_end if clef is not None else None
    default = HeaderAnalysis(
        header_accidentals=[],
        key_signature=KeySignature(fifths=0, mode="major"),
        time_signature=None,
        content_start_x=default_content_start,
    )

    if not staffs or clef is None:
        return default

    detection_crop = clef_key_crops.get(0)
    raw_crop = raw_clef_key_crops.get(0)
    if detection_crop is None or raw_crop is None:
        return default

    staff = staffs[0]
    min_x, max_x, bar_limit_x = _header_search_window(
        crop=detection_crop,
        clef=clef,
        staff=staff,
        bars=bars,
    )

    try:
        accidentals = detect_header_key_signature(
            clef_key_crop=detection_crop,
            staff=staff,
            staff_index=0,
            x_start=min_x,
            x_end=max_x + 1,
        )
    except FileNotFoundError:
        accidentals = []

    time_signature = None
    time_x_start = max(0, int(round(raw_crop.shape[1] * 0.40)))
    time_x_end = min(raw_crop.shape[1], bar_limit_x + 1)
    if time_x_end - time_x_start >= max(6, int(round(staff.spacing * 0.8))):
        try:
            time_signature = _detect_time_signature_from_roi(
                raw_crop[:, time_x_start:time_x_end]
            )
        except pytesseract.TesseractNotFoundError:
            pass

    return HeaderAnalysis(
        header_accidentals=accidentals,
        key_signature=_key_signature_from_header_accidentals(accidentals),
        time_signature=time_signature,
        content_start_x=default_content_start,
        search_min_x=min_x,
        search_max_x=max_x,
    )


def _header_search_window(
    crop: MatLike,
    clef: Clef,
    staff: Staff,
    bars: list[BarLine],
) -> tuple[int, int, int]:
    crop_width = crop.shape[1]
    margin = max(1, int(round(staff.spacing * 0.15)))
    bar_limit_x = crop_width - 1

    staff_bars = sorted(
        [bar for bar in bars if bar.staff_index == 0], key=lambda bar: bar.x
    )
    if staff_bars:
        bar_limit_x = min(
            bar_limit_x,
            max(
                0,
                staff_bars[0].x
                - clef.x_start
                - max(1, int(round(staff.spacing * 0.2))),
            ),
        )

    count, _, stats, _ = cv.connectedComponentsWithStats(crop, connectivity=8)
    components = []
    min_area = max(8, int(round(staff.spacing * staff.spacing * 0.20)))
    for index in range(1, count):
        if int(stats[index, cv.CC_STAT_AREA]) < min_area:
            continue
        left = int(stats[index, cv.CC_STAT_LEFT])
        components.append((left, left + int(stats[index, cv.CC_STAT_WIDTH])))

    min_x = 0
    max_x = bar_limit_x
    if components:
        components.sort()
        min_x = min(crop_width - 1, components[0][1] + margin)
        if len(components) > 1:
            max_x = min(max_x, components[-1][0] - margin)

    return min_x, max(min_x, min(crop_width - 1, max_x)), bar_limit_x


def _detect_time_signature_from_roi(time_roi: MatLike) -> TimeSignature | None:
    if time_roi.size == 0:
        return None

    gray = to_gray(time_roi)
    prepared = cv.resize(gray, None, fx=10.0, fy=10.0, interpolation=cv.INTER_CUBIC)
    _, prepared = cv.threshold(prepared, 0, 255, cv.THRESH_BINARY + cv.THRESH_OTSU)
    prepared = cv.copyMakeBorder(prepared, 8, 8, 8, 8, cv.BORDER_CONSTANT, value=255)
    if prepared.size == 0:
        return None

    def ocr_numbers(image: MatLike, psm: int) -> list[int]:
        if image.size == 0:
            return []
        config = f"--oem 3 --psm {psm} -c tessedit_char_whitelist=0123456789"
        text = pytesseract.image_to_string(image, config=config)
        return [int(token) for token in re.findall(r"\d+", text)]

    mid = prepared.shape[0] // 2
    top_tokens = ocr_numbers(prepared[:mid, :], psm=10)
    bottom_tokens = ocr_numbers(prepared[mid:, :], psm=10)
    if len(top_tokens) == 1 and len(bottom_tokens) == 1:
        return TimeSignature(numerator=top_tokens[0], denominator=bottom_tokens[0])

    tokens = ocr_numbers(prepared, psm=6)
    if len(tokens) == 2:
        return TimeSignature(numerator=tokens[0], denominator=tokens[1])

    common_time = cv.resize(gray, None, fx=8.0, fy=8.0, interpolation=cv.INTER_CUBIC)
    config = "--oem 3 --psm 8 -c tessedit_char_whitelist=Cc"
    if pytesseract.image_to_string(common_time, config=config).strip().lower() == "c":
        return TimeSignature(numerator=4, denominator=4)

    return None


def _refine_first_measure_start(
    measures_map: dict[int, list],
    measure_crops: dict[int, list[MatLike]],
    notes_mask: MatLike,
    staffs: list[Staff],
) -> None:
    if not staffs or not measures_map.get(0) or not measure_crops.get(0):
        return

    staff = staffs[0]
    first_measure = measures_map[0][0]
    first_crop = measure_crops[0][0]
    if first_crop is None or first_crop.size == 0:
        return

    detected_notes, _ = find_notes(
        mask=first_crop,
        staff=staff,
        measure=first_measure,
        measure_index=0,
    )
    if len(detected_notes) < 3:
        return

    detected_notes.sort(key=lambda note: note.center_x)
    left_edge_threshold = max(2, int(round(staff.spacing * 0.40)))
    if detected_notes[0].center_x > left_edge_threshold:
        return

    typical_step = median(note.step for note in detected_notes[1:])
    if detected_notes[0].step < typical_step + 2:
        return

    shift = detected_notes[0].center_x + max(1, int(round(staff.spacing * 0.40)))
    new_start = min(first_measure.x_end - 1, first_measure.x_start + shift)
    if new_start <= first_measure.x_start:
        return

    first_measure.x_start = new_start
    measure_crops[0][0] = notes_mask[
        first_measure.y_top : first_measure.y_bottom + 1,
        first_measure.x_start : first_measure.x_end,
    ]


def _key_signature_from_header_accidentals(
    header_accidentals: list[Accidental],
) -> KeySignature:
    sharp_count = sum(1 for g in header_accidentals if g.kind == "sharp")
    flat_count = sum(1 for g in header_accidentals if g.kind == "flat")

    if sharp_count and not flat_count:
        fifths = sharp_count
    elif flat_count and not sharp_count:
        fifths = -flat_count
    else:
        fifths = 0

    return KeySignature(fifths=fifths, mode="major")


def _populate_notes(score: Score) -> dict[tuple[int, int], dict]:
    score.notes = []
    intermediates_by_measure = {}

    for staff_index, staff in enumerate(score.staffs):
        clef = score.clefs.get(staff_index)
        for measure_index, measure in enumerate(
            score.get_measures_for_staff(staff_index)
        ):
            if measure.crop is None:
                continue

            detected_notes, intermediates = find_notes(
                mask=measure.crop,
                staff=staff,
                measure=measure,
                measure_index=measure_index,
            )
            intermediates_by_measure[(staff_index, measure_index)] = intermediates

            detected_notes = refine_beamed_durations(
                mask=measure.crop, notes=detected_notes, staff=staff
            )
            resolve_pitches(detected_notes, clef)

            measure.notes = detected_notes
            score.notes.extend(detected_notes)

    return intermediates_by_measure
