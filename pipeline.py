"""Complete sheet music processing pipeline."""

import re
from statistics import median

import cv2 as cv
import pytesseract
from cv2.typing import MatLike

from abc_export import write_abc_file
from accidental_detection import detect_key_signature_accidentals
from artifact_writer import ArtifactWriter
from bar_detection import find_bars
from clef_detection import detect_clef
from detection_logs import abc_key_from_score, detection_logs_text, meter_from_score
from constants import DEFAULT_TITLE, DEFAULT_UNIT_NOTE_LENGTH
from image_utils import to_gray
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
    ClefDetection,
    KeySignature,
    Note,
    Score,
    Staff,
    TimeSignature,
)
from score_tree import build_score
from staff_detection import (
    erase_staff_for_bars,
    erase_staff_for_notes,
    find_staffs,
)
from visualization import (
    choose_clef_overlay_rect,
    draw_bars_overlay,
    draw_clef_match_box,
    save_accidental_visualization,
    save_bar_visualization,
    save_clef_visualization,
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

    notes_mask = erase_staff_for_notes(gray, staffs)
    bars_mask = erase_staff_for_bars(binary, staffs)
    artifacts.write_image(
        artifacts.sections.masks, "02_notes_mask_erased.jpg", cv.bitwise_not(notes_mask)
    )
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

    header_accidentals = _detect_first_staff_header_accidentals(
        clef_key_crops=clef_key_crops,
        clefs_by_staff=clefs_by_staff,
        staffs=staffs,
        bars=bars,
    )
    first_staff_time_sig = _detect_first_staff_time_signature(
        clef_key_crops=raw_clef_key_crops,
        clefs_by_staff=clefs_by_staff,
        staffs=staffs,
        bars=bars,
    )
    if first_staff_time_sig is not None and 0 in clefs_by_staff:
        clefs_by_staff[0].time_signature = first_staff_time_sig
    _apply_detected_key_signature(clefs_by_staff, header_accidentals)

    print("Splitting into measures...")
    measures_map = split_measures(
        bars=bars,
        staffs=staffs,
        left_header_spacings=5.2,
        first_staff_conservative_spacings=7.0,
    )
    measure_crops = crop_measures(
        measures_map=measures_map,
        staffs=staffs,
        notes_image=notes_mask,
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
        clef_key_crops=clef_key_crops,
        measures_map=measures_map,
        measure_crops=measure_crops,
        notes_mask=notes_mask,
        bars_mask=bars_mask,
    )
    if header_accidentals and 0 in score.clefs:
        score.clefs[0].key_header_glyphs = header_accidentals

    print("Detecting notes...")
    note_intermediates = _populate_notes(score)
    print(f"  {len(score.notes)} note(s)")

    print("Creating visualizations...")
    _create_full_overlays(score, clefs_by_staff, clef_detections, artifacts)
    save_notes_visualization(
        notes_mask=notes_mask,
        score=score,
        artifacts=artifacts,
        intermediates_by_measure=note_intermediates,
    )
    _save_first_staff_accidental_visualization(
        raw_clef_key_crops=raw_clef_key_crops,
        clef_key_crops=clef_key_crops,
        header_accidentals=header_accidentals,
        bars=bars,
        clefs_by_staff=clefs_by_staff,
        staffs=staffs,
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


def _detect_first_staff_header_accidentals(
    clef_key_crops: dict[int, MatLike],
    clefs_by_staff: dict[int, Clef],
    staffs: list[Staff],
    bars: list[BarLine],
) -> list[Accidental]:
    if not staffs:
        return []

    crop = clef_key_crops.get(0)
    clef = clefs_by_staff.get(0)
    if crop is None or clef is None:
        return []

    try:
        min_x = _header_search_left_x(crop=crop, staff_spacing=staffs[0].spacing)
        max_x = _first_barline_x_limit_for_header(
            bars=bars,
            clef=clef,
            crop_width=crop.shape[1],
            staff_spacing=staffs[0].spacing,
        )
        max_x = _header_search_right_x(
            crop=crop, default_max_x=max_x, staff_spacing=staffs[0].spacing
        )
        accidentals = detect_key_signature_accidentals(
            clef_key_crop=crop,
            staff=staffs[0],
            staff_index=0,
            x_start=min_x,
            x_end=max_x + 1,
        )
        accidentals = _dedup_accidentals_by_x(
            accidentals=accidentals,
            x_tol=max(2, int(round(staffs[0].spacing * 0.60))),
        )
        accidentals = _reclassify_header_accidentals(
            accidentals=accidentals, crop=crop, staff_spacing=staffs[0].spacing
        )
        accidentals = _dedup_accidentals_by_component(
            accidentals=accidentals, crop=crop, staff_spacing=staffs[0].spacing
        )
        return _keep_dominant_accidental_kind(accidentals)
    except FileNotFoundError:
        return []


def _save_first_staff_accidental_visualization(
    raw_clef_key_crops: dict[int, MatLike],
    clef_key_crops: dict[int, MatLike],
    header_accidentals: list[Accidental],
    bars: list[BarLine],
    clefs_by_staff: dict[int, Clef],
    staffs: list[Staff],
    artifacts: ArtifactWriter,
) -> None:
    if not staffs:
        return

    crop = raw_clef_key_crops.get(0)
    detection_crop = clef_key_crops.get(0)
    clef = clefs_by_staff.get(0)
    if crop is None or detection_crop is None or clef is None:
        return

    overlay = crop.copy()
    min_x = _header_search_left_x(crop=detection_crop, staff_spacing=staffs[0].spacing)
    max_x = _first_barline_x_limit_for_header(
        bars=bars, clef=clef, crop_width=crop.shape[1], staff_spacing=staffs[0].spacing
    )
    max_x = _header_search_right_x(
        crop=detection_crop, default_max_x=max_x, staff_spacing=staffs[0].spacing
    )
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
    save_accidental_visualization(overlay, header_accidentals, artifacts)


def _header_search_left_x(crop: MatLike, staff_spacing: float) -> int:
    if crop.size == 0:
        return 0

    count, _, stats, _ = cv.connectedComponentsWithStats(crop, connectivity=8)
    min_area = max(8, int(round(staff_spacing * staff_spacing * 0.20)))
    best_right = 0
    best_x = None

    for i in range(1, count):
        area = int(stats[i, cv.CC_STAT_AREA])
        if area < min_area:
            continue
        x = int(stats[i, cv.CC_STAT_LEFT])
        w = int(stats[i, cv.CC_STAT_WIDTH])
        if best_x is None or x < best_x:
            best_x = x
            best_right = x + w

    margin = max(1, int(round(staff_spacing * 0.15)))
    return max(0, min(crop.shape[1] - 1, best_right + margin))


def _header_search_right_x(
    crop: MatLike, default_max_x: int, staff_spacing: float
) -> int:
    if crop.size == 0:
        return default_max_x

    count, _, stats, _ = cv.connectedComponentsWithStats(crop, connectivity=8)
    min_area = max(8, int(round(staff_spacing * staff_spacing * 0.20)))
    components = []

    for i in range(1, count):
        area = int(stats[i, cv.CC_STAT_AREA])
        if area < min_area:
            continue
        x = int(stats[i, cv.CC_STAT_LEFT])
        w = int(stats[i, cv.CC_STAT_WIDTH])
        components.append((x, x + w, area))

    components.sort()
    if len(components) < 2:
        return default_max_x

    margin = max(1, int(round(staff_spacing * 0.15)))
    return max(0, min(default_max_x, components[-1][0] - margin))


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


def _detect_first_staff_time_signature(
    clef_key_crops: dict[int, MatLike],
    clefs_by_staff: dict[int, Clef],
    staffs: list[Staff],
    bars: list[BarLine],
) -> TimeSignature | None:
    if not staffs:
        return None

    crop = clef_key_crops.get(0)
    clef = clefs_by_staff.get(0)
    if crop is None or clef is None or crop.size == 0:
        return None

    max_x = _first_barline_x_limit_for_header(
        bars=bars, clef=clef, crop_width=crop.shape[1], staff_spacing=staffs[0].spacing
    )
    time_roi = _time_signature_roi(
        crop=crop, max_x=max_x, staff_spacing=staffs[0].spacing
    )
    if time_roi is None:
        return None

    try:
        return _ocr_time_signature(time_roi)
    except pytesseract.TesseractNotFoundError:
        return None


def _apply_detected_key_signature(
    clefs_by_staff: dict[int, Clef], header_accidentals: list[Accidental]
) -> None:
    key_signature = _key_signature_from_header_accidentals(header_accidentals)
    for clef in clefs_by_staff.values():
        clef.key_signature = key_signature


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


def _time_signature_roi(
    crop: MatLike, max_x: int, staff_spacing: float
) -> MatLike | None:
    if crop.size == 0:
        return None

    x_start = max(0, int(round(crop.shape[1] * 0.40)))
    x_end = min(crop.shape[1], max_x + 1)
    if x_end - x_start < max(6, int(round(staff_spacing * 0.8))):
        return None

    roi = crop[:, x_start:x_end]
    return roi if roi.size > 0 else None


def _ocr_time_signature(time_roi: MatLike) -> TimeSignature | None:
    prepared = _prepare_time_signature_for_ocr(time_roi)
    if prepared.size == 0:
        return None

    mid = prepared.shape[0] // 2
    top_number = _ocr_single_number(prepared[:mid, :])
    bottom_number = _ocr_single_number(prepared[mid:, :])
    if top_number is not None and bottom_number is not None:
        return TimeSignature(numerator=top_number, denominator=bottom_number)

    tokens = _ocr_number_tokens(prepared, psm=6)
    if len(tokens) == 2:
        return TimeSignature(numerator=tokens[0], denominator=tokens[1])

    if _looks_like_common_time(time_roi):
        return TimeSignature(numerator=4, denominator=4)

    return None


def _prepare_time_signature_for_ocr(time_roi: MatLike) -> MatLike:
    gray = to_gray(time_roi)
    scaled = cv.resize(gray, None, fx=10.0, fy=10.0, interpolation=cv.INTER_CUBIC)
    _, thresholded = cv.threshold(scaled, 0, 255, cv.THRESH_BINARY + cv.THRESH_OTSU)
    return cv.copyMakeBorder(thresholded, 8, 8, 8, 8, cv.BORDER_CONSTANT, value=255)


def _ocr_single_number(image: MatLike) -> int | None:
    tokens = _ocr_number_tokens(image, psm=10)
    return tokens[0] if len(tokens) == 1 else None


def _ocr_number_tokens(image: MatLike, psm: int) -> list[int]:
    if image.size == 0:
        return []
    config = f"--oem 3 --psm {psm} -c tessedit_char_whitelist=0123456789"
    text = pytesseract.image_to_string(image, config=config)
    return [int(t) for t in re.findall(r"\d+", text)]


def _looks_like_common_time(time_roi: MatLike) -> bool:
    scaled = cv.resize(
        to_gray(time_roi), None, fx=8.0, fy=8.0, interpolation=cv.INTER_CUBIC
    )
    config = "--oem 3 --psm 8 -c tessedit_char_whitelist=Cc"
    return pytesseract.image_to_string(scaled, config=config).strip().lower() == "c"


def _first_barline_x_limit_for_header(
    bars: list[BarLine], clef: Clef, crop_width: int, staff_spacing: float
) -> int:
    staff_bars = sorted(
        [b for b in bars if b.staff_index == clef.staff_index], key=lambda b: b.x
    )
    if not staff_bars:
        return max(0, crop_width - 1)

    margin = max(1, int(round(staff_spacing * 0.2)))
    max_x = staff_bars[0].x - clef.x_start - margin
    return max(0, min(crop_width - 1, max_x))


def _dedup_accidentals_by_x(
    accidentals: list[Accidental], x_tol: int
) -> list[Accidental]:
    if len(accidentals) < 2:
        return accidentals

    kept = []
    for glyph in sorted(accidentals, key=lambda g: g.confidence, reverse=True):
        if any(abs(glyph.center_x - other.center_x) <= x_tol for other in kept):
            continue
        kept.append(glyph)

    kept.sort(key=lambda g: g.center_x)
    return kept


def _keep_dominant_accidental_kind(accidentals: list[Accidental]) -> list[Accidental]:
    if len(accidentals) < 2:
        return accidentals

    counts: dict[str, list] = {"sharp": [0, 0.0], "flat": [0, 0.0]}
    for g in accidentals:
        counts[g.kind][0] += 1
        counts[g.kind][1] += g.confidence

    if counts["sharp"][0] > counts["flat"][0]:
        dominant = "sharp"
    elif counts["flat"][0] > counts["sharp"][0]:
        dominant = "flat"
    else:
        dominant = "sharp" if counts["sharp"][1] >= counts["flat"][1] else "flat"

    return [g for g in accidentals if g.kind == dominant]


def _reclassify_header_accidentals(
    accidentals: list[Accidental], crop: MatLike, staff_spacing: float
) -> list[Accidental]:
    if not accidentals or crop.size == 0:
        return accidentals

    count, labels, stats, _ = cv.connectedComponentsWithStats(crop, connectivity=8)
    min_area = max(8, int(round(staff_spacing * staff_spacing * 0.10)))
    updated = []

    for glyph in accidentals:
        x = max(0, min(crop.shape[1] - 1, glyph.center_x))
        y = max(0, min(crop.shape[0] - 1, glyph.center_y))
        label = int(labels[y, x])
        if label <= 0 or int(stats[label, cv.CC_STAT_AREA]) < min_area:
            updated.append(glyph)
            continue

        left = int(stats[label, cv.CC_STAT_LEFT])
        top = int(stats[label, cv.CC_STAT_TOP])
        w = int(stats[label, cv.CC_STAT_WIDTH])
        h = int(stats[label, cv.CC_STAT_HEIGHT])
        comp = (labels[top : top + h, left : left + w] == label).astype("uint8")
        tall_threshold = max(3, int(round(h * 0.75)))
        tall_cols = [i for i, v in enumerate(comp.sum(axis=0)) if v >= tall_threshold]

        tall_clusters = 0
        if tall_cols:
            tall_clusters = 1
            prev = tall_cols[0]
            for v in tall_cols[1:]:
                if v - prev > 1:
                    tall_clusters += 1
                prev = v

        if tall_clusters >= 2:
            kind = "sharp"
        elif tall_clusters == 1:
            kind = "flat"
        else:
            kind = glyph.kind

        updated.append(
            Accidental(
                kind=kind,
                staff_index=glyph.staff_index,
                measure_index=glyph.measure_index,
                center_x=glyph.center_x,
                center_y=glyph.center_y,
                confidence=glyph.confidence,
                region=glyph.region,
            )
        )

    return updated


def _dedup_accidentals_by_component(
    accidentals: list[Accidental], crop: MatLike, staff_spacing: float
) -> list[Accidental]:
    if len(accidentals) < 2 or crop.size == 0:
        return accidentals

    count, labels, stats, _ = cv.connectedComponentsWithStats(crop, connectivity=8)
    min_area = max(8, int(round(staff_spacing * staff_spacing * 0.10)))
    best_by_label: dict[int, Accidental] = {}
    no_label = []

    for glyph in accidentals:
        x = max(0, min(crop.shape[1] - 1, glyph.center_x))
        y = max(0, min(crop.shape[0] - 1, glyph.center_y))
        label = int(labels[y, x])
        if label <= 0 or int(stats[label, cv.CC_STAT_AREA]) < min_area:
            no_label.append(glyph)
            continue
        current = best_by_label.get(label)
        if current is None or glyph.confidence > current.confidence:
            best_by_label[label] = glyph

    kept = list(best_by_label.values()) + no_label
    kept.sort(key=lambda g: g.center_x)
    return kept


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

            detected_notes = _remove_left_edge_header_bleed(
                detected_notes, staff, measure_index
            )
            detected_notes = refine_beamed_durations(
                mask=measure.crop, notes=detected_notes, staff=staff
            )
            resolve_pitches(detected_notes, clef)

            measure.notes = detected_notes
            score.notes.extend(detected_notes)

    return intermediates_by_measure


def _remove_left_edge_header_bleed(
    notes: list[Note], staff: Staff, measure_index: int
) -> list[Note]:
    if measure_index != 0 or len(notes) < 3:
        return notes
    if notes[0].center_x > max(2, int(round(staff.spacing * 0.40))):
        return notes
    typical_step = median(note.step for note in notes[1:])
    return notes[1:] if notes[0].step >= typical_step + 2 else notes


def _create_full_overlays(
    score: Score,
    clefs_by_staff: dict[int, Clef],
    clef_detections: dict[int, ClefDetection],
    artifacts: ArtifactWriter,
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
