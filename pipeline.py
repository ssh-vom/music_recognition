"""Complete sheet music processing pipeline with full visualization."""

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
    Measure,
    Note,
    Score,
    Staff,
    TimeSignature,
)
from score_tree import build_score
from staff_detection import (
    binarize,
    erase_staff_for_bars,
    erase_staff_for_notes,
    find_staves,
    to_gray,
)
from visualization import (
    draw_bars_overlay,
    save_accidental_visualization,
    save_bar_visualization,
    save_clef_visualization,
    save_measure_visualization,
    save_notes_visualization,
    save_staff_detection,
)

DEFAULT_TITLE = "Sheet Music"
DEFAULT_METER = "4/4"
DEFAULT_UNIT_NOTE_LENGTH = "1/4"
DEFAULT_KEY = "C"
DEFAULT_TEMPO_QPM = 120


def run_pipeline(image_path: str, show_windows: bool = False) -> Score:
    """Run complete sheet music processing pipeline with full visualization.

    Args:
        image_path: Path to the input sheet music image
        show_windows: If True, display OpenCV windows (default False for batch processing)

    Returns:
        Score with all detected elements and notes
    """
    print(f"\n{'=' * 60}")
    print(f"Processing: {image_path}")
    print(f"{'=' * 60}\n")

    artifacts = ArtifactWriter(image_path=image_path)

    raw_bgr = cv.imread(filename=image_path)
    if raw_bgr is None:
        raise FileNotFoundError(f"Could not load image: {image_path}")

    print("Step 1: Detecting staff lines...")
    staffs, binary, line_mask = find_staves(raw_bgr)
    gray = to_gray(raw_bgr)
    staff_intermediates = save_staff_detection(
        image=raw_bgr,
        gray=gray,
        binary=binary,
        line_mask=line_mask,
        staffs=staffs,
        artifacts=artifacts,
    )
    print(f"  Found {len(staffs)} staff(s)")

    print("Step 2: Preparing staff-erased masks...")
    notes_mask = _build_notes_mask(raw_bgr, staffs)
    bars_mask = _build_bars_mask(binary, staffs)

    artifacts.write_image(
        artifacts.sections.masks, "02_notes_mask_erased.jpg", cv.bitwise_not(notes_mask)
    )
    artifacts.write_image(
        artifacts.sections.masks, "03_bars_mask_erased.jpg", cv.bitwise_not(bars_mask)
    )

    print("Step 3: Detecting bar lines...")
    bars = find_bars(image=bars_mask, staffs=staffs)
    bar_intermediates = save_bar_visualization(
        image=raw_bgr,
        bars_mask=bars_mask,
        staffs=staffs,
        bars=bars,
        artifacts=artifacts,
    )
    print(f"  Found {len(bars)} bar line(s)")

    print("Step 4: Detecting clefs...")
    clefs_by_staff, clef_key_crops = _extract_clef_crops(
        raw_bgr, notes_mask, staffs, bars
    )
    raw_clef_key_crops = crop_clef_regions(clefs_by_staff, raw_bgr)
    clef_detections: dict[int, ClefDetection] = {}
    for staff_index, crop in clef_key_crops.items():
        detection = detect_clef(crop)
        clef_detections[staff_index] = detection
        clefs_by_staff[staff_index].kind = detection.clef

    # Save clef intermediates
    clef_intermediates = save_clef_visualization(
        clef_key_crops, clefs_by_staff, clef_detections, artifacts
    )
    print(f"  Detected clefs: {[clef.kind for clef in clefs_by_staff.values()]}")
    header_accidentals = _detect_first_staff_header_accidentals(
        clef_key_crops=clef_key_crops,
        clefs_by_staff=clefs_by_staff,
        staffs=staffs,
        bars=bars,
    )
    first_staff_time_signature = _detect_first_staff_time_signature(
        clef_key_crops=raw_clef_key_crops,
        clefs_by_staff=clefs_by_staff,
        staffs=staffs,
        bars=bars,
    )
    if first_staff_time_signature is not None and 0 in clefs_by_staff:
        clefs_by_staff[0].time_signature = first_staff_time_signature
    _apply_detected_key_signature(clefs_by_staff, header_accidentals)

    print("Step 5: Splitting into measures...")
    measures_map, measure_crops = _extract_measures(raw_bgr, notes_mask, staffs, bars)

    measure_intermediates = save_measure_visualization(
        sheet_image=raw_bgr,
        measures_map=measures_map,
        measure_crops=measure_crops,
        artifacts=artifacts,
    )

    total_measures = sum(len(measures) for measures in measures_map.values())
    print(f"  Split into {total_measures} measure(s)")

    print("Step 6: Building score...")
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

    print("Step 7: Detecting notes...")
    note_intermediates = _populate_notes(score)
    total_notes = len(score.notes)
    print(f"  Detected {total_notes} note(s)")

    print("Step 8: Creating visualizations...")
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

    print("Step 9: Generating logs...")
    clef_log = _build_clef_log(score)
    accidental_log = _build_accidental_log(score)
    key_signature_log = _build_key_signature_log(score)
    time_signature_log = _build_time_signature_log(score)
    note_log = _build_note_log(score)
    artifacts.write_text(
        artifacts.sections.logs,
        "detections.txt",
        clef_log + accidental_log + key_signature_log + time_signature_log + note_log,
    )

    print("Step 10: Exporting to ABC notation...")
    abc_path = artifacts.text_path(artifacts.sections.export, "output.abc")
    polyphonic_abc_path = artifacts.text_path(
        artifacts.sections.export, "output_polyphonic.abc"
    )
    meter = _meter_from_score(score)
    key = _abc_key_from_score(score)
    write_abc_file(
        score_tree=score,
        output_path=abc_path,
        title=DEFAULT_TITLE,
        meter=meter,
        unit_note_length=DEFAULT_UNIT_NOTE_LENGTH,
        key=key,
        tempo_qpm=DEFAULT_TEMPO_QPM,
        melody_only=True,
    )
    write_abc_file(
        score_tree=score,
        output_path=polyphonic_abc_path,
        title=f"{DEFAULT_TITLE} (Polyphonic)",
        meter=meter,
        unit_note_length=DEFAULT_UNIT_NOTE_LENGTH,
        key=key,
        tempo_qpm=DEFAULT_TEMPO_QPM,
    )

    # Print summary
    print(f"\n{'=' * 60}")
    print(f"Summary for {image_path}")
    print(f"{'=' * 60}")
    print(f"Staves: {len(staffs)}")
    print(f"Bar lines: {len(bars)}")
    print(f"Measures: {total_measures}")
    print(f"Notes: {total_notes}")
    print(f"Artifacts written to: {artifacts.root}")
    print(f"ABC file: {abc_path}")
    print(f"Polyphonic ABC file: {polyphonic_abc_path}")
    print(f"{'=' * 60}\n")

    if show_windows:
        cv.imshow(
            "Staff Detection", staff_intermediates.get("04_staff_overlay", raw_bgr)
        )
        bar_overlay = draw_bars_overlay(raw_bgr, bars)
        cv.imshow("Bar Detection", bar_overlay)
        cv.waitKey(0)
        cv.destroyAllWindows()

    return score


def _build_notes_mask(image: MatLike, staffs: list[Staff]) -> MatLike:
    """Build staff-erased mask for note detection."""
    gray = to_gray(image)
    return erase_staff_for_notes(gray, staffs)


def _build_bars_mask(binary: MatLike, staffs: list[Staff]) -> MatLike:
    """Build staff-erased mask for bar detection."""
    return erase_staff_for_bars(binary, staffs)


def _extract_clef_crops(
    sheet_bgr: MatLike,
    notes_mask: MatLike,
    staffs: list,
    bars: list[BarLine],
) -> tuple[dict[int, Clef], dict[int, MatLike]]:
    """Extract clef header crops for each staff."""
    clefs_by_staff = extract_clef_regions(staffs)
    crops = crop_clef_regions(clefs_by_staff, sheet_bgr, notes_mask)
    return clefs_by_staff, crops


def _extract_measures(
    sheet_bgr: MatLike,
    notes_mask: MatLike,
    staffs: list[Staff],
    bars: list[BarLine],
) -> tuple[dict[int, list[Measure]], dict[int, list[MatLike]]]:
    """Extract measures and their crops."""
    measures_map = split_measures(
        bars=bars,
        staffs=staffs,
        left_header_spacings=5.2,
        first_staff_conservative_spacings=7.0,
    )

    measure_crops = crop_measures(
        measures_map=measures_map,
        image=sheet_bgr,
        staffs=staffs,
        notes_image=notes_mask,
    )

    return measures_map, measure_crops


def _detect_first_staff_header_accidentals(
    clef_key_crops: dict[int, MatLike],
    clefs_by_staff: dict[int, Clef],
    staffs: list[Staff],
    bars: list[BarLine],
) -> list[Accidental]:
    """Detect header sharps/flats only for first staff."""
    if not staffs:
        return []

    first_staff_index = 0
    crop = clef_key_crops.get(first_staff_index)
    clef = clefs_by_staff.get(first_staff_index)
    if crop is None or clef is None:
        return []

    try:
        min_x = _header_search_left_x(
            crop=crop,
            staff_spacing=staffs[first_staff_index].spacing,
        )
        max_x = _first_barline_x_limit_for_header(
            bars=bars,
            clef=clef,
            crop_width=crop.shape[1],
            staff_spacing=staffs[first_staff_index].spacing,
        )
        max_x = _header_search_right_x(
            crop=crop,
            default_max_x=max_x,
            staff_spacing=staffs[first_staff_index].spacing,
        )
        accidentals = detect_key_signature_accidentals(
            clef_key_crop=crop,
            staff=staffs[first_staff_index],
            staff_index=first_staff_index,
            x_start=min_x,
            x_end=max_x + 1,
        )
        accidentals = _dedup_accidentals_by_x(
            accidentals=accidentals,
            x_tol=max(2, int(round(staffs[first_staff_index].spacing * 0.60))),
        )
        accidentals = _reclassify_header_accidentals(
            accidentals=accidentals,
            crop=crop,
            staff_spacing=staffs[first_staff_index].spacing,
        )
        accidentals = _dedup_accidentals_by_component(
            accidentals=accidentals,
            crop=crop,
            staff_spacing=staffs[first_staff_index].spacing,
        )
        accidentals = _keep_dominant_accidental_kind(accidentals)
        return accidentals
    except FileNotFoundError:
        # If templates are missing, keep pipeline usable and log no accidentals.
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
    """Save a first-staff header overlay for accidental debugging."""
    if not staffs:
        return

    first_staff_index = 0
    crop = raw_clef_key_crops.get(first_staff_index)
    detection_crop = clef_key_crops.get(first_staff_index)
    clef = clefs_by_staff.get(first_staff_index)
    if crop is None or detection_crop is None or clef is None:
        return

    overlay = crop.copy()
    min_x = _header_search_left_x(
        crop=detection_crop,
        staff_spacing=staffs[first_staff_index].spacing,
    )
    max_x = _first_barline_x_limit_for_header(
        bars=bars,
        clef=clef,
        crop_width=crop.shape[1],
        staff_spacing=staffs[first_staff_index].spacing,
    )
    max_x = _header_search_right_x(
        crop=detection_crop,
        default_max_x=max_x,
        staff_spacing=staffs[first_staff_index].spacing,
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
    """Start accidental search just to the right of the clef component."""
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
    crop: MatLike,
    default_max_x: int,
    staff_spacing: float,
) -> int:
    """Stop accidental search before the rightmost non-clef header symbol."""
    if crop.size == 0:
        return default_max_x

    count, _, stats, _ = cv.connectedComponentsWithStats(crop, connectivity=8)
    min_area = max(8, int(round(staff_spacing * staff_spacing * 0.20)))
    components: list[tuple[int, int, int]] = []

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
    rightmost_x = components[-1][0]
    return max(0, min(default_max_x, rightmost_x - margin))


def _draw_header_accidental_boxes(
    overlay: MatLike,
    detection_crop: MatLike,
    accidentals: list[Accidental],
) -> None:
    """Draw a bounding box around each accidental's connected component."""
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
    """Detect the first staff time signature with pytesseract OCR."""
    if not staffs:
        return None

    first_staff_index = 0
    crop = clef_key_crops.get(first_staff_index)
    clef = clefs_by_staff.get(first_staff_index)
    if crop is None or clef is None or crop.size == 0:
        return None

    max_x = _first_barline_x_limit_for_header(
        bars=bars,
        clef=clef,
        crop_width=crop.shape[1],
        staff_spacing=staffs[first_staff_index].spacing,
    )
    time_roi = _time_signature_roi(
        crop=crop,
        max_x=max_x,
        staff_spacing=staffs[first_staff_index].spacing,
    )
    if time_roi is None:
        return None

    try:
        return _ocr_time_signature(time_roi)
    except pytesseract.TesseractNotFoundError:
        return None


def _apply_detected_key_signature(
    clefs_by_staff: dict[int, Clef],
    header_accidentals: list[Accidental],
) -> None:
    """Apply one detected key signature to all staff clefs."""
    key_signature = _key_signature_from_header_accidentals(header_accidentals)
    for clef in clefs_by_staff.values():
        clef.key_signature = key_signature


def _key_signature_from_header_accidentals(
    header_accidentals: list[Accidental],
) -> KeySignature:
    """Convert counted header accidentals into a circle-of-fifths key signature."""
    sharp_count = sum(1 for glyph in header_accidentals if glyph.kind == "sharp")
    flat_count = sum(1 for glyph in header_accidentals if glyph.kind == "flat")

    if sharp_count and not flat_count:
        fifths = sharp_count
    elif flat_count and not sharp_count:
        fifths = -flat_count
    else:
        fifths = 0

    return KeySignature(fifths=fifths, mode="major")


def _time_signature_roi(
    crop: MatLike,
    max_x: int,
    staff_spacing: float,
) -> MatLike | None:
    """Build a simple OCR ROI in the right side of the first-staff header."""
    if crop.size == 0:
        return None

    x_start = max(0, int(round(crop.shape[1] * 0.40)))
    x_end = min(crop.shape[1], max_x + 1)
    if x_end - x_start < max(6, int(round(staff_spacing * 0.8))):
        return None

    roi = crop[:, x_start:x_end]
    if roi.size == 0:
        return None
    return roi


def _ocr_time_signature(time_roi: MatLike) -> TimeSignature | None:
    """Read numerator/denominator from a stacked time-signature ROI."""
    prepared = _prepare_time_signature_for_ocr(time_roi)
    if prepared.size == 0:
        return None

    top_half, bottom_half = _split_time_signature_halves(prepared)
    top_number = _ocr_single_number(top_half)
    bottom_number = _ocr_single_number(bottom_half)
    if top_number is not None and bottom_number is not None:
        return TimeSignature(numerator=top_number, denominator=bottom_number)

    tokens = _ocr_number_tokens(prepared, psm=6)
    if len(tokens) == 2:
        return TimeSignature(numerator=tokens[0], denominator=tokens[1])

    if _looks_like_common_time(time_roi):
        return TimeSignature(numerator=4, denominator=4)

    return None


def _prepare_time_signature_for_ocr(time_roi: MatLike) -> MatLike:
    """Convert header ROI into a simple black-on-white OCR image."""
    gray = to_gray(time_roi)
    scaled = cv.resize(gray, None, fx=10.0, fy=10.0, interpolation=cv.INTER_CUBIC)
    _, thresholded = cv.threshold(
        scaled, 0, 255, cv.THRESH_BINARY + cv.THRESH_OTSU
    )
    return cv.copyMakeBorder(
        thresholded, 8, 8, 8, 8, cv.BORDER_CONSTANT, value=255
    )


def _split_time_signature_halves(image: MatLike) -> tuple[MatLike, MatLike]:
    """Split OCR image into numerator and denominator halves."""
    mid = image.shape[0] // 2
    top = image[:mid, :]
    bottom = image[mid:, :]
    return top, bottom


def _ocr_single_number(image: MatLike) -> int | None:
    """Read a single integer from an OCR region."""
    tokens = _ocr_number_tokens(image, psm=10)
    if len(tokens) == 1:
        return tokens[0]
    return None


def _ocr_number_tokens(image: MatLike, psm: int) -> list[int]:
    """Extract integer tokens from pytesseract output."""
    if image.size == 0:
        return []

    config = f"--oem 3 --psm {psm} -c tessedit_char_whitelist=0123456789"
    text = pytesseract.image_to_string(image, config=config)
    return [int(token) for token in re.findall(r"\d+", text)]


def _looks_like_common_time(time_roi: MatLike) -> bool:
    """Detect common-time symbol C with a simple OCR pass."""
    gray = to_gray(time_roi)
    scaled = cv.resize(gray, None, fx=8.0, fy=8.0, interpolation=cv.INTER_CUBIC)

    config = "--oem 3 --psm 8 -c tessedit_char_whitelist=Cc"
    text = pytesseract.image_to_string(scaled, config=config).strip().lower()
    return text == "c"


def _first_barline_x_limit_for_header(
    bars: list[BarLine],
    clef: Clef,
    crop_width: int,
    staff_spacing: float,
) -> int:
    """Get crop-local x limit for header search, bounded by first barline."""
    staff_bars = sorted(
        [bar for bar in bars if bar.staff_index == clef.staff_index], key=lambda bar: bar.x
    )
    if not staff_bars:
        return max(0, crop_width - 1)

    margin = max(1, int(round(staff_spacing * 0.2)))
    first_bar_x = staff_bars[0].x
    max_x = first_bar_x - clef.x_start - margin
    return max(0, min(crop_width - 1, max_x))


def _dedup_accidentals_by_x(accidentals: list[Accidental], x_tol: int) -> list[Accidental]:
    """Keep highest-confidence glyph per nearby x-cluster."""
    if len(accidentals) < 2:
        return accidentals

    by_conf = sorted(accidentals, key=lambda glyph: glyph.confidence, reverse=True)
    kept: list[Accidental] = []

    for glyph in by_conf:
        if any(abs(glyph.center_x - other.center_x) <= x_tol for other in kept):
            continue
        kept.append(glyph)

    kept.sort(key=lambda glyph: glyph.center_x)
    return kept


def _keep_dominant_accidental_kind(accidentals: list[Accidental]) -> list[Accidental]:
    """Keep only the dominant accidental family (sharp or flat)."""
    if len(accidentals) < 2:
        return accidentals

    stats = {"sharp": [0, 0.0], "flat": [0, 0.0]}
    for glyph in accidentals:
        stats[glyph.kind][0] += 1
        stats[glyph.kind][1] += glyph.confidence

    if stats["sharp"][0] > stats["flat"][0]:
        dominant = "sharp"
    elif stats["flat"][0] > stats["sharp"][0]:
        dominant = "flat"
    else:
        dominant = "sharp" if stats["sharp"][1] >= stats["flat"][1] else "flat"

    return [glyph for glyph in accidentals if glyph.kind == dominant]


def _reclassify_header_accidentals(
    accidentals: list[Accidental],
    crop: MatLike,
    staff_spacing: float,
) -> list[Accidental]:
    """Use local component stroke structure to correct sharp/flat mislabels."""
    if not accidentals or crop.size == 0:
        return accidentals

    count, labels, stats, _ = cv.connectedComponentsWithStats(crop, connectivity=8)
    min_area = max(8, int(round(staff_spacing * staff_spacing * 0.10)))
    updated: list[Accidental] = []

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
        col_counts = comp.sum(axis=0)
        tall_threshold = max(3, int(round(h * 0.75)))
        tall_cols = [i for i, value in enumerate(col_counts) if value >= tall_threshold]
        tall_clusters = 0
        if tall_cols:
            tall_clusters = 1
            prev = tall_cols[0]
            for value in tall_cols[1:]:
                if value - prev > 1:
                    tall_clusters += 1
                prev = value

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
    accidentals: list[Accidental],
    crop: MatLike,
    staff_spacing: float,
) -> list[Accidental]:
    """Keep the strongest accidental per connected component."""
    if len(accidentals) < 2 or crop.size == 0:
        return accidentals

    count, labels, stats, _ = cv.connectedComponentsWithStats(crop, connectivity=8)
    min_area = max(8, int(round(staff_spacing * staff_spacing * 0.10)))
    best_by_label: dict[int, Accidental] = {}
    no_label: list[Accidental] = []

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
    kept.sort(key=lambda glyph: glyph.center_x)
    return kept


def _populate_notes(score: Score) -> dict[tuple[int, int], dict]:
    """Populate score with detected notes using pure functions.

    In the flat structure, notes are stored both:
    - In their respective measure (measure.notes)
    - In the flat score.notes list

    Returns:
        Dictionary mapping (staff_index, measure_index) to intermediates dict
    """
    score.notes = []  # Start fresh
    intermediates_by_measure: dict[tuple[int, int], dict] = {}

    for staff_index, staff in enumerate(score.staffs):
        clef = score.clefs.get(staff_index)
        staff_measures = score.get_measures_for_staff(staff_index)

        for measure_index, measure in enumerate(staff_measures):
            if measure.crop is None:
                continue

            result = find_notes(
                mask=measure.crop,
                staff=staff,
                measure=measure,
                measure_index=measure_index,
                return_intermediates=True,
            )
            # result is tuple when return_intermediates=True
            assert isinstance(result, tuple)
            detected_notes = result[0]
            intermediates = result[1]
            assert isinstance(intermediates, dict)
            intermediates_by_measure[(staff_index, measure_index)] = intermediates

            detected_notes = _remove_left_edge_header_bleed(
                notes=detected_notes,
                staff=staff,
                measure_index=measure_index,
            )

            # Phase 2: Beam-aware rhythm refinement
            # This only updates duration_class, never adds new notes
            detected_notes = refine_beamed_durations(
                mask=measure.crop,
                notes=detected_notes,
                staff=staff,
            )

            resolve_pitches(detected_notes, clef)

            measure.notes = detected_notes
            score.notes.extend(detected_notes)

    return intermediates_by_measure


def _remove_left_edge_header_bleed(
    notes: list[Note],
    staff: Staff,
    measure_index: int,
) -> list[Note]:
    """Remove notes that are actually clef/key header artifacts."""
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


def _create_full_overlays(
    score: Score,
    clefs_by_staff: dict[int, Clef],
    clef_detections: dict[int, ClefDetection],
    artifacts: ArtifactWriter,
) -> None:
    """Create full-sheet overlays for clefs and notes."""
    # Clef overlay on full sheet
    clef_overlay = score.sheet_image.copy()
    font = cv.FONT_HERSHEY_SIMPLEX

    for staff_index, staff in enumerate(score.staffs):
        clef = clefs_by_staff.get(staff_index)
        det = clef_detections.get(staff_index)
        if clef is None or det is None:
            continue

        x1, y1 = clef.x_start, clef.y_top
        x2, y2 = clef.x_end, clef.y_bottom
        cv.rectangle(clef_overlay, (x1, y1), (x2, y2), (100, 100, 100), 1)

        # Draw template matching boxes
        choice = _choose_clef_overlay_rect(clef, det)
        if choice is not None:
            rect, color = choice
            _draw_overlay_box(clef_overlay, rect, color, origin_x=x1, origin_y=y1)

        # Add label
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


def _choose_clef_overlay_rect(clef: Clef, detection: ClefDetection):
    """Choose which template match box to draw for clef overlay."""
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


def _draw_overlay_box(
    image, rect, color, *, origin_x: int = 0, origin_y: int = 0, thickness: int = 3
):
    """Draw a rectangle with optional origin offset."""
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


def _build_clef_log(score: Score) -> str:
    """Build log string for clef detections."""
    lines: list[str] = []
    for staff_index, detection in score.clef_detections.items():
        lines.append(
            f"staff {staff_index}: {detection.clef!r}  "
            f"letter T/B={detection.letter_score_treble:.3f}/{detection.letter_score_bass:.3f}  "
            f"slide T/B={detection.slide_score_treble:.3f}/{detection.slide_score_bass:.3f}\n"
        )
    return "".join(lines)


def _build_accidental_log(score: Score) -> str:
    """Build log string for first-staff header accidental counts."""
    clef = score.clefs.get(0)
    if clef is None:
        return "staff 0 header accidentals: sharps=0 flats=0 total=0\n"

    glyphs = clef.key_header_glyphs
    sharp_count = sum(1 for glyph in glyphs if glyph.kind == "sharp")
    flat_count = sum(1 for glyph in glyphs if glyph.kind == "flat")

    lines = [
        f"staff 0 header accidentals: sharps={sharp_count} flats={flat_count} total={len(glyphs)}\n"
    ]
    for glyph in glyphs:
        lines.append(
            f"  kind={glyph.kind:<5} x={glyph.center_x:>4}, y={glyph.center_y:>4}, conf={glyph.confidence:.3f}\n"
        )
    return "".join(lines)


def _build_time_signature_log(score: Score) -> str:
    """Build log string for first-staff time signature."""
    clef = score.clefs.get(0)
    if clef is None:
        return "staff 0 time signature: ?\n"

    time_signature = clef.time_signature
    if (
        time_signature.numerator is None
        or time_signature.denominator is None
    ):
        return "staff 0 time signature: ?\n"

    return (
        f"staff 0 time signature: "
        f"{time_signature.numerator}/{time_signature.denominator}\n"
    )


def _build_key_signature_log(score: Score) -> str:
    """Build log string for first-staff key signature."""
    clef = score.clefs.get(0)
    if clef is None:
        return "staff 0 key signature: C\n"

    return f"staff 0 key signature: {_abc_key_from_score(score)}\n"


def _meter_from_score(score: Score) -> str:
    """Build ABC meter from detected first-staff time signature."""
    clef = score.clefs.get(0)
    if clef is None:
        return DEFAULT_METER

    time_signature = clef.time_signature
    if (
        time_signature.numerator is None
        or time_signature.denominator is None
    ):
        return DEFAULT_METER

    return f"{time_signature.numerator}/{time_signature.denominator}"


def _abc_key_from_score(score: Score) -> str:
    """Build ABC key name from detected first-staff key signature."""
    clef = score.clefs.get(0)
    if clef is None or clef.key_signature.fifths is None:
        return DEFAULT_KEY

    return _abc_key_from_fifths(clef.key_signature.fifths)


def _abc_key_from_fifths(fifths: int) -> str:
    """Map circle-of-fifths count to ABC major key."""
    major_keys = {
        -7: "Cb",
        -6: "Gb",
        -5: "Db",
        -4: "Ab",
        -3: "Eb",
        -2: "Bb",
        -1: "F",
        0: "C",
        1: "G",
        2: "D",
        3: "A",
        4: "E",
        5: "B",
        6: "F#",
        7: "C#",
    }
    return major_keys.get(fifths, DEFAULT_KEY)


def _build_note_log(score: Score) -> str:
    """Build log string for note detections."""
    lines: list[str] = []

    # Group notes by staff and measure
    notes_by_staff_measure: dict[tuple[int, int], list[Note]] = {}
    for note in score.notes:
        key = (note.staff_index, note.measure_index)
        if key not in notes_by_staff_measure:
            notes_by_staff_measure[key] = []
        notes_by_staff_measure[key].append(note)

    for staff_index, staff in enumerate(score.staffs):
        staff_measures = score.get_measures_for_staff(staff_index)
        for measure_index, measure in enumerate(staff_measures):
            notes = notes_by_staff_measure.get((staff_index, measure_index), [])
            lines.append(
                f"staff {staff_index}, measure {measure_index}: {len(notes)} noteheads detected\n"
            )
            for note in notes:
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
