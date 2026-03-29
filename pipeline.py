"""Complete sheet music processing pipeline with full visualization."""

from statistics import median

import cv2 as cv
from cv2.typing import MatLike

from abc_export import write_abc_file
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
from schema import BarLine, Clef, ClefDetection, Measure, Note, Score, Staff
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

    # Initialize artifact writer
    artifacts = ArtifactWriter(image_path=image_path)

    # Load image
    raw_bgr = cv.imread(filename=image_path)
    if raw_bgr is None:
        raise FileNotFoundError(f"Could not load image: {image_path}")

    # Step 1: Staff Detection
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

    # Step 2: Prepare masks
    print("Step 2: Preparing staff-erased masks...")
    notes_mask = _build_notes_mask(raw_bgr, staffs)
    bars_mask = _build_bars_mask(binary, staffs)

    # Save masks
    artifacts.write_image(
        artifacts.sections.masks, "02_notes_mask_erased.jpg", cv.bitwise_not(notes_mask)
    )
    artifacts.write_image(
        artifacts.sections.masks, "03_bars_mask_erased.jpg", cv.bitwise_not(bars_mask)
    )

    # Step 3: Bar Detection
    print("Step 3: Detecting bar lines...")
    bars = find_bars(image=bars_mask, staffs=staffs)
    bar_intermediates = save_bar_visualization(
        image=raw_bgr,  # Original sheet for overlay
        bars_mask=bars_mask,  # Binary mask for debugging
        staffs=staffs,
        bars=bars,
        artifacts=artifacts,
    )
    print(f"  Found {len(bars)} bar line(s)")

    # Step 4: Clef Detection
    print("Step 4: Detecting clefs...")
    clefs_by_staff, clef_key_crops = _extract_clef_crops(
        raw_bgr, notes_mask, staffs, bars
    )
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

    # Step 5: Measure Splitting
    print("Step 5: Splitting into measures...")
    measures_map, measure_crops = _extract_measures(raw_bgr, notes_mask, staffs, bars)

    # Save measure intermediates
    measure_intermediates = save_measure_visualization(
        sheet_image=raw_bgr,
        measures_map=measures_map,
        measure_crops=measure_crops,
        artifacts=artifacts,
    )

    total_measures = sum(len(measures) for measures in measures_map.values())
    print(f"  Split into {total_measures} measure(s)")

    # Step 6: Build Score
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

    # Step 7: Note Detection
    print("Step 7: Detecting notes...")
    note_intermediates = _populate_notes(score)
    total_notes = len(score.notes)
    print(f"  Detected {total_notes} note(s)")

    # Step 8: Draw full overlays
    print("Step 8: Creating visualizations...")
    _create_full_overlays(score, clefs_by_staff, clef_detections, artifacts)
    save_notes_visualization(
        notes_mask=notes_mask,
        score=score,
        artifacts=artifacts,
        intermediates_by_measure=note_intermediates,
    )

    # Step 9: Generate logs
    print("Step 9: Generating logs...")
    clef_log = _build_clef_log(score)
    note_log = _build_note_log(score)
    artifacts.write_text(artifacts.sections.logs, "detections.txt", clef_log + note_log)

    # Step 10: Export to ABC notation
    print("Step 10: Exporting to ABC notation...")
    abc_path = artifacts.text_path(artifacts.sections.export, "output.abc")
    write_abc_file(
        score_tree=score,  # Score is backward compatible with ScoreTree
        output_path=abc_path,
        title=DEFAULT_TITLE,
        meter=DEFAULT_METER,
        unit_note_length=DEFAULT_UNIT_NOTE_LENGTH,
        key=DEFAULT_KEY,
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
    print(f"{'=' * 60}\n")

    # Show windows if requested
    if show_windows:
        cv.imshow(
            "Staff Detection", staff_intermediates.get("04_staff_overlay", raw_bgr)
        )
        bar_overlay = draw_bars_overlay(raw_bgr, bars)
        cv.imshow("Bar Detection", bar_overlay)
        # TODO: Re-add note and clef visualization after visualization module refactor
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
    # Split into measures with custom settings
    # Use a slightly earlier general content start for note evaluation.
    # Keep first staff conservative to avoid clef/key bleed in bar 1.
    measures_map = split_measures(
        bars=bars,
        staffs=staffs,
        left_header_spacings=5.2,
        first_staff_conservative_spacings=7.0,
    )

    # Crop the measures from notes_mask
    measure_crops = crop_measures(
        measures_map=measures_map,
        image=sheet_bgr,
        staffs=staffs,
        notes_image=notes_mask,
    )

    return measures_map, measure_crops


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
            resolve_pitches(detected_notes, clef)

            # Store in measure
            measure.notes = detected_notes
            # Also add to flat list
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
