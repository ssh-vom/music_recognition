"""Note detection - find noteheads and figure out pitch/duration."""

import math

import cv2 as cv
import numpy as np
from cv2.typing import MatLike

from schema import (
    Clef,
    DurationClass,
    KeySignature,
    Measure,
    Note,
    Staff,
    StepConfidence,
)

LETTER_ORDER = ("C", "D", "E", "F", "G", "A", "B")
LETTER_TO_INDEX = {letter: index for index, letter in enumerate(LETTER_ORDER)}
SHARP_ORDER = ("F", "C", "G", "D", "A", "E", "B")
FLAT_ORDER = ("B", "E", "A", "D", "G", "C", "F")


def find_notes(
    mask: MatLike,
    staff: Staff,
    measure: Measure,
    measure_index: int,
    return_intermediates: bool = False,
):
    intermediates = {} if return_intermediates else None

    kernel_diameter = max(1, int(round(staff.spacing * 0.45)))
    if kernel_diameter % 2 == 0:
        kernel_diameter += 1

    notehead_kernel = cv.getStructuringElement(
        cv.MORPH_ELLIPSE, (kernel_diameter, kernel_diameter)
    )
    opened_mask = cv.morphologyEx(mask, cv.MORPH_OPEN, notehead_kernel)

    close_kernel = cv.getStructuringElement(cv.MORPH_ELLIPSE, (3, 3))
    notehead_mask = cv.morphologyEx(opened_mask, cv.MORPH_CLOSE, close_kernel)
    secondary_mask = cv.morphologyEx(mask, cv.MORPH_CLOSE, close_kernel)

    if intermediates is not None:
        intermediates["opened_mask"] = opened_mask.copy()
        intermediates["notehead_mask"] = notehead_mask.copy()
        intermediates["secondary_mask"] = secondary_mask.copy()

    count, _, stats, centroids = cv.connectedComponentsWithStats(
        notehead_mask, connectivity=8
    )
    secondary_count, _, secondary_stats, secondary_centroids = (
        cv.connectedComponentsWithStats(secondary_mask, connectivity=8)
    )

    if intermediates is not None:
        intermediates["connected_components"] = {
            "count": count,
            "stats": stats.copy(),
            "centroids": centroids.copy(),
        }

    min_area = staff.spacing * staff.spacing * 0.08
    max_area = staff.spacing * staff.spacing * 1.8
    min_size = int(round(staff.spacing * 0.35))
    max_size = int(round(staff.spacing * 1.9))
    min_aspect = 0.45
    max_aspect = 2.2
    tiny_area = staff.spacing * staff.spacing * 0.22

    raw_centers = []
    filtered_info = [] if intermediates is not None else None

    for i in range(1, count):
        w = int(stats[i, cv.CC_STAT_WIDTH])
        h = int(stats[i, cv.CC_STAT_HEIGHT])
        area = float(stats[i, cv.CC_STAT_AREA])

        passed = True
        if area < min_area or area > max_area:
            passed = False
        if w < min_size or h < min_size or w > max_size or h > max_size:
            passed = False

        aspect = w / float(h)
        if aspect < min_aspect or aspect > max_aspect:
            passed = False

        cx = int(round(centroids[i][0]))
        cy = int(round(centroids[i][1]))

        if area <= tiny_area:
            refined = _refine_from_secondary_mask(
                cx,
                cy,
                secondary_count,
                secondary_stats,
                secondary_centroids,
                staff.spacing,
                mask.shape[1] - 1,
                mask.shape[0] - 1,
            )
            if refined:
                cx, cy = refined

        if passed:
            raw_centers.append((cx, cy))

        if filtered_info is not None:
            filtered_info.append(
                {
                    "id": i,
                    "x": cx,
                    "y": cy,
                    "w": w,
                    "h": h,
                    "area": area,
                    "aspect": aspect,
                    "passed": passed,
                }
            )

    if intermediates is not None:
        intermediates["filtered_components"] = filtered_info
        intermediates["raw_centers_before_merge"] = raw_centers.copy()

    merge_dist = max(2, int(round(staff.spacing * 0.75)))
    raw_centers.sort(key=lambda c: c[0])

    merged = _merge_centers(raw_centers, merge_dist)

    if intermediates is not None:
        intermediates["centers_after_merge"] = merged.copy()

    stem_info = {} if intermediates is not None else None
    merged = _augment_from_stems(
        mask, merged, staff.spacing, mask.shape[1], merge_dist, stem_info
    )

    if intermediates is not None:
        intermediates["stem_augmentation"] = stem_info
        intermediates["centers_after_stems"] = merged.copy()

    bottom_line_y = int(round(staff.lines[4].y - measure.y_top))
    half_step = staff.spacing / 2.0

    notes = []
    for cx, cy, _ in merged:
        # Classify duration first to detect hollow noteheads
        duration = _classify_duration(mask, cx, cy, staff.spacing)
        
        # For hollow noteheads (half/whole), centroid tends to be biased
        # upward due to the empty center. Apply a small correction.
        # This correction compensates for the geometric center of hollow ovals
        # being higher than the visual center due to ink distribution.
        cy_adjusted = cy
        if duration in ("half", "whole"):
            # Conservative correction: 0.15 * spacing instead of 0.25
            cy_adjusted = cy + int(round(staff.spacing * 0.15))
        
        step_float = (bottom_line_y - cy_adjusted) / half_step
        step = _quantize_step(step_float)
        residual = abs(step_float - step)
        confidence = _step_confidence(residual)

        notes.append(
            Note(
                kind="notehead",
                staff_index=measure.staff_index,
                measure_index=measure_index,
                center_x=cx,
                center_y=cy,  # Keep original cy for visualization
                step=step,
                step_confidence=confidence,
                duration_class=duration,
            )
        )

    notes.sort(key=lambda n: n.center_x)
    final_notes = _collapse_duplicates(notes, staff.spacing, mask)

    if intermediates is not None:
        intermediates["final_notes"] = final_notes

    if return_intermediates:
        return final_notes, intermediates
    return final_notes


def _merge_centers(centers, merge_dist: int):
    if not centers:
        return []

    merged = [[centers[0][0], centers[0][1], 1]]

    for cx, cy in centers[1:]:
        last_x, last_y, last_count = merged[-1]

        if abs(cx - last_x) <= merge_dist and abs(cy - last_y) <= merge_dist:
            new_count = last_count + 1
            new_x = int(round((last_x * last_count + cx) / new_count))
            new_y = int(round((last_y * last_count + cy) / new_count))
            merged[-1] = [new_x, new_y, new_count]
        else:
            merged.append([cx, cy, 1])

    return merged


def _quantize_step(step_float: float) -> int:
    # Bias downward since notehead center is often above geometric center
    STEP_ROUND_UP_THRESHOLD = 0.58

    lower = math.floor(step_float)
    if step_float - lower >= STEP_ROUND_UP_THRESHOLD:
        return lower + 1
    return lower


def _step_confidence(residual: float) -> StepConfidence:
    if residual <= 0.20:
        return "high"
    if residual <= 0.40:
        return "medium"
    return "low"


def _augment_from_stems(
    mask: MatLike,
    centers,
    spacing: float,
    width: int,
    merge_dist: int,
    stem_info: dict | None = None,
):
    if len(centers) > 2:
        if stem_info is not None:
            stem_info["skipped"] = True
            stem_info["reason"] = "too_many_notes"
        return centers

    count, _, stats, _ = cv.connectedComponentsWithStats(mask, connectivity=8)
    augmented = [c.copy() for c in centers]

    margin = max(2, int(round(spacing * 0.6)))
    added = 0
    all_stems = [] if stem_info is not None else None
    added_stems = [] if stem_info is not None else None

    for i in range(1, count):
        x = int(stats[i, cv.CC_STAT_LEFT])
        y = int(stats[i, cv.CC_STAT_TOP])
        w = int(stats[i, cv.CC_STAT_WIDTH])
        h = int(stats[i, cv.CC_STAT_HEIGHT])
        area = float(stats[i, cv.CC_STAT_AREA])

        stem_data = {"id": i, "x": x, "y": y, "w": w, "h": h, "area": area}

        if h < int(round(spacing * 2.0)):
            stem_data["rejected"] = "too_short"
            if all_stems is not None:
                all_stems.append(stem_data)
            continue
        if w > int(round(spacing * 1.5)):
            stem_data["rejected"] = "too_wide"
            if all_stems is not None:
                all_stems.append(stem_data)
            continue
        if area < spacing * spacing * 0.35:
            stem_data["rejected"] = "too_small"
            if all_stems is not None:
                all_stems.append(stem_data)
            continue
        if x + w >= width - margin:
            stem_data["rejected"] = "at_edge"
            if all_stems is not None:
                all_stems.append(stem_data)
            continue

        cx = x + w // 2
        cy = y + h - max(1, int(round(spacing * 0.55)))

        overlaps = False
        overlap_x_dist = int(round(merge_dist * 1.2))
        overlap_y_dist = int(round(spacing * 1.2))

        for ex, ey, _ in augmented:
            if abs(cx - ex) <= overlap_x_dist and abs(cy - ey) <= overlap_y_dist:
                overlaps = True
                break

        if overlaps:
            stem_data["rejected"] = "overlaps"
            if all_stems is not None:
                all_stems.append(stem_data)
            continue

        augmented.append([cx, cy, 1])
        added += 1
        stem_data["added"] = True
        stem_data["note_x"] = cx
        stem_data["note_y"] = cy
        if all_stems is not None:
            all_stems.append(stem_data)
        if added_stems is not None:
            added_stems.append(stem_data)
        if added >= 1:
            break

    if stem_info is not None:
        stem_info["all_stems"] = all_stems
        stem_info["added_stems"] = added_stems
        stem_info["original_count"] = len(centers)
        stem_info["final_count"] = len(augmented)

    augmented.sort(key=lambda c: c[0])
    return augmented


def _collapse_duplicates(notes: list[Note], spacing: float, mask: MatLike | None = None) -> list[Note]:
    if len(notes) < 2:
        return notes

    x_tol = max(2, int(round(spacing * 1.45)))
    y_tol = max(2, int(round(spacing * 0.75)))

    collapsed = [notes[0]]

    for note in notes[1:]:
        prev = collapsed[-1]

        # Original condition: both have no duration assigned
        is_duplicate = (
            prev.duration_class is None
            and note.duration_class is None
            and abs(note.center_x - prev.center_x) <= x_tol
            and abs(note.center_y - prev.center_y) <= y_tol
            and abs(note.step - prev.step) <= 1
        )
        
        # New condition: two filled noteheads without stems at same position
        # likely form a hollow notehead (whole/half) that was split
        # Only merge if they're close (hollow center gap is typically < 1.8 * spacing)
        hollow_x_tol = max(2, int(round(spacing * 1.8)))
        is_hollow_pair = (
            prev.duration_class in ("quarter", "whole")
            and note.duration_class in ("quarter", "whole")
            and abs(note.center_x - prev.center_x) <= hollow_x_tol
            and abs(note.center_y - prev.center_y) <= y_tol
            and abs(note.step - prev.step) <= 1
            and _is_likely_hollow_split(mask, prev, note, spacing)
        )

        if is_duplicate or is_hollow_pair:
            prev.center_x = int(round((prev.center_x + note.center_x) / 2.0))
            prev.center_y = int(round((prev.center_y + note.center_y) / 2.0))
            prev.step = int(round((prev.step + note.step) / 2.0))
            prev.step_confidence = (
                prev.step_confidence
                if prev.step_confidence == "high"
                else note.step_confidence
            )
            # If we're merging two filled components, this is likely a whole note
            # that was split by its hollow center
            if is_hollow_pair:
                prev.duration_class = "whole"
        else:
            collapsed.append(note)

    return collapsed


def _is_likely_hollow_split(
    mask: MatLike | None, 
    note1: Note, 
    note2: Note, 
    spacing: float
) -> bool:
    """Check if two close noteheads are likely halves of a hollow notehead."""
    if mask is None:
        # Without mask, use distance heuristic: hollow notehead halves are 
        # typically 0.5-1.5x spacing apart
        dist = abs(note1.center_x - note2.center_x)
        return spacing * 0.4 <= dist <= spacing * 1.5
    
    # Check if there's a gap (white space) between the two components
    # This indicates a hollow center
    x1 = min(note1.center_x, note2.center_x)
    x2 = max(note1.center_x, note2.center_x)
    y_mid = (note1.center_y + note2.center_y) // 2
    
    # Sample the region between the two noteheads
    mid_x = (x1 + x2) // 2
    roi_y_start = max(0, y_mid - int(spacing * 0.3))
    roi_y_end = min(mask.shape[0], y_mid + int(spacing * 0.3))
    roi_x_start = max(0, mid_x - 2)
    roi_x_end = min(mask.shape[1], mid_x + 3)
    
    if roi_y_start >= roi_y_end or roi_x_start >= roi_x_end:
        return False
        
    roi = mask[roi_y_start:roi_y_end, roi_x_start:roi_x_end]
    if roi.size == 0:
        return False
    
    # If the middle region has low ink density, it's likely hollow
    ink_ratio = cv.countNonZero(roi) / float(roi.size)
    return ink_ratio < 0.3  # Less than 30% ink suggests a gap/hollow center


def _refine_from_secondary_mask(
    cx: int,
    cy: int,
    count: int,
    stats,
    centroids,
    spacing: float,
    max_x: int,
    max_y: int,
):
    tol = max(2, int(round(spacing * 0.95)))
    min_h = max(6, int(round(spacing * 2.0)))
    min_area = spacing * spacing * 0.30

    best_idx = -1
    best_dx = float("inf")

    for i in range(1, count):
        w = int(stats[i, cv.CC_STAT_WIDTH])
        h = int(stats[i, cv.CC_STAT_HEIGHT])
        area = float(stats[i, cv.CC_STAT_AREA])
        center_x = int(round(float(centroids[i][0])))

        if h < min_h or area < min_area or w > int(round(spacing * 1.7)):
            continue

        dx = abs(center_x - cx)
        if dx > tol or dx >= best_dx:
            continue

        best_dx = dx
        best_idx = i

    if best_idx < 0:
        return None

    x = int(stats[best_idx, cv.CC_STAT_LEFT])
    y = int(stats[best_idx, cv.CC_STAT_TOP])
    w = int(stats[best_idx, cv.CC_STAT_WIDTH])
    h = int(stats[best_idx, cv.CC_STAT_HEIGHT])

    rx = x + w // 2
    ry = y + h - int(round(spacing * 0.90))
    rx = max(0, min(max_x, rx))
    ry = max(0, min(max_y, ry))

    return rx, ry


def _classify_duration(mask: MatLike, cx: int, cy: int, spacing: float) -> DurationClass | None:
    filled = _is_filled(mask, cx, cy, spacing)
    has_stem = _has_stem(mask, cx, cy, spacing)

    if not filled and not has_stem:
        return "whole"
    if not filled and has_stem:
        return "half"
    if filled and has_stem:
        return "quarter"
    if filled and not has_stem:
        # Edge case: filled notehead without detected stem
        # Could be a whole note misclassified as filled due to staff remnants,
        # or a quarter note with broken stem connectivity.
        # Use size heuristic: whole notes are typically larger than quarter noteheads.
        rx = max(2, int(round(spacing * 0.36)))
        ry = max(2, int(round(spacing * 0.28)))
        x1 = max(0, cx - rx)
        x2 = min(mask.shape[1], cx + rx + 1)
        y1 = max(0, cy - ry)
        y2 = min(mask.shape[0], cy + ry + 1)
        roi = mask[y1:y2, x1:x2]
        if roi.size > 0:
            # Whole noteheads are typically larger in appearance
            # Check the actual ink extent
            ink_pixels = cv.countNonZero(roi)
            roi_area = roi.shape[0] * roi.shape[1]
            fill_ratio = ink_pixels / float(roi_area) if roi_area > 0 else 0
            # Whole notes (hollow) have lower fill ratio than filled noteheads
            # but higher than true hollow due to staff remnants
            if fill_ratio < 0.35:
                return "whole"
        return "quarter"
    return None


def _is_filled(mask: MatLike, cx: int, cy: int, spacing: float) -> bool:
    rx = max(2, int(round(spacing * 0.36)))
    ry = max(2, int(round(spacing * 0.28)))

    x1 = max(0, cx - rx)
    x2 = min(mask.shape[1], cx + rx + 1)
    y1 = max(0, cy - ry)
    y2 = min(mask.shape[0], cy + ry + 1)

    roi = mask[y1:y2, x1:x2]
    if roi.size == 0:
        return False

    ellipse_mask = np.zeros(roi.shape, dtype=np.uint8)
    local_center = (cx - x1, cy - y1)
    cv.ellipse(
        ellipse_mask,
        local_center,
        (max(1, rx - 1), max(1, ry - 1)),
        0,
        0,
        360,
        255,
        -1,
    )

    ellipse_area = cv.countNonZero(ellipse_mask)
    if ellipse_area <= 0:
        return False

    ink = cv.countNonZero(cv.bitwise_and(roi, roi, mask=ellipse_mask))
    ink_ratio = ink / float(ellipse_area)

    return ink_ratio >= 0.55


def _has_stem(mask: MatLike, cx: int, cy: int, spacing: float) -> bool:
    x_radius = max(2, int(round(spacing * 0.85)))
    y_radius = max(3, int(round(spacing * 2.6)))

    x1 = max(0, cx - x_radius)
    x2 = min(mask.shape[1], cx + x_radius + 1)
    y1 = max(0, cy - y_radius)
    y2 = min(mask.shape[0], cy + y_radius + 1)

    roi = mask[y1:y2, x1:x2]
    if roi.size == 0 or roi.shape[1] < 2:
        return False

    min_run = max(3, int(round(spacing * 1.2)))

    for x in range(roi.shape[1]):
        run = 0
        best = 0
        for y in range(roi.shape[0]):
            if roi[y, x] > 0:
                run += 1
                best = max(best, run)
            else:
                run = 0
        if best >= min_run:
            return True

    return False


def resolve_pitches(notes: list[Note], clef: Clef | None) -> None:
    if clef is None or clef.kind is None:
        return

    if clef.kind == "treble":
        base_letter, base_octave = "E", 4
    elif clef.kind == "bass":
        base_letter, base_octave = "G", 2
    else:
        return

    key_accidentals = _key_signature_accidentals(clef.key_signature)

    for note in notes:
        letter, octave = _step_to_letter_octave(base_letter, base_octave, note.step)
        accidental = key_accidentals.get(letter, "")
        note.pitch_letter = f"{letter}{accidental}"
        note.octave = octave


def _step_to_letter_octave(
    base_letter: str, base_octave: int, step: int
) -> tuple[str, int]:
    base_index = LETTER_TO_INDEX[base_letter]
    absolute = base_octave * 7 + base_index + step
    octave = absolute // 7
    letter_index = absolute % 7
    return LETTER_ORDER[letter_index], octave


def _key_signature_accidentals(key_sig: KeySignature) -> dict[str, str]:
    accidentals = {}
    fifths = key_sig.fifths if key_sig.fifths is not None else 0

    if fifths > 0:
        for letter in SHARP_ORDER[:fifths]:
            accidentals[letter] = "#"
    elif fifths < 0:
        for letter in FLAT_ORDER[: abs(fifths)]:
            accidentals[letter] = "b"

    return accidentals
