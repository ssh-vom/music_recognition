"""Note detection - find noteheads and resolve pitch/duration."""

import math

import cv2 as cv
import numpy as np
from cv2.typing import MatLike

from constants import (
    DUPLICATE_MAX_STEP_DIFF,
    DUPLICATE_X_TOLERANCE_FRAC,
    DUPLICATE_Y_TOLERANCE_FRAC,
    FILL_ELLIPSE_X_RADIUS_FRAC,
    FILL_ELLIPSE_Y_RADIUS_FRAC,
    FILL_RATIO_THRESHOLD,
    HOLLOW_NOTE_Y_OFFSET_FRAC,
    HOLLOW_SPLIT_INK_RATIO_MAX,
    HOLLOW_SPLIT_X_TOLERANCE_FRAC,
    NOTEHEAD_CLEANUP_KERNEL,
    NOTEHEAD_KERNEL_DIAMETER_FRAC,
    NOTEHEAD_KERNEL_MIN,
    NOTE_MAX_AREA_FRAC,
    NOTE_MAX_ASPECT,
    NOTE_MAX_SIZE_FRAC,
    NOTE_MERGE_DISTANCE_FRAC,
    NOTE_MIN_AREA_FRAC,
    NOTE_MIN_ASPECT,
    NOTE_MIN_SIZE_FRAC,
    NOTE_TINY_AREA_FRAC,
    STEP_CONFIDENCE_HIGH,
    STEP_CONFIDENCE_MEDIUM,
    STEP_ROUND_UP_THRESHOLD,
    STEM_MIN_RUN_FRAC,
    STEM_X_RADIUS_FRAC,
    STEM_Y_RADIUS_FRAC,
    WHOLE_NOTE_FILL_RATIO_MAX,
)
from schema import Clef, DurationClass, KeySignature, Measure, Note, Staff, StepConfidence

# Pitch letter mappings
PITCH_CYCLE = ("C", "D", "E", "F", "G", "A", "B")
PITCH_TO_INDEX = {letter: idx for idx, letter in enumerate(PITCH_CYCLE)}
CIRCLE_OF_FIFTHS_SHARPS = ("F", "C", "G", "D", "A", "E", "B")
CIRCLE_OF_FIFTHS_FLATS = ("B", "E", "A", "D", "G", "C", "F")

# Clef base positions: which letter/octave sits on the bottom line
CLEF_BASE_POSITIONS = {
    "treble": ("E", 4),
    "bass": ("G", 2),
}


def find_notes(
    mask: MatLike,
    staff: Staff,
    measure: Measure,
    measure_index: int,
    return_intermediates: bool = False,
):
    """Detect noteheads in a measure and resolve their pitch and duration.
    
    Args:
        mask: Binary image with staff lines removed (ink = 255)
        staff: Staff containing this measure
        measure: Measure boundaries
        measure_index: Index of this measure within the staff
        return_intermediates: If True, return debug visualization data
        
    Returns:
        List of detected Note objects, or (notes, intermediates) tuple
    """
    intermediates = {} if return_intermediates else None

    # Isolate noteheads via morphological opening with elliptical kernel
    notehead_mask = _extract_notehead_mask(mask, staff.spacing)
    secondary_mask = _create_secondary_mask(mask)

    if intermediates:
        intermediates["notehead_mask"] = notehead_mask.copy()
        intermediates["secondary_mask"] = secondary_mask.copy()

    # Find connected components (candidate noteheads)
    components = cv.connectedComponentsWithStats(notehead_mask, connectivity=8)
    secondary = cv.connectedComponentsWithStats(secondary_mask, connectivity=8)
    
    # Filter components by geometric properties
    centers = _filter_notehead_candidates(
        components, secondary, staff.spacing, mask.shape, intermediates
    )

    # Merge nearby centers (handles chords and touching noteheads)
    merge_distance = max(2, int(round(staff.spacing * NOTE_MERGE_DISTANCE_FRAC)))
    centers = _merge_nearby_centers(centers, merge_distance)
    
    if intermediates:
        intermediates["centers_after_merge"] = centers.copy()

    # Add missing stems that weren't captured by notehead morphology
    centers = _add_stem_centers(mask, centers, staff.spacing, merge_distance, intermediates)

    # Convert centers to Note objects with pitch and duration
    notes = _resolve_notes(centers, mask, staff, measure, measure_index)
    
    # Merge duplicate detections (hollow noteheads split into two components)
    notes = _merge_duplicate_detections(notes, staff.spacing, mask)

    if intermediates is not None:
        intermediates["final_notes"] = notes
        return notes, intermediates
    return notes


def _extract_notehead_mask(mask: MatLike, spacing: float) -> MatLike:
    """Apply morphological opening to isolate elliptical noteheads."""
    diameter = max(NOTEHEAD_KERNEL_MIN, int(round(spacing * NOTEHEAD_KERNEL_DIAMETER_FRAC)))
    if diameter % 2 == 0:
        diameter += 1  # Ensure odd for symmetry
    
    kernel = cv.getStructuringElement(cv.MORPH_ELLIPSE, (diameter, diameter))
    opened = cv.morphologyEx(mask, cv.MORPH_OPEN, kernel)
    
    # Small closing to heal minor gaps
    cleanup_kernel = cv.getStructuringElement(cv.MORPH_ELLIPSE, NOTEHEAD_CLEANUP_KERNEL)
    return cv.morphologyEx(opened, cv.MORPH_CLOSE, cleanup_kernel)


def _create_secondary_mask(mask: MatLike) -> MatLike:
    """Create a less aggressively filtered mask for tiny notehead recovery."""
    cleanup_kernel = cv.getStructuringElement(cv.MORPH_ELLIPSE, NOTEHEAD_CLEANUP_KERNEL)
    return cv.morphologyEx(mask, cv.MORPH_CLOSE, cleanup_kernel)


def _filter_notehead_candidates(
    components, secondary, spacing: float, shape: tuple, intermediates: dict | None
) -> list[tuple[int, int]]:
    """Filter connected components by geometric criteria for noteheads."""
    count, _, stats, centroids = components
    s_count, _, s_stats, s_centroids = secondary
    
    # Size thresholds relative to staff spacing
    min_area = spacing * spacing * NOTE_MIN_AREA_FRAC
    max_area = spacing * spacing * NOTE_MAX_AREA_FRAC
    min_size = int(round(spacing * NOTE_MIN_SIZE_FRAC))
    max_size = int(round(spacing * NOTE_MAX_SIZE_FRAC))
    tiny_area = spacing * spacing * NOTE_TINY_AREA_FRAC
    
    centers = []
    filtered_log = [] if intermediates else None
    
    for i in range(1, count):  # Skip background (0)
        w = int(stats[i, cv.CC_STAT_WIDTH])
        h = int(stats[i, cv.CC_STAT_HEIGHT])
        area = float(stats[i, cv.CC_STAT_AREA])
        aspect = w / float(h) if h > 0 else float('inf')
        
        # Geometric filtering
        valid_area = min_area <= area <= max_area
        valid_size = min_size <= w <= max_size and min_size <= h <= max_size
        valid_aspect = NOTE_MIN_ASPECT <= aspect <= NOTE_MAX_ASPECT
        
        cx = int(round(centroids[i][0]))
        cy = int(round(centroids[i][1]))
        
        # Refine tiny components using the less filtered secondary mask
        if area <= tiny_area and valid_area:
            refined = _refine_tiny_center(
                cx, cy, s_count, s_stats, s_centroids, spacing, shape[1] - 1, shape[0] - 1
            )
            if refined:
                cx, cy = refined
        
        if valid_area and valid_size and valid_aspect:
            centers.append((cx, cy))
        
        if filtered_log is not None:
            filtered_log.append({
                "id": i, "x": cx, "y": cy, "w": w, "h": h,
                "area": area, "aspect": aspect, "passed": valid_area and valid_size and valid_aspect
            })
    
    if intermediates:
        intermediates["filtered_components"] = filtered_log
        intermediates["raw_centers"] = centers.copy()
    
    return centers


def _refine_tiny_center(cx, cy, count, stats, centroids, spacing, max_x, max_y):
    """Refine center position for tiny noteheads using secondary mask."""
    tolerance = max(2, int(round(spacing * 0.95)))
    min_height = max(6, int(round(spacing * 2.0)))
    min_area = spacing * spacing * 0.30
    max_width = int(round(spacing * 1.7))
    
    best_idx, best_dx = -1, float('inf')
    
    for i in range(1, count):
        w = int(stats[i, cv.CC_STAT_WIDTH])
        h = int(stats[i, cv.CC_STAT_HEIGHT])
        area = float(stats[i, cv.CC_STAT_AREA])
        center_x = int(round(float(centroids[i][0])))
        
        if h < min_height or area < min_area or w > max_width:
            continue
            
        dx = abs(center_x - cx)
        if dx <= tolerance and dx < best_dx:
            best_dx = dx
            best_idx = i
    
    if best_idx < 0:
        return None
    
    x = int(stats[best_idx, cv.CC_STAT_LEFT])
    y = int(stats[best_idx, cv.CC_STAT_TOP])
    w = int(stats[best_idx, cv.CC_STAT_WIDTH])
    h = int(stats[best_idx, cv.CC_STAT_HEIGHT])
    
    # Adjust y downward since notehead bottom is more reliable than center
    rx = x + w // 2
    ry = y + h - int(round(spacing * 0.90))
    return max(0, min(max_x, rx)), max(0, min(max_y, ry))


def _merge_nearby_centers(centers: list[tuple[int, int]], merge_dist: int) -> list[list]:
    """Merge centers that are close together (handles chords and touching noteheads)."""
    if not centers:
        return []
    
    centers = sorted(centers, key=lambda c: c[0])
    merged = [[centers[0][0], centers[0][1], 1]]  # [x, y, count]
    
    for cx, cy in centers[1:]:
        last_x, last_y, last_count = merged[-1]
        
        if abs(cx - last_x) <= merge_dist and abs(cy - last_y) <= merge_dist:
            # Weighted average of position
            new_count = last_count + 1
            new_x = int(round((last_x * last_count + cx) / new_count))
            new_y = int(round((last_y * last_count + cy) / new_count))
            merged[-1] = [new_x, new_y, new_count]
        else:
            merged.append([cx, cy, 1])
    
    return merged


def _add_stem_centers(
    mask: MatLike,
    centers: list[list],
    spacing: float,
    merge_dist: int,
    intermediates: dict | None
) -> list[list]:
    """Detect stem-only components and add them as note centers."""
    # Skip if we already have enough notes (avoid false positives)
    if len(centers) > 2:
        if intermediates:
            intermediates["stem_augmentation_skipped"] = "too_many_notes"
        return centers
    
    count, _, stats, _ = cv.connectedComponentsWithStats(mask, connectivity=8)
    result = [c.copy() for c in centers]
    
    margin = max(2, int(round(spacing * 0.6)))
    min_stem_height = int(round(spacing * 2.0))
    max_stem_width = int(round(spacing * 1.5))
    min_stem_area = spacing * spacing * 0.35
    
    # Overlap tolerances
    overlap_x = int(round(merge_dist * 1.2))
    overlap_y = int(round(spacing * 1.2))
    
    added = 0
    
    for i in range(1, count):
        x = int(stats[i, cv.CC_STAT_LEFT])
        y = int(stats[i, cv.CC_STAT_TOP])
        w = int(stats[i, cv.CC_STAT_WIDTH])
        h = int(stats[i, cv.CC_STAT_HEIGHT])
        area = float(stats[i, cv.CC_STAT_AREA])
        
        # Stem geometric criteria
        if h < min_stem_height:
            continue
        if w > max_stem_width:
            continue
        if area < min_stem_area:
            continue
        if x + w >= mask.shape[1] - margin:  # At edge, likely truncated
            continue
        
        # Position stem tip as note center
        cx = x + w // 2
        cy = y + h - max(1, int(round(spacing * 0.55)))
        
        # Check for overlap with existing centers
        overlaps = any(
            abs(cx - ex) <= overlap_x and abs(cy - ey) <= overlap_y
            for ex, ey, _ in result
        )
        if overlaps:
            continue
        
        result.append([cx, cy, 1])
        added += 1
        
        if added >= 1:  # Limit to one added stem per measure
            break
    
    if intermediates:
        intermediates["stem_augmentation"] = {"original_count": len(centers), "final_count": len(result)}
    
    return sorted(result, key=lambda c: c[0])


def _resolve_notes(
    centers: list[list],
    mask: MatLike,
    staff: Staff,
    measure: Measure,
    measure_index: int
) -> list[Note]:
    """Convert center points to Note objects with pitch and duration."""
    # Pitch calculation: position relative to staff lines
    bottom_line_y = int(round(staff.lines[4].y - measure.y_top))
    half_step_px = staff.spacing / 2.0
    
    notes = []
    for cx, cy, _ in centers:
        # Classify duration (filled vs hollow, stem presence)
        duration = _classify_duration(mask, cx, cy, staff.spacing)
        
        # Adjust y for hollow noteheads (centroid bias due to empty center)
        cy_pitch = cy
        if duration in ("half", "whole"):
            cy_pitch = cy + int(round(staff.spacing * HOLLOW_NOTE_Y_OFFSET_FRAC))
        
        # Calculate pitch step (0 = bottom line, positive = up)
        step_float = (bottom_line_y - cy_pitch) / half_step_px
        step = _quantize_to_step(step_float)
        residual = abs(step_float - step)
        
        notes.append(Note(
            kind="notehead",
            staff_index=measure.staff_index,
            measure_index=measure_index,
            center_x=cx,
            center_y=cy,
            step=step,
            step_confidence=_step_confidence(residual),
            duration_class=duration,
        ))
    
    return notes


def _quantize_to_step(step_float: float) -> int:
    """Round to nearest step, with bias for notehead geometry."""
    lower = math.floor(step_float)
    if step_float - lower >= STEP_ROUND_UP_THRESHOLD:
        return lower + 1
    return lower


def _step_confidence(residual: float) -> StepConfidence:
    """Confidence based on distance from exact step boundary."""
    if residual <= STEP_CONFIDENCE_HIGH:
        return "high"
    if residual <= STEP_CONFIDENCE_MEDIUM:
        return "medium"
    return "low"


def _merge_duplicate_detections(
    notes: list[Note], spacing: float, mask: MatLike | None
) -> list[Note]:
    """Collapse duplicate detections, including hollow noteheads split by their center."""
    if len(notes) < 2:
        return notes
    
    x_tol = max(2, int(round(spacing * DUPLICATE_X_TOLERANCE_FRAC)))
    y_tol = max(2, int(round(spacing * DUPLICATE_Y_TOLERANCE_FRAC)))
    hollow_x_tol = max(2, int(round(spacing * HOLLOW_SPLIT_X_TOLERANCE_FRAC)))
    
    result = [notes[0]]
    
    for note in notes[1:]:
        prev = result[-1]
        
        # Case 1: Both notes lack duration (shouldn't happen after classification, but safety check)
        is_unclassified_pair = (
            prev.duration_class is None and note.duration_class is None
            and abs(note.center_x - prev.center_x) <= x_tol
            and abs(note.center_y - prev.center_y) <= y_tol
            and abs(note.step - prev.step) <= DUPLICATE_MAX_STEP_DIFF
        )
        
        # Case 2: Two filled detections forming a hollow notehead split by its empty center
        is_hollow_split = (
            prev.duration_class in ("quarter", "whole")
            and note.duration_class in ("quarter", "whole")
            and abs(note.center_x - prev.center_x) <= hollow_x_tol
            and abs(note.center_y - prev.center_y) <= y_tol
            and abs(note.step - prev.step) <= DUPLICATE_MAX_STEP_DIFF
            and _has_hollow_center_gap(mask, prev, note, spacing)
        )
        
        if is_unclassified_pair or is_hollow_split:
            # Merge into single note at averaged position
            prev.center_x = (prev.center_x + note.center_x) // 2
            prev.center_y = (prev.center_y + note.center_y) // 2
            prev.step = (prev.step + note.step) // 2
            if note.step_confidence != "high":
                prev.step_confidence = note.step_confidence
            if is_hollow_split:
                prev.duration_class = "whole"
        else:
            result.append(note)
    
    return result


def _has_hollow_center_gap(
    mask: MatLike | None, note1: Note, note2: Note, spacing: float
) -> bool:
    """Check if there's a gap (low ink) between two note detections indicating hollow center."""
    if mask is None:
        # Without mask, rely on distance heuristic
        dist = abs(note1.center_x - note2.center_x)
        return spacing * 0.4 <= dist <= spacing * 1.5
    
    # Sample ink density in the region between the two note centers
    x_mid = (note1.center_x + note2.center_x) // 2
    y_mid = (note1.center_y + note2.center_y) // 2
    
    roi_half_height = int(spacing * 0.3)
    roi_half_width = 2
    
    y0 = max(0, y_mid - roi_half_height)
    y1 = min(mask.shape[0], y_mid + roi_half_height)
    x0 = max(0, x_mid - roi_half_width)
    x1 = min(mask.shape[1], x_mid + roi_half_width + 1)
    
    if y0 >= y1 or x0 >= x1:
        return False
    
    roi = mask[y0:y1, x0:x1]
    ink_ratio = cv.countNonZero(roi) / float(roi.size)
    return ink_ratio < HOLLOW_SPLIT_INK_RATIO_MAX


def _classify_duration(mask: MatLike, cx: int, cy: int, spacing: float) -> DurationClass | None:
    """Classify note duration based on fill state and stem presence."""
    filled = _detect_fill(mask, cx, cy, spacing)
    has_stem = _detect_stem(mask, cx, cy, spacing)
    
    if not filled and not has_stem:
        return "whole"
    if not filled and has_stem:
        return "half"
    if filled and has_stem:
        return "quarter"
    
    # Edge case: filled but no stem detected
    # Could be a whole note misclassified due to staff remnants,
    # or a quarter note with broken stem
    if filled and not has_stem:
        return _resolve_ambiguous_filled_note(mask, cx, cy, spacing)
    
    return None


def _detect_fill(mask: MatLike, cx: int, cy: int, spacing: float) -> bool:
    """Detect if notehead is filled (solid) or hollow (whole/half)."""
    rx = max(2, int(round(spacing * FILL_ELLIPSE_X_RADIUS_FRAC)))
    ry = max(2, int(round(spacing * FILL_ELLIPSE_Y_RADIUS_FRAC)))
    
    x0, x1 = max(0, cx - rx), min(mask.shape[1], cx + rx + 1)
    y0, y1 = max(0, cy - ry), min(mask.shape[0], cy + ry + 1)
    
    roi = mask[y0:y1, x0:x1]
    if roi.size == 0:
        return False
    
    # Create elliptical mask matching notehead shape
    ellipse = np.zeros(roi.shape, dtype=np.uint8)
    local_cx, local_cy = cx - x0, cy - y0
    cv.ellipse(ellipse, (local_cx, local_cy), (max(1, rx-1), max(1, ry-1)), 0, 0, 360, 255, -1)
    
    ellipse_area = cv.countNonZero(ellipse)
    if ellipse_area == 0:
        return False
    
    ink = cv.countNonZero(cv.bitwise_and(roi, roi, mask=ellipse))
    return ink / float(ellipse_area) >= FILL_RATIO_THRESHOLD


def _detect_stem(mask: MatLike, cx: int, cy: int, spacing: float) -> bool:
    """Detect vertical stem attached to notehead."""
    rx = max(2, int(round(spacing * STEM_X_RADIUS_FRAC)))
    ry = max(3, int(round(spacing * STEM_Y_RADIUS_FRAC)))
    
    x0, x1 = max(0, cx - rx), min(mask.shape[1], cx + rx + 1)
    y0, y1 = max(0, cy - ry), min(mask.shape[0], cy + ry + 1)
    
    roi = mask[y0:y1, x0:x1]
    if roi.size == 0 or roi.shape[1] < 2:
        return False
    
    # Look for long vertical runs (stem must be taller than notehead)
    min_run = max(3, int(round(spacing * STEM_MIN_RUN_FRAC)))
    
    for col in range(roi.shape[1]):
        run_len = 0
        max_run = 0
        for row in range(roi.shape[0]):
            if roi[row, col] > 0:
                run_len += 1
                max_run = max(max_run, run_len)
            else:
                run_len = 0
        if max_run >= min_run:
            return True
    
    return False


def _resolve_ambiguous_filled_note(
    mask: MatLike, cx: int, cy: int, spacing: float
) -> DurationClass:
    """Resolve filled notehead without detected stem - whole or quarter?"""
    rx = max(2, int(round(spacing * FILL_ELLIPSE_X_RADIUS_FRAC)))
    ry = max(2, int(round(spacing * FILL_ELLIPSE_Y_RADIUS_FRAC)))
    
    x0, x1 = max(0, cx - rx), min(mask.shape[1], cx + rx + 1)
    y0, y1 = max(0, cy - ry), min(mask.shape[0], cy + ry + 1)
    
    roi = mask[y0:y1, x0:x1]
    if roi.size == 0:
        return "quarter"
    
    # Whole notes (hollow) misclassified as filled will have lower ink density
    # due to their empty center being partially filled by staff remnants
    ink_pixels = cv.countNonZero(roi)
    fill_ratio = ink_pixels / float(roi.size)
    
    if fill_ratio < WHOLE_NOTE_FILL_RATIO_MAX:
        return "whole"
    return "quarter"


def resolve_pitches(notes: list[Note], clef: Clef | None) -> None:
    """Convert step positions to pitch letters and octaves based on clef."""
    if clef is None or clef.kind not in CLEF_BASE_POSITIONS:
        return
    
    base_letter, base_octave = CLEF_BASE_POSITIONS[clef.kind]
    key_accidentals = _get_key_accidentals(clef.key_signature)
    
    for note in notes:
        letter, octave = _step_to_pitch(base_letter, base_octave, note.step)
        accidental = key_accidentals.get(letter, "")
        note.pitch_letter = f"{letter}{accidental}"
        note.octave = octave


def _step_to_pitch(base_letter: str, base_octave: int, step: int) -> tuple[str, int]:
    """Convert staff step to pitch letter and octave."""
    base_idx = PITCH_TO_INDEX[base_letter]
    absolute = base_octave * 7 + base_idx + step
    octave = absolute // 7
    letter_idx = absolute % 7
    return PITCH_CYCLE[letter_idx], octave


def _get_key_accidentals(key_sig: KeySignature) -> dict[str, str]:
    """Get accidentals for each pitch letter based on key signature."""
    accidentals = {}
    fifths = key_sig.fifths or 0
    
    if fifths > 0:
        for letter in CIRCLE_OF_FIFTHS_SHARPS[:fifths]:
            accidentals[letter] = "#"
    elif fifths < 0:
        for letter in CIRCLE_OF_FIFTHS_FLATS[:abs(fifths)]:
            accidentals[letter] = "b"
    
    return accidentals
