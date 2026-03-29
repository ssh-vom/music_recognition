"""Note detection - pure functions for finding noteheads in sheet music.

Converts binary measure images into note objects with pitch and duration.
"""

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

# Pitch constants
LETTER_ORDER = ("C", "D", "E", "F", "G", "A", "B")
LETTER_TO_INDEX = {letter: index for index, letter in enumerate(LETTER_ORDER)}
SHARP_ORDER = ("F", "C", "G", "D", "A", "E", "B")  # Circle of fifths order
FLAT_ORDER = ("B", "E", "A", "D", "G", "C", "F")  # Reverse circle of fifths


def find_notes(
    mask: MatLike, staff: Staff, measure: Measure, measure_index: int
) -> list[Note]:
    """Find all noteheads in a measure.

    Process:
    1. Morphological open to isolate notehead-shaped blobs
    2. Connected components to find candidate blobs
    3. Filter by size, aspect ratio, and area
    4. Merge nearby detections
    5. Classify pitch and duration

    All thresholds are expressed as ratios of staff spacing for scale invariance.
    """
    # STEP 1: Morphological processing
    # Kernel diameter = 45% of staff spacing - targets notehead-sized regions
    kernel_diameter = max(1, int(round(staff.spacing * 0.45)))
    if kernel_diameter % 2 == 0:
        kernel_diameter += 1

    notehead_kernel = cv.getStructuringElement(
        cv.MORPH_ELLIPSE, (kernel_diameter, kernel_diameter)
    )
    opened_mask = cv.morphologyEx(mask, cv.MORPH_OPEN, notehead_kernel)

    # Close small gaps within noteheads
    close_kernel = cv.getStructuringElement(cv.MORPH_ELLIPSE, (3, 3))
    notehead_mask = cv.morphologyEx(opened_mask, cv.MORPH_CLOSE, close_kernel)
    secondary_mask = cv.morphologyEx(mask, cv.MORPH_CLOSE, close_kernel)

    # STEP 2: Find connected components
    count, _, stats, centroids = cv.connectedComponentsWithStats(
        notehead_mask, connectivity=8
    )
    secondary_count, _, secondary_stats, secondary_centroids = (
        cv.connectedComponentsWithStats(secondary_mask, connectivity=8)
    )

    # STEP 3: Filter candidates by geometry
    # Area thresholds: 8% to 180% of staff spacing squared
    # Captures noteheads while filtering noise and merged regions
    min_area = staff.spacing * staff.spacing * 0.08
    max_area = staff.spacing * staff.spacing * 1.8

    # Size thresholds: 35% to 190% of staff spacing
    # Ensures blobs are notehead-sized, not too small or large
    min_size = int(round(staff.spacing * 0.35))
    max_size = int(round(staff.spacing * 1.9))

    # Aspect ratio thresholds: 0.45 to 2.2
    # Noteheads are roughly circular (aspect near 1.0)
    min_aspect = 0.45
    max_aspect = 2.2

    # Tiny notehead threshold: 22% of spacing squared
    # These may need position refinement from secondary mask
    tiny_area = staff.spacing * staff.spacing * 0.22

    raw_centers = []

    for i in range(1, count):
        w = int(stats[i, cv.CC_STAT_WIDTH])
        h = int(stats[i, cv.CC_STAT_HEIGHT])
        area = float(stats[i, cv.CC_STAT_AREA])

        if area < min_area or area > max_area:
            continue
        if w < min_size or h < min_size or w > max_size or h > max_size:
            continue

        aspect = w / float(h)
        if aspect < min_aspect or aspect > max_aspect:
            continue

        cx = int(round(centroids[i][0]))
        cy = int(round(centroids[i][1]))

        # Refine tiny noteheads using larger secondary mask
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

        raw_centers.append((cx, cy))

    # STEP 4: Merge nearby detections
    # Merge distance: 75% of staff spacing
    # Groups split noteheads or multiple detections of same note
    merge_dist = max(2, int(round(staff.spacing * 0.75)))
    raw_centers.sort(key=lambda c: c[0])

    merged = _merge_centers(raw_centers, merge_dist)

    # STEP 5: Augment from stems (if few notes found)
    # Looks for tall vertical components that might be missed stems
    merged = _augment_from_stems(mask, merged, staff.spacing, mask.shape[1], merge_dist)

    # STEP 6: Convert to notes with pitch and duration
    bottom_line_y = int(round(staff.lines[4].y - measure.y_top))
    half_step = staff.spacing / 2.0

    notes = []
    for cx, cy, _ in merged:
        # Convert y-position to staff step
        # step 0 = bottom line, step 1 = first space above, etc.
        step_float = (bottom_line_y - cy) / half_step
        step = _quantize_step(step_float)
        residual = abs(step_float - step)
        confidence = _step_confidence(residual)
        duration = _classify_duration(mask, cx, cy, staff.spacing)

        notes.append(
            Note(
                kind="notehead",
                staff_index=measure.staff_index,
                measure_index=measure_index,
                center_x=cx,
                center_y=cy,
                step=step,
                step_confidence=confidence,
                duration_class=duration,
            )
        )

    notes.sort(key=lambda n: n.center_x)
    return _collapse_duplicates(notes, staff.spacing)


def _merge_centers(centers: list[tuple[int, int]], merge_dist: int) -> list[list]:
    """Merge nearby center points by averaging.

    Returns list of [x, y, count] where count is number of merged points.
    """
    if not centers:
        return []

    merged: list[list] = [[centers[0][0], centers[0][1], 1]]

    for cx, cy in centers[1:]:
        last_x, last_y, last_count = merged[-1]

        if abs(cx - last_x) <= merge_dist and abs(cy - last_y) <= merge_dist:
            # Average with existing point
            new_count = last_count + 1
            new_x = int(round((last_x * last_count + cx) / new_count))
            new_y = int(round((last_y * last_count + cy) / new_count))
            merged[-1] = [new_x, new_y, new_count]
        else:
            merged.append([cx, cy, 1])

    return merged


def _quantize_step(step_float: float) -> int:
    """Quantize floating-point step to integer with downward bias.

    STEP_ROUND_UP_THRESHOLD = 0.58 (instead of 0.5)
    Requires slightly more evidence before rounding up to next half-step.
    This compensates for notehead center being slightly above geometric center.
    """
    STEP_ROUND_UP_THRESHOLD = 0.58

    lower = math.floor(step_float)
    if step_float - lower >= STEP_ROUND_UP_THRESHOLD:
        return lower + 1
    return lower


def _step_confidence(residual: float) -> StepConfidence:
    """Classify step detection confidence based on quantization residual.

    Thresholds:
    - high: residual <= 0.20 (very close to step line)
    - medium: residual <= 0.40 (moderately close)
    - low: residual > 0.40 (far from step line, ambiguous)
    """
    if residual <= 0.20:
        return "high"
    if residual <= 0.40:
        return "medium"
    return "low"


def _augment_from_stems(
    mask: MatLike, centers: list, spacing: float, width: int, merge_dist: int
) -> list:
    """Look for missed notes by finding tall vertical components (stems).

    Only runs when few notes detected (< 3).
    Finds tall thin components and adds noteheads at their base.

    Detection criteria for stem components:
    - Height: at least 2x staff spacing
    - Width: at most 1.5x staff spacing
    - Area: at least 35% of spacing squared
    - Position: not at right edge (margin = 60% of spacing)
    """
    if len(centers) > 2:
        return centers

    count, _, stats, _ = cv.connectedComponentsWithStats(mask, connectivity=8)
    augmented = [c.copy() for c in centers]

    # Right edge margin to avoid page edge artifacts
    margin = max(2, int(round(spacing * 0.6)))
    added = 0

    for i in range(1, count):
        x = int(stats[i, cv.CC_STAT_LEFT])
        y = int(stats[i, cv.CC_STAT_TOP])
        w = int(stats[i, cv.CC_STAT_WIDTH])
        h = int(stats[i, cv.CC_STAT_HEIGHT])
        area = float(stats[i, cv.CC_STAT_AREA])

        # Tall component check: height >= 2x spacing
        if h < int(round(spacing * 2.0)):
            continue
        # Width check: not too wide (<= 1.5x spacing)
        if w > int(round(spacing * 1.5)):
            continue
        # Area check: substantial enough (>= 35% spacing^2)
        if area < spacing * spacing * 0.35:
            continue
        # Edge check: not at right margin
        if x + w >= width - margin:
            continue

        # Place notehead at stem base, slightly above bottom
        cx = x + w // 2
        cy = y + h - max(1, int(round(spacing * 0.55)))

        # Check for overlap with existing detections
        overlaps = False
        overlap_x_dist = int(round(merge_dist * 1.2))  # 120% of merge distance
        overlap_y_dist = int(round(spacing * 1.2))

        for ex, ey, _ in augmented:
            if abs(cx - ex) <= overlap_x_dist and abs(cy - ey) <= overlap_y_dist:
                overlaps = True
                break

        if overlaps:
            continue

        augmented.append([cx, cy, 1])
        added += 1
        if added >= 1:
            break

    augmented.sort(key=lambda c: c[0])
    return augmented


def _collapse_duplicates(notes: list[Note], spacing: float) -> list[Note]:
    """Collapse duplicate notes that are spatially overlapping.

    Two notes are considered duplicates if:
    - Both have no duration class assigned (ambiguous detections)
    - X distance <= 145% of spacing
    - Y distance <= 75% of spacing
    - Step difference <= 1

    Duplicate notes are merged by averaging positions and steps.
    """
    if len(notes) < 2:
        return notes

    x_tol = max(2, int(round(spacing * 1.45)))
    y_tol = max(2, int(round(spacing * 0.75)))

    collapsed = [notes[0]]

    for note in notes[1:]:
        prev = collapsed[-1]

        is_duplicate = (
            prev.duration_class is None
            and note.duration_class is None
            and abs(note.center_x - prev.center_x) <= x_tol
            and abs(note.center_y - prev.center_y) <= y_tol
            and abs(note.step - prev.step) <= 1
        )

        if is_duplicate:
            # Merge by averaging
            prev.center_x = int(round((prev.center_x + note.center_x) / 2.0))
            prev.center_y = int(round((prev.center_y + note.center_y) / 2.0))
            prev.step = int(round((prev.step + note.step) / 2.0))
            # Keep higher confidence
            prev.step_confidence = (
                prev.step_confidence
                if prev.step_confidence == "high"
                else note.step_confidence
            )
        else:
            collapsed.append(note)

    return collapsed


def _refine_from_secondary_mask(
    cx: int,
    cy: int,
    count: int,
    stats,
    centroids,
    spacing: float,
    max_x: int,
    max_y: int,
) -> tuple[int, int] | None:
    """Refine notehead position using secondary (less processed) mask.

    Used for tiny noteheads that may have been over-processed.
    Looks for taller component near original position.

    Matching criteria:
    - Height: at least 2x spacing
    - Area: at least 30% of spacing squared
    - Width: at most 1.7x spacing
    - X proximity: within 95% of spacing
    """
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

    # Place notehead near bottom of tall component
    rx = x + w // 2
    ry = y + h - int(round(spacing * 0.90))
    rx = max(0, min(max_x, rx))
    ry = max(0, min(max_y, ry))

    return rx, ry


def _classify_duration(
    mask: MatLike, cx: int, cy: int, spacing: float
) -> DurationClass | None:
    """Classify note duration based on filled status and stem presence.

    Classification logic:
    - whole: hollow notehead, no stem
    - half: hollow notehead, has stem
    - quarter: filled notehead, has stem
    - None: filled notehead, no stem (unusual/invalid)
    """
    filled = _is_filled(mask, cx, cy, spacing)
    has_stem = _has_stem(mask, cx, cy, spacing)

    if not filled and not has_stem:
        return "whole"
    if not filled and has_stem:
        return "half"
    if filled and has_stem:
        return "quarter"
    return None


def _is_filled(mask: MatLike, cx: int, cy: int, spacing: float) -> bool:
    """Check if notehead is filled (black) or hollow (white).

    Creates elliptical ROI around notehead center and measures ink density.

    Ellipse size: 36% x 28% of spacing (typical notehead proportions)
    Filled threshold: 55% ink coverage within ellipse
    """
    # Elliptical region around notehead
    rx = max(2, int(round(spacing * 0.36)))
    ry = max(2, int(round(spacing * 0.28)))

    x1 = max(0, cx - rx)
    x2 = min(mask.shape[1], cx + rx + 1)
    y1 = max(0, cy - ry)
    y2 = min(mask.shape[0], cy + ry + 1)

    roi = mask[y1:y2, x1:x2]
    if roi.size == 0:
        return False

    # Create elliptical mask
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

    # Filled threshold: 55% ink coverage
    return ink_ratio >= 0.55


def _has_stem(mask: MatLike, cx: int, cy: int, spacing: float) -> bool:
    """Check if notehead has an attached stem.

    Searches for vertical line above or below notehead.

    Search region: 85% horizontal, 260% vertical of spacing
    Stem criteria: vertical run of at least 120% of spacing
    """
    # Search region around notehead
    x_radius = max(2, int(round(spacing * 0.85)))
    y_radius = max(3, int(round(spacing * 2.6)))

    x1 = max(0, cx - x_radius)
    x2 = min(mask.shape[1], cx + x_radius + 1)
    y1 = max(0, cy - y_radius)
    y2 = min(mask.shape[0], cy + y_radius + 1)

    roi = mask[y1:y2, x1:x2]
    if roi.size == 0 or roi.shape[1] < 2:
        return False

    # Minimum continuous vertical run to qualify as stem
    min_run = max(3, int(round(spacing * 1.2)))

    # Check each column for vertical runs
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


# Pitch resolution functions


def resolve_pitches(notes: list[Note], clef: Clef | None) -> None:
    """Resolve note steps to pitch letters and octaves based on clef.

    Modifies notes in-place, setting pitch_letter and octave fields.
    """
    if clef is None or clef.kind is None:
        return

    # Clef-specific base pitches
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
    """Convert staff step to letter and octave.

    Uses base pitch (bottom staff line) and counts up by step.
    Each step = one half-step on staff (line to space or vice versa).
    """
    base_index = LETTER_TO_INDEX[base_letter]
    absolute = base_octave * 7 + base_index + step
    octave = absolute // 7
    letter_index = absolute % 7
    return LETTER_ORDER[letter_index], octave


def _key_signature_accidentals(key_sig: KeySignature) -> dict[str, str]:
    """Get accidentals from key signature.

    Returns dict mapping note letters to accidental symbols (# or b).
    Uses circle of fifths order for sharp/flat positions.
    """
    accidentals = {}
    fifths = key_sig.fifths if key_sig.fifths is not None else 0

    if fifths > 0:
        # Sharps: F, C, G, D, A, E, B order
        for letter in SHARP_ORDER[:fifths]:
            accidentals[letter] = "#"
    elif fifths < 0:
        # Flats: B, E, A, D, G, C, F order
        for letter in FLAT_ORDER[: abs(fifths)]:
            accidentals[letter] = "b"

    return accidentals
