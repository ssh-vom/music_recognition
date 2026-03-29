"""Rhythm detection - beam-aware duration refinement for already-detected notes.

This module implements the beam detection strategy from PLAN.md:
1. Uses existing note detections (never adds new ones)
2. Analyzes connected components in the staff-erased mask
3. Groups notes that share ink (connected components)
4. Detects beam-like horizontal connectors between stems
5. Updates duration_class for beamed groups (eighth/sixteenth)
"""

import cv2 as cv
import numpy as np
from cv2.typing import MatLike

from schema import DurationClass, Note, Staff

EVENT_X_TOLERANCE_PX = 5
COMPACT_GAP_MIN_FRAC = 1.2
COMPACT_GAP_MAX_FRAC = 3.6
COMPACT_RUN_MIN_EVENTS = 4


def refine_beamed_durations(
    mask: MatLike, notes: list[Note], staff: Staff
) -> list[Note]:
    """Refine note durations by detecting beam connections between existing notes.

    Args:
        mask: Staff-erased binary mask (measure.crop)
        notes: Already-detected notes from note_detection.find_notes()
        staff: Staff info for spacing reference

    Returns:
        Notes with potentially updated duration_class (quarter -> eighth/sixteenth)
        Never adds new notes, only annotates existing ones.
    """
    if len(notes) < 2:
        return notes

    # Build connected components from the mask
    num_labels, labels, stats, _ = cv.connectedComponentsWithStats(
        mask, connectivity=8
    )

    # Assign each note to its connected component
    note_components: dict[int, list[tuple[int, Note]]] = {}
    for i, note in enumerate(notes):
        cx, cy = note.center_x, note.center_y
        # Check bounds
        if 0 <= cy < labels.shape[0] and 0 <= cx < labels.shape[1]:
            comp_id = int(labels[cy, cx])
            if comp_id > 0:  # 0 is background
                if comp_id not in note_components:
                    note_components[comp_id] = []
                note_components[comp_id].append((i, note))

    # Analyze components with 2+ notes for beam groups
    for comp_id, note_group in note_components.items():
        if len(note_group) < 2:
            continue

        # Get beam info for this group
        beam_count = _detect_beam_count(mask, note_group, staff)
        if beam_count > 0:
            # Update durations for notes in this beamed group
            for idx, note in note_group:
                # Only upgrade filled notes with stems (not whole/half)
                if note.duration_class == "quarter":
                    if beam_count == 1:
                        notes[idx].duration_class = "eighth"
                    elif beam_count >= 2:
                        notes[idx].duration_class = "sixteenth"

    # Fallback for broken beam connectivity:
    # on fast printed tunes, a long run of tightly spaced note events usually
    # indicates beamed eighths even if staff removal split the beam component.
    _apply_compact_spacing_fallback(notes, staff.spacing)

    return notes


def _apply_compact_spacing_fallback(notes: list[Note], spacing: float) -> None:
    """Upgrade compact runs of quarter-like events to eighths.

    This is a conservative fallback for cases where beam ink is broken across
    connected components after preprocessing. It intentionally targets long,
    evenly spaced melodic runs common in fast fiddle tunes and should stay off
    simpler quarter-note songs whose event spacing is much larger.
    """
    if len(notes) < COMPACT_RUN_MIN_EVENTS:
        return

    events = _group_notes_into_events(notes, x_tol=EVENT_X_TOLERANCE_PX)
    if len(events) < COMPACT_RUN_MIN_EVENTS:
        return

    anchors = [
        int(round(sum(note.center_x for note in event) / float(len(event))))
        for event in events
    ]
    compact_min = spacing * COMPACT_GAP_MIN_FRAC
    compact_max = spacing * COMPACT_GAP_MAX_FRAC

    start = 0
    while start < len(events) - 1:
        gap = anchors[start + 1] - anchors[start]
        if gap < compact_min or gap > compact_max:
            start += 1
            continue

        end = start + 1
        while end < len(events) - 1:
            next_gap = anchors[end + 1] - anchors[end]
            if compact_min <= next_gap <= compact_max:
                end += 1
            else:
                break

        run_events = events[start : end + 1]
        if len(run_events) >= COMPACT_RUN_MIN_EVENTS and _run_can_be_eighths(
            run_events, spacing
        ):
            for event in run_events:
                for note in event:
                    if note.duration_class in ("quarter", None):
                        note.duration_class = "eighth"

        start = end + 1



def _group_notes_into_events(notes: list[Note], x_tol: int) -> list[list[Note]]:
    """Group notes that share nearly the same onset x-position."""
    ordered = sorted(notes, key=lambda note: (note.center_x, note.center_y))
    events: list[list[Note]] = []

    for note in ordered:
        if not events:
            events.append([note])
            continue

        last_event = events[-1]
        anchor_x = round(
            sum(existing.center_x for existing in last_event) / float(len(last_event))
        )
        if abs(note.center_x - anchor_x) <= x_tol:
            last_event.append(note)
        else:
            events.append([note])

    return events



def _run_can_be_eighths(run_events: list[list[Note]], spacing: float) -> bool:
    """Check whether a compact onset run looks like an eighth-note passage."""
    if len(run_events) < COMPACT_RUN_MIN_EVENTS:
        return False

    # Reject runs with obvious long durations already present.
    for event in run_events:
        for note in event:
            if note.duration_class in ("whole", "half"):
                return False

    # Require that the run is mostly single melodic events or tight stacks, not
    # huge clusters. This keeps the heuristic focused on monophonic passages.
    if any(len(event) > 2 for event in run_events):
        return False

    # Require a modest pitch contour variation so we do not relabel repeated wide
    # quarter-note blocks in simpler songs.
    steps = [event[-1].step for event in run_events if event]
    if len(set(steps)) < 2:
        return False

    return True



def _detect_beam_count(
    mask: MatLike, note_group: list[tuple[int, Note]], staff: Staff
) -> int:
    """Detect how many beam bands connect a group of notes (0, 1, or 2).

    Args:
        mask: Binary mask
        note_group: List of (index, note) tuples sharing a component
        staff: Staff for spacing reference

    Returns:
        Number of beam bands detected (0 = no beam, 1 = eighth, 2 = sixteenth)
    """
    if len(note_group) < 2:
        return 0

    notes = [n for _, n in note_group]
    spacing = staff.spacing

    # Determine if stems are mostly up or down
    stem_dirs = [_estimate_stem_direction(mask, n, spacing) for n in notes]
    up_count = sum(1 for d in stem_dirs if d == "up")
    down_count = sum(1 for d in stem_dirs if d == "down")

    # For beamed groups, use the majority direction
    # Mixed directions can occur in legitimate beamed groups when notes
    # cross the staff middle line
    if up_count == 0 and down_count == 0:
        return 0
    
    beam_direction = "up" if up_count > down_count else "down"

    # Get the region between the outermost notes
    min_x = min(n.center_x for n in notes)
    max_x = max(n.center_x for n in notes)
    x_range = max_x - min_x

    # Need some horizontal spread for a beam
    if x_range < spacing * 0.5:
        return 0

    # Find the beam y-level based on the component's ink distribution
    # Look for horizontal bands with high ink density that span the note group
    
    # First, get the vertical bounds of this component by looking at note positions
    min_note_y = min(n.center_y for n in notes)
    max_note_y = max(n.center_y for n in notes)
    
    # Get stem tips for the dominant direction
    stem_tips = [
        _find_stem_endpoint(mask, n, spacing, beam_direction) for n in notes
    ]
    valid_tips = [y for y in stem_tips if y is not None]
    
    if not valid_tips:
        return 0
    
    if beam_direction == "up":
        # For up stems, beam connects at the top (minimum y)
        beam_y = min(valid_tips)
        # Search around the beam area
        search_y_start = max(0, beam_y - int(spacing * 0.5))
        search_y_end = min(mask.shape[0], beam_y + int(spacing * 0.5))
    else:
        # For down stems, beam connects at the bottom (maximum y)
        beam_y = max(valid_tips)
        # Search around the beam area
        search_y_start = max(0, beam_y - int(spacing * 0.5))
        search_y_end = min(mask.shape[0], beam_y + int(spacing * 0.5))
    
    if search_y_start >= search_y_end:
        return 0

    # Extract horizontal projection of ink in the search band
    band_region = mask[search_y_start:search_y_end, min_x:max_x]
    if band_region.size == 0:
        return 0

    # Sum horizontally to find dense horizontal lines
    horizontal_density = np.sum(band_region > 0, axis=1)

    # Find peaks in density - each peak is a potential beam
    # Lower threshold: beam can be thinner than a full notehead width
    threshold = max(2, spacing * 0.15)  # Reduced from 0.3
    peaks = _find_peaks(horizontal_density, threshold)

    # Filter peaks that have good horizontal span (beam-like, not just a dot)
    beam_count = 0
    beam_rows = []
    for peak_y_local, density in peaks:
        peak_y = search_y_start + peak_y_local
        # Check if this horizontal run spans enough of the note group
        row_mask = mask[peak_y, min_x:max_x] > 0
        runs = _find_ink_runs(row_mask)
        total_ink = sum(length for _, length in runs)
        longest_run = max((length for _, length in runs), default=0)
        
        # A beam should:
        # 1. Span a significant portion of the note group (20% instead of 50%)
        # 2. Have a long continuous run (beam is continuous, not fragmented)
        ink_ratio = total_ink / x_range
        if ink_ratio > 0.20 and longest_run > spacing * 0.8:
            beam_rows.append((peak_y, density, ink_ratio))
            beam_count += 1
            if beam_count >= 2:
                break

    return min(beam_count, 2)  # Cap at 2 for now


def _estimate_stem_direction(
    mask: MatLike, note: Note, spacing: float
) -> str:
    """Estimate whether the note's stem goes up or down.

    Returns 'up' or 'down' based on which side has more vertical ink above/below.
    """
    cx, cy = note.center_x, note.center_y
    h, w = mask.shape

    # Define search windows
    x_radius = int(spacing * 0.6)
    y_radius = int(spacing * 2.5)

    x1 = max(0, cx - x_radius)
    x2 = min(w, cx + x_radius)

    # Check ink above notehead
    y_above_start = max(0, cy - y_radius)
    y_above_end = max(0, cy - int(spacing * 0.3))
    above_region = mask[y_above_start:y_above_end, x1:x2]
    above_ink = np.sum(above_region > 0) if above_region.size > 0 else 0

    # Check ink below notehead
    y_below_start = min(h, cy + int(spacing * 0.3))
    y_below_end = min(h, cy + y_radius)
    below_region = mask[y_below_start:y_below_end, x1:x2]
    below_ink = np.sum(below_region > 0) if below_region.size > 0 else 0

    # Stem direction is where there's significantly more ink
    if above_ink > below_ink * 1.5:
        return "up"
    elif below_ink > above_ink * 1.5:
        return "down"
    else:
        # Default based on pitch - higher notes tend to have down stems
        # step is relative to staff: positive = higher
        return "down" if note.step > 0 else "up"


def _find_stem_endpoint(
    mask: MatLike, note: Note, spacing: float, direction: str
) -> int | None:
    """Find the y-coordinate of the stem end (where beam connects).

    Args:
        mask: Binary mask
        note: Note to analyze
        spacing: Staff spacing
        direction: 'up' or 'down'

    Returns:
        Y coordinate of stem endpoint, or None if not found
    """
    cx, cy = note.center_x, note.center_y
    h, w = mask.shape
    x_radius = int(spacing * 0.4)

    x1 = max(0, cx - x_radius)
    x2 = min(w, cx + x_radius)

    search_range = int(spacing * 3)

    if direction == "up":
        y_start = max(0, cy - search_range)
        y_end = max(0, cy - int(spacing * 0.4))
        # Search upward for last ink pixel
        for y in range(y_end - 1, y_start - 1, -1):
            if np.any(mask[y, x1:x2] > 0):
                return y
    else:
        y_start = min(h, cy + int(spacing * 0.4))
        y_end = min(h, cy + search_range)
        # Search downward for last ink pixel
        for y in range(y_start, y_end):
            if np.any(mask[y, x1:x2] > 0):
                return y

    return None


def _find_peaks(data: np.ndarray, threshold: float) -> list[tuple[int, float]]:
    """Find peaks in 1D data above threshold.

    Returns list of (index, value) tuples.
    """
    peaks = []
    i = 0
    n = len(data)

    while i < n:
        if data[i] > threshold:
            # Start of a peak region
            start = i
            max_val = data[i]
            max_idx = i
            while i < n and data[i] > threshold:
                if data[i] > max_val:
                    max_val = data[i]
                    max_idx = i
                i += 1
            peaks.append((max_idx, max_val))
        else:
            i += 1

    return peaks


def _find_ink_runs(binary_1d: np.ndarray) -> list[tuple[int, int]]:
    """Find contiguous runs of True/ink in a 1D binary array.

    Returns list of (start_index, length) tuples.
    """
    runs = []
    in_run = False
    start = 0

    for i, val in enumerate(binary_1d):
        if val and not in_run:
            in_run = True
            start = i
        elif not val and in_run:
            in_run = False
            runs.append((start, i - start))

    if in_run:
        runs.append((start, len(binary_1d) - start))

    return runs
