"""Rhythm detection - beam-aware duration refinement for already-detected notes."""

import cv2 as cv
import numpy as np
from cv2.typing import MatLike

from note_grouping import EVENT_X_TOLERANCE_PX, group_notes_into_events
from schema import DurationClass, Note, Staff
COMPACT_GAP_MIN_FRAC = 1.2
COMPACT_GAP_MAX_FRAC = 3.6
COMPACT_RUN_MIN_EVENTS = 4


def refine_beamed_durations(mask: MatLike, notes: list[Note], staff: Staff) -> list[Note]:
    """Update duration_class for notes connected by beams (never adds new notes)."""
    if len(notes) < 2:
        return notes

    num_labels, labels, stats, _ = cv.connectedComponentsWithStats(mask, connectivity=8)

    note_components: dict[int, list[tuple[int, Note]]] = {}
    for i, note in enumerate(notes):
        cx, cy = note.center_x, note.center_y
        if 0 <= cy < labels.shape[0] and 0 <= cx < labels.shape[1]:
            comp_id = int(labels[cy, cx])
            if comp_id > 0:
                note_components.setdefault(comp_id, []).append((i, note))

    for comp_id, note_group in note_components.items():
        if len(note_group) < 2:
            continue
        beam_count = _detect_beam_count(mask, note_group, staff)
        if beam_count > 0:
            for idx, note in note_group:
                if note.duration_class == "quarter":
                    notes[idx].duration_class = "eighth" if beam_count == 1 else "sixteenth"

    _apply_compact_spacing_fallback(notes, staff.spacing)
    return notes


def _apply_compact_spacing_fallback(notes: list[Note], spacing: float) -> None:
    """Upgrade compact runs of quarter-like events to eighths.

    Conservative fallback for when beam ink is broken after preprocessing.
    Targets long, evenly spaced melodic runs common in fast fiddle tunes.
    """
    if len(notes) < COMPACT_RUN_MIN_EVENTS:
        return

    events = group_notes_into_events(notes, x_tol=EVENT_X_TOLERANCE_PX)
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
        if len(run_events) >= COMPACT_RUN_MIN_EVENTS and _run_can_be_eighths(run_events, spacing):
            for event in run_events:
                for note in event:
                    if note.duration_class in ("quarter", None):
                        note.duration_class = "eighth"

        start = end + 1


def _run_can_be_eighths(run_events: list[list[Note]], spacing: float) -> bool:
    if len(run_events) < COMPACT_RUN_MIN_EVENTS:
        return False
    for event in run_events:
        for note in event:
            if note.duration_class in ("whole", "half"):
                return False
    if any(len(event) > 2 for event in run_events):
        return False
    steps = [event[-1].step for event in run_events if event]
    return len(set(steps)) >= 2


def _detect_beam_count(mask: MatLike, note_group: list[tuple[int, Note]], staff: Staff) -> int:
    if len(note_group) < 2:
        return 0

    notes = [n for _, n in note_group]
    spacing = staff.spacing

    stem_dirs = [_estimate_stem_direction(mask, n, spacing) for n in notes]
    up_count = sum(1 for d in stem_dirs if d == "up")
    down_count = sum(1 for d in stem_dirs if d == "down")

    if up_count == 0 and down_count == 0:
        return 0

    beam_direction = "up" if up_count > down_count else "down"

    min_x = min(n.center_x for n in notes)
    max_x = max(n.center_x for n in notes)
    x_range = max_x - min_x

    if x_range < spacing * 0.5:
        return 0

    stem_tips = [_find_stem_endpoint(mask, n, spacing, beam_direction) for n in notes]
    valid_tips = [y for y in stem_tips if y is not None]

    if not valid_tips:
        return 0

    if beam_direction == "up":
        beam_y = min(valid_tips)
    else:
        beam_y = max(valid_tips)

    search_y_start = max(0, beam_y - int(spacing * 0.5))
    search_y_end = min(mask.shape[0], beam_y + int(spacing * 0.5))

    if search_y_start >= search_y_end:
        return 0

    band_region = mask[search_y_start:search_y_end, min_x:max_x]
    if band_region.size == 0:
        return 0

    horizontal_density = np.sum(band_region > 0, axis=1)
    threshold = max(2, spacing * 0.15)
    peaks = _find_peaks(horizontal_density, threshold)

    beam_count = 0
    for peak_y_local, _ in peaks:
        peak_y = search_y_start + peak_y_local
        row_mask = mask[peak_y, min_x:max_x] > 0
        runs = _find_ink_runs(row_mask)
        total_ink = sum(length for _, length in runs)
        longest_run = max((length for _, length in runs), default=0)

        if total_ink / x_range > 0.20 and longest_run > spacing * 0.8:
            beam_count += 1
            if beam_count >= 2:
                break

    return min(beam_count, 2)


def _estimate_stem_direction(mask: MatLike, note: Note, spacing: float) -> str:
    cx, cy = note.center_x, note.center_y
    h, w = mask.shape
    x_radius = int(spacing * 0.6)
    y_radius = int(spacing * 2.5)

    x1 = max(0, cx - x_radius)
    x2 = min(w, cx + x_radius)

    above = mask[max(0, cy - y_radius) : max(0, cy - int(spacing * 0.3)), x1:x2]
    below = mask[min(h, cy + int(spacing * 0.3)) : min(h, cy + y_radius), x1:x2]
    above_ink = np.sum(above > 0) if above.size > 0 else 0
    below_ink = np.sum(below > 0) if below.size > 0 else 0

    if above_ink > below_ink * 1.5:
        return "up"
    if below_ink > above_ink * 1.5:
        return "down"
    return "down" if note.step > 0 else "up"


def _find_stem_endpoint(mask: MatLike, note: Note, spacing: float, direction: str) -> int | None:
    cx, cy = note.center_x, note.center_y
    h, w = mask.shape
    x_radius = int(spacing * 0.4)
    x1 = max(0, cx - x_radius)
    x2 = min(w, cx + x_radius)
    search_range = int(spacing * 3)

    if direction == "up":
        y_start = max(0, cy - search_range)
        y_end = max(0, cy - int(spacing * 0.4))
        for y in range(y_end - 1, y_start - 1, -1):
            if np.any(mask[y, x1:x2] > 0):
                return y
    else:
        y_start = min(h, cy + int(spacing * 0.4))
        y_end = min(h, cy + search_range)
        for y in range(y_start, y_end):
            if np.any(mask[y, x1:x2] > 0):
                return y

    return None


def _find_peaks(data: np.ndarray, threshold: float) -> list[tuple[int, float]]:
    peaks = []
    i = 0
    n = len(data)

    while i < n:
        if data[i] > threshold:
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
