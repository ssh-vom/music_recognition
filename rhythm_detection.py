import cv2 as cv
import numpy as np
from cv2.typing import MatLike

from schema import Note, Staff


def refine_beamed_durations(
    mask: MatLike,
    notes: list[Note],
    staff: Staff,
) -> list[Note]:
    """Update duration_class for notes connected by beams (never adds new notes)."""
    if len(notes) < 2:
        return notes

    _, labels, _, _ = cv.connectedComponentsWithStats(mask, connectivity=8)

    # beamed notes share the same connected component because the beam joins their stems;
    # group notes by component id so we can check each beam group together
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
                # only promote quarter notes — halves and wholes sharing a component are left alone
                if note.duration_class == "quarter":
                    notes[idx].duration_class = (
                        "eighth" if beam_count == 1 else "sixteenth"
                    )

    return notes


def _detect_beam_count(
    mask: MatLike,
    note_group: list[tuple[int, Note]],
    staff: Staff,
) -> int:
    notes = [n for _, n in note_group]
    spacing = staff.spacing

    stem_dirs = [_estimate_stem_direction(mask, n, spacing) for n in notes]
    up_count = sum(1 for d in stem_dirs if d == "up")
    down_count = sum(1 for d in stem_dirs if d == "down")

    if not (up_count or down_count):
        return 0

    # use the majority stem direction to decide which end of the stems to look for the beam
    beam_direction = "up" if up_count > down_count else "down"

    min_x = min(n.center_x for n in notes)
    max_x = max(n.center_x for n in notes)
    # notes that are too close together horizontally are likely the same note detected twice, not a beam group
    if max_x - min_x < spacing * 0.5:
        return 0

    # find where each stem ends in the beam direction
    tips = [
        y
        for note in notes
        if (y := _find_stem_endpoint(mask, note, spacing, beam_direction)) is not None
    ]
    if not tips:
        return 0

    # the beam sits at the extreme tip — smallest y for upward stems, largest for downward
    beam_y = min(tips) if beam_direction == "up" else max(tips)
    padding = int(spacing * 0.5)
    y0 = max(0, beam_y - padding)
    y1 = min(mask.shape[0], beam_y + padding)
    if y0 >= y1 or min_x >= max_x:
        return 0

    band = mask[y0:y1, min_x:max_x]
    if band.size == 0:
        return 0

    # count how many pixels are lit in each row of the band; each beam appears as a dense horizontal peak
    horizontal_density = np.sum(band > 0, axis=1)
    threshold = max(2, spacing * 0.15)
    peaks = _find_peaks(horizontal_density, threshold)

    beam_count = 0
    x_span = max_x - min_x
    for local_y, _ in peaks:
        row_mask = mask[y0 + local_y, min_x:max_x] > 0
        runs = _find_ink_runs(row_mask)
        total_ink = sum(length for _, length in runs)
        longest_run = max((length for _, length in runs), default=0)

        # a real beam covers most of the x span in a single long run; short or sparse rows are noise
        if total_ink / x_span > 0.20 and longest_run > spacing * 0.8:
            beam_count += 1
            if beam_count >= 2:
                break

    return min(beam_count, 2)


def _estimate_stem_direction(
    mask: MatLike,
    note: Note,
    spacing: float,
) -> str:
    cx, cy = note.center_x, note.center_y
    h, w = mask.shape
    x_radius = int(spacing * 0.6)
    y_radius = int(spacing * 2.5)

    x1 = max(0, cx - x_radius)
    x2 = min(w, cx + x_radius)

    # compare ink above vs below the notehead; whichever side has more ink is where the stem goes
    above = mask[max(0, cy - y_radius) : max(0, cy - int(spacing * 0.3)), x1:x2]
    below = mask[min(h, cy + int(spacing * 0.3)) : min(h, cy + y_radius), x1:x2]
    above_ink = np.sum(above > 0) if above.size > 0 else 0
    below_ink = np.sum(below > 0) if below.size > 0 else 0

    if above_ink > below_ink * 1.5:
        return "up"
    if below_ink > above_ink * 1.5:
        return "down"
    # when ink is ambiguous, use pitch position as a tiebreaker — notes above the middle line
    # conventionally have downward stems and notes below have upward stems
    return "down" if note.step > 0 else "up"


def _find_stem_endpoint(
    mask: MatLike, note: Note, spacing: float, direction: str
) -> int | None:
    cx, cy = note.center_x, note.center_y
    h, w = mask.shape
    x_radius = int(spacing * 0.4)
    x1 = max(0, cx - x_radius)
    x2 = min(w, cx + x_radius)
    search_range = int(spacing * 3)

    # walk outward from the notehead along the stem until we find the last row with ink
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
    # returns (index_of_peak, peak_value) for each contiguous run above the threshold
    peaks = []
    i = 0
    n = len(data)

    while i < n:
        if data[i] > threshold:
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
    # returns (start, length) for each contiguous run of True values
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
