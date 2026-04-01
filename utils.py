"""Shared lightweight helpers and constants."""

from pathlib import Path

import cv2 as cv
from cv2.typing import MatLike

from schema import Note

ROOT = Path(__file__).resolve().parent / "templates"
CLEF_DIR = ROOT / "clef"
ACCIDENTALS_DIR = ROOT / "accidentals"

CLEF_TREBLE = CLEF_DIR / "treble.png"
CLEF_BASS = CLEF_DIR / "bass.png"
ACCIDENTAL_SHARP = ACCIDENTALS_DIR / "sharp.png"
ACCIDENTAL_FLAT = ACCIDENTALS_DIR / "flat.png"

EVENT_X_TOLERANCE_PX = 5


def to_gray(image: MatLike) -> MatLike:
    if len(image.shape) == 2:
        return image.copy()
    return cv.cvtColor(image, cv.COLOR_BGR2GRAY)


def resize_to_height(template: MatLike, target_h: int) -> MatLike:
    th, tw = template.shape[:2]
    if th < 1 or target_h < 1:
        return template
    scale = target_h / th
    new_w = max(1, int(round(tw * scale)))
    new_h = max(1, int(round(th * scale)))
    interp = cv.INTER_AREA if scale < 1.0 else cv.INTER_CUBIC
    return cv.resize(template, (new_w, new_h), interpolation=interp)


def fit_to_roi(template: MatLike, roi_h: int, roi_w: int) -> MatLike:
    th, tw = template.shape[:2]
    if th <= roi_h and tw <= roi_w:
        return template
    scale = max(min((roi_h - 1) / th, (roi_w - 1) / tw) * 0.99, 1e-3)
    new_w = max(1, int(round(tw * scale)))
    new_h = max(1, int(round(th * scale)))
    return cv.resize(template, (new_w, new_h), interpolation=cv.INTER_AREA)


def group_notes_into_events(notes: list[Note], x_tol: int) -> list[list[Note]]:
    ordered = sorted(notes, key=lambda note: (note.center_x, note.center_y))
    events: list[list[Note]] = []

    for note in ordered:
        if not events:
            events.append([note])
            continue
        last_event = events[-1]
        anchor_x = round(sum(n.center_x for n in last_event) / float(len(last_event)))
        if abs(note.center_x - anchor_x) <= x_tol:
            last_event.append(note)
        else:
            events.append([note])

    return events
