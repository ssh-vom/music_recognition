"""Export detected music to ABC notation."""

from pathlib import Path

from note_grouping import EVENT_X_TOLERANCE_PX, group_notes_into_events
from schema import BarLine

BEAM_BREAK_EPSILON = 1e-6


def write_abc_file(
    score_tree,
    output_path,
    *,
    title="Sheet Music",
    reference_number=1,
    meter="4/4",
    unit_note_length="1/4",
    key="C",
    tempo_qpm: int | None = None,
):
    abc_text = build_abc_text(
        score=score_tree,
        title=title,
        reference_number=reference_number,
        meter=meter,
        unit_note_length=unit_note_length,
        key=key,
        tempo_qpm=tempo_qpm,
    )
    path = Path(output_path)
    path.write_text(abc_text, encoding="utf-8")
    return abc_text


def build_abc_text(
    score, *, title, reference_number, meter, unit_note_length, key, tempo_qpm
):
    header = [
        f"X:{reference_number}",
        f"T:{title}",
        f"M:{meter}",
        f"L:{unit_note_length}",
        *([f"Q:1/4={tempo_qpm}"] if tempo_qpm is not None else []),
        f"K:{key}",
    ]
    return (
        "\n".join(header)
        + "\n"
        + notes_to_abc_body(score=score, meter=meter, key=key)
        + "\n"
    )


def notes_to_abc_body(score, *, meter, key):
    default_rest = _default_measure_rest(meter)
    beats_per_measure = _meter_numerator(meter)
    key_accidentals = _abc_key_signature_accidentals(key)
    staff_lines = []

    for staff_index, _ in enumerate(score.staffs):
        measures = score.get_measures_for_staff(staff_index)
        if not measures:
            continue

        bars = score.get_bars_for_staff(staff_index)
        end_bar = _find_end_bar(bars, measures)
        segments = ["|:" if _has_left_begin_repeat_flat(bars, measures) else "|"]

        for i, measure in enumerate(measures):
            notes = score.get_notes_for_measure(staff_index, i)
            tokens = _notes_to_measure_tokens(
                notes=notes,
                beats_per_measure=beats_per_measure,
                key_accidentals=key_accidentals,
            )
            segments += [
                " ".join(tokens) if tokens else default_rest,
                _boundary_separator(measure.closing_bar)
                if i < len(measures) - 1
                else ":|"
                if end_bar and end_bar.repeat == "end"
                else "|",
            ]

        staff_lines.append(" ".join(segments))

    return "\n".join(staff_lines) if staff_lines else f"| {default_rest} |"


def _has_left_begin_repeat_flat(
    staff_bars: list[BarLine], staff_measures: list
) -> bool:
    if not staff_bars or not staff_measures:
        return False
    first_measure = staff_measures[0]
    return any(
        bar.x <= first_measure.x_start + 8 and bar.repeat == "begin"
        for bar in staff_bars
    )


def _find_end_bar(staff_bars: list[BarLine], staff_measures: list) -> BarLine | None:
    if not staff_measures:
        return None
    last_measure = staff_measures[-1]
    right_candidates = [bar for bar in staff_bars if bar.x >= (last_measure.x_end - 10)]
    return max(right_candidates, key=lambda bar: bar.x) if right_candidates else None


def _default_measure_rest(meter):
    numerator = _meter_numerator(meter)
    return "z" if numerator == 1 else f"z{numerator}"


def _meter_numerator(meter):
    numerator = 4
    if "/" in meter:
        left = meter.split("/", maxsplit=1)[0].strip()
        if left.isdigit():
            numerator = max(1, int(left))
    return numerator


def _boundary_separator(bar):
    if bar is None:
        return "|"
    if bar.repeat == "begin":
        return "|:"
    if bar.repeat == "end":
        return ":|"
    return "|"


def _notes_to_measure_tokens(notes, beats_per_measure, key_accidentals):
    if not notes:
        return []

    events = group_notes_into_events(notes, x_tol=EVENT_X_TOLERANCE_PX)
    if not events:
        return []

    event_tokens = []
    beats = []
    for event_notes in events:
        token = _event_to_abc_pitch(event_notes, key_accidentals=key_accidentals)
        if token is None:
            continue
        event_tokens.append(token)
        beats.append(_event_beats(event_notes))

    if not event_tokens:
        return []

    total = sum(beats)
    deficit = beats_per_measure - total
    has_subquarter = any(beat < 1.0 for beat in beats)
    if not has_subquarter and 0.75 <= deficit <= 1.25:
        beats[-1] += deficit

    return _format_tokens_with_beams(event_tokens, beats)


def _format_tokens_with_beams(event_tokens, beats):
    formatted = []
    beam_run = ""
    beam_run_beats = 0.0

    for token, beat in zip(event_tokens, beats):
        rendered = f"{token}{_beats_to_abc_suffix(beat)}"
        is_beamable = 0.0 < beat < 1.0

        if not is_beamable:
            if beam_run:
                formatted.append(beam_run)
                beam_run = ""
                beam_run_beats = 0.0
            formatted.append(rendered)
            continue

        beam_run += rendered
        beam_run_beats += beat

        if beam_run_beats >= 1.0 - BEAM_BREAK_EPSILON:
            formatted.append(beam_run)
            beam_run = ""
            beam_run_beats = 0.0

    if beam_run:
        formatted.append(beam_run)

    return formatted


def _event_to_abc_pitch(event_notes, key_accidentals):
    """One ABC token per vertical slice: top note when several align at the same x."""
    pitch_tokens = []
    seen = set()

    for note in sorted(event_notes, key=_chord_note_sort_key):
        token = _note_to_abc_pitch(note, key_accidentals)
        if token is None or token in seen:
            continue
        seen.add(token)
        pitch_tokens.append(token)

    if not pitch_tokens:
        return None
    return pitch_tokens[-1]


def _chord_note_sort_key(note):
    return (
        note.octave if note.octave is not None else 0,
        note.pitch_letter or "",
        note.center_y,
    )


def _event_beats(event_notes):
    beats = [
        _note_base_beats(note)
        for note in event_notes
        if note.duration_class is not None or note.pitch_letter is not None
    ]
    return min(beats) if beats else 1.0


def _note_to_abc_pitch(note, key_accidentals):
    if note.pitch_letter is None or note.octave is None:
        return None
    return _pitch_to_abc(note.pitch_letter, note.octave, key_accidentals)


def _pitch_to_abc(pitch_letter, octave, key_accidentals):
    base = pitch_letter[0].upper()
    accidental_char = pitch_letter[1:2]
    key_accidental = key_accidentals.get(base, "")

    accidental = ""
    if accidental_char == "#" and key_accidental != "#":
        accidental = "^"
    elif accidental_char == "b" and key_accidental != "b":
        accidental = "_"

    if octave >= 5:
        apostrophes = "'" * (octave - 5)
        return f"{accidental}{base.lower()}{apostrophes}"

    return f"{accidental}{base}{',' * max(0, 4 - octave)}"


SHARP_ORDER = ("F", "C", "G", "D", "A", "E", "B")
FLAT_ORDER = ("B", "E", "A", "D", "G", "C", "F")


def _abc_key_signature_accidentals(key: str) -> dict[str, str]:
    major_fifths = {
        "CB": -7,
        "GB": -6,
        "DB": -5,
        "AB": -4,
        "EB": -3,
        "BB": -2,
        "F": -1,
        "C": 0,
        "G": 1,
        "D": 2,
        "A": 3,
        "E": 4,
        "B": 5,
        "F#": 6,
        "C#": 7,
    }

    normalized = (key or "C").strip() or "C"
    tonic = normalized.split()[0].replace("♭", "b").replace("♯", "#")
    tonic_upper = tonic[0].upper() + tonic[1:]
    fifths = major_fifths.get(tonic_upper.upper(), major_fifths.get(tonic_upper, 0))

    accidentals = {}
    if fifths > 0:
        for letter in SHARP_ORDER[:fifths]:
            accidentals[letter] = "#"
    elif fifths < 0:
        for letter in FLAT_ORDER[: abs(fifths)]:
            accidentals[letter] = "b"
    return accidentals


def _note_base_beats(note):
    if note.duration_class == "whole":
        return 4.0
    if note.duration_class == "half":
        return 2.0
    if note.duration_class == "eighth":
        return 0.5
    if note.duration_class == "sixteenth":
        return 0.25
    return 1.0


def _beats_to_abc_suffix(beats):
    rounded = int(round(beats))
    if abs(beats - rounded) < 1e-6:
        return "" if rounded <= 1 else str(rounded)
    if abs(beats - 0.5) < 1e-6:
        return "/2"
    if abs(beats - 0.25) < 1e-6:
        return "/4"
    if beats > 0:
        for denom in [2, 4, 8]:
            numer = beats * denom
            if abs(numer - round(numer)) < 1e-6:
                numer = int(round(numer))
                if numer == denom:
                    return ""
                if numer == 1:
                    return f"/{denom}"
                return f"{numer}/{denom}"
    return ""
