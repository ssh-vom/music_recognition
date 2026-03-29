"""ABC notation export - compatible with flat Score structure."""

from pathlib import Path

from schema import BarLine

EVENT_X_TOLERANCE_PX = 5
BEAM_BREAK_EPSILON = 1e-6


def write_abc_file(
    score_tree,  # Can be Score or old ScoreTree
    output_path,
    *,
    title="Sheet Music",
    reference_number=1,
    meter="4/4",
    unit_note_length="1/4",
    key="C",
    tempo_qpm=120,
    melody_only=False,
):
    abc_text = build_abc_text(
        score=score_tree,
        title=title,
        reference_number=reference_number,
        meter=meter,
        unit_note_length=unit_note_length,
        key=key,
        tempo_qpm=tempo_qpm,
        melody_only=melody_only,
    )
    path = Path(output_path)
    path.write_text(abc_text, encoding="utf-8")
    return abc_text


def build_abc_text(
    score,
    *,
    title,
    reference_number,
    meter,
    unit_note_length,
    key,
    tempo_qpm,
    melody_only=False,
):
    header_lines = [
        f"X:{reference_number}",
        f"T:{title}",
        f"M:{meter}",
        f"L:{unit_note_length}",
        f"Q:1/4={tempo_qpm}",
        f"K:{key}",
    ]
    body = notes_to_abc_body(
        score=score,
        meter=meter,
        key=key,
        melody_only=melody_only,
    )
    return "\n".join(header_lines) + "\n" + body + "\n"


def notes_to_abc_body(score, *, meter, key, melody_only=False):
    """Convert Score to ABC body text.

    Works with both flat Score structure and old nested ScoreTree.
    """
    # Handle both old ScoreTree (has staff_nodes) and new Score (has staffs)
    if hasattr(score, "staff_nodes"):
        # Old structure - use legacy path
        return _notes_to_abc_body_legacy(
            score,
            meter,
            key,
            melody_only=melody_only,
        )

    # New flat Score structure
    staff_lines = []
    default_rest = _default_measure_rest(meter)
    beats_per_measure = _meter_numerator(meter)
    key_accidentals = _abc_key_signature_accidentals(key)

    # Group measures and bars by staff
    for staff_index, staff in enumerate(score.staffs):
        staff_measures = score.get_measures_for_staff(staff_index)
        staff_bars = score.get_bars_for_staff(staff_index)

        if not staff_measures:
            continue

        segments = []

        # Check for repeat at beginning
        start_repeat = _has_left_begin_repeat_flat(staff_bars, staff_measures)
        if start_repeat:
            segments.append("|:")
        else:
            segments.append("|")

        for measure_index, measure in enumerate(staff_measures):
            # Get notes for this measure from the flat list
            notes = score.get_notes_for_measure(staff_index, measure_index)

            tokens = _notes_to_measure_tokens(
                notes=notes,
                beats_per_measure=beats_per_measure,
                key_accidentals=key_accidentals,
                melody_only=melody_only,
            )
            if not tokens:
                tokens = [default_rest]

            segments.append(" ".join(tokens))

            # Add bar separator
            if measure_index < len(staff_measures) - 1:
                boundary_bar = measure.closing_bar
                segments.append(_boundary_separator(boundary_bar))
            else:
                # End of staff - check for end repeat
                end_bar = _find_end_bar(staff_bars, staff_measures)
                if end_bar is not None and end_bar.repeat == "end":
                    segments.append(":|")
                else:
                    segments.append("|")

        staff_lines.append(" ".join(segments))

    if len(staff_lines) == 0:
        return f"| {default_rest} |"

    return "\n".join(staff_lines)


def _notes_to_abc_body_legacy(score_tree, meter, key, melody_only=False):
    """Legacy path for old nested ScoreTree structure."""
    staff_lines = []
    default_rest = _default_measure_rest(meter)
    beats_per_measure = _meter_numerator(meter)
    key_accidentals = _abc_key_signature_accidentals(key)

    for staff_node in sorted(score_tree.staff_nodes, key=lambda node: node.index):
        if not staff_node.measures:
            continue
        segments = []

        start_repeat = _has_left_begin_repeat_legacy(staff_node)
        if start_repeat:
            segments.append("|:")
        else:
            segments.append("|")

        for measure_index, measure_node in enumerate(staff_node.measures):
            tokens = _notes_to_measure_tokens(
                notes=measure_node.notes,
                beats_per_measure=beats_per_measure,
                key_accidentals=key_accidentals,
                melody_only=melody_only,
            )
            if not tokens:
                tokens = [default_rest]

            segments.append(" ".join(tokens))

            if measure_index < len(staff_node.measures) - 1:
                boundary_bar = measure_node.closing_bar
                segments.append(_boundary_separator(boundary_bar))
            else:
                staff_end_bar = staff_node.end_bar
                if staff_end_bar is not None and staff_end_bar.repeat == "end":
                    segments.append(":|")
                else:
                    segments.append("|")

        staff_lines.append(" ".join(segments))

    if len(staff_lines) == 0:
        return f"| {default_rest} |"

    return "\n".join(staff_lines)


def _has_left_begin_repeat_legacy(staff_node):
    """Check if staff starts with a begin repeat (legacy)."""
    if not staff_node.bars:
        return False
    if not staff_node.measures:
        return False

    first_measure = staff_node.measures[0].measure
    tol = 8
    for bar in staff_node.bars:
        if bar.x <= first_measure.x_start + tol and bar.repeat == "begin":
            return True
    return False


def _has_left_begin_repeat_flat(
    staff_bars: list[BarLine], staff_measures: list
) -> bool:
    """Check if staff starts with a begin repeat (flat structure)."""
    if not staff_bars or not staff_measures:
        return False

    first_measure = staff_measures[0]
    tol = 8
    for bar in staff_bars:
        if bar.x <= first_measure.x_start + tol and bar.repeat == "begin":
            return True
    return False


def _find_end_bar(staff_bars: list[BarLine], staff_measures: list) -> BarLine | None:
    """Find the bar line at the end of the staff."""
    if not staff_measures:
        return None

    last_measure = staff_measures[-1]
    tol = 10

    right_candidates = [
        bar for bar in staff_bars if bar.x >= (last_measure.x_end - tol)
    ]
    if not right_candidates:
        return None
    return max(right_candidates, key=lambda bar: bar.x)


def _default_measure_rest(meter):
    numerator = 4
    if "/" in meter:
        left = meter.split("/", maxsplit=1)[0].strip()
        if left.isdigit():
            numerator = max(1, int(left))
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


def _notes_to_measure_tokens(
    notes,
    beats_per_measure,
    key_accidentals,
    melody_only=False,
):
    if not notes:
        return []

    events = _group_notes_into_events(notes)
    if not events:
        return []

    event_tokens = []
    beats = []
    for event_notes in events:
        token = _event_to_abc_pitch(
            event_notes,
            key_accidentals=key_accidentals,
            melody_only=melody_only,
        )
        if token is None:
            continue
        event_tokens.append(token)
        beats.append(_event_beats(event_notes))

    if not event_tokens:
        return []

    # Conservative carry rule:
    # preserve the old "stretch last note" behavior only for simple quarter/half
    # measures that are about one beat short. Do not apply it when we already have
    # eighth/sixteenth rhythms, because that creates nonsense like 5/2 endings.
    total = sum(beats)
    deficit = beats_per_measure - total
    has_subquarter = any(beat < 1.0 for beat in beats)
    if not has_subquarter and 0.75 <= deficit <= 1.25:
        beats[-1] += deficit

    return _format_tokens_with_beams(event_tokens, beats)



def _format_tokens_with_beams(event_tokens, beats):
    """Format ABC tokens so beamable notes are physically adjacent.

    In ABC, adjacent short notes (eighths/sixteenths) should be written without
    spaces to allow the renderer to beam them together, e.g. `g/2g/2`.

    We keep grouping simple and conservative:
    - quarter/half/whole notes always stand alone
    - consecutive sub-quarter events are concatenated
    - insert a space whenever we hit a 1-beat boundary
    """
    formatted: list[str] = []
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

        # Break beam runs at quarter-note beat boundaries.
        if beam_run_beats >= 1.0 - BEAM_BREAK_EPSILON:
            formatted.append(beam_run)
            beam_run = ""
            beam_run_beats = 0.0

    if beam_run:
        formatted.append(beam_run)

    return formatted



def _group_notes_into_events(notes):
    """Group notes that occur at nearly the same x-position into one ABC event.

    This prevents stacked noteheads / near-duplicate detections from being exported
    as sequential notes that inflate the beat count. Single-note events stay
    monophonic; multi-note events become ABC chords.
    """
    ordered = sorted(notes, key=lambda note: (note.center_x, note.center_y))
    events = []

    for note in ordered:
        if not events:
            events.append([note])
            continue

        last_event = events[-1]
        anchor_x = round(
            sum(existing.center_x for existing in last_event) / float(len(last_event))
        )
        if abs(note.center_x - anchor_x) <= EVENT_X_TOLERANCE_PX:
            last_event.append(note)
        else:
            events.append([note])

    return events



def _event_to_abc_pitch(event_notes, key_accidentals, melody_only=False):
    """Convert one note event to an ABC token.

    A single detected note becomes a normal ABC pitch. Multiple notes sharing the
    same onset become an ABC chord like [CEG].
    """
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
    if melody_only:
        return pitch_tokens[-1]
    if len(pitch_tokens) == 1:
        return pitch_tokens[0]
    return "[" + "".join(pitch_tokens) + "]"



def _chord_note_sort_key(note):
    octave = note.octave if note.octave is not None else 0
    letter = note.pitch_letter or ""
    return (octave, letter, note.center_y)



def _event_beats(event_notes):
    """Choose one duration for a simultaneous event.

    For chords this should usually be identical across notes. If OCR disagrees,
    prefer the shortest duration so we do not overfill the measure in ABC.
    """
    beats = [
        _note_base_beats(note)
        for note in event_notes
        if note.duration_class is not None or note.pitch_letter is not None
    ]
    if not beats:
        return 1.0
    return min(beats)


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
        letter = base.lower()
        apostrophes = "'" * (octave - 5)
        return f"{accidental}{letter}{apostrophes}"

    letter = base
    commas = "," * max(0, 4 - octave)
    return f"{accidental}{letter}{commas}"


SHARP_ORDER = ("F", "C", "G", "D", "A", "E", "B")
FLAT_ORDER = ("B", "E", "A", "D", "G", "C", "F")


def _abc_key_signature_accidentals(key: str) -> dict[str, str]:
    """Return the implied accidentals from an ABC key signature.

    This keeps note spellings relative to K:, so in K:F we emit B instead of _B,
    and in K:G we emit F/f instead of ^F/^f.
    """
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

    normalized = (key or "C").strip()
    if not normalized:
        normalized = "C"
    tonic = normalized.split()[0]
    tonic = tonic.replace("♭", "b").replace("♯", "#")
    tonic_upper = tonic[0].upper() + tonic[1:]
    fifths = major_fifths.get(tonic_upper.upper(), major_fifths.get(tonic_upper, 0))

    accidentals: dict[str, str] = {}
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
    # Handle simple integer cases (whole, half, quarter multiples)
    rounded = int(round(beats))
    if abs(beats - rounded) < 1e-6:
        if rounded <= 1:
            return ""  # Quarter note is default
        return str(rounded)

    # Handle fractional beats (eighth, sixteenth)
    if abs(beats - 0.5) < 1e-6:
        return "/2"  # Eighth note
    if abs(beats - 0.25) < 1e-6:
        return "/4"  # Sixteenth note

    # Handle dotted rhythms or other fractions as multiples
    # e.g., 1.5 beats = quarter + eighth = "3/2"
    if beats > 0:
        # Express as numerator/denominator
        # Try to find a clean fraction representation
        for denom in [2, 4, 8]:
            numer = beats * denom
            if abs(numer - round(numer)) < 1e-6:
                numer = int(round(numer))
                if numer == denom:
                    return ""  # Quarter
                if numer == 1:
                    return f"/{denom}"
                return f"{numer}/{denom}"

    return ""
