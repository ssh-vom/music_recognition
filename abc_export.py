"""ABC notation export - compatible with flat Score structure."""

from pathlib import Path

from schema import BarLine, Score


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
    header_lines = [
        f"X:{reference_number}",
        f"T:{title}",
        f"M:{meter}",
        f"L:{unit_note_length}",
        f"Q:1/4={tempo_qpm}",
        f"K:{key}",
    ]
    body = notes_to_abc_body(score=score, meter=meter)
    return "\n".join(header_lines) + "\n" + body + "\n"


def notes_to_abc_body(score, *, meter):
    """Convert Score to ABC body text.

    Works with both flat Score structure and old nested ScoreTree.
    """
    # Handle both old ScoreTree (has staff_nodes) and new Score (has staffs)
    if hasattr(score, "staff_nodes"):
        # Old structure - use legacy path
        return _notes_to_abc_body_legacy(score, meter)

    # New flat Score structure
    staff_lines = []
    default_rest = _default_measure_rest(meter)
    beats_per_measure = _meter_numerator(meter)

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


def _notes_to_abc_body_legacy(score_tree, meter):
    """Legacy path for old nested ScoreTree structure."""
    staff_lines = []
    default_rest = _default_measure_rest(meter)
    beats_per_measure = _meter_numerator(meter)

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


def _notes_to_measure_tokens(notes, beats_per_measure):
    if not notes:
        return []

    pitch_tokens = []
    beats = []
    for note in notes:
        token = _note_to_abc_pitch(note)
        if token is None:
            continue
        pitch_tokens.append(token)
        beats.append(_note_base_beats(note))

    if not pitch_tokens:
        return []

    total = sum(beats)
    deficit = beats_per_measure - total
    if deficit >= 1:
        beats[-1] += deficit

    tokens = []
    for token, beat in zip(pitch_tokens, beats):
        tokens.append(f"{token}{_beats_to_abc_suffix(beat)}")
    return tokens


def _note_to_abc_pitch(note):
    if note.pitch_letter is None or note.octave is None:
        return None

    return _pitch_to_abc(note.pitch_letter, note.octave)


def _pitch_to_abc(pitch_letter, octave):
    base = pitch_letter[0].upper()
    accidental_char = pitch_letter[1:2]
    accidental = ""
    if accidental_char == "#":
        accidental = "^"
    elif accidental_char == "b":
        accidental = "_"

    if octave >= 5:
        letter = base.lower()
        apostrophes = "'" * (octave - 5)
        return f"{accidental}{letter}{apostrophes}"

    letter = base
    commas = "," * max(0, 4 - octave)
    return f"{accidental}{letter}{commas}"


def _note_base_beats(note):
    if note.duration_class == "whole":
        return 4.0
    if note.duration_class == "half":
        return 2.0
    return 1.0


def _beats_to_abc_suffix(beats):
    rounded = int(round(beats))
    if abs(beats - rounded) < 1e-6:
        if rounded <= 1:
            return ""
        return str(rounded)
    return ""
