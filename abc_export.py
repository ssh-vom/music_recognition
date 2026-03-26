from pathlib import Path

from schema import BarLine, Note
from score_tree import ScoreTree, StaffNode


def write_abc_file(
    score_tree: ScoreTree,
    output_path: str | Path,
    *,
    title: str = "Twinkle Twinkle Little Star",
    reference_number: int = 1,
    meter: str = "4/4",
    unit_note_length: str = "1/4",
    key: str = "C",
    tempo_qpm: int = 120,
) -> str:
    abc_text = build_abc_text(
        score_tree=score_tree,
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
    score_tree: ScoreTree,
    *,
    title: str,
    reference_number: int,
    meter: str,
    unit_note_length: str,
    key: str,
    tempo_qpm: int,
) -> str:
    header_lines = [
        f"X:{reference_number}",
        f"T:{title}",
        f"M:{meter}",
        f"L:{unit_note_length}",
        f"Q:1/4={tempo_qpm}",
        f"K:{key}",
    ]
    body = notes_to_abc_body(
        score_tree=score_tree,
        meter=meter,
    )
    return "\n".join(header_lines) + "\n" + body + "\n"


def notes_to_abc_body(
    score_tree: ScoreTree,
    *,
    meter: str,
) -> str:
    staff_lines: list[str] = []
    default_rest = _default_measure_rest(meter)
    beats_per_measure = _meter_numerator(meter)

    for staff_node in sorted(score_tree.staff_nodes, key=lambda node: node.index):
        if not staff_node.measures:
            continue
        segments: list[str] = []

        start_repeat = _has_left_begin_repeat(staff_node)
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
                # Emit right-edge repeat/final bar markers when detected.
                staff_end_bar = staff_node.end_bar
                if staff_end_bar is not None and staff_end_bar.repeat == "end":
                    segments.append(":|")
                else:
                    segments.append("|")

        staff_lines.append(" ".join(segments))

    if len(staff_lines) == 0:
        return f"| {default_rest} |"

    return "\n".join(staff_lines)


def _default_measure_rest(meter: str) -> str:
    numerator = 4
    if "/" in meter:
        left = meter.split("/", maxsplit=1)[0].strip()
        if left.isdigit():
            numerator = max(1, int(left))
    return "z" if numerator == 1 else f"z{numerator}"


def _meter_numerator(meter: str) -> int:
    numerator = 4
    if "/" in meter:
        left = meter.split("/", maxsplit=1)[0].strip()
        if left.isdigit():
            numerator = max(1, int(left))
    return numerator


def _has_left_begin_repeat(staff_node: StaffNode) -> bool:
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


def _boundary_separator(bar: BarLine | None) -> str:
    if bar is None:
        return "|"
    if bar.repeat == "begin":
        return "|:"
    if bar.repeat == "end":
        return ":|"
    return "|"


def _notes_to_measure_tokens(
    notes: list[Note],
    beats_per_measure: int,
) -> list[str]:
    if not notes:
        return []

    pitch_tokens: list[str] = []
    beats: list[float] = []
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

    tokens: list[str] = []
    for token, beat in zip(pitch_tokens, beats):
        tokens.append(f"{token}{_beats_to_abc_suffix(beat)}")
    return tokens


def _note_to_abc_pitch(note: Note) -> str | None:
    if note.pitch_letter is None or note.octave is None:
        return None

    return _pitch_to_abc(note.pitch_letter, note.octave)


def _pitch_to_abc(pitch_letter: str, octave: int) -> str:
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


def _note_base_beats(note: Note) -> float:
    if note.duration_class == "whole":
        return 4.0
    if note.duration_class == "half":
        return 2.0
    return 1.0


def _beats_to_abc_suffix(beats: float) -> str:
    rounded = int(round(beats))
    if abs(beats - rounded) < 1e-6:
        if rounded <= 1:
            return ""
        return str(rounded)
    return ""
