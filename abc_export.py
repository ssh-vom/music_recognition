from pathlib import Path

from schema import BarLine, Measure, Note


def write_abc_file(
    notes_by_staff: dict[int, list[list[Note]]],
    measures_map: dict[int, list[Measure]],
    bars: list[BarLine],
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
        notes_by_staff=notes_by_staff,
        measures_map=measures_map,
        bars=bars,
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
    notes_by_staff: dict[int, list[list[Note]]],
    measures_map: dict[int, list[Measure]],
    bars: list[BarLine],
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
        notes_by_staff=notes_by_staff,
        measures_map=measures_map,
        bars=bars,
        meter=meter,
    )
    return "\n".join(header_lines) + "\n" + body + "\n"


def notes_to_abc_body(
    notes_by_staff: dict[int, list[list[Note]]],
    measures_map: dict[int, list[Measure]],
    bars: list[BarLine],
    *,
    meter: str,
) -> str:
    staff_lines: list[str] = []
    default_rest = _default_measure_rest(meter)
    beats_per_measure = _meter_numerator(meter)
    bars_by_staff = _group_bars_by_staff(bars)

    for staff_index in sorted(notes_by_staff.keys()):
        segments: list[str] = []
        staff_notes = notes_by_staff[staff_index]
        staff_measures = measures_map.get(staff_index, [])
        if not staff_measures:
            continue

        start_repeat = _has_left_begin_repeat(
            staff_bars=bars_by_staff.get(staff_index, []),
            first_measure=staff_measures[0],
        )
        if start_repeat:
            segments.append("|:")
        else:
            segments.append("|")

        closing_bars = _closing_bars_for_measures(
            staff_bars=bars_by_staff.get(staff_index, []),
            staff_measures=staff_measures,
        )

        for measure_index, notes in enumerate(staff_notes):
            tokens = _notes_to_measure_tokens(
                notes=notes,
                beats_per_measure=beats_per_measure,
            )
            if not tokens:
                tokens = [default_rest]

            segments.append(" ".join(tokens))

            if measure_index < len(staff_notes) - 1:
                boundary_bar = (
                    closing_bars[measure_index]
                    if measure_index < len(closing_bars)
                    else None
                )
                segments.append(_boundary_separator(boundary_bar))
            else:
                # Emit right-edge repeat/final bar markers when detected.
                staff_end_bar = _staff_end_bar(
                    staff_bars=bars_by_staff.get(staff_index, []),
                    last_measure=staff_measures[-1],
                )
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


def _group_bars_by_staff(bars: list[BarLine]) -> dict[int, list[BarLine]]:
    grouped: dict[int, list[BarLine]] = {}
    for bar in bars:
        grouped.setdefault(bar.staff_index, []).append(bar)
    for staff_bars in grouped.values():
        staff_bars.sort(key=lambda bar: bar.x)
    return grouped


def _has_left_begin_repeat(staff_bars: list[BarLine], first_measure: Measure) -> bool:
    if not staff_bars:
        return False
    tol = 8
    for bar in staff_bars:
        if bar.x <= first_measure.x_start + tol and bar.repeat == "begin":
            return True
    return False


def _closing_bars_for_measures(
    staff_bars: list[BarLine],
    staff_measures: list[Measure],
) -> list[BarLine]:
    if not staff_bars or not staff_measures:
        return []

    first_start = staff_measures[0].x_start
    last_end = staff_measures[-1].x_end
    usable = [bar for bar in staff_bars if first_start < bar.x < (last_end + 12)]

    closers: list[BarLine] = []
    for index, bar in enumerate(usable):
        if bar.kind == "double_left":
            if index + 1 < len(usable) and usable[index + 1].kind == "double_right":
                continue
        closers.append(bar)

    # Splitter appends a final measure after iterating closure bars, so at most
    # measure_count - 1 bars are separators.
    needed = max(0, len(staff_measures) - 1)
    return closers[:needed]


def _staff_end_bar(
    staff_bars: list[BarLine],
    last_measure: Measure,
) -> BarLine | None:
    tol = 10
    right_candidates = [
        bar for bar in staff_bars if bar.x >= (last_measure.x_end - tol)
    ]
    if not right_candidates:
        return None
    return max(right_candidates, key=lambda bar: bar.x)


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
