from constants import Constants as const
from schema import Score


def detection_logs_text(score: Score) -> str:
    return (
        format_clef_log(score)
        + format_accidental_log(score)
        + format_key_signature_log(score)
        + format_time_signature_log(score)
        + format_note_log(score)
    )


def format_clef_log(score: Score) -> str:
    lines = []
    for staff_index, det in score.clef_detections.items():
        lines.append(
            f"staff {staff_index}: {det.clef!r}  "
            f"letter T/B={det.letter_score_treble:.3f}/{det.letter_score_bass:.3f}  "
            f"slide T/B={det.slide_score_treble:.3f}/{det.slide_score_bass:.3f}\n"
        )
    return "".join(lines)


def format_accidental_log(score: Score) -> str:
    clef = score.clefs.get(0)
    glyphs = clef.key_header_glyphs
    sharps = sum(1 for g in glyphs if g.kind == "sharp")
    flats = sum(1 for g in glyphs if g.kind == "flat")
    lines = [
        f"staff 0 header accidentals: sharps={sharps} flats={flats} total={len(glyphs)}\n"
    ]
    for g in glyphs:
        lines.append(
            f"  kind={g.kind:<5} x={g.center_x:>4}, y={g.center_y:>4}, conf={g.confidence:.3f}\n"
        )
    return "".join(lines)


def format_time_signature_log(score: Score) -> str:
    clef = score.clefs.get(0)
    ts = clef.time_signature
    if ts.numerator is None or ts.denominator is None:
        return "staff 0 time signature: ?\n"
    return f"staff 0 time signature: {ts.numerator}/{ts.denominator}\n"


def format_key_signature_log(score: Score) -> str:
    return f"staff 0 key signature: {abc_key_from_score(score)}\n"


def meter_from_score(score: Score) -> str:
    clef = score.clefs.get(0)
    ts = clef.time_signature
    if ts.numerator is None or ts.denominator is None:
        return const.DEFAULT_METER
    return f"{ts.numerator}/{ts.denominator}"


def abc_key_from_score(score: Score) -> str:
    clef = score.clefs.get(0)
    if clef.key_signature.fifths is None:
        return const.DEFAULT_KEY
    return abc_key_from_fifths(clef.key_signature.fifths)


def abc_key_from_fifths(fifths: int) -> str:
    major_keys = {
        -7: "Cb",
        -6: "Gb",
        -5: "Db",
        -4: "Ab",
        -3: "Eb",
        -2: "Bb",
        -1: "F",
        0: "C",
        1: "G",
        2: "D",
        3: "A",
        4: "E",
        5: "B",
        6: "F#",
        7: "C#",
    }
    return major_keys.get(fifths, const.DEFAULT_KEY)


def format_note_log(score: Score) -> str:
    notes_by_measure: dict[tuple[int, int], list] = {}
    for note in score.notes:
        notes_by_measure.setdefault((note.staff_index, note.measure_index), []).append(
            note
        )

    lines = []
    for staff_index in range(len(score.staffs)):
        for measure_index, _ in enumerate(score.get_measures_for_staff(staff_index)):
            notes = notes_by_measure.get((staff_index, measure_index), [])
            lines.append(
                f"staff {staff_index}, measure {measure_index}: {len(notes)} noteheads detected\n"
            )
            for note in notes:
                pitch = (
                    f"{note.pitch_letter}{note.octave}"
                    if note.pitch_letter is not None and note.octave is not None
                    else "?"
                )
                lines.append(
                    f"  x={note.center_x:>4}, y={note.center_y:>4}, step={note.step:>3}, "
                    f"conf={note.step_confidence or '?':<6}, pitch={pitch:<4}, duration={note.duration_class or '?'}\n"
                )
    return "".join(lines)
