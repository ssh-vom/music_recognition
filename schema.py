"""
Flattened score schema and small score-construction helpers.

These support the detections we need for each of the different components:

Staffs,
Clefs,
Bars,
Measures,
Notes,
Duration
"""

from dataclasses import dataclass, field
from typing import Literal

from cv2.typing import MatLike

# double_left / double_right are the two strokes of a double bar, kept separate so the exporter can render repeat signs correctly
BarKind = Literal["single", "double_left", "double_right"]
RepeatKind = Literal["none", "begin", "end"]
NoteKind = Literal["notehead"]
ClefKind = Literal["treble", "bass"]
AccidentalKind = Literal["sharp", "flat"]
AccidentalRegion = Literal["measure", "header"]
StepConfidence = Literal["high", "medium", "low"]
DurationClass = Literal["whole", "half", "quarter", "eighth", "sixteenth"]


@dataclass
class StaffLine:
    y: int        # centre row in the full image
    x_start: int
    x_end: int


@dataclass
class Staff:
    lines: list[StaffLine]
    spacing: float  # pixel distance between adjacent lines; used as the scale unit throughout the pipeline
    top: int
    bottom: int


@dataclass
class BarLine:
    x: int
    y_top: int
    y_bottom: int
    kind: BarKind
    repeat: RepeatKind
    staff_index: int


@dataclass
class Measure:
    x_start: int
    x_end: int
    y_top: int
    y_bottom: int
    staff_index: int
    notes: list["Note"] = field(default_factory=list)
    closing_bar: "BarLine | None" = None  # bar that ends this measure; drives repeat symbol export
    crop: MatLike | None = None           # binary image slice used for note detection


@dataclass(frozen=True)
class KeySignature:
    fifths: int | None = None  # positive = sharps, negative = flats, 0 = C major
    mode: Literal["major", "minor"] | None = None


@dataclass(frozen=True)
class TimeSignature:
    numerator: int | None = None
    denominator: int | None = None


@dataclass
class Clef:
    staff_index: int
    kind: ClefKind | None  # None if detection confidence was too low
    x_start: int
    x_end: int  # notes to the right of this are musical content, not header symbols
    y_top: int
    y_bottom: int
    key_signature: KeySignature
    time_signature: TimeSignature
    key_header_glyphs: list["Accidental"] = field(default_factory=list)


@dataclass(frozen=True)
class ClefDetection:
    clef: ClefKind | None
    confidence: float
    letter_score_treble: float = 0.0  # score from letterbox match (template fitted to ROI)
    letter_score_bass: float = 0.0
    slide_score_treble: float = 0.0   # score from multi-scale sliding match
    slide_score_bass: float = 0.0
    treble_match_top_left: tuple[int, int] | None = None
    treble_match_size: tuple[int, int] | None = None
    bass_match_top_left: tuple[int, int] | None = None
    bass_match_size: tuple[int, int] | None = None


@dataclass(frozen=True)
class Accidental:
    kind: AccidentalKind
    staff_index: int
    measure_index: int  # -1 for key signature accidentals in the header
    center_x: int
    center_y: int
    confidence: float
    region: AccidentalRegion = "measure"


@dataclass
class Note:
    kind: NoteKind
    staff_index: int
    measure_index: int
    center_x: int
    center_y: int
    step: int  # half-steps up from the bottom staff line; negative = below the staff
    step_confidence: StepConfidence | None = None
    pitch_letter: str | None = None  # e.g. "C" or "F#"; set by resolve_pitches
    octave: int | None = None        # scientific octave; set by resolve_pitches
    duration_class: DurationClass | None = None


@dataclass(frozen=True)
class HeaderAnalysis:
    header_accidentals: list[Accidental]
    key_signature: KeySignature
    time_signature: TimeSignature | None
    content_start_x: int | None  # x where musical content begins, after clef/key/time
    search_min_x: int | None = None
    search_max_x: int | None = None


@dataclass
class Score:
    """All detections for one sheet image stored in flat lists, filtered by index on the way out."""

    image_path: str
    sheet_image: MatLike
    staffs: list[Staff]
    measures: list[Measure]
    bars: list[BarLine]
    notes: list[Note]
    clefs: dict[int, Clef]
    clef_detections: dict[int, ClefDetection]

    def get_measures_for_staff(self, staff_index: int) -> list[Measure]:
        return [m for m in self.measures if m.staff_index == staff_index]

    def get_notes_for_measure(self, staff_index: int, measure_index: int) -> list[Note]:
        return [
            n
            for n in self.notes
            if n.staff_index == staff_index and n.measure_index == measure_index
        ]

    def get_bars_for_staff(self, staff_index: int) -> list[BarLine]:
        return [b for b in self.bars if b.staff_index == staff_index]


def build_score(
    *,
    image_path: str,
    sheet_image: MatLike,
    staffs: list[Staff],
    bars: list[BarLine],
    clefs_by_staff: dict[int, Clef],
    clef_detections: dict[int, ClefDetection],
    measures_map: dict[int, list[Measure]],
    measure_crops: dict[int, list[MatLike]],
) -> Score:
    # group bars by staff index so we can assign closing bars efficiently
    bars_by_staff: dict[int, list[BarLine]] = {i: [] for i in range(len(staffs))}
    for bar in bars:
        bars_by_staff[bar.staff_index].append(bar)
    for staff_bars in bars_by_staff.values():
        staff_bars.sort(key=lambda bar: bar.x)

    all_measures: list[Measure] = []

    for staff_index, _ in enumerate(staffs):
        staff_measures = measures_map.get(staff_index, [])
        staff_crops = measure_crops.get(staff_index, [])
        staff_bars = bars_by_staff.get(staff_index, [])

        # attach the image crop so note detection doesn't need to re-slice later
        for measure_index, measure in enumerate(staff_measures):
            if measure_index < len(staff_crops):
                measure.crop = staff_crops[measure_index]
            all_measures.append(measure)

        # the last measure has no closing bar — it ends at the staff edge
        closing_bars = _closing_bars_for_measures(staff_bars, staff_measures)
        staff_measure_list = [m for m in all_measures if m.staff_index == staff_index]
        for measure_index, measure in enumerate(staff_measure_list):
            if measure_index < len(closing_bars):
                measure.closing_bar = closing_bars[measure_index]

    return Score(
        image_path=image_path,
        sheet_image=sheet_image,
        staffs=staffs,
        measures=all_measures,
        bars=bars,
        notes=[],  # populated later by _populate_notes in pipeline.py
        clefs=clefs_by_staff,
        clef_detections=clef_detections,
    )


def _closing_bars_for_measures(
    staff_bars: list[BarLine], staff_measures: list[Measure]
) -> list[BarLine]:
    if not staff_bars or not staff_measures:
        return []

    first_start = staff_measures[0].x_start
    last_end = staff_measures[-1].x_end
    # +12 px overshoot so the final bar line isn't missed due to rounding
    usable = [bar for bar in staff_bars if first_start < bar.x < (last_end + 12)]

    closers: list[BarLine] = []
    for index, bar in enumerate(usable):
        # skip the left stroke of a double bar; the right stroke is the real boundary
        if bar.kind == "double_left":
            if index + 1 < len(usable) and usable[index + 1].kind == "double_right":
                continue
        closers.append(bar)

    # one fewer closer than measures since the last measure closes at the staff edge
    return closers[: max(0, len(staff_measures) - 1)]
