"""
Flattened Tree Structure of the notes
described further in score_tree.py
These support the detections we need for each of the different components

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
    y: int
    x_start: int
    x_end: int


@dataclass
class Staff:
    lines: list[StaffLine]
    spacing: float
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
    closing_bar: "BarLine | None" = None
    crop: MatLike | None = None


@dataclass(frozen=True)
class KeySignature:
    fifths: int | None = None
    mode: Literal["major", "minor"] | None = None


@dataclass(frozen=True)
class TimeSignature:
    numerator: int | None = None
    denominator: int | None = None


@dataclass
class Clef:
    staff_index: int
    kind: ClefKind | None
    x_start: int
    x_end: int
    y_top: int
    y_bottom: int
    key_signature: KeySignature
    time_signature: TimeSignature
    key_header_glyphs: list["Accidental"] = field(default_factory=list)


@dataclass(frozen=True)
class ClefDetection:
    clef: ClefKind | None
    confidence: float
    letter_score_treble: float = 0.0
    letter_score_bass: float = 0.0
    slide_score_treble: float = 0.0
    slide_score_bass: float = 0.0
    treble_match_top_left: tuple[int, int] | None = None
    treble_match_size: tuple[int, int] | None = None
    bass_match_top_left: tuple[int, int] | None = None
    bass_match_size: tuple[int, int] | None = None


@dataclass(frozen=True)
class Accidental:
    kind: AccidentalKind
    staff_index: int
    measure_index: int
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
    step: int
    step_confidence: StepConfidence | None = None
    pitch_letter: str | None = None
    octave: int | None = None
    duration_class: DurationClass | None = None


@dataclass
class Score:
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
