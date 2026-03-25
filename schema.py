from dataclasses import dataclass, field
from typing import Literal


BarKind = Literal["single", "double_left", "double_right"]
RepeatKind = Literal["none", "begin", "end"]
NoteKind = Literal["notehead"]
ClefKind = Literal["treble", "bass"]
AccidentalKind = Literal["sharp", "flat"]
AccidentalRegion = Literal["measure", "header"]
StepConfidence = Literal["high", "medium", "low"]
DurationClass = Literal["whole", "half", "quarter"]


@dataclass
class Measure:
    x_start: int
    x_end: int
    y_top: int
    y_bottom: int
    staff_index: int


@dataclass
class BarLine:
    x: int
    y_top: int
    y_bottom: int
    kind: BarKind
    repeat: RepeatKind
    staff_index: int  # which staff the barline belongs to


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


@dataclass(frozen=True)
class KeySignature:
    """Pitch key for this staff (or measure, when overridden later).

    `fifths` is the circle-of-fifths offset: 0 = C major / A minor, positive =
    sharps, negative = flats. Fields are None until detection fills them.
    """

    fifths: int | None = None
    mode: Literal["major", "minor"] | None = None


@dataclass(frozen=True)
class TimeSignature:
    """Meter. None until detected (e.g. OCR or symbol recognition)."""

    numerator: int | None = None
    denominator: int | None = None


@dataclass
class Clef:
    """Header context for one staff: clef symbol plus key and time for notation.

    `staff_index` ties this to the same staff across all measures on that line.
    Bounds cover the left header crop (clef + key-signature zone). Key and time
    usually describe the opening signature; if they change mid-piece, future
    code can record overrides on `Measure` or separate events while this keeps
    the initial staff context.
    """

    staff_index: int
    kind: ClefKind | None
    x_start: int
    x_end: int
    y_top: int
    y_bottom: int
    key_signature: KeySignature
    time_signature: TimeSignature
    # Sharp/flat glyphs in the key-signature strip (crop-local x,y; ``measure_index`` -1).
    key_header_glyphs: list["Accidental"] = field(default_factory=list)


@dataclass(frozen=True)
class ClefDetection:
    """Clef classification plus per-template diagnostic scores and boxes."""

    clef: ClefKind | None
    # Winner letterbox score (the value used for confidence thresholding).
    confidence: float
    # Letterbox NCC on the left header ROI (drives treble vs bass decision).
    letter_score_treble: float = 0.0
    letter_score_bass: float = 0.0
    # Sliding matchTemplate NCC per template (diagnostic only).
    slide_score_treble: float = 0.0
    slide_score_bass: float = 0.0
    # Letterbox boxes for each template in clef-key crop coordinates.
    treble_match_top_left: tuple[int, int] | None = None
    treble_match_size: tuple[int, int] | None = None
    bass_match_top_left: tuple[int, int] | None = None
    bass_match_size: tuple[int, int] | None = None


@dataclass(frozen=True)
class Accidental:
    """A sharp or flat (measure-local or header-crop-local pixel coordinates)."""

    kind: AccidentalKind
    staff_index: int
    measure_index: int  # -1 = key header / clef crop (see ``region``)
    center_x: int
    center_y: int
    confidence: float
    region: AccidentalRegion = "measure"


@dataclass
class Note:
    kind: NoteKind
    staff_index: int
    measure_index: int
    center_x: int  # measure local center
    center_y: int  # measure local center
    step: int
    step_confidence: StepConfidence | None = None
    pitch_letter: str | None = None
    octave: int | None = None
    duration_class: DurationClass | None = None
