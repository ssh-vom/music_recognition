"""Simplified data model for sheet music - flat structure with clear relationships.

All entities reference each other by IDs rather than deep nesting.
"""

from dataclasses import dataclass, field
from typing import Literal

from cv2.typing import MatLike


# Type aliases for clarity
BarKind = Literal["single", "double_left", "double_right"]
RepeatKind = Literal["none", "begin", "end"]
NoteKind = Literal["notehead"]
ClefKind = Literal["treble", "bass"]
AccidentalKind = Literal["sharp", "flat"]
AccidentalRegion = Literal["measure", "header"]
StepConfidence = Literal["high", "medium", "low"]
DurationClass = Literal["whole", "half", "quarter"]


@dataclass
class StaffLine:
    """A single staff line (one of 5 per staff)."""

    y: int
    x_start: int
    x_end: int


@dataclass
class Staff:
    """A staff containing 5 lines with spacing and vertical bounds."""

    lines: list[StaffLine]
    spacing: float  # Average gap between lines
    top: int
    bottom: int


@dataclass
class BarLine:
    """Vertical bar line separating measures."""

    x: int
    y_top: int
    y_bottom: int
    kind: BarKind
    repeat: RepeatKind
    staff_index: int  # Which staff this bar belongs to


@dataclass
class Measure:
    """A single measure/bar region within a staff.

    Contains both the geometry and detected notes (flattened from old MeasureNode).
    """

    x_start: int
    x_end: int
    y_top: int
    y_bottom: int
    staff_index: int
    # Notes are stored directly here (not nested in separate MeasureNode)
    notes: list["Note"] = field(default_factory=list)
    # Bar line that closes this measure (optional)
    closing_bar: "BarLine | None" = None
    # Staff-local crop image (measure region only)
    crop: MatLike | None = None


@dataclass(frozen=True)
class KeySignature:
    """Pitch key for this staff (circle-of-fifths representation)."""

    fifths: int | None = (
        None  # 0 = C major/A minor, positive = sharps, negative = flats
    )
    mode: Literal["major", "minor"] | None = None


@dataclass(frozen=True)
class TimeSignature:
    """Meter (e.g., 4/4, 3/4)."""

    numerator: int | None = None
    denominator: int | None = None


@dataclass
class Clef:
    """Clef symbol with position and associated key/time signatures."""

    staff_index: int
    kind: ClefKind | None
    x_start: int
    x_end: int
    y_top: int
    y_bottom: int
    key_signature: KeySignature
    time_signature: TimeSignature
    # Sharp/flat glyphs in the key-signature strip (crop-local x,y)
    key_header_glyphs: list["Accidental"] = field(default_factory=list)


@dataclass(frozen=True)
class ClefDetection:
    """Clef classification results with template matching scores."""

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
    """A sharp or flat (measure-local or header coordinates)."""

    kind: AccidentalKind
    staff_index: int
    measure_index: int  # -1 = key header / clef crop
    center_x: int
    center_y: int
    confidence: float
    region: AccidentalRegion = "measure"


@dataclass
class Note:
    """A detected notehead with pitch and duration."""

    kind: NoteKind
    staff_index: int
    measure_index: int
    center_x: int  # Measure-local center
    center_y: int  # Measure-local center
    step: int  # Staff step (0 = bottom line, +1 = next space/line up)
    step_confidence: StepConfidence | None = None
    pitch_letter: str | None = None  # e.g., "C", "F#"
    octave: int | None = None
    duration_class: DurationClass | None = None


@dataclass
class Score:
    """Complete sheet music analysis result - flat structure.

    The old ScoreTree with nested StaffNode/MeasureNode hierarchy is now flattened:
    - staffs: list of Staff objects (by staff_index order)
    - measures: flat list of Measure objects (can filter by staff_index)
    - notes: flat list of Note objects (can filter by staff_index/measure_index)

    This makes it easy to:
    - Get all notes: score.notes
    - Get notes for staff 0: [n for n in score.notes if n.staff_index == 0]
    - Get measures for staff 0: [m for m in score.measures if m.staff_index == 0]
    """

    image_path: str
    # Original sheet image
    sheet_image: MatLike
    # All detected staves (index matches staff_index in other entities)
    staffs: list[Staff]
    # All measures across all staves (flat list)
    measures: list[Measure]
    # All bar lines
    bars: list[BarLine]
    # All detected notes
    notes: list[Note]
    # Clef info per staff (dict keyed by staff_index)
    clefs: dict[int, Clef]
    # Clef detection metadata
    clef_detections: dict[int, ClefDetection]
    # Processing masks for debugging
    notes_mask: MatLike | None = None
    bars_mask: MatLike | None = None

    # Helper methods for common access patterns
    def get_measures_for_staff(self, staff_index: int) -> list[Measure]:
        """Get all measures belonging to a specific staff."""
        return [m for m in self.measures if m.staff_index == staff_index]

    def get_notes_for_staff(self, staff_index: int) -> list[Note]:
        """Get all notes belonging to a specific staff."""
        return [n for n in self.notes if n.staff_index == staff_index]

    def get_notes_for_measure(self, staff_index: int, measure_index: int) -> list[Note]:
        """Get all notes in a specific measure."""
        return [
            n
            for n in self.notes
            if n.staff_index == staff_index and n.measure_index == measure_index
        ]

    def get_bars_for_staff(self, staff_index: int) -> list[BarLine]:
        """Get all bar lines for a specific staff."""
        return [b for b in self.bars if b.staff_index == staff_index]


# Keep ScoreTree alias for backward compatibility during transition
ScoreTree = Score
