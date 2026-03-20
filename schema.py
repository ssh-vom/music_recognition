from dataclasses import dataclass
from typing import Literal


BarKind = Literal["single", "double_left", "double_right"]
RepeatKind = Literal["none", "begin", "end"]


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
