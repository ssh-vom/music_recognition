from dataclasses import dataclass


@dataclass
class BarLine:
    x: int
    y_top: int
    y_bottom: int
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
