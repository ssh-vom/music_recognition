import cv2 as cv
import numpy as np
from cv2.typing import MatLike

from schema import Clef, KeySignature, Measure, Note, Staff

LETTER_ORDER = ("C", "D", "E", "F", "G", "A", "B")
LETTER_TO_INDEX = {letter: index for index, letter in enumerate(LETTER_ORDER)}
SHARP_ORDER = ("F", "C", "G", "D", "A", "E", "B")
FLAT_ORDER = ("B", "E", "A", "D", "G", "C", "F")


class NoteDetector:
    def detect(
        self,
        cleaned_measure_mask: MatLike,
        staff: Staff,
        measure: Measure,
        measure_index: int,
    ) -> list[Note]:
        assert len(cleaned_measure_mask.shape) == 2
        assert len(staff.lines) == 5

        notehead_kernel_diameter_px = max(1, int(round(staff.spacing * 0.45)))
        if notehead_kernel_diameter_px % 2 == 0:
            notehead_kernel_diameter_px += 1

        notehead_kernel = cv.getStructuringElement(
            cv.MORPH_ELLIPSE,
            (notehead_kernel_diameter_px, notehead_kernel_diameter_px),
        )
        opened_notehead_mask = cv.morphologyEx(
            cleaned_measure_mask,
            cv.MORPH_OPEN,
            notehead_kernel,
        )
        close_kernel = cv.getStructuringElement(cv.MORPH_ELLIPSE, (3, 3))
        notehead_mask = cv.morphologyEx(
            opened_notehead_mask,
            cv.MORPH_CLOSE,
            close_kernel,
        )

        component_count, _, component_stats, component_centroids = (
            cv.connectedComponentsWithStats(notehead_mask, connectivity=8)
        )

        min_notehead_area_px = staff.spacing * staff.spacing * 0.08
        max_notehead_area_px = staff.spacing * staff.spacing * 1.8
        min_notehead_size_px = int(round(staff.spacing * 0.35))
        max_notehead_size_px = int(round(staff.spacing * 1.9))
        min_notehead_aspect_ratio = 0.45
        max_notehead_aspect_ratio = 2.2

        bottom_staff_line_local_y = int(round(staff.lines[4].y - measure.y_top))
        staff_half_step_px = staff.spacing / 2.0
        merge_distance_px = max(2, int(round(staff.spacing * 0.75)))

        raw_note_centers: list[tuple[int, int]] = []

        for component_index in range(1, component_count):
            component_width_px = int(component_stats[component_index, cv.CC_STAT_WIDTH])
            component_height_px = int(
                component_stats[component_index, cv.CC_STAT_HEIGHT]
            )
            component_area_px = float(component_stats[component_index, cv.CC_STAT_AREA])

            if component_area_px < min_notehead_area_px:
                continue
            if component_area_px > max_notehead_area_px:
                continue
            if component_width_px < min_notehead_size_px:
                continue
            if component_height_px < min_notehead_size_px:
                continue
            if component_width_px > max_notehead_size_px:
                continue
            if component_height_px > max_notehead_size_px:
                continue

            notehead_width_to_height_ratio = component_width_px / float(
                component_height_px
            )
            if notehead_width_to_height_ratio < min_notehead_aspect_ratio:
                continue
            if notehead_width_to_height_ratio > max_notehead_aspect_ratio:
                continue

            note_center_x_local_px = int(round(component_centroids[component_index][0]))
            note_center_y_local_px = int(round(component_centroids[component_index][1]))
            raw_note_centers.append((note_center_x_local_px, note_center_y_local_px))

        raw_note_centers.sort(key=lambda center: center[0])

        merged_note_centers: list[list[int]] = []
        for note_center_x_local_px, note_center_y_local_px in raw_note_centers:
            if not merged_note_centers:
                merged_note_centers.append(
                    [note_center_x_local_px, note_center_y_local_px, 1]
                )
                continue

            previous_center_x_local_px = merged_note_centers[-1][0]
            previous_center_y_local_px = merged_note_centers[-1][1]
            previous_count = merged_note_centers[-1][2]

            if (
                abs(note_center_x_local_px - previous_center_x_local_px)
                <= merge_distance_px
                and abs(note_center_y_local_px - previous_center_y_local_px)
                <= merge_distance_px
            ):
                merged_count = previous_count + 1
                merged_center_x_local_px = int(
                    round(
                        (
                            previous_center_x_local_px * previous_count
                            + note_center_x_local_px
                        )
                        / merged_count
                    )
                )
                merged_center_y_local_px = int(
                    round(
                        (
                            previous_center_y_local_px * previous_count
                            + note_center_y_local_px
                        )
                        / merged_count
                    )
                )
                merged_note_centers[-1] = [
                    merged_center_x_local_px,
                    merged_center_y_local_px,
                    merged_count,
                ]
                continue

            merged_note_centers.append(
                [note_center_x_local_px, note_center_y_local_px, 1]
            )

        detected_notes: list[Note] = []
        for note_center_x_local_px, note_center_y_local_px, _ in merged_note_centers:
            note_step_float = (
                bottom_staff_line_local_y - note_center_y_local_px
            ) / staff_half_step_px
            note_step = int(round(note_step_float))
            step_residual = abs(note_step_float - note_step)
            step_confidence = self._step_confidence(step_residual)
            duration_class = self._classify_duration(
                cleaned_measure_mask=cleaned_measure_mask,
                center_x=note_center_x_local_px,
                center_y=note_center_y_local_px,
                staff_spacing=staff.spacing,
            )

            detected_notes.append(
                Note(
                    kind="notehead",
                    staff_index=measure.staff_index,
                    measure_index=measure_index,
                    center_x=note_center_x_local_px,
                    center_y=note_center_y_local_px,
                    step=note_step,
                    step_confidence=step_confidence,
                    duration_class=duration_class,
                )
            )

        detected_notes.sort(key=lambda note: note.center_x)
        return detected_notes

    @staticmethod
    def _step_confidence(step_residual: float) -> str:
        if step_residual <= 0.20:
            return "high"
        if step_residual <= 0.40:
            return "medium"
        return "low"

    def _classify_duration(
        self,
        cleaned_measure_mask: MatLike,
        center_x: int,
        center_y: int,
        staff_spacing: float,
    ) -> str | None:
        is_filled = self._is_filled_notehead(
            cleaned_measure_mask=cleaned_measure_mask,
            center_x=center_x,
            center_y=center_y,
            staff_spacing=staff_spacing,
        )
        has_stem = self._has_stem(
            cleaned_measure_mask=cleaned_measure_mask,
            center_x=center_x,
            center_y=center_y,
            staff_spacing=staff_spacing,
        )

        if not is_filled and not has_stem:
            return "whole"
        if not is_filled and has_stem:
            return "half"
        if is_filled and has_stem:
            return "quarter"
        return None

    @staticmethod
    def _is_filled_notehead(
        cleaned_measure_mask: MatLike,
        center_x: int,
        center_y: int,
        staff_spacing: float,
    ) -> bool:
        radius_x = max(2, int(round(staff_spacing * 0.36)))
        radius_y = max(2, int(round(staff_spacing * 0.28)))

        x1 = max(0, center_x - radius_x)
        x2 = min(cleaned_measure_mask.shape[1], center_x + radius_x + 1)
        y1 = max(0, center_y - radius_y)
        y2 = min(cleaned_measure_mask.shape[0], center_y + radius_y + 1)

        roi = cleaned_measure_mask[y1:y2, x1:x2]
        if roi.size == 0:
            return False

        ellipse_mask = np.zeros(roi.shape, dtype=np.uint8)
        local_center = (center_x - x1, center_y - y1)
        cv.ellipse(
            ellipse_mask,
            local_center,
            (max(1, radius_x - 1), max(1, radius_y - 1)),
            0,
            0,
            360,
            255,
            -1,
        )

        ellipse_area = cv.countNonZero(ellipse_mask)
        if ellipse_area <= 0:
            return False

        ink_inside = cv.countNonZero(cv.bitwise_and(roi, roi, mask=ellipse_mask))
        ink_ratio = ink_inside / float(ellipse_area)
        return ink_ratio >= 0.55

    @staticmethod
    def _has_stem(
        cleaned_measure_mask: MatLike,
        center_x: int,
        center_y: int,
        staff_spacing: float,
    ) -> bool:
        x_radius = max(2, int(round(staff_spacing * 0.85)))
        y_radius = max(3, int(round(staff_spacing * 2.6)))

        x1 = max(0, center_x - x_radius)
        x2 = min(cleaned_measure_mask.shape[1], center_x + x_radius + 1)
        y1 = max(0, center_y - y_radius)
        y2 = min(cleaned_measure_mask.shape[0], center_y + y_radius + 1)

        roi = cleaned_measure_mask[y1:y2, x1:x2]
        if roi.size == 0 or roi.shape[1] < 2:
            return False

        min_vertical_run = max(3, int(round(staff_spacing * 1.2)))
        for x in range(roi.shape[1]):
            run_length = 0
            best_run = 0
            for y in range(roi.shape[0]):
                if roi[y, x] > 0:
                    run_length += 1
                    if run_length > best_run:
                        best_run = run_length
                else:
                    run_length = 0

            if best_run >= min_vertical_run:
                return True

        return False

    def draw_overlay(
        self,
        cleaned_measure_mask: MatLike,
        detected_notes: list[Note],
    ) -> MatLike:
        assert len(cleaned_measure_mask.shape) == 2

        overlay = cv.cvtColor(cv.bitwise_not(cleaned_measure_mask), cv.COLOR_GRAY2BGR)

        for note in detected_notes:
            match note.kind:
                case "notehead":
                    pass
                case _:
                    assert False, f"Unknown note kind: {note.kind}"

            note_center = (note.center_x, note.center_y)
            cv.circle(overlay, note_center, 3, (0, 0, 255), 1)
            confidence_label = note.step_confidence if note.step_confidence else "?"
            pitch_label = (
                f"{note.pitch_letter}{note.octave}"
                if note.pitch_letter is not None and note.octave is not None
                else "?"
            )
            duration_label = note.duration_class if note.duration_class else "?"
            label = f"{note.step} {confidence_label} {pitch_label} {duration_label}"
            cv.putText(
                overlay,
                label,
                (note.center_x + 4, note.center_y - 4),
                cv.FONT_HERSHEY_SIMPLEX,
                0.35,
                (0, 255, 0),
                1,
                cv.LINE_AA,
            )

        return overlay


def resolve_note_pitches(notes: list[Note], clef: Clef | None) -> None:
    if clef is None or clef.kind is None:
        return

    if clef.kind == "treble":
        base_letter = "E"
        base_octave = 4
    elif clef.kind == "bass":
        base_letter = "G"
        base_octave = 2
    else:
        return

    key_accidentals = _key_signature_accidentals(clef.key_signature)

    for note in notes:
        letter, octave = _step_to_letter_octave(base_letter, base_octave, note.step)
        accidental = key_accidentals.get(letter, "")
        note.pitch_letter = f"{letter}{accidental}"
        note.octave = octave


def _step_to_letter_octave(
    base_letter: str,
    base_octave: int,
    step: int,
) -> tuple[str, int]:
    base_index = LETTER_TO_INDEX[base_letter]
    absolute_index = base_octave * 7 + base_index + step
    octave = absolute_index // 7
    letter_index = absolute_index % 7
    letter = LETTER_ORDER[letter_index]
    return letter, octave


def _key_signature_accidentals(key_signature: KeySignature) -> dict[str, str]:
    accidentals: dict[str, str] = {}
    fifths = key_signature.fifths if key_signature.fifths is not None else 0

    if fifths > 0:
        for letter in SHARP_ORDER[:fifths]:
            accidentals[letter] = "#"
    elif fifths < 0:
        for letter in FLAT_ORDER[: abs(fifths)]:
            accidentals[letter] = "b"

    return accidentals
