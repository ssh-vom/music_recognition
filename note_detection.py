import cv2 as cv
import math
import numpy as np
from cv2.typing import MatLike

from schema import Clef, KeySignature, Measure, Note, Staff

LETTER_ORDER = ("C", "D", "E", "F", "G", "A", "B")
LETTER_TO_INDEX = {letter: index for index, letter in enumerate(LETTER_ORDER)}
SHARP_ORDER = ("F", "C", "G", "D", "A", "E", "B")
FLAT_ORDER = ("B", "E", "A", "D", "G", "C", "F")


class NoteDetector:
    # Slight downward bias: require a bit more evidence before rounding up to
    # the next half-step bin. This helps borderline line/space cases where the
    # detected center is a touch too high.
    STEP_ROUND_UP_THRESHOLD = 0.58

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
        secondary_note_mask = cv.morphologyEx(
            cleaned_measure_mask,
            cv.MORPH_CLOSE,
            close_kernel,
        )

        component_count, _, component_stats, component_centroids = (
            cv.connectedComponentsWithStats(notehead_mask, connectivity=8)
        )
        secondary_count, _, secondary_stats, secondary_centroids = (
            cv.connectedComponentsWithStats(secondary_note_mask, connectivity=8)
        )

        min_notehead_area_px = staff.spacing * staff.spacing * 0.08
        max_notehead_area_px = staff.spacing * staff.spacing * 1.8
        min_notehead_size_px = int(round(staff.spacing * 0.35))
        max_notehead_size_px = int(round(staff.spacing * 1.9))
        min_notehead_aspect_ratio = 0.45
        max_notehead_aspect_ratio = 2.2
        tiny_component_area_px = staff.spacing * staff.spacing * 0.22

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
            if component_area_px <= tiny_component_area_px:
                recovered_center = self._recover_center_from_secondary_component(
                    center_x=note_center_x_local_px,
                    center_y=note_center_y_local_px,
                    secondary_count=secondary_count,
                    secondary_stats=secondary_stats,
                    secondary_centroids=secondary_centroids,
                    staff_spacing=staff.spacing,
                    max_x=int(cleaned_measure_mask.shape[1] - 1),
                    max_y=int(cleaned_measure_mask.shape[0] - 1),
                )
                if recovered_center is not None:
                    note_center_x_local_px, note_center_y_local_px = recovered_center
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

        merged_note_centers = self._augment_centers_from_stem_components(
            cleaned_measure_mask=cleaned_measure_mask,
            merged_note_centers=merged_note_centers,
            staff_spacing=staff.spacing,
            measure_width_px=cleaned_measure_mask.shape[1],
            merge_distance_px=merge_distance_px,
        )

        detected_notes: list[Note] = []
        for note_center_x_local_px, note_center_y_local_px, _ in merged_note_centers:
            note_step_float = (
                bottom_staff_line_local_y - note_center_y_local_px
            ) / staff_half_step_px
            note_step = self._quantize_step(note_step_float)
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
        detected_notes = self._collapse_close_duplicate_notes(
            detected_notes=detected_notes,
            staff_spacing=staff.spacing,
        )
        return detected_notes

    @classmethod
    def _quantize_step(cls, note_step_float: float) -> int:
        lower = math.floor(note_step_float)
        fraction = note_step_float - lower
        if fraction >= cls.STEP_ROUND_UP_THRESHOLD:
            return lower + 1
        return lower

    @staticmethod
    def _step_confidence(step_residual: float) -> str:
        if step_residual <= 0.20:
            return "high"
        if step_residual <= 0.40:
            return "medium"
        return "low"

    def _augment_centers_from_stem_components(
        self,
        *,
        cleaned_measure_mask: MatLike,
        merged_note_centers: list[list[int]],
        staff_spacing: float,
        measure_width_px: int,
        merge_distance_px: int,
    ) -> list[list[int]]:
        if len(merged_note_centers) > 2:
            return merged_note_centers

        component_count, _, stats, _ = cv.connectedComponentsWithStats(
            cleaned_measure_mask, connectivity=8
        )
        augmented = [center.copy() for center in merged_note_centers]
        x_edge_margin = max(2, int(round(staff_spacing * 0.6)))
        added_candidates = 0

        for component_index in range(1, component_count):
            x = int(stats[component_index, cv.CC_STAT_LEFT])
            y = int(stats[component_index, cv.CC_STAT_TOP])
            width = int(stats[component_index, cv.CC_STAT_WIDTH])
            height = int(stats[component_index, cv.CC_STAT_HEIGHT])
            area = float(stats[component_index, cv.CC_STAT_AREA])

            if height < int(round(staff_spacing * 2.0)):
                continue
            if width > int(round(staff_spacing * 1.5)):
                continue
            if area < staff_spacing * staff_spacing * 0.35:
                continue
            if x + width >= measure_width_px - x_edge_margin:
                continue

            candidate_x = x + width // 2
            candidate_y = y + height - max(1, int(round(staff_spacing * 0.55)))

            overlaps_existing = False
            for center_x, center_y, _ in augmented:
                if (
                    abs(candidate_x - center_x) <= int(round(merge_distance_px * 1.2))
                    and abs(candidate_y - center_y) <= int(round(staff_spacing * 1.2))
                ):
                    overlaps_existing = True
                    break
            if overlaps_existing:
                continue

            augmented.append([candidate_x, candidate_y, 1])
            added_candidates += 1
            if added_candidates >= 1:
                break

        augmented.sort(key=lambda center: center[0])
        return augmented

    def _collapse_close_duplicate_notes(
        self,
        *,
        detected_notes: list[Note],
        staff_spacing: float,
    ) -> list[Note]:
        if len(detected_notes) < 2:
            return detected_notes

        x_tol = max(2, int(round(staff_spacing * 1.45)))
        y_tol = max(2, int(round(staff_spacing * 0.75)))

        collapsed: list[Note] = [detected_notes[0]]
        for note in detected_notes[1:]:
            previous = collapsed[-1]
            if (
                previous.duration_class is None
                and note.duration_class is None
                and abs(note.center_x - previous.center_x) <= x_tol
                and abs(note.center_y - previous.center_y) <= y_tol
                and abs(note.step - previous.step) <= 1
            ):
                merged_center_x = int(round((previous.center_x + note.center_x) / 2.0))
                merged_center_y = int(round((previous.center_y + note.center_y) / 2.0))
                previous.center_x = merged_center_x
                previous.center_y = merged_center_y
                previous.step = int(round((previous.step + note.step) / 2.0))
                previous.step_confidence = (
                    previous.step_confidence
                    if previous.step_confidence == "high"
                    else note.step_confidence
                )
                continue

            collapsed.append(note)

        return collapsed

    def _recover_center_from_secondary_component(
        self,
        *,
        center_x: int,
        center_y: int,
        secondary_count: int,
        secondary_stats: MatLike,
        secondary_centroids: MatLike,
        staff_spacing: float,
        max_x: int,
        max_y: int,
    ) -> tuple[int, int] | None:
        x_tol = max(2, int(round(staff_spacing * 0.95)))
        min_h = max(6, int(round(staff_spacing * 2.0)))
        min_area = staff_spacing * staff_spacing * 0.30
        best_idx = -1
        best_dx = 10**9

        for component_index in range(1, secondary_count):
            width = int(secondary_stats[component_index, cv.CC_STAT_WIDTH])
            height = int(secondary_stats[component_index, cv.CC_STAT_HEIGHT])
            area = float(secondary_stats[component_index, cv.CC_STAT_AREA])
            cx = int(round(float(secondary_centroids[component_index][0])))

            if height < min_h or area < min_area:
                continue
            if width > int(round(staff_spacing * 1.7)):
                continue
            dx = abs(cx - center_x)
            if dx > x_tol:
                continue
            if dx < best_dx:
                best_dx = dx
                best_idx = component_index

        if best_idx < 0:
            return None

        x = int(secondary_stats[best_idx, cv.CC_STAT_LEFT])
        y = int(secondary_stats[best_idx, cv.CC_STAT_TOP])
        width = int(secondary_stats[best_idx, cv.CC_STAT_WIDTH])
        height = int(secondary_stats[best_idx, cv.CC_STAT_HEIGHT])
        refined_x = x + width // 2
        refined_y = y + height - int(round(staff_spacing * 0.90))
        refined_x = max(0, min(max_x, refined_x))
        refined_y = max(0, min(max_y, refined_y))
        return refined_x, refined_y

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
