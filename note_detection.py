import cv2 as cv
import math
import numpy as np

from schema import Clef, KeySignature, Measure, Note, Staff

LETTER_ORDER = ("C", "D", "E", "F", "G", "A", "B")
LETTER_TO_INDEX = {letter: index for index, letter in enumerate(LETTER_ORDER)}
SHARP_ORDER = ("F", "C", "G", "D", "A", "E", "B")
FLAT_ORDER = ("B", "E", "A", "D", "G", "C", "F")


class NoteDetector:
    # Slight downward bias: require a bit more evidence before rounding up to
    # the next half-step bin.
    STEP_ROUND_UP_THRESHOLD = 0.58

    def detect(self, mask, staff, measure, measure_index) -> list[Note]:
        kernel_diameter = max(1, int(round(staff.spacing * 0.45)))
        if kernel_diameter % 2 == 0:
            kernel_diameter += 1

        notehead_kernel = cv.getStructuringElement(
            cv.MORPH_ELLIPSE, (kernel_diameter, kernel_diameter)
        )
        opened_mask = cv.morphologyEx(mask, cv.MORPH_OPEN, notehead_kernel)

        close_kernel = cv.getStructuringElement(cv.MORPH_ELLIPSE, (3, 3))
        notehead_mask = cv.morphologyEx(opened_mask, cv.MORPH_CLOSE, close_kernel)
        secondary_mask = cv.morphologyEx(mask, cv.MORPH_CLOSE, close_kernel)

        count, _, stats, centroids = cv.connectedComponentsWithStats(
            notehead_mask, connectivity=8
        )
        secondary_count, _, secondary_stats, secondary_centroids = (
            cv.connectedComponentsWithStats(secondary_mask, connectivity=8)
        )

        min_area = staff.spacing * staff.spacing * 0.08
        max_area = staff.spacing * staff.spacing * 1.8
        min_size = int(round(staff.spacing * 0.35))
        max_size = int(round(staff.spacing * 1.9))
        min_aspect = 0.45
        max_aspect = 2.2
        tiny_area = staff.spacing * staff.spacing * 0.22

        bottom_line_y = int(round(staff.lines[4].y - measure.y_top))
        half_step = staff.spacing / 2.0
        merge_dist = max(2, int(round(staff.spacing * 0.75)))

        raw_centers = []

        for i in range(1, count):
            w = int(stats[i, cv.CC_STAT_WIDTH])
            h = int(stats[i, cv.CC_STAT_HEIGHT])
            area = float(stats[i, cv.CC_STAT_AREA])

            if area < min_area or area > max_area:
                continue
            if w < min_size or h < min_size or w > max_size or h > max_size:
                continue

            aspect = w / float(h)
            if aspect < min_aspect or aspect > max_aspect:
                continue

            cx = int(round(centroids[i][0]))
            cy = int(round(centroids[i][1]))

            if area <= tiny_area:
                refined = self._recover_from_secondary(
                    cx,
                    cy,
                    secondary_count,
                    secondary_stats,
                    secondary_centroids,
                    staff.spacing,
                    mask.shape[1] - 1,
                    mask.shape[0] - 1,
                )
                if refined:
                    cx, cy = refined

            raw_centers.append((cx, cy))

        raw_centers.sort(key=lambda c: c[0])

        merged = []
        for cx, cy in raw_centers:
            if not merged:
                merged.append([cx, cy, 1])
                continue

            last_x, last_y, last_count = merged[-1]
            if abs(cx - last_x) <= merge_dist and abs(cy - last_y) <= merge_dist:
                new_count = last_count + 1
                new_x = int(round((last_x * last_count + cx) / new_count))
                new_y = int(round((last_y * last_count + cy) / new_count))
                merged[-1] = [new_x, new_y, new_count]
                continue

            merged.append([cx, cy, 1])

        merged = self._augment_from_stems(
            mask, merged, staff.spacing, mask.shape[1], merge_dist
        )

        notes = []
        for cx, cy, _ in merged:
            step_float = (bottom_line_y - cy) / half_step
            step = self._quantize(step_float)
            residual = abs(step_float - step)
            confidence = self._confidence(residual)
            duration = self._classify_duration(mask, cx, cy, staff.spacing)

            notes.append(
                Note(
                    kind="notehead",
                    staff_index=measure.staff_index,
                    measure_index=measure_index,
                    center_x=cx,
                    center_y=cy,
                    step=step,
                    step_confidence=confidence,
                    duration_class=duration,
                )
            )

        notes.sort(key=lambda n: n.center_x)
        return self._collapse_duplicates(notes, staff.spacing)

    @classmethod
    def _quantize(cls, step_float: float) -> int:
        lower = math.floor(step_float)
        if step_float - lower >= cls.STEP_ROUND_UP_THRESHOLD:
            return lower + 1
        return lower

    @staticmethod
    def _confidence(residual: float):
        if residual <= 0.20:
            return "high"
        if residual <= 0.40:
            return "medium"
        return "low"

    def _augment_from_stems(self, mask, centers, spacing, width, merge_dist) -> list:
        if len(centers) > 2:
            return centers

        count, _, stats, _ = cv.connectedComponentsWithStats(mask, connectivity=8)
        augmented = [c.copy() for c in centers]
        margin = max(2, int(round(spacing * 0.6)))
        added = 0

        for i in range(1, count):
            x = int(stats[i, cv.CC_STAT_LEFT])
            y = int(stats[i, cv.CC_STAT_TOP])
            w = int(stats[i, cv.CC_STAT_WIDTH])
            h = int(stats[i, cv.CC_STAT_HEIGHT])
            area = float(stats[i, cv.CC_STAT_AREA])

            if h < int(round(spacing * 2.0)):
                continue
            if w > int(round(spacing * 1.5)):
                continue
            if area < spacing * spacing * 0.35:
                continue
            if x + w >= width - margin:
                continue

            cx = x + w // 2
            cy = y + h - max(1, int(round(spacing * 0.55)))

            overlaps = False
            for ex, ey, _ in augmented:
                if abs(cx - ex) <= int(round(merge_dist * 1.2)) and abs(cy - ey) <= int(
                    round(spacing * 1.2)
                ):
                    overlaps = True
                    break

            if overlaps:
                continue

            augmented.append([cx, cy, 1])
            added += 1
            if added >= 1:
                break

        augmented.sort(key=lambda c: c[0])
        return augmented

    def _collapse_duplicates(self, notes, spacing) -> list[Note]:
        if len(notes) < 2:
            return notes

        x_tol = max(2, int(round(spacing * 1.45)))
        y_tol = max(2, int(round(spacing * 0.75)))

        collapsed = [notes[0]]
        for note in notes[1:]:
            prev = collapsed[-1]
            if (
                prev.duration_class is None
                and note.duration_class is None
                and abs(note.center_x - prev.center_x) <= x_tol
                and abs(note.center_y - prev.center_y) <= y_tol
                and abs(note.step - prev.step) <= 1
            ):
                prev.center_x = int(round((prev.center_x + note.center_x) / 2.0))
                prev.center_y = int(round((prev.center_y + note.center_y) / 2.0))
                prev.step = int(round((prev.step + note.step) / 2.0))
                prev.step_confidence = (
                    prev.step_confidence
                    if prev.step_confidence == "high"
                    else note.step_confidence
                )
                continue

            collapsed.append(note)

        return collapsed

    def _recover_from_secondary(
        self, cx, cy, count, stats, centroids, spacing, max_x, max_y
    ):
        tol = max(2, int(round(spacing * 0.95)))
        min_h = max(6, int(round(spacing * 2.0)))
        min_area = spacing * spacing * 0.30
        best_idx = -1
        best_dx = 10**9

        for i in range(1, count):
            w = int(stats[i, cv.CC_STAT_WIDTH])
            h = int(stats[i, cv.CC_STAT_HEIGHT])
            area = float(stats[i, cv.CC_STAT_AREA])
            center_x = int(round(float(centroids[i][0])))

            if h < min_h or area < min_area or w > int(round(spacing * 1.7)):
                continue

            dx = abs(center_x - cx)
            if dx > tol or dx >= best_dx:
                continue

            best_dx = dx
            best_idx = i

        if best_idx < 0:
            return None

        x = int(stats[best_idx, cv.CC_STAT_LEFT])
        y = int(stats[best_idx, cv.CC_STAT_TOP])
        w = int(stats[best_idx, cv.CC_STAT_WIDTH])
        h = int(stats[best_idx, cv.CC_STAT_HEIGHT])

        rx = x + w // 2
        ry = y + h - int(round(spacing * 0.90))
        rx = max(0, min(max_x, rx))
        ry = max(0, min(max_y, ry))
        return rx, ry

    def _classify_duration(self, mask, cx, cy, spacing):
        filled = self._is_filled(mask, cx, cy, spacing)
        has_stem = self._has_stem(mask, cx, cy, spacing)

        if not filled and not has_stem:
            return "whole"
        if not filled and has_stem:
            return "half"
        if filled and has_stem:
            return "quarter"
        return None

    @staticmethod
    def _is_filled(mask, cx, cy, spacing) -> bool:
        rx = max(2, int(round(spacing * 0.36)))
        ry = max(2, int(round(spacing * 0.28)))

        x1 = max(0, cx - rx)
        x2 = min(mask.shape[1], cx + rx + 1)
        y1 = max(0, cy - ry)
        y2 = min(mask.shape[0], cy + ry + 1)

        roi = mask[y1:y2, x1:x2]
        if roi.size == 0:
            return False

        ellipse_mask = np.zeros(roi.shape, dtype=np.uint8)
        local_center = (cx - x1, cy - y1)
        cv.ellipse(
            ellipse_mask,
            local_center,
            (max(1, rx - 1), max(1, ry - 1)),
            0,
            0,
            360,
            255,
            -1,
        )

        ellipse_area = cv.countNonZero(ellipse_mask)
        if ellipse_area <= 0:
            return False

        ink = cv.countNonZero(cv.bitwise_and(roi, roi, mask=ellipse_mask))
        return ink / float(ellipse_area) >= 0.55

    @staticmethod
    def _has_stem(mask, cx, cy, spacing) -> bool:
        x_radius = max(2, int(round(spacing * 0.85)))
        y_radius = max(3, int(round(spacing * 2.6)))

        x1 = max(0, cx - x_radius)
        x2 = min(mask.shape[1], cx + x_radius + 1)
        y1 = max(0, cy - y_radius)
        y2 = min(mask.shape[0], cy + y_radius + 1)

        roi = mask[y1:y2, x1:x2]
        if roi.size == 0 or roi.shape[1] < 2:
            return False

        min_run = max(3, int(round(spacing * 1.2)))
        for x in range(roi.shape[1]):
            run = 0
            best = 0
            for y in range(roi.shape[0]):
                if roi[y, x] > 0:
                    run += 1
                    if run > best:
                        best = run
                else:
                    run = 0
            if best >= min_run:
                return True

        return False

    def draw_overlay(self, mask, notes):
        overlay = cv.cvtColor(cv.bitwise_not(mask), cv.COLOR_GRAY2BGR)

        for note in notes:
            assert note.kind == "notehead"
            center = (note.center_x, note.center_y)
            cv.circle(overlay, center, 3, (0, 0, 255), 1)

            conf = note.step_confidence if note.step_confidence else "?"
            pitch = (
                f"{note.pitch_letter}{note.octave}"
                if note.pitch_letter and note.octave
                else "?"
            )
            dur = note.duration_class if note.duration_class else "?"
            label = f"{note.step} {conf} {pitch} {dur}"

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
        base_letter, base_octave = "E", 4
    elif clef.kind == "bass":
        base_letter, base_octave = "G", 2
    else:
        return

    key_accidentals = _key_signature_accidentals(clef.key_signature)

    for note in notes:
        letter, octave = _step_to_letter_octave(base_letter, base_octave, note.step)
        accidental = key_accidentals.get(letter, "")
        note.pitch_letter = f"{letter}{accidental}"
        note.octave = octave


def _step_to_letter_octave(
    base_letter: str, base_octave: int, step: int
) -> tuple[str, int]:
    base_index = LETTER_TO_INDEX[base_letter]
    absolute = base_octave * 7 + base_index + step
    octave = absolute // 7
    letter_index = absolute % 7
    return LETTER_ORDER[letter_index], octave


def _key_signature_accidentals(key_sig: KeySignature) -> dict[str, str]:
    accidentals = {}
    fifths = key_sig.fifths if key_sig.fifths is not None else 0

    if fifths > 0:
        for letter in SHARP_ORDER[:fifths]:
            accidentals[letter] = "#"
    elif fifths < 0:
        for letter in FLAT_ORDER[: abs(fifths)]:
            accidentals[letter] = "b"

    return accidentals
