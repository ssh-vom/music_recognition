import cv2 as cv
from cv2.typing import MatLike

from schema import Measure, Note, Staff


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
            note_step = int(
                round(
                    (bottom_staff_line_local_y - note_center_y_local_px)
                    / staff_half_step_px
                )
            )

            detected_notes.append(
                Note(
                    kind="notehead",
                    staff_index=measure.staff_index,
                    measure_index=measure_index,
                    center_x=note_center_x_local_px,
                    center_y=note_center_y_local_px,
                    step=note_step,
                )
            )

        detected_notes.sort(key=lambda note: note.center_x)
        return detected_notes

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
            cv.putText(
                overlay,
                str(note.step),
                (note.center_x + 4, note.center_y - 4),
                cv.FONT_HERSHEY_SIMPLEX,
                0.35,
                (0, 255, 0),
                1,
                cv.LINE_AA,
            )

        return overlay
