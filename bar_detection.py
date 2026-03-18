from dataclasses import dataclass
from cv2.typing import MatLike
from schema import BarLine, Staff
from constants import MASK_ON, MASK_OFF
import cv2 as cv


@dataclass(frozen=True)
class BarOverlayConfig:
    def __init__(self):
        pass


@dataclass(frozen=True)
class BarDetectionConfig:
    def __init__(self):
        pass


class BarDetector:
    """Detects bar lines in sheet music images
    this class takes in a sheet that has been preprocessed to
    staff lines, making it easier to group those elements
    """

    def __init__(
        self,
        binary_img: MatLike,
        original_img: MatLike,
        staffs: list[Staff],
        config: BarDetectionConfig | None = None,
        overlay_config: BarOverlayConfig | None = None,
    ):
        """
        Args:
            binary_img: Input image (Binarized using otsu) of sheet music
            config: Detection parameters (uses defaults if not provided)
            overlay_config: Visualization parameters (uses defaults if not provided)
        """

        self.original = original_img
        self.image = binary_img
        self.staffs = staffs
        self.config = BarDetectionConfig()
        self.overlay_config = BarOverlayConfig()

    def detect(self) -> List[BarLine]:
        self.bars: list[BarLine] = []

        for staff_index, staff in enumerate(self.staffs):
            y0 = staff.top
            y1 = staff.bottom + 1
            region = self.image[y0:y1, :]

            kernel_height = max(2, int(round(4 * staff.spacing)))
            opening_kernel = cv.getStructuringElement(cv.MORPH_RECT, (1, kernel_height))
            closing_kernel = cv.getStructuringElement(cv.MORPH_RECT, (1, 3))

            vertical = cv.morphologyEx(region, cv.MORPH_OPEN, opening_kernel)
            vertical = cv.morphologyEx(vertical, cv.MORPH_CLOSE, closing_kernel)

            contours, _ = cv.findContours(
                vertical,
                cv.RETR_EXTERNAL,
                cv.CHAIN_APPROX_SIMPLE,
            )

            for contour in contours:
                x, y, width, height = cv.boundingRect(contour)
                min_height = int(0.8 * (staff.bottom - staff.top + 1))
                max_width = max(3, int(round(0.35 * staff.spacing)))

                if height < min_height or width > max_width:
                    continue

                found = 0
                for line in staff.lines:
                    local_y = line.y - staff.top
                    if y <= local_y <= y + height:
                        found += 1
                if found < 4:
                    continue

                self.bars.append(
                    BarLine(
                        x=x + width // 2,
                        y_top=y0 + y,
                        y_bottom=y0 + y + height - 1,
                        staff_index=staff_index,
                    )
                )

            # Sort the bars inplace first by staff then by x coordinate
        self.bars.sort(key=lambda b: (b.staff_index, b.x))
        return self.bars

    def draw_overlay(self) -> MatLike:
        """Draw detected staff lines and bounding boxes on the image."""
        overlay = self.original.copy()

        for bar in self.bars:
            cv.line(
                overlay,
                (bar.x, bar.y_top),
                (bar.x, bar.y_bottom),
                (MASK_OFF, MASK_OFF, MASK_ON),
                2,
            )
        return overlay
