from dataclasses import dataclass

import cv2 as cv
from cv2.typing import MatLike

from constants import MASK_OFF, MASK_ON
from schema import BarLine, Staff

# Configuration and detection logic for vertical barline detection


@dataclass(frozen=True)
class BarOverlayConfig:
    line_color: tuple[int, int, int] = (MASK_OFF, MASK_OFF, MASK_ON)
    line_thickness: int = 1


@dataclass(frozen=True)
class BarDetectionConfig:
    left_skip_spacings: float = (
        5.0  # staff spacings to skip from the left to ignore clef/key signature
    )
    # Multiplier for staff spacing to determine closing kernel height for joining broken bars
    vertical_close_height_ratio: float = 2.0
    # Minimum bar height as ratio of staff height (filters out short stems)
    min_height_ratio: float = 0.4
    # Minimum contour density (area/width*height) to accept as a bar (filters out noise)
    min_density: float = 0.55
    # Maximum bar width as ratio of staff spacing (thicker bars need higher density)
    max_width_ratio: float = 0.6
    # Minimum density for bars wider than max_width_ratio (allows thick final bars)
    thick_bar_min_density: float = 0.75
    # Maximum distance between bar centers to merge into single bar (in spacings)
    merge_distance_ratio: float = 0.5


class BarDetector:
    """
    Detects measure bar lines using staff geometry and vertical blob detection.

    Process:
    1. For each staff, crop the image to staff vertical bounds
    2. Skip left area to ignore clef/key signature
    3. Apply vertical closing to join broken bar fragments
    4. Find contours of vertical components
    5. Filter contours by height, width, and density to isolate barlines
    6. Extract x positions and merge nearby detections
    7. Return sorted barline positions
    """

    def __init__(
        self,
        binary_img: MatLike,
        original_img: MatLike,
        staffs: list[Staff],
    ):
        """Initialize bar detector with processed images and detected staffs.

        Args:
            binary_img: Binarized image (staff lines removed preferred)
            original_img: Original grayscale/color image for drawing overlays
            staffs: List of detected staff objects with geometry information
        """
        self.original = original_img
        self.image = binary_img
        self.staffs = staffs
        self.config = BarDetectionConfig()
        self.overlay_config = BarOverlayConfig()
        self.bars: list[BarLine] = []

    def detect(self) -> list[BarLine]:
        """Detect barline positions for each staff using vertical blob analysis.

        Returns:
            List of BarLine objects sorted by staff index then x position.
        """
        bars: list[BarLine] = []

        for staff_index, staff in enumerate(self.staffs):
            # Get staff vertical bounds and height
            y0 = staff.top
            y1 = staff.bottom + 1
            staff_height = y1 - y0
            roi = self.image[y0:y1, :]

            # Skip clef/key signature area on the left
            left_skip = min(
                roi.shape[1], int(round(self.config.left_skip_spacings * staff.spacing))
            )
            work = roi[:, left_skip:]
            if work.size == 0:
                continue

            # Create vertical closing kernel to join broken bar fragments
            # Kernel height is based on staff spacing to adapt to different scales
            close_kernel = cv.getStructuringElement(
                cv.MORPH_RECT,
                (
                    1,  # width: keep vertical orientation
                    max(
                        5,  # minimum kernel size
                        int(
                            round(
                                self.config.vertical_close_height_ratio * staff.spacing
                            )
                        ),
                    ),
                ),
            )
            # Apply closing to connect vertically aligned components
            joined = cv.morphologyEx(work, cv.MORPH_CLOSE, close_kernel)

            # Find all connected components in the processed image
            contours, _ = cv.findContours(
                joined, cv.RETR_EXTERNAL, cv.CHAIN_APPROX_SIMPLE
            )

            # Collect x positions of valid barline candidates
            candidates: list[int] = []
            for contour in contours:
                # Get bounding box of component
                x, _, width, height = cv.boundingRect(contour)

                # Filter by minimum height (remove short stems and noise)
                if height < int(round(self.config.min_height_ratio * staff_height)):
                    continue

                # Calculate density (area/width*height) to distinguish solid bars from noise
                density = cv.contourArea(contour) / float(width * height)
                if density < self.config.min_density:
                    continue

                # For wider components, require higher density to avoid false positives
                # from wide noise blobs
                max_width = max(
                    3,  # minimum width in pixels
                    int(round(self.config.max_width_ratio * staff.spacing)),
                )
                if width > max_width and density < self.config.thick_bar_min_density:
                    continue

                # Convert to absolute x coordinate and store
                candidates.append(left_skip + x + width // 2)

            # Skip staff if no valid barlines found
            if not candidates:
                continue

            # Sort and merge nearby detections to handle slight variations
            candidates.sort()
            merge_distance = max(
                3,  # minimum merge distance in pixels
                int(round(self.config.merge_distance_ratio * staff.spacing)),
            )

            merged_candidates: list[int] = [candidates[0]]
            for x in candidates[1:]:
                # If current detection is far from last merged one, start new group
                if x - merged_candidates[-1] > merge_distance:
                    merged_candidates.append(x)
                else:
                    # Otherwise, update position to average of merged detections
                    merged_candidates[-1] = (merged_candidates[-1] + x) // 2

            # Create BarLine objects for each merged detection
            for x in merged_candidates:
                bars.append(
                    BarLine(
                        x=x,
                        y_top=y0,
                        y_bottom=y1 - 1,
                        staff_index=staff_index,
                    )
                )

        # Sort all detected barlines by staff then x position
        bars.sort(key=lambda bar: (bar.staff_index, bar.x))
        self.bars = bars
        return bars

    def draw_overlay(self) -> MatLike:
        """Draw detected barlines on the original image for visualization.

        Returns:
            Copy of original image with barlines drawn as red vertical lines.
        """
        overlay = self.original.copy()

        for bar in self.bars:
            # Draw a vertical line at the detected barline position
            cv.line(
                overlay,
                (bar.x, bar.y_top),  # Start point (top of staff)
                (bar.x, bar.y_bottom),  # End point (bottom of staff)
                self.overlay_config.line_color,  # Red color for visibility
                self.overlay_config.line_thickness,
            )

        return overlay
