from dataclasses import dataclass
import cv2 as cv
from cv2.typing import MatLike


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


class StaffDetector:
    def __init__(self, sheet_img: MatLike):
        self.I = sheet_img
        pass

    def to_gray(self) -> MatLike:
        if len(self.I) == 2:
            return self.I.copy()
        return cv.cvtColor(self.I, cv.COLOR_BGR2GRAY)

    def binarize(self, gray_image: MatLike) -> MatLike:

        blurred = cv.GaussianBlur(gray_image, (5, 5), 0)
        _, binary = cv.threshold(
            blurred,
            0,
            255,
            cv.THRES_BINARY_INV + cv.THRESH_OTSU,
        )
        return binary

    def extract_horizonal_lines(self, binary_image: MatLike) -> MatLike:
        kernel_width = max(25, binary_image.shape[1] // 12)
        kernel = cv.getStructuringElement(cv.MORPH_RECT, (kernel_width, 1))
        return cv.morphologyEx(binary_image, cv.MORPH_OPEN, kernel)
