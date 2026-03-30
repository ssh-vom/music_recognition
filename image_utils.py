"""Shared image conversion helpers."""

import cv2 as cv
from cv2.typing import MatLike


def to_gray(image: MatLike) -> MatLike:
    if len(image.shape) == 2:
        return image.copy()
    return cv.cvtColor(image, cv.COLOR_BGR2GRAY)
