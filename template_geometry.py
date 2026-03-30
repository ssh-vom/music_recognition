"""Shared OpenCV helpers for template matching (grayscale, resize-to-height, fit-to-ROI)."""

import cv2 as cv
from cv2.typing import MatLike


def to_gray(image: MatLike) -> MatLike:
    if len(image.shape) == 2:
        return image
    return cv.cvtColor(image, cv.COLOR_BGR2GRAY)


def resize_to_height(template: MatLike, target_h: int) -> MatLike:
    th, tw = template.shape[:2]
    if th < 1 or target_h < 1:
        return template
    scale = target_h / th
    new_w = max(1, int(round(tw * scale)))
    new_h = max(1, int(round(th * scale)))
    interp = cv.INTER_AREA if scale < 1.0 else cv.INTER_CUBIC
    return cv.resize(template, (new_w, new_h), interpolation=interp)


def fit_to_roi(template: MatLike, roi_h: int, roi_w: int) -> MatLike:
    th, tw = template.shape[:2]
    if th <= 0 or tw <= 0 or (th <= roi_h and tw <= roi_w):
        return template
    scale = max(min((roi_h - 1) / th, (roi_w - 1) / tw) * 0.99, 1e-3)
    new_w = max(1, int(round(tw * scale)))
    new_h = max(1, int(round(th * scale)))
    return cv.resize(template, (new_w, new_h), interpolation=cv.INTER_AREA)
