"""Helpers for Template matching"""

import cv2 as cv
from cv2.typing import MatLike


def resize_to_height(template: MatLike, target_h: int) -> MatLike:
    th, tw = template.shape[:2]  # grab the templates height and width
    if th < 1 or target_h < 1:
        return template
    scale = target_h / th  # find scaling factor
    new_w = max(1, int(round(tw * scale)))  # resize width
    new_h = max(1, int(round(th * scale)))  # resize height
    interp = (
        cv.INTER_AREA if scale < 1.0 else cv.INTER_CUBIC
    )  # set interpolation method
    return cv.resize(template, (new_w, new_h), interpolation=interp)


def fit_to_roi(template: MatLike, roi_h: int, roi_w: int) -> MatLike:
    th, tw = template.shape[:2]  # template height and width
    if th <= roi_h and tw <= roi_w:
        return template
    scale = max(min((roi_h - 1) / th, (roi_w - 1) / tw) * 0.99, 1e-3)  # scale factor
    new_w = max(1, int(round(tw * scale)))  # resize width
    new_h = max(1, int(round(th * scale)))  # resize height
    return cv.resize(template, (new_w, new_h), interpolation=cv.INTER_AREA)
