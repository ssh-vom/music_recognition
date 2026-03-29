"""Treble vs bass clef via OpenCV matchTemplate — same idea as the official tutorial:

  https://docs.opencv.org/4.x/d4/dc6/tutorial_py_template_matching.html

``cv.matchTemplate(image, templ, cv.TM_CCOEFF_NORMED)`` then ``cv.minMaxLoc`` on the
result map. Templates are trimmed of white margins before matching so the letterbox
fit targets the glyph, not empty PNG padding. Boxes drawn in the UI use the **letterbox**
rectangle (whole glyph at one scale), not the sliding-window peak (which often latches
onto a small curl of the clef).
"""

from pathlib import Path

import cv2 as cv
import numpy as np

from schema import ClefDetection
from symbol_templates import CLEF_BASS, CLEF_TREBLE


class ClefDetector:
    def __init__(self):
        self.treble_template = self._load_and_trim(CLEF_TREBLE)
        self.bass_template = self._load_and_trim(CLEF_BASS)
        self.clef_horizontal_fraction = 0.42
        self.min_confidence = 0.15
        self.input_is_staff_erased_binary = True
        self.letterbox_tie_margin = 0.06

    def detect(self, clef_key_crop):
        left = self._prepare_left_roi(clef_key_crop)
        if left is None:
            return ClefDetection(
                clef=None,
                confidence=0.0,
                letter_score_treble=0.0,
                letter_score_bass=0.0,
                slide_score_treble=0.0,
                slide_score_bass=0.0,
                treble_match_top_left=(0, 0),
                treble_match_size=(0, 0),
                bass_match_top_left=(0, 0),
                bass_match_size=(0, 0),
            )

        treble_score, treble_rect = self._letterbox_match(left, self.treble_template)
        bass_score, bass_rect = self._letterbox_match(left, self.bass_template)

        slide_treble, _ = self._multi_scale_match(left, self.treble_template)
        slide_bass, _ = self._multi_scale_match(left, self.bass_template)

        kind, confidence = self._decide(treble_score, bass_score)

        tx, ty, tw, th = treble_rect
        bx, by, bw, bh = bass_rect
        return ClefDetection(
            clef=kind,
            confidence=confidence,
            letter_score_treble=treble_score,
            letter_score_bass=bass_score,
            slide_score_treble=slide_treble,
            slide_score_bass=slide_bass,
            treble_match_top_left=(tx, ty),
            treble_match_size=(tw, th),
            bass_match_top_left=(bx, by),
            bass_match_size=(bw, bh),
        )

    def _prepare_left_roi(self, clef_key_crop):
        gray = self._to_gray(clef_key_crop)
        if self.input_is_staff_erased_binary:
            gray = cv.bitwise_not(gray)

        width = gray.shape[1]
        left = gray[:, : max(1, int(width * self.clef_horizontal_fraction))]
        if left.shape[0] < 8 or left.shape[1] < 8:
            return None
        return left

    def _decide(self, treble_score, bass_score):
        if treble_score + self.letterbox_tie_margin >= bass_score:
            winner = "treble"
            confidence = treble_score
        else:
            winner = "bass"
            confidence = bass_score

        if confidence < self.min_confidence:
            return None, confidence
        return winner, confidence

    @staticmethod
    def _to_gray(image):
        if len(image.shape) == 2:
            return image
        return cv.cvtColor(image, cv.COLOR_BGR2GRAY)

    @staticmethod
    def _load_and_trim(path):
        img = cv.imread(str(path), cv.IMREAD_COLOR)
        if img is None:
            raise FileNotFoundError(f"Cannot read clef template: {path}")
        gray = ClefDetector._to_gray(img)
        return ClefDetector._trim_white(gray)

    @staticmethod
    def _trim_white(gray, white_thresh=248):
        _, inv = cv.threshold(gray, white_thresh, 255, cv.THRESH_BINARY_INV)
        pts = cv.findNonZero(inv)
        if pts is None:
            return gray
        x, y, w, h = cv.boundingRect(pts)
        if w < 2 or h < 2:
            return gray
        return gray[y : y + h, x : x + w]

    @staticmethod
    def _letterbox_match(roi_gray, template_gray):
        roi_h, roi_w = roi_gray.shape[:2]
        th, tw = template_gray.shape[:2]
        if th < 1 or tw < 1:
            return 0.0, (0, 0, 0, 0)
        scale = min((roi_w - 1) / tw, (roi_h - 1) / th) * 0.99
        scale = max(scale, 1e-6)
        new_w = max(1, int(round(tw * scale)))
        new_h = max(1, int(round(th * scale)))
        resized = cv.resize(
            template_gray,
            (new_w, new_h),
            interpolation=cv.INTER_AREA if scale < 1.0 else cv.INTER_CUBIC,
        )
        canvas = np.full((roi_h, roi_w), 255, dtype=np.uint8)
        y0 = max(0, (roi_h - new_h) // 2)
        x0 = 0
        y1 = min(roi_h, y0 + new_h)
        x1 = min(roi_w, x0 + new_w)
        rw = x1 - x0
        rh = y1 - y0
        canvas[y0:y1, x0:x1] = resized[:rh, :rw]
        result = cv.matchTemplate(roi_gray, canvas, cv.TM_CCOEFF_NORMED)
        score = float(result[0, 0])
        rect = (x0, y0, rw, rh)
        return score, rect

    @staticmethod
    def _multi_scale_match(roi_gray, template_gray):
        roi_h, roi_w = roi_gray.shape[:2]
        best_score = 0.0
        best_rect = (0, 0, 0, 0)

        for frac in (0.78, 0.88, 0.95, 1.0):
            target_h = max(12, min(roi_h - 1, int(round(roi_h * frac))))
            scaled = ClefDetector._resize_to_height(template_gray, target_h)
            scaled = ClefDetector._fit_to_roi(scaled, roi_h, roi_w)
            sh, sw = scaled.shape[:2]
            if sh < 4 or sw < 4 or sh > roi_h or sw > roi_w:
                continue
            result = cv.matchTemplate(roi_gray, scaled, cv.TM_CCOEFF_NORMED)
            _, max_val, _, max_loc = cv.minMaxLoc(result)
            score = float(max_val)
            if score > best_score:
                best_score = score
                x, y = int(max_loc[0]), int(max_loc[1])
                best_rect = (x, y, sw, sh)

        return best_score, best_rect

    @staticmethod
    def _resize_to_height(template_gray, target_h):
        th, tw = template_gray.shape[:2]
        if th < 1 or target_h < 1:
            return template_gray
        scale = target_h / th
        new_w = max(1, int(round(tw * scale)))
        new_h = max(1, int(round(th * scale)))
        interp = cv.INTER_AREA if scale < 1.0 else cv.INTER_CUBIC
        return cv.resize(template_gray, (new_w, new_h), interpolation=interp)

    @staticmethod
    def _fit_to_roi(template_gray, roi_h, roi_w):
        th, tw = template_gray.shape[:2]
        if th <= 0 or tw <= 0:
            return template_gray
        if th <= roi_h and tw <= roi_w:
            return template_gray
        scale = min((roi_h - 1) / th, (roi_w - 1) / tw) * 0.99
        scale = max(scale, 1e-3)
        new_w = max(1, int(round(tw * scale)))
        new_h = max(1, int(round(th * scale)))
        return cv.resize(template_gray, (new_w, new_h), interpolation=cv.INTER_AREA)

    def draw_overlay(self, clef_crop, detection, clef):
        """Draw clef detection overlay on the clef crop image."""
        if len(clef_crop.shape) == 2:
            overlay = cv.cvtColor(cv.bitwise_not(clef_crop), cv.COLOR_GRAY2BGR)
        else:
            overlay = clef_crop.copy()

        if clef is None or detection is None:
            return overlay

        # Draw clef bounding box
        x1, y1 = 0, 0
        x2, y2 = overlay.shape[1], overlay.shape[0]
        cv.rectangle(overlay, (x1, y1), (x2, y2), (100, 100, 100), 1)

        # Draw template matching boxes
        choice = self._choose_overlay_rect(clef, detection)
        if choice is not None:
            rect, color = choice
            self._draw_box(overlay, rect, color, thickness=3)

        # Add label
        name = clef.kind if clef.kind else "?"
        label = f"{name}  T={detection.letter_score_treble:.2f} B={detection.letter_score_bass:.2f}"
        cv.putText(
            overlay,
            label,
            (6, 22),
            cv.FONT_HERSHEY_SIMPLEX,
            0.55,
            (30, 30, 30),
            2,
            cv.LINE_AA,
        )

        return overlay

    @staticmethod
    def _choose_overlay_rect(clef, detection):
        """Choose which template match box to draw."""
        if (
            clef.kind == "treble"
            and detection.treble_match_top_left is not None
            and detection.treble_match_size is not None
        ):
            x, y = detection.treble_match_top_left
            w, h = detection.treble_match_size
            return (x, y, w, h), (0, 200, 100)
        if (
            clef.kind == "bass"
            and detection.bass_match_top_left is not None
            and detection.bass_match_size is not None
        ):
            x, y = detection.bass_match_top_left
            w, h = detection.bass_match_size
            return (x, y, w, h), (0, 120, 255)
        if (
            detection.treble_match_top_left is not None
            and detection.treble_match_size is not None
        ):
            x, y = detection.treble_match_top_left
            w, h = detection.treble_match_size
            return (x, y, w, h), (180, 180, 180)
        return None

    @staticmethod
    def _draw_box(image, rect, color, *, thickness=3):
        """Draw a rectangle on the image."""
        x, y, w, h = rect
        if w < 2 or h < 2:
            return
        cv.rectangle(
            image,
            (x, y),
            (x + w - 1, y + h - 1),
            color,
            thickness,
        )

    def save_intermediates(
        self, clef_key_crops, clefs_by_staff, clef_detections, artifacts
    ) -> dict:
        """Save all intermediate processing steps for visualization."""
        paths = {}

        # Save individual clef header crops
        clef_crops_dir = artifacts.ensure_subdir(
            artifacts.sections.clef, "01_clef_header_crops"
        )
        for staff_index, crop in clef_key_crops.items():
            crop_path = clef_crops_dir / f"staff_{staff_index}.jpg"
            if len(crop.shape) == 2:
                display_crop = cv.bitwise_not(crop)
            else:
                display_crop = crop
            cv.imwrite(str(crop_path), display_crop)
        paths["01_clef_header_crops"] = clef_crops_dir

        # Save detection results with overlays
        for staff_index, crop in clef_key_crops.items():
            clef = clefs_by_staff.get(staff_index)
            detection = clef_detections.get(staff_index)
            if clef is not None and detection is not None:
                overlay = self.draw_overlay(crop, detection, clef)
                overlay_path = artifacts.write_image(
                    artifacts.sections.clef,
                    f"02_detection_staff_{staff_index}.jpg",
                    overlay,
                )
                paths[f"02_detection_staff_{staff_index}"] = overlay_path

        # Create combined clef overlay on full sheet
        # This will be handled by pipeline.py since it needs the full sheet image

        return paths
