"""Treble vs bass clef via OpenCV matchTemplate — same idea as the official tutorial:

  https://docs.opencv.org/4.x/d4/dc6/tutorial_py_template_matching.html

``cv.matchTemplate(image, templ, cv.TM_CCOEFF_NORMED)`` then ``cv.minMaxLoc`` on the
result map. Templates are trimmed of white margins before matching so the letterbox
fit targets the glyph, not empty PNG padding. Boxes drawn in the UI use the **letterbox**
rectangle (whole glyph at one scale), not the sliding-window peak (which often latches
onto a small curl of the clef).
"""

from dataclasses import dataclass
from pathlib import Path

import cv2 as cv
import numpy as np
from cv2.typing import MatLike

from schema import ClefDetection, ClefKind
from symbol_templates import CLEF_BASS, CLEF_TREBLE


@dataclass(frozen=True)
class ClefDetectorConfig:
    treble_template: Path = CLEF_TREBLE
    bass_template: Path = CLEF_BASS
    # Left header slice (clef + key / meter). All matching uses this ROI width so
    # templates keep a realistic aspect ratio (do not match on a very narrow strip).
    clef_horizontal_fraction: float = 0.42
    # Staff-erased NCC is often modest; 0.24 can still be a valid treble vs bass pick.
    min_confidence: float = 0.15
    input_is_staff_erased_binary: bool = True
    # Template heights as fractions of the left-ROI height (tutorial-style multi-scale).
    scale_fracs: tuple[float, ...] = (0.78, 0.88, 0.95, 1.0)
    letterbox_tie_margin: float = 0.06
    # Crop near-white borders off PNG templates so scale-to-fit uses real ink bounds.
    trim_template_whitespace: bool = True


Rect = tuple[int, int, int, int]


@dataclass(frozen=True)
class TemplateScore:
    letter_score: float
    letter_rect: Rect
    slide_score: float


@dataclass(frozen=True)
class ClefDecision:
    kind: ClefKind | None
    confidence: float


class ClefDetector:
    def __init__(self, config: ClefDetectorConfig | None = None):
        self.config = config or ClefDetectorConfig()

    def detect(self, clef_key_crop: MatLike) -> ClefDetection:
        cfg = self.config
        left = self._prepare_left_roi(clef_key_crop, cfg)
        if left is None:
            return self._build_detection(
                decision=ClefDecision(kind=None, confidence=0.0),
                treble_score=TemplateScore(0.0, (0, 0, 0, 0), 0.0),
                bass_score=TemplateScore(0.0, (0, 0, 0, 0), 0.0),
            )

        treble_t = self._prepare_template(
            cfg.treble_template, cfg.trim_template_whitespace
        )
        bass_t = self._prepare_template(cfg.bass_template, cfg.trim_template_whitespace)

        treble_score = self._score_template(left, treble_t, cfg.scale_fracs)
        bass_score = self._score_template(left, bass_t, cfg.scale_fracs)
        decision = self._decide_clef(treble_score, bass_score, cfg)
        return self._build_detection(
            decision=decision,
            treble_score=treble_score,
            bass_score=bass_score,
        )

    @staticmethod
    def _prepare_left_roi(
        clef_key_crop: MatLike,
        cfg: ClefDetectorConfig,
    ) -> MatLike | None:
        gray = ClefDetector._to_gray(clef_key_crop)
        if cfg.input_is_staff_erased_binary:
            gray = ClefDetector._ink_on_white_for_templates(gray)

        width = gray.shape[1]
        left = gray[:, : max(1, int(width * cfg.clef_horizontal_fraction))]
        if left.shape[0] < 8 or left.shape[1] < 8:
            return None
        return left

    @staticmethod
    def _score_template(
        left_roi: MatLike,
        template_gray: MatLike,
        scale_fracs: tuple[float, ...],
    ) -> TemplateScore:
        letter_score, letter_rect = ClefDetector._letterbox_match(left_roi, template_gray)
        slide_score, _slide_rect = ClefDetector._multi_scale_match_template(
            left_roi, template_gray, scale_fracs
        )
        return TemplateScore(
            letter_score=letter_score,
            letter_rect=letter_rect,
            slide_score=slide_score,
        )

    @staticmethod
    def _decide_clef(
        treble_score: TemplateScore,
        bass_score: TemplateScore,
        cfg: ClefDetectorConfig,
    ) -> ClefDecision:
        if treble_score.letter_score + cfg.letterbox_tie_margin >= bass_score.letter_score:
            winner: ClefKind = "treble"
            confidence = treble_score.letter_score
        else:
            winner = "bass"
            confidence = bass_score.letter_score

        if confidence < cfg.min_confidence:
            return ClefDecision(kind=None, confidence=confidence)
        return ClefDecision(kind=winner, confidence=confidence)

    @staticmethod
    def _build_detection(
        decision: ClefDecision,
        treble_score: TemplateScore,
        bass_score: TemplateScore,
    ) -> ClefDetection:
        tx, ty, tw, th = treble_score.letter_rect
        bx, by, bw, bh = bass_score.letter_rect
        return ClefDetection(
            clef=decision.kind,
            confidence=decision.confidence,
            letter_score_treble=treble_score.letter_score,
            letter_score_bass=bass_score.letter_score,
            slide_score_treble=treble_score.slide_score,
            slide_score_bass=bass_score.slide_score,
            treble_match_top_left=(tx, ty),
            treble_match_size=(tw, th),
            bass_match_top_left=(bx, by),
            bass_match_size=(bw, bh),
        )

    @staticmethod
    def _ink_on_white_for_templates(gray: MatLike) -> MatLike:
        return cv.bitwise_not(gray)

    @staticmethod
    def _to_gray(image: MatLike) -> MatLike:
        if len(image.shape) == 2:
            return image
        return cv.cvtColor(image, cv.COLOR_BGR2GRAY)

    @staticmethod
    def _load_template_gray(path: Path) -> MatLike:
        image = cv.imread(str(path), cv.IMREAD_COLOR)
        if image is None:
            raise FileNotFoundError(f"Cannot read clef template: {path}")
        return ClefDetector._to_gray(image)

    @classmethod
    def _prepare_template(cls, path: Path, trim: bool) -> MatLike:
        g = cls._load_template_gray(path)
        if trim:
            g = cls._trim_white_borders(g)
        return g

    @staticmethod
    def _trim_white_borders(gray: MatLike, white_thresh: int = 248) -> MatLike:
        """Tight crop to non-white pixels so letterbox scales the glyph, not the PNG canvas."""
        _, inv = cv.threshold(gray, white_thresh, 255, cv.THRESH_BINARY_INV)
        pts = cv.findNonZero(inv)
        if pts is None:
            return gray
        x, y, w, h = cv.boundingRect(pts)
        if w < 2 or h < 2:
            return gray
        return gray[y : y + h, x : x + w]

    @staticmethod
    def _letterbox_match(
        roi_gray: MatLike,
        template_gray: MatLike,
    ) -> tuple[float, tuple[int, int, int, int]]:
        """
        Scale template uniformly to fit inside roi, left-align, single TM_CCOEFF_NORMED
        at the only valid position (same as a 1×1 result map). Coordinates are relative
        to ``roi_gray`` (and thus to the clef crop if roi is the left slice from x=0).
        """
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
        # Use actual placed size (handles edge clipping vs ROI).
        rect = (x0, y0, rw, rh)
        return score, rect

    @staticmethod
    def _multi_scale_match_template(
        roi_gray: MatLike,
        template_gray: MatLike,
        scale_fracs: tuple[float, ...],
    ) -> tuple[float, tuple[int, int, int, int]]:
        """
        For each scale, resize template to ``frac * roi_h`` (clamped), ensure it fits
        in the ROI, then ``matchTemplate`` + ``minMaxLoc`` (OpenCV tutorial pattern).
        Returns the best score and the winning (x, y, w, h) in ``roi_gray`` coords.
        """
        roi_h, roi_w = roi_gray.shape[:2]
        best_score = 0.0
        best_rect = (0, 0, 0, 0)

        for frac in scale_fracs:
            target_h = max(12, min(roi_h - 1, int(round(roi_h * frac))))
            scaled = ClefDetector._resize_template_to_height(template_gray, target_h)
            scaled = ClefDetector._fit_template_inside_roi(scaled, roi_h, roi_w)
            sh, sw = scaled.shape[:2]
            if sh < 4 or sw < 4 or sh > roi_h or sw > roi_w:
                continue
            result = cv.matchTemplate(roi_gray, scaled, cv.TM_CCOEFF_NORMED)
            _min_val, max_val, _min_loc, max_loc = cv.minMaxLoc(result)
            score = float(max_val)
            if score > best_score:
                best_score = score
                x, y = int(max_loc[0]), int(max_loc[1])
                best_rect = (x, y, sw, sh)

        return best_score, best_rect

    @staticmethod
    def _resize_template_to_height(template_gray: MatLike, target_h: int) -> MatLike:
        th, tw = template_gray.shape[:2]
        if th < 1 or target_h < 1:
            return template_gray
        scale = target_h / th
        new_w = max(1, int(round(tw * scale)))
        new_h = max(1, int(round(th * scale)))
        interp = cv.INTER_AREA if scale < 1.0 else cv.INTER_CUBIC
        return cv.resize(template_gray, (new_w, new_h), interpolation=interp)

    @staticmethod
    def _fit_template_inside_roi(template_gray: MatLike, roi_h: int, roi_w: int) -> MatLike:
        """Uniform shrink until both dimensions fit (wide ROI ⇒ no tiny sliver)."""
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
