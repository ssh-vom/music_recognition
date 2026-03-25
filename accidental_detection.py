"""Sharp / flat detection via OpenCV matchTemplate on staff-erased measure images."""

from dataclasses import dataclass
from pathlib import Path

import cv2 as cv
from cv2.typing import MatLike

from schema import Accidental, AccidentalKind, Measure, Note, Staff
from symbol_templates import ACCIDENTAL_FLAT, ACCIDENTAL_SHARP


@dataclass(frozen=True)
class AccidentalDetectorConfig:
    sharp_template: Path = ACCIDENTAL_SHARP
    flat_template: Path = ACCIDENTAL_FLAT
    match_threshold: float = 0.5
    # Template height as a fraction of staff line spacing (try several).
    scale_fracs: tuple[float, ...] = (0.35, 0.5, 0.65, 0.8)
    # Minimum distance between reported accidentals (pixels).
    min_peak_distance_frac: float = 0.55
    # Horizontal gap between accidental search window and first notehead (staff spacing).
    notehead_clearance_frac: float = 0.22


class AccidentalDetector:
    def __init__(self, config: AccidentalDetectorConfig | None = None):
        self.config = config or AccidentalDetectorConfig()

    @staticmethod
    def exclusive_x_before_first_note(
        staff: Staff,
        detected_notes: list[Note],
        notehead_clearance_frac: float,
    ) -> int | None:
        """
        Right boundary (exclusive) for accidental search: left of the first notehead
        in this measure, with a gap so the notehead is not inside the template ROI.
        """
        if not detected_notes:
            return None
        first_x = min(n.center_x for n in detected_notes)
        margin = max(1, int(round(staff.spacing * notehead_clearance_frac)))
        end = first_x - margin
        if end < 4:
            return None
        return end

    def detect(
        self,
        cleaned_measure_mask: MatLike,
        staff: Staff,
        measure: Measure,
        measure_index: int,
        *,
        first_note_x_exclusive: int | None = None,
    ) -> list[Accidental]:
        """
        Run template matching only on columns ``[0, first_note_x_exclusive)`` (measure-local x),
        i.e. strictly to the left of the first notehead. Call after note detection and pass
        ``first_note_x_exclusive`` from :meth:`exclusive_x_before_first_note`.

        The measure mask uses white ink on black; we invert to black-on-white for the PNG templates.
        """
        cfg = self.config
        assert len(cleaned_measure_mask.shape) == 2

        roi = cv.bitwise_not(cleaned_measure_mask)
        if roi.size == 0:
            return []

        if first_note_x_exclusive is None or first_note_x_exclusive < 4:
            return []

        roi_w_full = roi.shape[1]
        x_end = min(first_note_x_exclusive, roi_w_full)
        if x_end < 4:
            return []
        roi = roi[:, :x_end]

        merged = self._match_in_black_on_white_roi(roi, staff)
        staff_index = measure.staff_index

        accidentals: list[Accidental] = []
        for score, cx, cy, kind in merged:
            accidentals.append(
                Accidental(
                    kind=kind,
                    staff_index=staff_index,
                    measure_index=measure_index,
                    center_x=cx,
                    center_y=cy,
                    confidence=score,
                    region="measure",
                )
            )

        accidentals.sort(key=lambda a: (a.center_x, a.kind))
        return accidentals

    def detect_key_header_glyphs(
        self,
        clef_key_crop: MatLike,
        staff: Staff,
        staff_index: int,
        *,
        clef_horizontal_fraction: float = 0.42,
        clef_strip_fraction: float = 0.55,
    ) -> list[Accidental]:
        """
        Sharp/flat in the key-signature strip: ``left`` header slice, columns after the
        clef-heavy strip (same split as :class:`ClefDetectorConfig`). Coordinates are
        **clef+key crop-local** (same frame as the header image).
        """
        assert len(clef_key_crop.shape) == 2
        roi = cv.bitwise_not(clef_key_crop)
        if roi.size == 0:
            return []

        width = roi.shape[1]
        left = roi[:, : max(1, int(width * clef_horizontal_fraction))]
        lw = left.shape[1]
        strip_end = max(1, int(lw * clef_strip_fraction))
        key_roi = left[:, strip_end:]
        if key_roi.shape[1] < 4:
            return []

        merged = self._match_in_black_on_white_roi(key_roi, staff)
        accidentals: list[Accidental] = []
        for score, cx, cy, kind in merged:
            accidentals.append(
                Accidental(
                    kind=kind,
                    staff_index=staff_index,
                    measure_index=-1,
                    center_x=strip_end + cx,
                    center_y=cy,
                    confidence=score,
                    region="header",
                )
            )
        accidentals.sort(key=lambda a: (a.center_x, a.kind))
        return accidentals

    def _match_in_black_on_white_roi(
        self,
        roi: MatLike,
        staff: Staff,
    ) -> list[tuple[float, int, int, AccidentalKind]]:
        """``roi`` is black ink on white; returns peaks in ROI pixel coordinates."""
        cfg = self.config
        if roi.shape[0] < 4 or roi.shape[1] < 4:
            return []

        sharp_t = self._load_template_gray(cfg.sharp_template)
        flat_t = self._load_template_gray(cfg.flat_template)

        min_dist = max(
            4,
            int(round(staff.spacing * cfg.min_peak_distance_frac)),
        )

        candidates: list[tuple[float, int, int, AccidentalKind]] = []
        sharp_flat: list[tuple[AccidentalKind, MatLike]] = [
            ("sharp", sharp_t),
            ("flat", flat_t),
        ]
        for kind, template in sharp_flat:
            for frac in cfg.scale_fracs:
                target_h = max(4, int(round(staff.spacing * frac)))
                scaled = self._resize_template_to_height(template, target_h)
                scaled = self._fit_template_to_roi(scaled, roi.shape[0], roi.shape[1])
                th, tw = scaled.shape[:2]
                if th < 3 or tw < 3:
                    continue
                result = cv.matchTemplate(roi, scaled, cv.TM_CCOEFF_NORMED)
                peaks = self._gather_peaks(result, cfg.match_threshold, min_dist, tw, th)
                for score, cx, cy in peaks:
                    candidates.append((score, cx, cy, kind))

        return self._nms_by_kind(candidates, min_dist)

    def draw_overlay(
        self,
        base_bgr: MatLike,
        accidentals: list[Accidental],
    ) -> MatLike:
        """Optional debug draw: template matches are noisy until tuned—do not enable on the main grid by default."""
        out = base_bgr.copy()
        for acc in accidentals:
            color = (255, 0, 255) if acc.kind == "sharp" else (255, 128, 0)
            cv.drawMarker(
                out,
                (acc.center_x, acc.center_y),
                color,
                markerType=cv.MARKER_CROSS,
                markerSize=10,
                thickness=1,
                line_type=cv.LINE_AA,
            )
            cv.putText(
                out,
                acc.kind[0].upper(),
                (acc.center_x + 6, acc.center_y + 4),
                cv.FONT_HERSHEY_SIMPLEX,
                0.4,
                color,
                1,
                cv.LINE_AA,
            )
        return out

    @staticmethod
    def _to_gray(image: MatLike) -> MatLike:
        if len(image.shape) == 2:
            return image
        return cv.cvtColor(image, cv.COLOR_BGR2GRAY)

    @staticmethod
    def _load_template_gray(path: Path) -> MatLike:
        image = cv.imread(str(path), cv.IMREAD_COLOR)
        if image is None:
            raise FileNotFoundError(f"Cannot read accidental template: {path}")
        return AccidentalDetector._to_gray(image)

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
    def _fit_template_to_roi(template_gray: MatLike, roi_h: int, roi_w: int) -> MatLike:
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

    @staticmethod
    def _gather_peaks(
        result: MatLike,
        threshold: float,
        min_dist: int,
        tw: int,
        th: int,
    ) -> list[tuple[float, int, int]]:
        work = result.copy()
        peaks: list[tuple[float, int, int]] = []
        while True:
            _min_val, max_val, _min_loc, max_loc = cv.minMaxLoc(work)
            if max_val < threshold:
                break
            x, y = int(max_loc[0]), int(max_loc[1])
            cx = x + tw // 2
            cy = y + th // 2
            peaks.append((float(max_val), cx, cy))
            cv.circle(work, (cx, cy), min_dist, 0, thickness=-1)
        return peaks

    @staticmethod
    def _nms_by_kind(
        candidates: list[tuple[float, int, int, AccidentalKind]],
        min_dist: int,
    ) -> list[tuple[float, int, int, AccidentalKind]]:
        """Keep strongest peaks that are not spatially overlapping (any kind)."""
        candidates = sorted(candidates, key=lambda t: -t[0])
        kept: list[tuple[float, int, int, AccidentalKind]] = []
        for score, cx, cy, kind in candidates:
            if any(
                (cx - ox) ** 2 + (cy - oy) ** 2 < (min_dist * min_dist)
                for _, ox, oy, _ in kept
            ):
                continue
            kept.append((score, cx, cy, kind))
        return kept
