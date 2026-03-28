"""Sharp / flat detection via OpenCV matchTemplate on staff-erased measure images."""

from pathlib import Path

import cv2 as cv

from schema import Accidental
from symbol_templates import ACCIDENTAL_FLAT, ACCIDENTAL_SHARP


class AccidentalDetector:
    def __init__(self):
        self.sharp_template = self._load_template(ACCIDENTAL_SHARP)
        self.flat_template = self._load_template(ACCIDENTAL_FLAT)
        self.match_threshold = 0.5
        self.scale_fracs = (0.35, 0.5, 0.65, 0.8)
        self.min_peak_distance_frac = 0.55
        self.notehead_clearance_frac = 0.22

    def exclusive_x_before_first_note(self, staff, detected_notes):
        """Right boundary for accidental search: left of the first notehead with a gap."""
        if not detected_notes:
            return None
        first_x = min(n.center_x for n in detected_notes)
        margin = max(1, int(round(staff.spacing * self.notehead_clearance_frac)))
        end = first_x - margin
        if end < 4:
            return None
        return end

    def detect(
        self, mask, staff, measure, measure_index, *, first_note_x_exclusive=None
    ):
        """
        Run template matching only on columns [0, first_note_x_exclusive).
        Call after note detection and pass first_note_x_exclusive from exclusive_x_before_first_note.
        """
        roi = cv.bitwise_not(mask)
        if roi.size == 0:
            return []

        if first_note_x_exclusive is None or first_note_x_exclusive < 4:
            return []

        x_end = min(first_note_x_exclusive, roi.shape[1])
        if x_end < 4:
            return []
        roi = roi[:, :x_end]

        merged = self._match_in_roi(roi, staff)
        accidentals = []
        for score, cx, cy, kind in merged:
            accidentals.append(
                Accidental(
                    kind=kind,
                    staff_index=measure.staff_index,
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
        clef_key_crop,
        staff,
        staff_index,
        *,
        clef_horizontal_fraction=0.42,
        clef_strip_fraction=0.55,
    ):
        """
        Sharp/flat in the key-signature strip: left header slice, columns after the
        clef-heavy strip. Coordinates are clef+key crop-local.
        """
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

        merged = self._match_in_roi(key_roi, staff)
        accidentals = []
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

    def _match_in_roi(self, roi, staff):
        """roi is black ink on white; returns peaks in ROI pixel coordinates."""
        if roi.shape[0] < 4 or roi.shape[1] < 4:
            return []

        min_dist = max(4, int(round(staff.spacing * self.min_peak_distance_frac)))
        candidates = []

        for kind, template in [
            ("sharp", self.sharp_template),
            ("flat", self.flat_template),
        ]:
            for frac in self.scale_fracs:
                target_h = max(4, int(round(staff.spacing * frac)))
                scaled = self._resize_to_height(template, target_h)
                scaled = self._fit_to_roi(scaled, roi.shape[0], roi.shape[1])
                th, tw = scaled.shape[:2]
                if th < 3 or tw < 3:
                    continue
                result = cv.matchTemplate(roi, scaled, cv.TM_CCOEFF_NORMED)
                peaks = self._gather_peaks(
                    result, self.match_threshold, min_dist, tw, th
                )
                for score, cx, cy in peaks:
                    candidates.append((score, cx, cy, kind))

        return self._nms(candidates, min_dist)

    def draw_overlay(self, base_bgr, accidentals):
        """Optional debug draw: template matches are noisy until tuned."""
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
    def _to_gray(image):
        if len(image.shape) == 2:
            return image
        return cv.cvtColor(image, cv.COLOR_BGR2GRAY)

    @staticmethod
    def _load_template(path):
        image = cv.imread(str(path), cv.IMREAD_COLOR)
        if image is None:
            raise FileNotFoundError(f"Cannot read accidental template: {path}")
        return AccidentalDetector._to_gray(image)

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

    @staticmethod
    def _gather_peaks(result, threshold, min_dist, tw, th):
        work = result.copy()
        peaks = []
        while True:
            _, max_val, _, max_loc = cv.minMaxLoc(work)
            if max_val < threshold:
                break
            x, y = int(max_loc[0]), int(max_loc[1])
            cx = x + tw // 2
            cy = y + th // 2
            peaks.append((float(max_val), cx, cy))
            cv.circle(work, (cx, cy), min_dist, 0, thickness=-1)
        return peaks

    @staticmethod
    def _nms(candidates, min_dist):
        """Keep strongest peaks that are not spatially overlapping (any kind)."""
        candidates = sorted(candidates, key=lambda t: -t[0])
        kept = []
        for score, cx, cy, kind in candidates:
            if any(
                (cx - ox) ** 2 + (cy - oy) ** 2 < (min_dist * min_dist)
                for _, ox, oy, _ in kept
            ):
                continue
            kept.append((score, cx, cy, kind))
        return kept
