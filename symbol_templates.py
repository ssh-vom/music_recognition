"""Paths to music-symbol templates for OpenCV matchTemplate. Time signatures use OCR."""

from pathlib import Path

ROOT = Path(__file__).resolve().parent / "templates"
CLEF_DIR = ROOT / "clef"
METER_DIR = ROOT / "meter"  # optional test assets; meter OCR does not require PNGs here
ACCIDENTALS_DIR = ROOT / "accidentals"

CLEF_TREBLE = CLEF_DIR / "treble.png"
CLEF_BASS = CLEF_DIR / "bass.png"
ACCIDENTAL_SHARP = ACCIDENTALS_DIR / "sharp.png"
ACCIDENTAL_FLAT = ACCIDENTALS_DIR / "flat.png"
