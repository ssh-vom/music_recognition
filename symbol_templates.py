"""Template paths for symbol matching."""

from pathlib import Path

ROOT = Path(__file__).resolve().parent / "templates"
CLEF_DIR = ROOT / "clef"
ACCIDENTALS_DIR = ROOT / "accidentals"

CLEF_TREBLE = CLEF_DIR / "treble.png"
CLEF_BASS = CLEF_DIR / "bass.png"
ACCIDENTAL_SHARP = ACCIDENTALS_DIR / "sharp.png"
ACCIDENTAL_FLAT = ACCIDENTALS_DIR / "flat.png"
