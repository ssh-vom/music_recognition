from datetime import datetime
from pathlib import Path

import cv2 as cv


SECTIONS = {
    "staff": "01_staff",
    "bars": "02_bars",
    "clef": "03_clef",
    "notes": "04_notes",
    "masks": "05_masks",
    "export": "06_export",
    "logs": "07_logs",
    "pipeline": "08_pipeline",
    "combined": "09_combined",
}


class _Sections:
    """Simple namespace to allow dot access to section names."""

    staff: str = "01_staff"
    bars: str = "02_bars"
    clef: str = "03_clef"
    notes: str = "04_notes"
    masks: str = "05_masks"
    export: str = "06_export"
    logs: str = "07_logs"
    pipeline: str = "08_pipeline"
    combined: str = "09_combined"

    def __init__(self):
        # Attributes defined above, init just ensures they exist
        pass


class ArtifactWriter:
    def __init__(self, image_path: str, root_dir: str = "artifacts"):
        self.sections = _Sections()
        image_stem = Path(image_path).stem
        timestamp = datetime.now().strftime("%Y%m%d")
        self.root = Path(root_dir) / f"{image_stem}_{timestamp}"

        for name in SECTIONS.values():
            (self.root / name).mkdir(parents=True, exist_ok=True)

    def section_dir(self, section: str) -> Path:
        return self.root / section

    def _file_path(self, section: str, filename: str) -> Path:
        return self.root / section / filename

    def text_path(self, section: str, filename: str) -> Path:
        return self._file_path(section, filename)

    def write_image(self, section: str, filename: str, image) -> Path:
        path = self._file_path(section, filename)
        cv.imwrite(str(path), image)
        return path

    def write_text(self, section: str, filename: str, content: str) -> Path:
        path = self._file_path(section, filename)
        path.write_text(content, encoding="utf-8")
        return path

    def ensure_subdir(self, section: str, subdir: str) -> Path:
        """Create and return a subdirectory within a section."""
        path = self.section_dir(section) / subdir
        path.mkdir(parents=True, exist_ok=True)
        return path
