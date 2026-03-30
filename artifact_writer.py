from datetime import datetime
from pathlib import Path

import cv2 as cv


class _Sections:
    staff = "01_staff"
    bars = "02_bars"
    clef = "03_clef"
    notes = "04_notes"
    masks = "05_masks"
    export = "06_export"
    logs = "07_logs"
    pipeline = "08_pipeline"
    combined = "09_combined"


class ArtifactWriter:
    def __init__(self, image_path: str, root_dir: str = "artifacts"):
        self.sections = _Sections()
        image_stem = Path(image_path).stem
        self.root = Path(root_dir) / f"{image_stem}"

        for name in [
            v
            for k, v in vars(_Sections).items()
            if not k.startswith("_") and isinstance(v, str)
        ]:
            (self.root / name).mkdir(parents=True, exist_ok=True)

    def section_dir(self, section: str) -> Path:
        return self.root / section

    def path(self, section: str, filename: str) -> Path:
        return self.root / section / filename

    def write_image(self, section: str, filename: str, image) -> Path:
        p = self.path(section, filename)
        cv.imwrite(str(p), image)
        return p

    def write_text(self, section: str, filename: str, content: str) -> Path:
        p = self.path(section, filename)
        p.write_text(content, encoding="utf-8")
        return p

    def ensure_subdir(self, section: str, subdir: str) -> Path:
        p = self.section_dir(section) / subdir
        p.mkdir(parents=True, exist_ok=True)
        return p
