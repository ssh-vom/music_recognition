from dataclasses import dataclass
from datetime import datetime
from pathlib import Path

import cv2 as cv
from cv2.typing import MatLike


@dataclass(frozen=True)
class ArtifactSections:
    staff: str = "01_staff"
    bars: str = "02_bars"
    clef: str = "03_clef"
    notes: str = "04_notes"
    masks: str = "05_masks"
    export: str = "06_export"
    logs: str = "07_logs"


class ArtifactWriter:
    def __init__(self, image_path: str, root_dir: str = "artifacts"):
        self.sections = ArtifactSections()
        image_stem = Path(image_path).stem
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        self.root = Path(root_dir) / f"{image_stem}_{timestamp}"

        for section_name in self.section_names():
            self.section_dir(section_name).mkdir(parents=True, exist_ok=True)

    def section_names(self) -> list[str]:
        return [
            self.sections.staff,
            self.sections.bars,
            self.sections.clef,
            self.sections.notes,
            self.sections.masks,
            self.sections.export,
            self.sections.logs,
        ]

    def section_dir(self, section: str) -> Path:
        return self.root / section

    def image_path(self, section: str, filename: str) -> Path:
        return self.section_dir(section) / filename

    def text_path(self, section: str, filename: str) -> Path:
        return self.section_dir(section) / filename

    def write_image(self, section: str, filename: str, image: MatLike) -> Path:
        path = self.image_path(section, filename)
        cv.imwrite(str(path), image)
        return path

    def write_text(self, section: str, filename: str, content: str) -> Path:
        path = self.text_path(section, filename)
        path.write_text(content, encoding="utf-8")
        return path
