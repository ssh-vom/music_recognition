from pathlib import Path
from pipeline import run_pipeline


def main():
    """Process all sheet music images in the music_sheets/ directory."""
    music_dir = Path("./music_sheets")
    image_paths = sorted(music_dir.glob("*.png"))
    for image_path in image_paths:
        run_pipeline(str(image_path), show_windows=False)


if __name__ == "__main__":
    main()
