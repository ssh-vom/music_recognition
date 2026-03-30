"""Main entry point - scans music_sheets/ directory and processes all sheet music."""

from pathlib import Path

from pipeline import run_pipeline


def main():
    """Process all sheet music images in the music_sheets/ directory."""
    music_dir = Path("./music_sheets")

    if not music_dir.exists():
        print(f"Error: Directory {music_dir} not found")
        return

    # Find all PNG images in the music_sheets directory
    image_paths = sorted(music_dir.glob("*.png"))

    if not image_paths:
        print(f"No PNG images found in {music_dir}")
        return

    print(f"Found {len(image_paths)} sheet music image(s) to process")
    print("=" * 60)

    processed_count = 0
    failed_count = 0

    # Process each image
    for image_path in image_paths:
        try:
            run_pipeline(str(image_path), show_windows=False)
            processed_count += 1
        except Exception as e:
            failed_count += 1
            print(f"Error processing {image_path}: {e}")
            import traceback

            traceback.print_exc()
            continue

    print("\n" + "=" * 60)
    if failed_count == 0:
        print(f"Processed {processed_count}/{len(image_paths)} images successfully.")
    else:
        print(
            f"Processed {processed_count}/{len(image_paths)} images; "
            f"{failed_count} failed."
        )
    print("=" * 60)


if __name__ == "__main__":
    main()
