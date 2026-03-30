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

    # Process each image
    for image_path in image_paths:
        try:
            run_pipeline(str(image_path), show_windows=False)
        except Exception as e:
            print(f"ERROR processing {image_path}: {e}")
            import traceback

            traceback.print_exc()
            continue

    print("\n" + "=" * 60)
    print("All images processed successfully!")
    print("=" * 60)


def play_music():
    """Reads a local text file containing ABC notation for a C major scale, parses it using music21, and plays the resulting MIDI output.
    Assumes all abc notation is formatted correctly"""
    # Define the URL and the local path for saving the file
    local_file_path = 'data/c_major_scale.txt'

    # Ensure the 'data' directory exists
    os.makedirs(os.path.dirname(local_file_path), exist_ok=True)
    
    # Read and print the content from the local file
    with open(local_file_path, 'r') as file:
        tunes = file.read()
    
    tune_list = []
    tune_list = tunes.split('\n\n\n')

    for i in range(len(tune_list)):
        # tune_list[i] = preprocess_tunes(tune_list[i])
        score = converter.parse(tune_list[i])
        score.show('midi')


if __name__ == "__main__":
    main()
