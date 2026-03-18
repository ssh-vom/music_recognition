import os
import re
import cv2 as cv
from cv2.typing import MatLike
import pytesseract
from music21 import converter
from tunes_processing import preprocess_tunes

def main():

    I = cv.imread(filename="./twinkle_twinkle_little_star.png")
    if I is None:
        raise FileNotFoundError(
            "Could not load image: ./twinkle_twinkle_little_star.png"
        )

    J = preprocess(I)
    bpm, raw = extract_bpm(J)
    print("BPM:", bpm, "| OCR:", raw)
    play_music()
    cv.imshow(winname="filtered", mat=J)

    cv.waitKey(0)
    cv.destroyAllWindows()
    return


def preprocess(I: MatLike):

    # CROPPING STAGE
    height = I.shape[0]
    top_crop = int(0.18 * height)
    bottom_crop = int(0.8 * height)
    J = I[top_crop:bottom_crop, :]

    # filter, J = cv.threshold(src=I, thresh=0.0, maxval=255.0, type=0)

    return J


def extract_bpm(I: MatLike) -> tuple[int | None, str | None]:
    # Grab tempo from the top left
    h, w = I.shape[0], I.shape[1]
    roi = I[0 : int(h * 0.1), 0 : int(w * 0.2)]
    txt = pytesseract.image_to_string(
        roi, config="--oem 3 --psm 7 -c tessedit_char_whitelist=0123456789="
    )
    m = re.search(r"(\d{2,3})", txt)

    if not m:
        return None, txt
    bpm = int(m.group(1))
    if not (20 <= bpm <= 320):
        return None, txt
    return bpm, txt


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
