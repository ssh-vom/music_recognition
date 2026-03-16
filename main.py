import cv2 as cv
import re
from cv2.typing import MatLike
import pytesseract
from staff_detection import StaffDetector


def main():

    I = cv.imread(filename="./twinkle_twinkle_little_star.png")
    if I is None:
        raise FileNotFoundError(
            "Could not load image: ./twinkle_twinkle_little_star.png"
        )

    J = preprocess(I)
    det = StaffDetector(I)
    staffs, binary, line_mask = det.detect()
    overlay = det.draw_overlay(staffs)
    bpm, raw = extract_bpm(J)
    print("BPM:", bpm, "| OCR:", raw)
    cv.imshow(winname="filtered", mat=J)
    cv.imshow("staff overlay", overlay)
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


if __name__ == "__main__":
    main()
