import cv2 as cv
import numpy as np
import re
from cv2.typing import MatLike
import pytesseract
from staff_detection import StaffDetector
from bar_detection import BarDetector
from measure_splitting import MeasureSplitter

# Optical Music Recognition pipeline for sheet music to ABC conversion


def main():

    I = cv.imread(filename="./twinkle_twinkle_little_star.png")
    if I is None:
        raise FileNotFoundError(
            "Could not load image: ./twinkle_twinkle_little_star.png"
        )

    J = preprocess(I)
    det = StaffDetector(I)
    staffs, _, _ = det.detect()
    overlay = det.draw_overlay(staffs)
    removed = det.remove_staffs(staffs)
    bar_det = BarDetector(removed, I, staffs)
    bars = bar_det.detect()
    overlay_bar = bar_det.draw_overlay()
    print(f"Number of bars detected {len(bars)}")
    for bar in bars:
        print(
            f"staff = {bar.staff_index} x = {bar.x} kind = {bar.kind} repeat = {bar.repeat}"
        )
        print(f"y= {bar.y_top},{bar.y_bottom}")
    bpm, raw = extract_bpm(J)
    ms = MeasureSplitter(bars, staffs, I)
    cropped = ms.crop_measures()
    show_measure_grid(cropped)

    print("BPM:", bpm, "| OCR:", raw)
    cv.imshow(winname="filtered", mat=J)
    cv.imwrite("staff_overlay.jpg", overlay)
    cv.imshow("Removed staff lines", removed)
    cv.imshow("Overlay bar", overlay_bar)
    cv.imwrite("removed.jpg", removed)
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


def show_measure_grid(cropped: dict[int, list[MatLike]]) -> None:
    tiles: list[MatLike] = []

    for staff_index, staff_crops in cropped.items():
        for measure_index, crop in enumerate(staff_crops):
            if len(crop.shape) == 2:
                tile = cv.cvtColor(crop, cv.COLOR_GRAY2BGR)
            else:
                tile = crop.copy()

            tile = cv.bitwise_not(tile)

            cv.putText(
                tile,
                f"s{staff_index} m{measure_index}",
                (6, 18),
                cv.FONT_HERSHEY_SIMPLEX,
                0.5,
                (0, 255, 0),
                1,
                cv.LINE_AA,
            )
            tiles.append(tile)

    if not tiles:
        return

    cols = 4
    rows = (len(tiles) + cols - 1) // cols
    cell_h = max(tile.shape[0] for tile in tiles)
    cell_w = max(tile.shape[1] for tile in tiles)

    blank = np.zeros((cell_h, cell_w, 3), dtype=np.uint8)
    padded_tiles: list[MatLike] = []

    for tile in tiles:
        padded = blank.copy()
        h, w = tile.shape[:2]
        padded[:h, :w] = tile
        padded_tiles.append(padded)

    while len(padded_tiles) < rows * cols:
        padded_tiles.append(blank.copy())

    grid_rows: list[MatLike] = []
    for row_index in range(rows):
        start = row_index * cols
        end = start + cols
        grid_rows.append(cv.hconcat(padded_tiles[start:end]))

    grid = cv.vconcat(grid_rows)
    cv.imshow("measures_grid", grid)


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
