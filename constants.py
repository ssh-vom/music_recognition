"""Sheet music detection constants."""

MASK_BACKGROUND = 0
MASK_FOREGROUND = 255

BLUR_KERNEL_SIZE = (5, 5)  # mild blur before thresholding to suppress scanner noise

# Staff line detection
STAFF_LINE_KERNEL_WIDTH_FRAC = 1 / 12  # horizontal open kernel spans ~1/12 of image width to survive small gaps in printed lines
STAFF_LINE_KERNEL_MIN_WIDTH = 25  # floor so the kernel is still useful on smaller images
LINE_DETECTION_MIN_RATIO = 0.35  # rows below 35% of the peak row's pixel count are treated as noise, not a staff line
LINE_CLUSTER_MAX_GAP = 1  # adjacent rows are only the same staff line if they touch; a gap of 2+ means it's a new line
LINES_PER_STAFF = 5  # standard five-line Western staff
STAFF_SPACING_TOLERANCE_FRAC = 0.35  # inter-line gaps can differ by up to 35% before we reject a candidate staff
STAFF_SPACING_TOLERANCE_MIN = 2.0  # prevents over-rejection when the image is small and spacing is only a few pixels
STAFF_VERTICAL_PADDING_FRAC = 2.0  # add 2x spacing above and below the outermost lines so stems and ledger lines are included
STAFF_ERASE_BAND_FRAC = 0.2  # only erase within +/- 20% of spacing around each line so nearby noteheads are not destroyed
SLIT_REPAIR_KERNEL_MIN = 3
SLIT_REPAIR_KERNEL_MAX = 7
SLIT_REPAIR_BAND_FRAC = 0.1  # narrower than the erase band; just enough to close the gap a stem leaves in a notehead after erasure

# Notehead detection and sizing (all fractions are relative to staff spacing)
NOTEHEAD_KERNEL_DIAMETER_FRAC = 0.45  # slightly under half-spacing so small noteheads survive the morphological open
NOTEHEAD_KERNEL_MIN = 3  # smallest ellipse kernel that makes morphological sense
NOTEHEAD_CLEANUP_KERNEL = (3, 3)  # tiny close kernel to smooth jagged edges that the open leaves behind
NOTE_MIN_AREA_FRAC = 0.08  # blobs smaller than 8% of spacing^2 are dust or scanner artifacts
NOTE_MAX_AREA_FRAC = 1.8  # blobs larger than 1.8x spacing^2 are likely merged symbols, not a single notehead
NOTE_MIN_SIZE_FRAC = 0.35  # minimum width or height along each axis
NOTE_MAX_SIZE_FRAC = 1.9  # maximum width or height; larger blobs are usually ties or slurs
NOTE_MIN_ASPECT = 0.45  # reject very tall/skinny blobs — those are almost always stems
NOTE_MAX_ASPECT = 2.2  # reject very wide blobs — those are usually ties or beams
NOTE_TINY_AREA_FRAC = 0.22  # if a blob passes area/size/aspect but is still tiny, try refining its center using the secondary mask
NOTE_MERGE_DISTANCE_FRAC = 0.75  # two detected centers within 75% of spacing are averaged into one
STEP_ROUND_UP_THRESHOLD = 0.58  # bias snapping slightly above 0.5 because noteheads tend to sit on lines more than spaces
STEP_CONFIDENCE_HIGH = 0.20  # centroid within 20% of a half-step grid position means high confidence pitch
STEP_CONFIDENCE_MEDIUM = 0.40  # within 40% means medium; beyond that we flag it as low confidence
HOLLOW_NOTE_Y_OFFSET_FRAC = 0.15  # morphological rounding trims the top of hollow noteheads, pulling the centroid up; nudge it back down by 15% of spacing
FILL_ELLIPSE_X_RADIUS_FRAC = 0.36  # horizontal radius of the elliptical fill-test window
FILL_ELLIPSE_Y_RADIUS_FRAC = 0.28  # vertical radius; noteheads are wider than tall so these differ
FILL_RATIO_THRESHOLD = 0.55  # more than 55% ink inside the ellipse means the notehead is filled (quarter or shorter)
WHOLE_NOTE_FILL_RATIO_MAX = 0.35  # a whole note has an open center so its ink ratio stays below 35%
STEM_X_RADIUS_FRAC = 0.85  # horizontal search radius when looking for a stem near the notehead center
STEM_Y_RADIUS_FRAC = 2.6  # vertical search radius; stems extend roughly 2-3 spacings above or below the head
STEM_MIN_RUN_FRAC = 1.2  # a vertical ink run must be at least 1.2x spacing tall to count as a stem
DUPLICATE_X_TOLERANCE_FRAC = 1.45  # two centers within 1.45x spacing horizontally are candidates for deduplication
DUPLICATE_Y_TOLERANCE_FRAC = 0.75  # they must also be close vertically
DUPLICATE_MAX_STEP_DIFF = 1  # only merge if the two pitch steps differ by at most 1 (same position or adjacent)
HOLLOW_SPLIT_X_TOLERANCE_FRAC = 1.8  # hollow noteheads split further apart than filled ones, so we use a wider tolerance
HOLLOW_SPLIT_INK_RATIO_MAX = 0.30  # the gap between the two halves of a split hollow note must be mostly empty (< 30% ink)

# Clef detection
CLEF_ROI_WIDTH_FRAC = 0.42  # clef symbol lives in the left ~42% of the header crop; the rest is key/time signature
CLEF_MIN_CONFIDENCE = 0.15  # below this template match score we return None rather than guess
CLEF_TIE_MARGIN = 0.06  # treble wins a tie unless bass beats it by more than 6 points; treble is far more common in our test set
CLEF_TRIM_WHITE_THRESH = 248  # pixels at or above 248 brightness are treated as background when trimming the template border
CLEF_MATCH_SCALES = (0.78, 0.88, 0.95, 1.0)  # try a few scale fractions in case the printed clef is smaller than our reference template

# Bar line detection
BAR_SEARCH_LEFT_SKIP_FRAC = 5.0  # skip 5x spacing from the left on the first staff to get past clef, key, and time signature
BAR_SEARCH_LEFT_SKIP_OTHER_FRAC = 2.0  # subsequent staves only repeat the clef, so a smaller skip is enough
BAR_CLOSE_KERNEL_HEIGHT_FRAC = 2.0  # close vertical gaps up to 2x spacing so a bar line broken by staff erasure stays connected
BAR_CLOSE_KERNEL_MIN = 5  # minimum kernel height for the close operation
BAR_MIN_HEIGHT_FRAC = 0.4  # a valid bar line must span at least 40% of the staff height
BAR_MIN_DENSITY = 0.55  # real bar lines are nearly solid; sparse blobs are leaked noteheads or stems
BAR_MAX_WIDTH_FRAC = 0.6  # anything wider than 0.6x spacing is probably not a thin bar line
BAR_DOUBLE_MIN_WIDTH_FRAC = 1.0  # a double bar spans at least one full spacing between its two strokes
BAR_RIGHT_MARGIN_FRAC = 2.0  # how close to the right staff edge counts as "near the end of the line"
BAR_LEFT_MARGIN_FRAC = 4.0  # similar margin on the left for detecting opening double bars
BAR_LEFT_RELAXED_EXTRA_FRAC = 0.7  # near the left edge we allow slightly wider blobs because clef artifacts can be thick
BAR_MERGE_DISTANCE_FRAC = 0.5  # two detected bar positions within half a spacing are merged into one
BAR_PAIR_GAP_FRAC = 1.5  # two singles within 1.5x spacing near a staff edge are reclassified as a double bar

# Repeat dot detection
REPEAT_DOT_SEARCH_WIDTH_FRAC = 1.8  # how far to look on each side of the bar line for repeat dots
REPEAT_DOT_BAR_GAP_FRAC = 0.25  # leave a small gap between the bar line itself and the search window
REPEAT_DOT_MIN_AREA_FRAC = 0.04  # minimum dot area relative to spacing^2
REPEAT_DOT_MAX_AREA_FRAC = 0.80  # maximum dot area; anything larger is probably a notehead
REPEAT_DOT_MAX_SIZE_FRAC = 0.95  # dots are smaller than one full spacing in each dimension
REPEAT_DOT_Y_TOLERANCE_FRAC = 0.65  # dots may sit up to 65% of spacing away from the expected y positions
REPEAT_DOT_X_ALIGNMENT_FRAC = 0.85  # top and bottom dots must be within 85% of spacing in x to count as aligned

DEFAULT_TITLE = "Sheet Music"
DEFAULT_METER = "4/4"
DEFAULT_UNIT_NOTE_LENGTH = "1/4"
DEFAULT_KEY = "C"
