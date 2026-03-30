"""Sheet music detection constants."""

# =============================================================================
# GENERAL IMAGE PROCESSING
# =============================================================================

# Binary mask values
MASK_BACKGROUND = 0
MASK_FOREGROUND = 255

# Gaussian blur for initial denoising
BLUR_KERNEL_SIZE = (5, 5)

# =============================================================================
# STAFF DETECTION
# =============================================================================

# Staff line morphology: kernel width as fraction of image width
STAFF_LINE_KERNEL_WIDTH_FRAC = 1 / 12  # ~8% of image width
STAFF_LINE_KERNEL_MIN_WIDTH = 25  # pixels

# Line detection: minimum strength relative to peak
LINE_DETECTION_MIN_RATIO = 0.35

# Line clustering: maximum gap between consecutive pixels to be considered same line
LINE_CLUSTER_MAX_GAP = 1  # pixels

# Staff validation
LINES_PER_STAFF = 5
STAFF_SPACING_TOLERANCE_FRAC = 0.35  # 35% variation allowed between line gaps
STAFF_SPACING_TOLERANCE_MIN = 2.0  # pixels

# Staff extent padding: extend staff region above/below by this factor of spacing
STAFF_VERTICAL_PADDING_FRAC = 2.0

# Line extent detection: vertical window for checking line continuity
LINE_EXTENT_HALF_WINDOW = 1

# =============================================================================
# STAFF REMOVAL
# =============================================================================

# Staff reconstruction kernel width as fraction of image width
STAFF_RECONSTRUCTION_WIDTH_FRAC = 1 / 30  # ~3.3% of image width

# Staff erasure band: only erase within this distance of known staff lines
STAFF_ERASE_BAND_FRAC = 0.2  # ±20% of spacing

# Slit repair: heal gaps where stems crossed staff lines
SLIT_REPAIR_KERNEL_MIN = 3
SLIT_REPAIR_KERNEL_MAX = 7
SLIT_REPAIR_BAND_FRAC = 0.1  # Only repair near staff lines

# Adaptive thresholding for note extraction
NOTE_ADAPTIVE_BLOCK_SIZE = 15
NOTE_ADAPTIVE_C = -2

# =============================================================================
# NOTEHEAD DETECTION
# =============================================================================

# Notehead kernel: elliptical opening to isolate noteheads
NOTEHEAD_KERNEL_DIAMETER_FRAC = 0.45  # relative to staff spacing
NOTEHEAD_KERNEL_MIN = 3  # minimum kernel size in pixels

# Cleanup kernel: small ellipse to close gaps after opening
NOTEHEAD_CLEANUP_KERNEL = (3, 3)

# Geometric filtering of connected components
NOTE_MIN_AREA_FRAC = 0.08  # min area relative to spacing²
NOTE_MAX_AREA_FRAC = 1.8  # max area relative to spacing²
NOTE_MIN_SIZE_FRAC = 0.35  # min width/height relative to spacing
NOTE_MAX_SIZE_FRAC = 1.9  # max width/height relative to spacing
NOTE_MIN_ASPECT = 0.45  # width/height ratio
NOTE_MAX_ASPECT = 2.2

# Tiny notehead refinement threshold
NOTE_TINY_AREA_FRAC = 0.22  # use secondary mask refinement below this area

# Center merging: merge nearby centers (handles chords/touching noteheads)
NOTE_MERGE_DISTANCE_FRAC = 0.75  # relative to spacing

# =============================================================================
# PITCH DETECTION
# =============================================================================

# Step quantization: round to nearest step boundary
# Bias for notehead center being above geometric center (ink pulls down)
STEP_ROUND_UP_THRESHOLD = 0.58  # round up if fractional part > 0.58

# Step confidence levels based on distance from exact step position
STEP_CONFIDENCE_HIGH = 0.20  # within 20% of step boundary = high confidence
STEP_CONFIDENCE_MEDIUM = 0.40  # within 40% = medium confidence

# Hollow notehead correction: centroid bias due to empty center
HOLLOW_NOTE_Y_OFFSET_FRAC = 0.15  # shift down by 15% of spacing

# =============================================================================
# DURATION CLASSIFICATION
# =============================================================================

# Fill detection: ellipse size relative to spacing
FILL_ELLIPSE_X_RADIUS_FRAC = 0.36
FILL_ELLIPSE_Y_RADIUS_FRAC = 0.28
FILL_RATIO_THRESHOLD = 0.55  # ink ratio above this = filled notehead

# Hollow notehead fill ratio (for distinguishing whole notes misclassified as filled)
WHOLE_NOTE_FILL_RATIO_MAX = 0.35  # below this = likely hollow/whole note

# Stem detection: search region relative to spacing
STEM_X_RADIUS_FRAC = 0.85
STEM_Y_RADIUS_FRAC = 2.6
STEM_MIN_RUN_FRAC = 1.2  # minimum vertical run length

# =============================================================================
# DUPLICATE NOTE COLLAPSE
# =============================================================================

# Tolerances for considering two detections as the same note
DUPLICATE_X_TOLERANCE_FRAC = 1.45  # relative to spacing
DUPLICATE_Y_TOLERANCE_FRAC = 0.75
DUPLICATE_MAX_STEP_DIFF = 1

# Hollow notehead split detection: two components forming one hollow note
HOLLOW_SPLIT_X_TOLERANCE_FRAC = 1.8  # relative to spacing
HOLLOW_SPLIT_INK_RATIO_MAX = 0.30  # max ink between halves

# =============================================================================
# CLEF DETECTION
# =============================================================================

# Region of interest: left portion of staff header containing clef
CLEF_ROI_WIDTH_FRAC = 0.42  # first 42% of staff width

# Template matching confidence thresholds
CLEF_MIN_CONFIDENCE = 0.15
CLEF_TIE_MARGIN = 0.06  # treble wins if score within this margin of bass

# Image preprocessing
CLEF_TRIM_WHITE_THRESH = 248  # threshold for trimming white borders

# Multi-scale matching scales (fractions of ROI height)
CLEF_MATCH_SCALES = (0.78, 0.88, 0.95, 1.0)

# =============================================================================
# ACCIDENTAL DETECTION (Sharps/Flats)
# =============================================================================

# Template matching scales
ACCIDENTAL_SCALES = (0.35, 0.5, 0.65, 0.8)
ACCIDENTAL_MATCH_THRESHOLD = 0.50

# Peak suppression: minimum distance between detections
ACCIDENTAL_MIN_DISTANCE_FRAC = 0.55  # relative to spacing

# Geometric detection fallback (for tiny headers where template matching fails)
ACCIDENTAL_GEOMETRIC_MIN_HEIGHT_FRAC = 1.8  # min component height
ACCIDENTAL_GEOMETRIC_MIN_AREA_FRAC = 0.18  # min area relative to spacing²

# Sharp vs flat classification based on vertical stroke structure
SHARP_MIN_TALL_CLUSTERS = 2  # need 2+ tall stroke clusters
FLAT_TALL_CLUSTERS = 1  # flat has 1 tall cluster
TALL_COLUMN_THRESHOLD_FRAC = 0.75  # column is "tall" if exceeds this fraction of height

# =============================================================================
# BAR LINE DETECTION
# =============================================================================

# Skip left region containing clef/key (as multiples of spacing)
BAR_SEARCH_LEFT_SKIP_FRAC = 5.0  # first staff
BAR_SEARCH_LEFT_SKIP_OTHER_FRAC = 2.0  # subsequent staves

# Morphological closing: join vertical bar line fragments
BAR_CLOSE_KERNEL_HEIGHT_FRAC = 2.0  # kernel height relative to spacing
BAR_CLOSE_KERNEL_MIN = 5  # minimum kernel height in pixels

# Contour filtering
BAR_MIN_HEIGHT_FRAC = 0.4  # relative to staff height
BAR_MIN_DENSITY = 0.55  # ink ratio within bounding rect
BAR_MAX_WIDTH_FRAC = 0.6  # single bar line max width
BAR_DOUBLE_MIN_WIDTH_FRAC = 1.0  # double bar line min width

# Edge margins for detecting repeat signs
BAR_RIGHT_MARGIN_FRAC = 2.0
BAR_LEFT_MARGIN_FRAC = 4.0
BAR_LEFT_RELAXED_EXTRA_FRAC = 0.7  # extra width allowed at left edge

# Merging nearby bar lines
BAR_MERGE_DISTANCE_FRAC = 0.5  # relative to spacing
BAR_PAIR_GAP_FRAC = 1.5  # max gap between double bar lines

# =============================================================================
# MEASURE SPLITTING
# =============================================================================

# Header width: region containing clef, key, time signature
MEASURE_HEADER_WIDTH_FRAC = 5.2  # relative to spacing
MEASURE_FIRST_STAFF_HEADER_FRAC = 7.0  # conservative for first staff

# Bar line trimming: shrink measure slightly to avoid including bar lines
BAR_TRIM_FRAC = 0.25  # relative to spacing

# Minimum measure width
MIN_MEASURE_WIDTH = 4  # pixels

# =============================================================================
# RHYTHM/BEAM DETECTION
# =============================================================================

# Event grouping: notes at roughly the same onset time
EVENT_X_TOLERANCE = 5  # pixels

# Compact run detection: tightly spaced notes likely beamed
COMPACT_GAP_MIN_FRAC = 1.2  # min spacing between beamed notes
COMPACT_GAP_MAX_FRAC = 3.6  # max spacing between beamed notes
COMPACT_RUN_MIN_LENGTH = 4  # minimum notes in a compact run

# Beam detection
BEAM_X_SPAN_MIN_FRAC = 0.5  # need some horizontal spread
BEAM_DENSITY_THRESHOLD_FRAC = 0.15  # ink density threshold
BEAM_INK_SPAN_MIN_FRAC = 0.20  # min ink span across note group
BEAM_CONTINUITY_MIN_FRAC = 0.8  # min continuous run length

# Stem direction estimation
STEM_DIR_Y_RADIUS_FRAC = 2.5
STEM_DIR_X_RADIUS_FRAC = 0.6
STEM_DIR_RATIO_THRESHOLD = 1.5  # ink ratio to determine direction

# =============================================================================
# TIME SIGNATURE OCR
# =============================================================================

# Image preprocessing for Tesseract
TIME_SIG_SCALE_FACTOR = 10.0
TIME_SIG_BORDER = 8  # pixels

# OCR page segmentation modes
PSM_UNIFORM_BLOCK = 6
PSM_SINGLE_CHAR = 10
PSM_SINGLE_WORD = 8

# Character whitelists
TIME_SIG_DIGITS = "0123456789"
TIME_SIG_COMMON_TIME = "Cc"

# Common time detection
COMMON_TIME_CHAR = "C"

# =============================================================================
# DEFAULT EXPORT VALUES
# =============================================================================

DEFAULT_TITLE = "Sheet Music"
DEFAULT_METER = "4/4"
DEFAULT_UNIT_NOTE_LENGTH = "1/4"
DEFAULT_KEY = "C"
DEFAULT_TEMPO_QPM = 120
