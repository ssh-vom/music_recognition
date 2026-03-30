"""
Methodology Visualization & Design Decisions
============================================

This module provides visual explanations and illustrations of the image processing
methods used in the sheet music analysis pipeline. It is designed to help explain
*why* certain kernel sizes, thresholds, and approaches were chosen for different
detection tasks.

Usage for Report Generation:
    from methodology_visualization import (
        visualize_staff_detection_methods,
        visualize_note_detection_methods,
        visualize_clef_detection_methods,
        generate_method_comparison_chart,
    )
    
    # Generate comparison figures for your report
    staff_fig = visualize_staff_detection_methods()
    note_fig = visualize_note_detection_methods()
    clef_fig = visualize_clef_detection_methods()
"""

import math
from dataclasses import dataclass
from typing import Callable

import cv2 as cv
import numpy as np
from cv2.typing import MatLike


# =============================================================================
# KERNEL SIZE REFERENCE TABLE
# =============================================================================
# This table documents all kernel sizes used in the pipeline and their purposes

KERNEL_SPECIFICATIONS = """
┌─────────────────────────────────────────────────────────────────────────────────┐
│                        KERNEL SIZE SPECIFICATIONS                               │
├──────────────────┬────────────────────────┬─────────────────────────────────────┤
│ Detection Task   │ Kernel Type            │ Rationale                           │
├──────────────────┼────────────────────────┼─────────────────────────────────────┤
│ STAFF LINES      │ (Wx1) Rectangle        │ W = max(25, image_width // 12)      │
│                  │                        │                                     │
│                  │ [BULLET] Width: ~8% of image  │ Staff lines are long horizontal     │
│                  │ [BULLET] Height: 1 pixel      │ structures - needs wide kernel to   │
│                  │                        │ bridge small gaps while preserving  │
│                  │ Operation: MORPH_OPEN  │ line continuity                     │
├──────────────────┼────────────────────────┼─────────────────────────────────────┤
│ NOTEHEADS        │ (DxD) Ellipse          │ D = round(staff_spacing x 0.45)     │
│                  │                        │                                     │
│                  │ [BULLET] Diameter: ~45% of    │ Noteheads are oval/elliptical       │
│                  │   staff line spacing   │ shapes with diameter roughly half   │
│                  │ [BULLET] Shape: cv.MORPH_     │ the staff spacing                   │
│                  │   ELLIPSE              │                                     │
│                  │                        │ Elliptical kernel matches notehead  │
│                  │ Operation: MORPH_OPEN  │ geometry better than rectangle      │
├──────────────────┼────────────────────────┼─────────────────────────────────────┤
│ NOTE CLEANUP     │ (3x3) Ellipse          │ Fixed small kernel                  │
│                  │                        │                                     │
│                  │ [BULLET] Small fixed size     │ Removes noise from opening while    │
│                  │                        │ preserving notehead structure       │
│                  │ Operation: MORPH_CLOSE │                                     │
├──────────────────┼────────────────────────┼─────────────────────────────────────┤
│ BAR LINES        │ (1xH) Rectangle        │ H = round(2.0 x staff_spacing)      │
│                  │                        │                                     │
│                  │ [BULLET] Tall vertical kernel │ Bar lines span ~2 staff heights     │
│                  │ [BULLET] Joins fragments      │ Morphological close joins vertical  │
│                  │                        │ fragments into solid bar lines      │
├──────────────────┼────────────────────────┼─────────────────────────────────────┤
│ STAFF ERASE      │ (Kx1) Rectangle        │ K = max(1, image_width // 30)       │
│ (for note det)   │                        │                                     │
│                  │ [BULLET] ~3.3% of image width │ Horizontal dilation reconstructs    │
│                  │                        │ staff lines for subtraction         │
├──────────────────┼────────────────────────┼─────────────────────────────────────┤
│ SLIT REPAIR      │ (1xK) Rectangle        │ K = clamp(3, vertical_extent, 7)    │
│                  │                        │                                     │
│                  │ [BULLET] Small vertical kernel│ Closes slits created when staff     │
│                  │                        │ lines intersect note stems          │
│                  │ Operation: MORPH_CLOSE │ Without this, filled noteheads      │
│                  │                        │ would appear hollow after staff     │
│                  │                        │ removal                             │
└──────────────────┴────────────────────────┴─────────────────────────────────────┘
"""


# =============================================================================
# DESIGN DECISION RATIONALE
# =============================================================================

DESIGN_DECISIONS = """
╔═══════════════════════════════════════════════════════════════════════════════════╗
║                     DESIGN DECISIONS & METHODOLOGY COMPARISON                      ║
╠═══════════════════════════════════════════════════════════════════════════════════╣

1. STAFF LINE DETECTION: Morphological vs. Hough Transform
═══════════════════════════════════════════════════════════════════════════════════

   APPROACH CHOSEN: Morphological opening with wide horizontal kernel
   
   WHY NOT HOUGH TRANSFORM?
   ┌──────────────────────────┬─────────────────────────────────────────────────┐
   │ Hough Transform Issues     │ Morphological Solution                          │
   ├──────────────────────────┼─────────────────────────────────────────────────┤
   │ Detects ALL lines          │ Selectively targets horizontal structures       │
   │ (including stems, beams)   │ by aspect ratio (width >> height)               │
   ├──────────────────────────┼─────────────────────────────────────────────────┤
   │ Requires angle clustering  │ No angle search - assumes near-horizontal       │
   │ for slight rotations       │ staff lines (valid for scanned sheet music)     │
   ├──────────────────────────┼─────────────────────────────────────────────────┤
   │ Vote accumulation is       │ Direct pixel analysis with gap tolerance        │
   │ computationally expensive  │ (max_gap=1 in clustering)                       │
   ├──────────────────────────┼─────────────────────────────────────────────────┤
   │ Returns line segments      │ Produces connected regions for easy extent      │
   │ needing post-processing    │ measurement (x_start, x_end per line)           │
   └──────────────────────────┴─────────────────────────────────────────────────┘

   KEY INSIGHT: Staff lines are the WIDEST horizontal structures in the image.
   A kernel of width ~8% of the image filters out note stems, beams, and text.


2. NOTEHEAD DETECTION: Geometric Filtering vs. Template Matching
═══════════════════════════════════════════════════════════════════════════════════

   APPROACH CHOSEN: Morphological opening + connected components + geometric filters
   
   WHY NOT TEMPLATE MATCHING?
   ┌──────────────────────────┬─────────────────────────────────────────────────┐
   │ Template Matching Issues │ Geometric Solution                              │
   ├──────────────────────────┼─────────────────────────────────────────────────┤
   │ Requires multiple scales │ Staff spacing provides natural scale reference  │
   │ (noteheads vary in size) │ All thresholds are relative to spacing          │
   ├──────────────────────────┼─────────────────────────────────────────────────┤
   │ Sensitive to rotation    │ Elliptical kernel handles rotation naturally    │
   │ (stems attach at angles) │ via morphological operations                    │
   ├──────────────────────────┼─────────────────────────────────────────────────┤
   │ Fails on degraded/       │ Geometric features (area, aspect) are more      │
   │ handwritten scores         │ robust than pixel-perfect correlation           │
   ├──────────────────────────┼─────────────────────────────────────────────────┤
   │ Computationally heavy      │ Single-pass connected components is O(n)        │
   │ (correlation across image) │ vs. O(nxmxscales) for template matching         │
   └──────────────────────────┴─────────────────────────────────────────────────┘

   KEY INSIGHT: Noteheads have CONSISTENT GEOMETRY relative to staff spacing:
   - Area: 0.08 to 1.8 x (staff_spacing)²
   - Size: 0.35 to 1.9 x staff_spacing
   - Aspect ratio: 0.45 to 2.2 (width/height)


3. CLEF DETECTION: Template Matching vs. Feature Detection
═══════════════════════════════════════════════════════════════════════════════════

   APPROACH CHOSEN: Template matching with letterbox scaling
   
   WHY TEMPLATE MATCHING HERE?
   ┌──────────────────────────┬─────────────────────────────────────────────────┐
   │ Clefs are DISTINCTIVE      │ Feature detection would require training data   │
   │ SYMBOLS with consistent    │ Templates encode expert knowledge directly    │
   │ appearance                 │                                                 │
   ├──────────────────────────┼─────────────────────────────────────────────────┤
   │ Only 2 classes needed      │ Simple correlation-based classifier           │
   │ (treble vs. bass)          │ (treble_score vs. bass_score)                 │
   ├──────────────────────────┼─────────────────────────────────────────────────┤
   │ Clef region is isolated    │ No interference from other symbols            │
   │ (left 42% of staff)        │ before bar line detection                     │
   ├──────────────────────────┼─────────────────────────────────────────────────┤
   │ Shape is complex/organic   │ SIFT/ORB keypoints would be sparse              │
   │ (not corner-rich)          │ on smooth curves of G-clef                    │
   └──────────────────────────┴─────────────────────────────────────────────────┘

   KEY INSIGHT: Clefs are LARGE, ISOLATED, CONSISTENT symbols at fixed positions.
   This is the ideal case for template matching - unlike scattered noteheads.


4. STAFF REMOVAL: Adaptive Threshold vs. Simple Subtraction
═══════════════════════════════════════════════════════════════════════════════════

   APPROACH CHOSEN: Adaptive threshold + morphological reconstruction + subtraction
   
   THE CHALLENGE: Remove staff lines while preserving:
   - Filled noteheads (can look like staff lines when filled)
   - Hollow noteheads (staff lines pass through them)
   - Stems and beams (touch/intersect staff lines)
   
   SOLUTION ARCHITECTURE:
   ┌────────────────────────────────────────────────────────────────────────────┐
   │ 1. INVERT + ADAPTIVE THRESHOLD                                               │
   │    [BULLET] blockSize=15, C=-2                                                      │
   │    [BULLET] Handles uneven lighting better than global Otsu                         │
   │                                                                              │
   │ 2. MORPHOLOGICAL OPENING (dilate then erode horizontal)                     │
   │    [BULLET] Reconstructs staff lines from fragments                               │
   │    [BULLET] Kernel: (width//30 x 1) rectangle                                       │
   │                                                                              │
   │ 3. RESTRICTED SUBTRACTION                                                     │
   │    [BULLET] Only subtract within band around known staff line positions             │
   │    [BULLET] band = +-0.2 x staff_spacing                                             │
   │    [BULLET] Prevents removing parts of noteheads that overlap staff                 │
   │                                                                              │
   │ 4. SLIT REPAIR                                                                │
   │    [BULLET] Morphological close with vertical kernel                                │
   │    [BULLET] Heals gaps where stems crossed staff lines                              │
   │    [BULLET] Without this: hollow noteheads become "pac-man" shapes                  │
   │                                                                              │
   │ 5. BLENDED REPAIR                                                             │
   │    [BULLET] Only apply slit repair near staff line locations                        │
   │    [BULLET] Preserve notehead details elsewhere                                     │
   └────────────────────────────────────────────────────────────────────────────┘


5. PITCH DETECTION: Step-Based vs. Staff Line Intersection
═══════════════════════════════════════════════════════════════════════════════════

   APPROACH CHOSEN: "Step" units (half line spacing) from bottom line
   
   TRADITIONAL APPROACH: Check which staff lines a notehead intersects
   
   WHY STEP-BASED IS BETTER:
   ┌──────────────────────────┬─────────────────────────────────────────────────┐
   │ Line Intersection Issues   │ Step-Based Solution                             │
   ├──────────────────────────┼─────────────────────────────────────────────────┤
   │ Ambiguous when note is     │ Continuous metric provides sub-line             │
   │ between lines              │ precision (step = 0.5 x spacing)              │
   ├──────────────────────────┼─────────────────────────────────────────────────┤
   │ Requires pixel-perfect     │ Step float is robust to small vertical          │
   │ staff removal              │ errors - rounded to nearest step                │
   ├──────────────────────────┼─────────────────────────────────────────────────┤
   │ Doesn't handle ledger      │ Same math extends to ledger lines               │
   │ lines naturally            │ (step beyond staff range)                       │
   ├──────────────────────────┼─────────────────────────────────────────────────┤
   │ Binary (on/off line)       │ Confidence metric based on residual             │
   │                            │ from nearest step (0.58 rounding threshold)     │
   └──────────────────────────┴─────────────────────────────────────────────────┘

   FORMULA: step = round((bottom_line_y - note_center_y) / (spacing / 2))
   
   ROUNDING THRESHOLD: 0.58 (not 0.5)
   [BULLET] Notehead center is often ABOVE geometric center (gravity pulls ink down)
   [BULLET] Bias of 0.08 compensates for this asymmetry


6. DURATION CLASSIFICATION: Region Analysis vs. Symbol Recognition
═══════════════════════════════════════════════════════════════════════════════════

   APPROACH CHOSEN: Local pixel analysis in elliptical region around notehead
   
   CLASSIFICATION LOGIC:
   ┌────────────────────────┬───────────────┬───────────────┬────────────────────┐
   │ Region Fill Ratio      │ Stem Detected │ Duration      │ Visual Appearance  │
   ├────────────────────────┼───────────────┼───────────────┼────────────────────┤
   │ ink_ratio >= 0.55      │ No            │ whole         │ Open oval, no stem │
   ├────────────────────────┼───────────────┼───────────────┼────────────────────┤
   │ ink_ratio < 0.55       │ No            │ half          │ Hollow oval, stem  │
   ├────────────────────────┼───────────────┼───────────────┼────────────────────┤
   │ ink_ratio >= 0.55      │ Yes           │ quarter       │ Filled oval, stem  │
   ├────────────────────────┼───────────────┼───────────────┼────────────────────┤
   │ (filled + stem)        │               │               │                    │
   │                        │ + beam check  │ eighth/16th   │ Filled + flags/beam│
   └────────────────────────┴───────────────┴───────────────┴────────────────────┘

   STEM DETECTION: Vertical run-length analysis
   [BULLET] Search region: +-0.85 x spacing horizontally, +-2.6 x spacing vertically
   [BULLET] Minimum run: 1.2 x spacing (must be taller than notehead)
   [BULLET] Threshold: Any column with vertical run >= min_run


╚═══════════════════════════════════════════════════════════════════════════════════╝
"""


# =============================================================================
# PARAMETER TOLERANCE TABLES
# =============================================================================

PARAMETER_RATIONALES = """
┌─────────────────────────────────────────────────────────────────────────────────────┐
│                    PARAMETER TOLERANCES AND RATIONALES                              │
├────────────────────────────────┬────────────────┬─────────────────────────────────┤
│ Parameter                      │ Value          │ Rationale                       │
├────────────────────────────────┼────────────────┼─────────────────────────────────┤
│ Staff Line Gap Tolerance       │ max_gap = 1    │ Lines may have 1-pixel gaps     │
│ (clustering)                   │                │ from scanning artifacts         │
├────────────────────────────────┼────────────────┼─────────────────────────────────┤
│ Staff Spacing Consistency      │ 0.35 (35%)     │ Enforces 5 evenly spaced lines  │
│ (grouping tolerance)           │                │ Rejects false positives         │
├────────────────────────────────┼────────────────┼─────────────────────────────────┤
│ Notehead Merge Distance        │ 0.75 x spacing │ Chords/beams create touching      │
│                                │                │ noteheads that should merge     │
├────────────────────────────────┼────────────────┼─────────────────────────────────┤
│ Pitch Rounding Threshold       │ 0.58           │ Bias for notehead center being  │
│                                │                │ above geometric center          │
├────────────────────────────────┼────────────────┼─────────────────────────────────┤
│ Duration Fill Threshold        │ 0.55           │ Distinguishes hollow vs filled  │
│ (ink ratio)                    │                │ noteheads reliably              │
├────────────────────────────────┼────────────────┼─────────────────────────────────┤
│ Step Confidence - High         │ residual <= 0.20│ Within 20% of step boundary    │
├────────────────────────────────┼────────────────┼─────────────────────────────────┤
│ Step Confidence - Medium       │ residual <= 0.40│ Within 40% of step boundary    │
├────────────────────────────────┼────────────────┼─────────────────────────────────┤
│ Clef Confidence Minimum        │ 0.15           │ Rejects false positives when    │
│                                │                │ neither template matches well   │
├────────────────────────────────┼────────────────┼─────────────────────────────────┤
│ Staff Removal Band             │ +-0.2 x spacing │ Only erase near known staff     │
│                                │                │ positions (protects notes)      │
├────────────────────────────────┼────────────────┼─────────────────────────────────┤
│ Slit Repair Band               │ +-0.1 x spacing │ Only repair near staff lines    │
│                                │                │ (don't over-repair noteheads)   │
└────────────────────────────────┴────────────────┴─────────────────────────────────┘
"""


# =============================================================================
# VISUALIZATION FUNCTIONS
# =============================================================================

@dataclass
class KernelVisualization:
    """Represents a kernel for visualization purposes."""
    name: str
    kernel: np.ndarray
    scale_factor: int = 20  # Pixels per kernel cell for display
    color_active: tuple = (0, 0, 0)  # BGR
    color_inactive: tuple = (255, 255, 255)  # BGR


def create_kernel_visualization(kv: KernelVisualization) -> MatLike:
    """
    Create a visual representation of a morphological kernel.
    
    Returns a colored image showing the kernel structure at enlarged scale
    for easy inclusion in reports.
    """
    h, w = kv.kernel.shape[:2]
    sf = kv.scale_factor
    
    # Create enlarged visualization
    vis = np.full((h * sf, w * sf, 3), kv.color_inactive, dtype=np.uint8)
    
    for y in range(h):
        for x in range(w):
            y1, y2 = y * sf, (y + 1) * sf
            x1, x2 = x * sf, (x + 1) * sf
            if kv.kernel[y, x] > 0:
                vis[y1:y2, x1:x2] = kv.color_active
                # Add grid lines
                cv2_rectangle(vis, (x1, y1), (x2-1, y2-1), (128, 128, 128), 1)
    
    # Add border
    cv2_rectangle(vis, (0, 0), (vis.shape[1]-1, vis.shape[0]-1), (0, 0, 255), 2)
    
    return vis


def visualize_staff_line_kernel(image_width: int) -> MatLike:
    """
    Visualize the staff line detection kernel.
    
    Shows why a wide horizontal kernel is used to isolate staff lines.
    """
    kernel_width = max(25, image_width // 12)
    kernel = cv.getStructuringElement(cv.MORPH_RECT, (kernel_width, 1))
    
    # Create visualization with annotations
    vis_width = min(800, kernel_width * 3)
    vis = np.full((200, vis_width, 3), 255, dtype=np.uint8)
    
    # Draw the kernel (scaled)
    display_width = min(kernel_width * 2, vis_width - 100)
    scale = display_width / kernel_width
    kernel_y = 100
    
    # Draw kernel representation
    for i in range(kernel_width):
        x = int(50 + i * scale)
        next_x = int(50 + (i + 1) * scale)
        cv2_rectangle(vis, (x, kernel_y - 5), (next_x - 1, kernel_y + 5), (0, 0, 0), -1)
    
    # Add annotations
    font = cv.FONT_HERSHEY_SIMPLEX
    cv2_put_text(vis, f"Staff Line Kernel: {kernel_width}x1", (50, 40), font, 0.6, (0, 0, 0), 2)
    cv2_put_text(vis, f"Width = max(25, image_width // 12) = {kernel_width}px", (50, 70), font, 0.5, (0, 0, 128), 1)
    cv2_put_text(vis, "Captures long horizontal structures (staff lines)", (50, 170), font, 0.5, (0, 128, 0), 1)
    cv2_put_text(vis, "Filters out notes, stems, text (too narrow)", (50, 190), font, 0.5, (0, 128, 0), 1)
    
    return vis


def visualize_notehead_kernel(staff_spacing: float) -> MatLike:
    """
    Visualize the notehead detection kernel.
    
    Shows why an elliptical kernel sized relative to staff spacing works.
    """
    diameter = max(1, int(round(staff_spacing * 0.45)))
    if diameter % 2 == 0:
        diameter += 1
    
    kernel = cv.getStructuringElement(cv.MORPH_ELLIPSE, (diameter, diameter))
    
    # Create side-by-side comparison
    vis_height = 300
    vis_width = 600
    vis = np.full((vis_height, vis_width, 3), 255, dtype=np.uint8)
    
    # Draw kernel
    kv = KernelVisualization(
        name="Notehead Kernel",
        kernel=kernel,
        scale_factor=15,
        color_active=(0, 0, 0),
        color_inactive=(220, 220, 220)
    )
    kernel_vis = create_kernel_visualization(kv)
    
    # Place kernel visualization
    y_offset = 50
    x_offset = 50
    kh, kw = kernel_vis.shape[:2]
    vis[y_offset:y_offset+kh, x_offset:x_offset+kw] = kernel_vis
    
    # Draw idealized notehead shape
    note_center = (400, 150)
    note_rx = int(staff_spacing * 0.35)
    note_ry = int(staff_spacing * 0.28)
    cv2_ellipse(vis, note_center, (note_rx, note_ry), 0, 0, 360, (0, 0, 200), 2)
    
    # Add annotations
    font = cv.FONT_HERSHEY_SIMPLEX
    cv2_put_text(vis, f"Notehead Kernel: {diameter}x{diameter} Ellipse", (50, 30), font, 0.6, (0, 0, 0), 2)
    cv2_put_text(vis, f"Diameter = 0.45 x staff_spacing ≈ {diameter}px", (50, 260), font, 0.5, (0, 0, 128), 1)
    
    # Draw connection between kernel and notehead
    cv2_line(vis, (x_offset + kw + 10, y_offset + kh//2), (note_center[0] - note_rx - 10, note_center[1]), (128, 128, 128), 1)
    
    return vis


def visualize_bar_line_kernel(staff_spacing: float) -> MatLike:
    """Visualize the bar line detection kernel."""
    kernel_h = max(5, int(round(2.0 * staff_spacing)))
    kernel = cv.getStructuringElement(cv.MORPH_RECT, (1, kernel_h))
    
    vis = np.full((400, 300, 3), 255, dtype=np.uint8)
    
    # Draw tall vertical kernel
    kv = KernelVisualization(
        name="Bar Line Kernel",
        kernel=kernel,
        scale_factor=8,
        color_active=(0, 0, 0),
        color_inactive=(220, 220, 220)
    )
    kernel_vis = create_kernel_visualization(kv)
    
    y_offset = 50
    x_offset = 100
    kh, kw = kernel_vis.shape[:2]
    vis[y_offset:y_offset+kh, x_offset:x_offset+kw] = kernel_vis
    
    # Add annotations
    font = cv.FONT_HERSHEY_SIMPLEX
    cv2_put_text(vis, f"Bar Line Kernel: 1x{kernel_h}", (50, 30), font, 0.6, (0, 0, 0), 2)
    cv2_put_text(vis, "Vertical close joins", (50, 250), font, 0.5, (0, 128, 0), 1)
    cv2_put_text(vis, "bar line fragments", (50, 270), font, 0.5, (0, 128, 0), 1)
    
    return vis


def create_method_comparison_figure() -> MatLike:
    """
    Create a comparison figure showing different approaches considered.
    
    This is useful for reports to justify design decisions.
    """
    width, height = 1200, 800
    vis = np.full((height, width, 3), 245, dtype=np.uint8)  # Light gray background
    
    font = cv.FONT_HERSHEY_SIMPLEX
    font_small = cv.FONT_HERSHEY_PLAIN
    
    # Title
    cv2_put_text(vis, "Method Comparison: Chosen vs. Alternative Approaches", (50, 40), font, 0.9, (0, 0, 0), 2)
    
    # Table headers
    y_start = 80
    row_height = 180
    col_widths = [200, 450, 450]
    col_starts = [50, 250, 700]
    
    headers = ["Task", "Chosen Method", "Alternative (Not Used)"]
    for i, (header, x) in enumerate(zip(headers, col_starts)):
        cv2_put_text(vis, header, (x, y_start), font, 0.7, (0, 0, 128), 2)
        cv2_line(vis, (x, y_start + 5), (x + col_widths[i] - 20, y_start + 5), (0, 0, 128), 2)
    
    # Row 1: Staff Detection
    y = y_start + 50
    cv2_put_text(vis, "Staff Lines", (col_starts[0], y), font, 0.6, (0, 0, 0), 1)
    
    chosen_staff = [
        "Morphological opening",
        "Wide horizontal kernel (~8% image width)",
        "Clustering with gap tolerance=1",
        "[OK] Fast, selective for long lines"
    ]
    alt_staff = [
        "Hough Transform",
        "Vote accumulation across angles",
        "Post-processing to filter stems",
        "[X] Detects all lines, needs clustering"
    ]
    
    for i, line in enumerate(chosen_staff):
        color = (0, 128, 0) if "[OK]" in line else (0, 0, 0)
        cv2_put_text(vis, line, (col_starts[1], y + i*25), font_small, 1.1, color, 1)
    
    for i, line in enumerate(alt_staff):
        color = (0, 0, 200) if "[X]" in line else (100, 100, 100)
        cv2_put_text(vis, line, (col_starts[2], y + i*25), font_small, 1.1, color, 1)
    
    cv2_line(vis, (50, y + 110), (width-50, y + 110), (200, 200, 200), 1)
    
    # Row 2: Note Detection
    y = y_start + 50 + row_height
    cv2_put_text(vis, "Noteheads", (col_starts[0], y), font, 0.6, (0, 0, 0), 1)
    
    chosen_note = [
        "Ellipse kernel (0.45xspacing)",
        "Connected components + filtering",
        "Geometric thresholds relative to spacing",
        "[OK] Scale-invariant, no templates needed"
    ]
    alt_note = [
        "Template matching",
        "Multiple scales for size variation",
        "Correlation across measure region",
        "[X] Slow, needs many templates"
    ]
    
    for i, line in enumerate(chosen_note):
        color = (0, 128, 0) if "[OK]" in line else (0, 0, 0)
        cv2_put_text(vis, line, (col_starts[1], y + i*25), font_small, 1.1, color, 1)
    
    for i, line in enumerate(alt_note):
        color = (0, 0, 200) if "[X]" in line else (100, 100, 100)
        cv2_put_text(vis, line, (col_starts[2], y + i*25), font_small, 1.1, color, 1)
    
    cv2_line(vis, (50, y + 110), (width-50, y + 110), (200, 200, 200), 1)
    
    # Row 3: Clef Detection
    y = y_start + 50 + 2*row_height
    cv2_put_text(vis, "Clefs", (col_starts[0], y), font, 0.6, (0, 0, 0), 1)
    
    chosen_clef = [
        "Template matching (letterbox)",
        "Treble vs. Bass correlation scores",
        "Restricted to left 42% of staff",
        "[OK] Few classes, consistent appearance"
    ]
    alt_clef = [
        "Feature detection (SIFT/ORB)",
        "Keypoint matching with RANSAC",
        "Requires training images",
        "[X] Clefs are smooth, corner-poor"
    ]
    
    for i, line in enumerate(chosen_clef):
        color = (0, 128, 0) if "[OK]" in line else (0, 0, 0)
        cv2_put_text(vis, line, (col_starts[1], y + i*25), font_small, 1.1, color, 1)
    
    for i, line in enumerate(alt_clef):
        color = (0, 0, 200) if "[X]" in line else (100, 100, 100)
        cv2_put_text(vis, line, (col_starts[2], y + i*25), font_small, 1.1, color, 1)
    
    # Footer
    cv2_put_text(vis, "Key Insight: Different tasks require different approaches based on symbol properties and variability.", 
                 (50, height - 30), font, 0.55, (80, 80, 80), 1)
    
    return vis


def create_kernel_size_cheatsheet() -> MatLike:
    """
    Create a quick reference figure showing all kernel sizes in one view.
    """
    width, height = 1000, 700
    vis = np.full((height, width, 3), 255, dtype=np.uint8)
    
    font = cv.FONT_HERSHEY_SIMPLEX
    font_small = cv.FONT_HERSHEY_PLAIN
    
    # Title
    cv2_put_text(vis, "Morphological Kernel Size Reference", (50, 40), font, 0.9, (0, 0, 0), 2)
    
    # Reference values
    ref_spacing = 20  # pixels
    ref_width = 600  # pixels
    
    kernels = [
        ("Staff Lines", max(25, ref_width // 12), 1, cv.MORPH_RECT, 
         "max(25, W//12) x 1", "Horizontal open"),
        ("Noteheads", int(round(ref_spacing * 0.45)), int(round(ref_spacing * 0.45)), cv.MORPH_ELLIPSE,
         "0.45S x 0.45S ellipse", "Note isolation"),
        ("Note Cleanup", 3, 3, cv.MORPH_ELLIPSE,
         "3 x 3 ellipse", "Noise removal"),
        ("Bar Lines", 1, max(5, int(round(2.0 * ref_spacing))), cv.MORPH_RECT,
         "1 x 2.0S", "Vertical close"),
        ("Staff Erase", max(1, ref_width // 30), 1, cv.MORPH_RECT,
         "W//30 x 1", "Staff reconstruction"),
        ("Slit Repair", 1, 5, cv.MORPH_RECT,
         "1 x 3-7", "Gap healing"),
    ]
    
    y = 100
    box_height = 80
    
    for name, kw, kh, ktype, formula, purpose in kernels:
        # Draw kernel visualization
        if ktype == cv.MORPH_RECT:
            kernel = cv.getStructuringElement(cv.MORPH_RECT, (kw, kh))
        else:
            kernel = cv.getStructuringElement(cv.MORPH_ELLIPSE, (kw, kh))
        
        # Scale for display
        max_display = 60
        scale = min(max_display / max(kw, kh), 3) if max(kw, kh) > 0 else 3
        dw, dh = int(kw * scale), int(kh * scale)
        
        # Draw box
        kernel_x, kernel_y = 80, y + 10
        
        # Draw simplified representation
        if ktype == cv.MORPH_ELLIPSE and kw == kh:
            # Draw circle
            cv2_circle(vis, (kernel_x + dw//2, kernel_y + dh//2), dw//2, (0, 0, 0), -1)
        else:
            # Draw rectangle
            cv2_rectangle(vis, (kernel_x, kernel_y), (kernel_x + dw, kernel_y + dh), (0, 0, 0), -1)
        
        # Text
        cv2_put_text(vis, name, (200, y + 25), font, 0.7, (0, 0, 128), 2)
        cv2_put_text(vis, f"Formula: {formula}", (200, y + 50), font_small, 1.2, (0, 0, 0), 1)
        cv2_put_text(vis, f"Purpose: {purpose}", (200, y + 70), font_small, 1.2, (80, 80, 80), 1)
        
        # Separator
        cv2_line(vis, (50, y + box_height - 5), (width - 50, y + box_height - 5), (220, 220, 220), 1)
        
        y += box_height
    
    # Legend
    legend_y = height - 60
    cv2_put_text(vis, "S = staff line spacing (pixels)    W = image width (pixels)", 
                 (50, legend_y), font, 0.55, (80, 80, 80), 1)
    cv2_put_text(vis, "All kernels use OpenCV cv.getStructuringElement() with cv.MORPH_OPEN/CLOSE",
                 (50, legend_y + 25), font, 0.55, (80, 80, 80), 1)
    
    return vis


def visualize_staff_detection_methods() -> MatLike:
    """
    Generate a figure comparing staff detection approaches for reports.
    
    Returns an image suitable for inclusion in documentation.
    """
    return create_method_comparison_figure()


def visualize_note_detection_methods() -> MatLike:
    """Generate a figure showing note detection methodology."""
    return create_kernel_size_cheatsheet()


def visualize_clef_detection_methods() -> MatLike:
    """Generate a figure showing clef detection methodology."""
    width, height = 900, 500
    vis = np.full((height, width, 3), 255, dtype=np.uint8)
    
    font = cv.FONT_HERSHEY_SIMPLEX
    
    cv2_put_text(vis, "Clef Detection: Template Matching Approach", (50, 40), font, 0.8, (0, 0, 0), 2)
    
    # Draw template representations
    cv2_put_text(vis, "Treble Template", (150, 100), font, 0.6, (0, 0, 128), 2)
    cv2_put_text(vis, "Bass Template", (550, 100), font, 0.6, (0, 0, 128), 2)
    
    # Draw simplified clef shapes
    # Treble (G-clef approximation)
    center_t = (200, 250)
    cv2_circle(vis, center_t, 40, (0, 0, 0), 3)
    cv2_line(vis, (center_t[0], center_t[1] - 60), (center_t[0], center_t[1] + 60), (0, 0, 0), 3)
    
    # Bass (F-clef approximation)
    center_b = (600, 250)
    cv2_circle(vis, (center_b[0] - 20, center_b[1] - 20), 15, (0, 0, 0), 3)
    cv2_circle(vis, (center_b[0] + 20, center_b[1] + 20), 15, (0, 0, 0), 3)
    cv2_line(vis, (center_b[0] - 30, center_b[1] - 10), (center_b[0] + 30, center_b[1] + 10), (0, 0, 0), 3)
    
    # Matching explanation
    cv2_put_text(vis, "Letterbox Scaling:", (50, 380), font, 0.6, (0, 0, 0), 1)
    cv2_put_text(vis, "[BULLET] Template scaled to fit clef region height", (70, 410), font, 0.5, (80, 80, 80), 1)
    cv2_put_text(vis, "[BULLET] Placed left-aligned (clefs appear at staff start)", (70, 435), font, 0.5, (80, 80, 80), 1)
    cv2_put_text(vis, "[BULLET] Correlation score computed (TM_CCOEFF_NORMED)", (70, 460), font, 0.5, (80, 80, 80), 1)
    
    cv2_put_text(vis, "Decision Rule:", (500, 380), font, 0.6, (0, 0, 0), 1)
    cv2_put_text(vis, "[BULLET] Compare treble_score vs bass_score", (520, 410), font, 0.5, (80, 80, 80), 1)
    cv2_put_text(vis, "[BULLET] Tie margin: 0.06 (prefer treble if close)", (520, 435), font, 0.5, (80, 80, 80), 1)
    cv2_put_text(vis, "[BULLET] Min confidence: 0.15 (reject if both low)", (520, 460), font, 0.5, (80, 80, 80), 1)
    
    return vis


def generate_method_comparison_chart() -> MatLike:
    """
    Generate a comprehensive comparison chart.
    
    This is the main entry point for creating report figures.
    """
    return create_method_comparison_figure()


# =============================================================================
# UTILITY FUNCTIONS (cv2 wrappers for type consistency)
# =============================================================================

def cv2_rectangle(img: MatLike, pt1: tuple, pt2: tuple, color: tuple, thickness: int = 1) -> None:
    """Wrapper for cv.rectangle."""
    cv.rectangle(img, pt1, pt2, color, thickness)


def cv2_put_text(img: MatLike, text: str, org: tuple, fontFace: int, fontScale: float, 
                 color: tuple, thickness: int = 1) -> None:
    """Wrapper for cv.putText."""
    cv.putText(img, text, org, fontFace, fontScale, color, thickness, cv.LINE_AA)


def cv2_line(img: MatLike, pt1: tuple, pt2: tuple, color: tuple, thickness: int = 1) -> None:
    """Wrapper for cv.line."""
    cv.line(img, pt1, pt2, color, thickness)


def cv2_circle(img: MatLike, center: tuple, radius: int, color: tuple, thickness: int = 1) -> None:
    """Wrapper for cv.circle."""
    cv.circle(img, center, radius, color, thickness)


def cv2_ellipse(img: MatLike, center: tuple, axes: tuple, angle: float, 
                startAngle: float, endAngle: float, color: tuple, thickness: int = 1) -> None:
    """Wrapper for cv.ellipse."""
    cv.ellipse(img, center, axes, angle, startAngle, endAngle, color, thickness)


# =============================================================================
# TEXT OUTPUT FOR REPORTS
# =============================================================================

def print_methodology_documentation() -> str:
    """
    Print complete methodology documentation as a formatted string.
    
    This can be copied directly into a report or thesis chapter.
    """
    output = []
    output.append("=" * 80)
    output.append("MUSICAL SYMBOL DETECTION METHODOLOGY")
    output.append("=" * 80)
    output.append("")
    output.append(KERNEL_SPECIFICATIONS)
    output.append("")
    output.append(DESIGN_DECISIONS)
    output.append("")
    output.append(PARAMETER_RATIONALES)
    output.append("")
    output.append("=" * 80)
    output.append("END OF METHODOLOGY DOCUMENTATION")
    output.append("=" * 80)
    
    return "\n".join(output)


# =============================================================================
# MAIN (for testing/standalone generation)
# =============================================================================

if __name__ == "__main__":
    import os
    
    # Create output directory
    output_dir = "/Users/shivom/courses/4tn4/project/methodology_figures"
    os.makedirs(output_dir, exist_ok=True)
    
    print("Generating methodology visualization figures...")
    
    # Generate kernel visualization
    staff_kernel_vis = visualize_staff_line_kernel(image_width=600)
    cv.imwrite(f"{output_dir}/01_staff_line_kernel.jpg", staff_kernel_vis)
    print(f"  [OK] Saved: 01_staff_line_kernel.jpg")
    
    # Generate notehead kernel visualization
    note_kernel_vis = visualize_notehead_kernel(staff_spacing=20)
    cv.imwrite(f"{output_dir}/02_notehead_kernel.jpg", note_kernel_vis)
    print(f"  [OK] Saved: 02_notehead_kernel.jpg")
    
    # Generate bar line kernel visualization
    bar_kernel_vis = visualize_bar_line_kernel(staff_spacing=20)
    cv.imwrite(f"{output_dir}/03_bar_line_kernel.jpg", bar_kernel_vis)
    print(f"  [OK] Saved: 03_bar_line_kernel.jpg")
    
    # Generate method comparison
    comparison_vis = generate_method_comparison_chart()
    cv.imwrite(f"{output_dir}/04_method_comparison.jpg", comparison_vis)
    print(f"  [OK] Saved: 04_method_comparison.jpg")
    
    # Generate kernel cheatsheet
    cheatsheet_vis = create_kernel_size_cheatsheet()
    cv.imwrite(f"{output_dir}/05_kernel_cheatsheet.jpg", cheatsheet_vis)
    print(f"  [OK] Saved: 05_kernel_cheatsheet.jpg")
    
    # Generate clef detection visualization
    clef_vis = visualize_clef_detection_methods()
    cv.imwrite(f"{output_dir}/06_clef_detection.jpg", clef_vis)
    print(f"  [OK] Saved: 06_clef_detection.jpg")
    
    # Save text documentation
    doc_text = print_methodology_documentation()
    with open(f"{output_dir}/methodology_documentation.txt", "w") as f:
        f.write(doc_text)
    print(f"  [OK] Saved: methodology_documentation.txt")
    
    print(f"\nAll figures saved to: {output_dir}")
    print("\nTo include in your report:")
    print("  1. Reference the kernel size tables when explaining morphological operations")
    print("  2. Use the method comparison figure to justify design decisions")
    print("  3. Cite the parameter tolerance table when discussing robustness")
    print("  4. Copy text from methodology_documentation.txt into your methodology section")
