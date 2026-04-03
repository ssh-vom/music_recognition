// Report Template - 8.5 x 11 inch format
#import "custom-counters.typ": eq-numbered, table-numbered
#import "@preview/codly:1.3.0": *
#import "@preview/codly-languages:0.1.1": *
#show: codly-init.with()
#set page(
  paper: "us-letter",
  margin: (top: 1in, bottom: 1in, left: 1in, right: 1in),
  numbering: "1",
  number-align: center + bottom,
)
#set text(font: "Times New Roman", size: 12pt)
// #set par(justify: true, linebreaks: "optimized", leading: 1.5em)

// Heading styles
#show heading.where(level: 1): it => [
  #v(0.5em)
  #text(size: 14pt, weight: "bold", it.body)
  #v(0.5em)
]

#show heading.where(level: 2): it => [
  #v(0.3em)
  #text(size: 12pt, weight: "bold", it.body)
  #v(0.3em)
]

#let sheet_figures(folder_name, display_name, first_sheet: false) = [
  === Staff Detection Images - #display_name

  #if first_sheet [
    #figure(
      image("../methodology_figures/01_staff_line_kernel.jpg", width: 55%),
      caption: [
        Horizontal staff-line kernel used for morphological opening. The wide
        rectangle preserves long horizontal staff lines while suppressing narrower
        symbols.
      ],
    ) <fig:staffline-kernel>

    #v(1em)
  ]

  #grid(
    columns: (1fr, 1fr),
    gutter: 1em,
    [
      #if first_sheet [
        #figure(
          image("../artifacts/" + folder_name + "/01_staff/01_grayscale.jpg", width: 100%),
          caption: [
            Input grayscale image (#display_name). Three staffs are visible with
            treble clef, barlines, and note symbols.
          ],
        ) <fig:twinkle-grayscale>
      ] else [
        #figure(
          image("../artifacts/" + folder_name + "/01_staff/01_grayscale.jpg", width: 100%),
          caption: [
            Input grayscale image (#display_name). Three staffs are visible with
            treble clef, barlines, and note symbols.
          ],
        )
      ]
    ],
    [
      #if first_sheet [
        #figure(
          image("../artifacts/" + folder_name + "/01_staff/02_otsu_binary.jpg", width: 100%),
          caption: [
            Otsu global binarization for #display_name, with foreground ink shown
            as white on black for morphological processing.
          ],
        ) <fig:twinkle-otsu>
      ] else [
        #figure(
          image("../artifacts/" + folder_name + "/01_staff/02_otsu_binary.jpg", width: 100%),
          caption: [
            Otsu global binarization for #display_name, with foreground ink shown
            as white on black for morphological processing.
          ],
        )
      ]
    ],
  )

  #grid(
    columns: (1fr, 1fr),
    gutter: 1em,
    [
      #if first_sheet [
        #figure(
          image("../artifacts/" + folder_name + "/01_staff/03_horizontal_lines.jpg", width: 100%),
          caption: [
            Staff lines isolated for #display_name by horizontal morphological
            opening (kernel $W times 1$).
          ],
        ) <fig:twinkle-staff-lines-only>
      ] else [
        #figure(
          image("../artifacts/" + folder_name + "/01_staff/03_horizontal_lines.jpg", width: 100%),
          caption: [
            Staff lines isolated for #display_name by horizontal morphological
            opening (kernel $W times 1$).
          ],
        )
      ]
    ],
    [
      #if first_sheet [
        #figure(
          image("../artifacts/" + folder_name + "/01_staff/04_staff_overlay.jpg", width: 100%),
          caption: [
            Detected staff groups overlaid on #display_name, with even staff
            spacings.
          ],
        ) <fig:twinkle-staff-overlay>
      ] else [
        #figure(
          image("../artifacts/" + folder_name + "/01_staff/04_staff_overlay.jpg", width: 100%),
          caption: [
            Detected staff groups overlaid on #display_name, with even staff
            spacings.
          ],
        )
      ]
    ],
  )
  #pagebreak()
  == Staff Removal Images - #display_name
  #if first_sheet [
    #figure(
      image("../artifacts/" + folder_name + "/05_masks/01_notes_mask.jpg", width: 80%),
      caption: [
        Raw adaptive-threshold mask for #display_name before staff removal.
        Staff lines, noteheads, stems, and barlines are all present.
      ],
    ) <fig:twinkle-adaptive-threshold-staff-removal>
  ] else [
    #figure(
      image("../artifacts/" + folder_name + "/05_masks/01_notes_mask.jpg", width: 80%),
      caption: [
        Raw adaptive-threshold mask for #display_name before staff removal.
        Staff lines, noteheads, stems, and barlines are all present.
      ],
    )
  ]

  #v(1em)

  #grid(
    columns: (1fr, 1fr),
    gutter: 1em,
    [
      #if first_sheet [
        #figure(
          image("../artifacts/" + folder_name + "/05_masks/02_notes_mask_erased.jpg", width: 100%),
          caption: [
            Notes mask for #display_name after band-restricted staff subtraction
            and slit repair. Staff lines are removed while notehead, stem, and
            beam structure is preserved.
          ],
        ) <fig:twinkle-staff-removal-notes>
      ] else [
        #figure(
          image("../artifacts/" + folder_name + "/05_masks/02_notes_mask_erased.jpg", width: 100%),
          caption: [
            Notes mask for #display_name after band-restricted staff subtraction
            and slit repair. Staff lines are removed while notehead, stem, and
            beam structure is preserved.
          ],
        )
      ]
    ],
    [
      #if first_sheet [
        #figure(
          image("../artifacts/" + folder_name + "/05_masks/03_bars_mask_erased.jpg", width: 100%),
          caption: [
            Bars mask for #display_name after staff removal, used for vertical
            bar line detection. Only tall vertical ink segments remain.
          ],
        ) <fig:twinkle-staff-removal-bars>
      ] else [
        #figure(
          image("../artifacts/" + folder_name + "/05_masks/03_bars_mask_erased.jpg", width: 100%),
          caption: [
            Bars mask for #display_name after staff removal, used for vertical
            bar line detection. Only tall vertical ink segments remain.
          ],
        )
      ]
    ],
  )

  #v(1em)

  #pagebreak()
  == Bar Line Detection Images - #display_name
  #if first_sheet [
    #figure(
      image("../methodology_figures/03_bar_line_kernel.jpg", width: 55%),
      caption: [
        Vertical bar-line kernel used for morphological close. The tall rectangle
        reconnects fragmented vertical ink into solid bar candidates.
      ],
    ) <fig:barline-kernel>

    #v(1em)
  ]

  #grid(
    columns: (1fr, 1fr),
    gutter: 1em,
    [
      #if first_sheet [
        #figure(
          image("../artifacts/" + folder_name + "/02_bars/02_vertical_close.jpg", width: 100%),
          caption: [
            Vertical morphological close for #display_name reconnects fragmented
            bar-line ink into solid vertical candidates.
          ],
        ) <fig:twinkle-bar-vertical-close>
      ] else [
        #figure(
          image("../artifacts/" + folder_name + "/02_bars/02_vertical_close.jpg", width: 100%),
          caption: [
            Vertical morphological close for #display_name reconnects fragmented
            bar-line ink into solid vertical candidates.
          ],
        )
      ]
    ],
    [
      #if first_sheet [
        #figure(
          image("../artifacts/" + folder_name + "/02_bars/04_bar_overlay.jpg", width: 100%),
          caption: [
            Detected bar lines overlaid on #display_name. The barlines partition
            each staff into measures, and repeat structure is preserved where present.
          ],
        ) <fig:twinkle-bar-overlay>
      ] else [
        #figure(
          image("../artifacts/" + folder_name + "/02_bars/04_bar_overlay.jpg", width: 100%),
          caption: [
            Detected bar lines overlaid on #display_name. The barlines partition
            each staff into measures, and repeat structure is preserved where present.
          ],
        )
      ]
    ],
  )

  #v(1em)

  == Clef Detection Images - #display_name
  #grid(
    columns: (1fr, 1fr),
    gutter: 1em,
    [
      #if first_sheet [
        #figure(
          grid(
            columns: (1fr, 1fr),
            gutter: 0.75em,
            [#image("../templates/clef/treble.png", width: 100%)],
            [#image("../templates/clef/bass.png", width: 100%)],
          ),
          caption: [
            Treble and bass clef templates used for template matching. Each
            template is scaled to the staff crop and scored with normalized
            cross-correlation.
          ],
        ) <fig:clef-templates>
      ]
    ],
    [
      #if first_sheet [
        #figure(
          image("../artifacts/" + folder_name + "/03_clef/03_full_clef_overlay.jpg", width: 100%),
          caption: [
            Full clef overlay for #display_name. The detected clef label is shown
            for each staff after template matching.
          ],
        ) <fig:twinkle-clef-overlay>
      ] else [
        #figure(
          image("../artifacts/" + folder_name + "/03_clef/03_full_clef_overlay.jpg", width: 100%),
          caption: [
            Full clef overlay for #display_name. The detected clef label is shown
            for each staff after template matching.
          ],
        )
      ]
    ],
  )

  #v(1em)

  #if first_sheet [
  #pagebreak()
    == Notehead Detection Images - #display_name

    #grid(
      columns: (1fr, 1fr),
      gutter: 1em,
      [
        #figure(
          image("../methodology_figures/02_notehead_kernel.jpg", width: 100%),
          caption: [
            Elliptical notehead kernel used for morphological opening. Its
            diameter is scaled relative to staff spacing.
          ],
        ) <fig:twinkle-notehead-kernel>
      ],
      [
        #figure(
          image("../artifacts/" + folder_name + "/04_notes/02_morphological/staff_0_measure_0_notehead.jpg", width: 100%),
          caption: [
            Morphological opening result for #display_name on staff 0, measure 0.
            Filled elliptical notehead blobs remain while stems and residue are suppressed.
          ],
        ) <fig:twinkle-morph-open>
      ],
    )

    #v(1em)

    #figure(
      image("../artifacts/" + folder_name + "/04_notes/03_geometric_filtering/staff_0_measure_0.jpg", width: 70%),
      caption: [
        Geometric filtering result for #display_name on the same measure. Area,
        size, and aspect-ratio thresholds retain only plausible noteheads.
      ],
    ) <fig:twinkle-geometric-filter>

    #v(1em)
  ]
]

#let detection_comparison(sheet_file, folder_name, caption_body) = [
  #figure(
    grid(
      columns: (1fr, 1fr),
      gutter: 1em,
      [image("../music_sheets/" + sheet_file + ".png", width: 100%)],
      [image("../artifacts/" + folder_name + "/04_notes/06_full_notes_overlay.jpg", width: 100%)],
    ),
    caption: caption_body,
  )
]


#align(center + top)[
  #v(2in)
  
  #text(size: 16pt)[Sheet Music Player Project]
  #v(0.5em)
  #text(size: 18pt, weight: "bold")[Optical Music Recognition via Morphology] \  \

  #text(size: 12pt)[Daniel Young],
  #text(size: 12pt)[Joseph Petrasek],
  #text(size: 12pt)[Shivom Sharma]
  #v(1.0em)
  
  #text(size: 12pt)[COMPENG 4TN4 - Image Processing]
  
  
  #text(size: 12pt)[Date: April 3rd, 2026]
  
  #v(1.5in)
]
#pagebreak()
#align(center)[
  #text(size: 14pt, weight: "bold")[Abstract]
]

#v(0.5em)

// Abstract - not to exceed 1/2 page
#par[
  This project presents an image processing pipeline for Optical Music Recognition (OMR) that converts scanned sheet music in 
  PNG format into playable music in ABC format. The central challenge faced in OMR is that many symbols
  share visual geometry. Filled Noteheads resemble staff line fragments, beams resemble note stems, and the note stems themselves intersect 
  with practically every other structure on a score. Our approach utilizes the distinct morphological properties of each type of symbol; 
  staff lines are the widest horizontal structures, bar lines are the tallest vertical structures and noteheads share an elliptical shape that 
  is proportional to the spacing between staff lines. With these properties in mind, we utilize morphological kernels and connected component analysis 
  to distinguish between each symbol independently.\ This pipeline was evaluated on five melodies from `abcnotation.com`, of varying complexity. _Twinkle Twinkle Little Star_ was reproduced 
  exactly. _Mary Had a Little Lamb_ correctly worked with a one-flat key signature, and Frere Jacques correctly tracked stepwise 
  melodies with eighth notes. The Sailors' Hornpipe, a more complex piece, preserved proper contours in pitch, while _The Boys of 45 Reel_, showcased
  significant false detections of noteheads based on misclassified note fragments. All music was played on #link("https://editor.drawthedots.com/")[abcjs]. These results confirm that morphology-based OMR is effective for standard music sheets with simple sparse layouts, 
  however, classifying rhythm and beams remains a difficult limitation to overcome. 
]

#pagebreak()

= Technical Discussion


The pipeline consists of 8 key steps:

#figure(
  image("assets/Pipeline_Steps.png", width: 80%),
  caption: [Steps of the Pipeline],
  supplement: [Figure],
) <fig-pipeline_steps>


=== Staff Line Detection:
We start by converting the input image to grayscale and binarizing using otsu's (@fig:twinkle-otsu).
We then perform opening with a horizontal morphological kernel of size ($W times 1$) shown in (@fig:staffline-kernel) and parameterized by (@eq:kernel-dimensions).
Staff lines are horizontal and wide enough to pass through this kernel, the other features are suppressed (@fig:twinkle-staff-lines-only)
These lines share a common spacing size $S$ (@fig:twinkle-staff-overlay), which we use for further note/rhythm detection.


=== Staff Line Removal:
To remove the lines for use in note extraction and other analysis, we used an adaptive threshold to binarize the image, and take into account local illumination. Then we used horizontal opening and subtract from the created mask, within a band around the staff spacing ($plus.minus 0.2 S$) around each staff line. This avoids breaking the notehead when it overlaps with the staff line. We then perform closing using a small vertical kernel ($1 times K$) where $K in [3,7]$, it fills in the gaps left by the subtraction. We perform this operation twice, once targeting notes (@fig:twinkle-staff-removal-notes) and once for vertical bars (@fig:twinkle-staff-removal-bars)

=== Bar Line and Clef Detection
Bar lines are discovered by running a `find_bars()` over the bars mask. A tall vertical morphological close with the kernel shown in (@fig:barline-kernel) reconnects fragmented vertical ink into solid bar candidates (@fig:twinkle-bar-vertical-close). Vertical components spanning most of the staff's height and falling within the width limits of the kernel are then classified as bar lines, producing the final overlay shown in (@fig:twinkle-bar-overlay). In this stage we also look to the right and left of a located bar, observing for top and bottom dots indicating whether we must repeat a section of the staff. Their x positions were stored and used to split measures.

Clef detection was done using template matching in `clef_detection.py`. A small region on the left side of each staff is cropped, and the treble and bass templates shown in (@fig:clef-templates) are compared against that crop using normalized cross-correlation. We can view one example of a clef assignment as follows: (@fig:twinkle-clef-overlay).

#pagebreak()
=== Key Signature and Time Signature Detection
Key signature detection looks at the region between the clef and the first bar line. In `accidental_detection.py`, connected components are extracted and classified as sharps or flats based on their shape, using a similar method of template matching. We count the number and determine the pitch based on a mapping of sharp counts or flat counts. We ensure any duplicate detections are removed before determining the signature.
When classifying durations, we use the ratio $r$ of filled pixels in the round regions of ink, as well as whether it has a stem or beam attached. The time signature is read using Tesseract OCR. The region is split into top and bottom halves for the numerator and denominator of the signature,
scaled up and then passed to OCR.

=== Measure Splitting and Score Trees
Bar line positions are used in `measure_splitting.py` to divide each staff into measures. A small offset is applied so the clef and key signature are not included in the first measure. The detected elements are assembled into a structured format in `schema.py` through the `build_score()` helper. The schema defines `Score`, `Staff`, `Measure`, `BarLine`, `Clef`, and `Note` objects, and this structure is used by the later pitch, rhythm, and ABC export stages.


=== Note Detection, Rhythm and Pitch Mapping
Notes are detected in `note_detection.py` using blob detection on the notes mask. An elliptical morphological opening with the kernel shown in (@fig:twinkle-notehead-kernel) isolates candidate notehead blobs; a sample intermediate result is shown in (@fig:twinkle-morph-open). Blobs are then filtered based on size and shape using thresholds based on staff spacing, as shown in (@fig:twinkle-geometric-filter). Each detected note head is mapped to a pitch by comparing its vertical position to the staff lines, 
this position gets converted into a step value, and the key signature is applied to adjust for accidentals. Rhythm is estimated within `rhythm_detection.py` by looking for beams connected to note stems, we then use the heuristic as follows. 
(@table:duration-classification)

=== ABC Notation Export
The final score is converted to ABC notation in `abc_export.py`. The system writes out header information (key, time signature, etc), and then outputs each note with its pitch and duration based on the metadata gathered prior.
This file is then played on the abcjs website to test.


#pagebreak()

= Discussion of Results

The analysis below will focus on the five songs that were tested with the image processing pipeline, then speaking to the overall success in terms of the project's objectives:

=== Twinkle Twinkle Little Star (Key C, 4/4)
All notes across three staffs were detected with high to medium confidence, and arrived at the correct ABC output. We can see this detailed in Figures @fig:twinkle-comparison and @fig:ttls_abc_js. The clef was determined correctly, and correct durations were found for each note. We were able to cleanly extract the correct 4/4 time signature, and each measure contained the proper amount of expected notes when split.

=== Mary Had a Little Lamb (Key F, 4/4)
Mary Had a Little Lamb correctly preserved the melody and also correctly identified the single flat key signature. The side-by-side comparison in @fig:mary-comparison shows a clean final overlay with the expected note structure, while the ABC rendering is shown in Figure @fig:mary_had_a_little_lamb_output_abc_js.

=== Frere Jacques (Key C, 4/4)
Frere Jacques preserved the correct melody and motion. As shown in @fig:frere-comparison, the repeated C-D-E-C phrase was tracked cleanly, although some beamed passages introduced small note-count errors. The final ABC rendering is shown in @fig:frere_jacques_abc_js.

=== Sailors' Hornpipe (Key G, 4/4)
Sailors' Hornpipe was more complex, but still preserved the overall pitch shape. @fig:sailors-comparison shows that most simpler measures are handled well, while denser passages begin to reveal beam-related false positives. The final ABC rendering is shown in @fig:sailors_abc_js.

=== Boys of 45 Reel (Key D, 4/4)
The Boys of 45 Reel illustrates the main failure mode of the system. In @fig:reel-comparison, beam fragments are misclassified as high-pitch noteheads, leading to repeated false positives in dense eighth-note passages. The degraded final ABC rendering is shown in @fig:boys_of_the_reel_abc_js.


#pagebreak()


=== Staff Detection and Bar Detection
Staff detection worked on every song, and the kernel choice and parameters were capable of extracting the spacings consistently, as illustrated by @fig:twinkle-staff-lines-only and @fig:twinkle-staff-overlay. Similarly, bar detection worked on each song; based on the closing operations performed in @fig:twinkle-bar-vertical-close, we were able to consistently find single bars, double bars, begin-repeats and end-repeats in overlays such as @fig:twinkle-bar-overlay.
=== Clef and Key Signature Detection
Clef detection was correct in all five songs, showing that template matching was robust enough for the task, likely due to the unique shapes of each of these components, as shown by the templates in @fig:clef-templates and the sample detection in @fig:twinkle-clef-overlay. Accidental detection correctly counted accidentals for four out of five of the songs, which aligns with the cleaner outputs in @fig:mary-comparison and @fig:sailors-comparison, beginning to fall apart on the most complex piece test _The Boys of 45 Reel_ as seen in @fig:reel-comparison. This was likely due to tight spacing causing merged accidentals.
=== Note Detection and Pitch Assignment
Note detection correctly produced accurate center positions for quarter and half notes when sheets were uncluttered, such as in _Twinkle Twinkle Little Star_, _Mary Had a Little Lamb_ and _Frere Jacques_, as reflected in @fig:twinkle-morph-open, @fig:twinkle-geometric-filter, @fig:twinkle-comparison, and @fig:mary-comparison. Notes going across ledger lines (both above and below) worked in implementation.
The beam counting approach worked successfully on _Frere Jacques_, and partially on _Sailors' Hornpipe_, as seen in @fig:frere-comparison and @fig:sailors-comparison, but was not robust enough to function when the sheet became largely crowded, which is most apparent in @fig:reel-comparison.
=== ABC Output and Audio Playback
The ABC files that were generated by the pipeline were valid and playable using the abcjs website. This process was deterministic, and worked on all songs we tested with; representative outputs are shown in @fig:ttls_abc_js, @fig:mary_had_a_little_lamb_output_abc_js, @fig:frere_jacques_abc_js, @fig:sailors_abc_js, and @fig:boys_of_the_reel_abc_js.
= Takeaways
Overall, when analyzing the image processing pipeline as a whole, it was able to correctly play three out of five of the songs tested using pure
morphology and connected-component analysis. This is reflected most clearly by the strong matches in @fig:twinkle-comparison,
@fig:mary-comparison, and @fig:frere-comparison, which show that the pipeline is robust on simpler monophonic songs with sparse note
placement. It begins to fall apart when confronted with crowded music such as _The Boys of 45 Reel_, as shown in @fig:reel-comparison. When
comparing this to the project's overall objectives, we would still call this a success, since without the use of a complex
Neural Network  design, and relying only on kernel-based morphology and connected components, the system was able to consistently
produce correct results on simple songs. In the future, a larger dataset of templated symbols could help improve detection in crowded sheets
rather than relying primarily on region growing with thresholding.


#pagebreak()

= Results

// Generate all pipeline figures for each sheet music piece
#sheet_figures("twinkle_twinkle_little_star", "Twinkle Twinkle Little Star", first_sheet: true)
#pagebreak()
#sheet_figures("mary_had_a_little_lamb", "Mary Had a Little Lamb")
#pagebreak()
#sheet_figures("frere-jacques", "Frere Jacques")
#pagebreak()
#sheet_figures("sailors-hornpipe", "Sailors' Hornpipe")
#pagebreak()
#sheet_figures("boys-of-45-reel-the", "The Boys of 45 Reel")



#pagebreak()
#figure(
  grid(
    columns: (1fr, 1fr),
    gutter: 1em,
    [#image("../music_sheets/twinkle_twinkle_little_star.png", width: 100%)],
    [#image("../artifacts/twinkle_twinkle_little_star/04_notes/06_full_notes_overlay.jpg", width: 100%)],
  ),
  caption: [
    Input sheet music and final detection overlay for Twinkle Twinkle Little Star. All 40 notes across three staves were detected with predominantly high or medium confidence, and the recovered melody matches the expected phrase structure exactly.
  ],
) <fig:twinkle-comparison>

#figure(
  grid(
    columns: (1fr, 1fr),
    gutter: 1em,
    [#image("../music_sheets/mary_had_a_little_lamb.png", width: 100%)],
    [#image("../artifacts/mary_had_a_little_lamb/04_notes/06_full_notes_overlay.jpg", width: 100%)],
  ),
  caption: [
    Input sheet music and final detection overlay for Mary Had a Little Lamb. The key-signature flat was detected at confidence 0.850 and mapped correctly to F major, and the final whole note is classified correctly.
  ],
) <fig:mary-comparison>

#figure(
  grid(
    columns: (1fr, 1fr),
    gutter: 1em,
    [#image("../music_sheets/frere-jacques.png", width: 100%)],
    [#image("../artifacts/frere-jacques/04_notes/06_full_notes_overlay.jpg", width: 100%)],
  ),
  caption: [
    Input sheet music and final detection overlay for Frere Jacques. The main stepwise melodic contour is preserved, though minor false detections appear in some beamed passages.
  ],
) <fig:frere-comparison>

#figure(
  grid(
    columns: (1fr, 1fr),
    gutter: 1em,
    [#image("../music_sheets/sailors-hornpipe.png", width: 100%)],
    [#image("../artifacts/sailors-hornpipe/04_notes/06_full_notes_overlay.jpg", width: 100%)],
  ),
  caption: [
    Input sheet music and final detection overlay for Sailors' Hornpipe. Key G was detected correctly with an F-sharp confidence of 0.900, and most measures preserve the correct melodic contour despite a few denser beam-related errors.
  ],
) <fig:sailors-comparison>

#figure(
  grid(
    columns: (1fr, 1fr),
    gutter: 1em,
    [#image("../music_sheets/boys-of-45-reel-the.png", width: 100%)],
    [#image("../artifacts/boys-of-45-reel-the/04_notes/06_full_notes_overlay.jpg", width: 100%)],
  ),
  caption: [
    Input sheet music and final detection overlay for The Boys of 45 Reel. Dense beamed passages generate systematic false positives, visible as extra high-pitch labels above the staff where beam fragments are mistaken for noteheads.
  ],
) <fig:reel-comparison>
#pagebreak()

#figure(
  image("assets/mary_had_a_little_lamb_output_abc_js.png", width: 80%),
  caption: [Mary had a little lamb output on abcjs],
) <fig:mary_had_a_little_lamb_output_abc_js>

#figure(
  image("assets/frere_jacques_abc_js.png", width: 80%),
  caption: [Frere Jacques output on abcjs],
) <fig:frere_jacques_abc_js>

#figure(
  image("assets/ttls_abc_js.png", width: 80%),
  caption: [Twinkle Twinkle Little Star output on abcjs],
) <fig:ttls_abc_js>

#figure(
  image("assets/sailors_abc_js.png", width: 80%),
  caption: [Sailors' Hornpipe output on abcjs],
) <fig:sailors_abc_js>

#figure(
  image("assets/boys_of_the_reel_abc_js.png", width: 80%),
  caption: [Boys of 45 Reel output on abcjs],
) <fig:boys_of_the_reel_abc_js>
#pagebreak()
= Appendix

== Principal Equations

#eq-numbered(
  $
  W_"staff" = max(25, [W_"img"/12]), 
  D_"note" = [0.45 times S], 
  H_"bar" = [2.0times S]
  $,
  caption: [Scale-Relative Kernel Dimensions],
  label-name: <eq:kernel-dimensions>
)

Where $W$ is width, $D$ is diameter and $H$ is height

#eq-numbered(
$
"step" = round[(y_"bottom" - y_"note") \/ (S /3)]
$,
  caption: [Pitch step formula],
)\
Where $y_"bottom"$ is the y-coordinate of the lowest staff line, $y_"note"$ is the detected centroid of the notehead. The `round` function uses a threshold of `0.58`, this bias compensated for the fill on a note being concentrated at the bottom (notehead position). 

#eq-numbered(
  $ "confidence" = cases(
    "high"   & quad lr(|"step" - "floor"("step")|) <= 0.20,
    "medium" & quad lr(|"step" - "floor"("step")|) <= 0.40,
    "low"    & quad "otherwise"
  ) $,
  caption: [Pitch Step Confidence],
  label-name: <eq:confidence>
)
#v(0.5em)
#align(center)[
  #table-numbered(
  table(
    columns: (1.6fr, 1.2fr, 1.2fr, 1.2fr),
    align: center,
    stroke: 1pt,
    inset: 3pt,
    [*Fill Ratio $r$*], [*Stem*], [*Beam*], [*Duration*],
    [$r >= 0.55$],   [No],  [-],   [Whole],
    [$r < 0.55$],    [Yes], [No],  [Half],
    [$r >= 0.55$],   [Yes], [No],  [Quarter],
    [$r >= 0.55$],   [Yes], [Yes], [Eighth / Sixteenth],
  ),
  caption: [Duration Classification],
  label-name: <table:duration-classification>
) 
]


== Acknowledgments

The code for this project is fully available on GitHub: #link("https://github.com/ssh-vom/music_recognition")["https://github.com/ssh-vom/music_recognition"], 
the code shown in the _Program Listings_ section is a simplified version removing most of the visualizations, and giving the key algorithms.
The libraries used within the code were as follows:
- OpenCV: for morphology, connected component analysis, template matching and loading/saving of images.
- pytesseract / Tesseract OCR: Digit recognition for time signature extraction
- ABC notation standard: we referenced #link("https://www.abcnotation.com")["https://www.abcnotation.com"]
- NumPy: For array operations 
- Matplotlib: visualization of intermediate detections 

== Program Listings



=== `abc_export.py` - ABC Notation Generation
#set text(size: 6.5pt)
#set par(leading: 0.4em)
#set raw(tab-size: 2)
```python
def build_abc_text(score, *, title, reference_number, meter, unit_note_length, key, tempo_qpm):
    header = [
        f"X:{reference_number}",
        f"T:{title}",
        f"M:{meter}",
        f"L:{unit_note_length}",
        *([f"Q:1/4={tempo_qpm}"] if tempo_qpm is not None else []),
        f"K:{key}",
    ]
    return (
        "\n".join(header)
        + "\n"
        + notes_to_abc_body(score=score, meter=meter, key=key)
        + "\n"
    )

def notes_to_abc_body(score, *, meter, key):
    beats_per_measure = _meter_numerator(meter)
    key_accidentals = _abc_key_signature_accidentals(key)
    staff_lines = []

    for staff_index, _ in enumerate(score.staffs):
        measures = score.get_measures_for_staff(staff_index)
        if not measures:
            continue
        bars = score.get_bars_for_staff(staff_index)
        segments = ["|:" if _has_left_begin_repeat_flat(bars, measures) else "|"]

        for i, measure in enumerate(measures):
            notes = score.get_notes_for_measure(staff_index, i)
            tokens = _notes_to_measure_tokens(notes, beats_per_measure, key_accidentals)
            segments += [
                " ".join(tokens) if tokens else default_rest,
                _boundary_separator(measure.closing_bar),
            ]
        staff_lines.append(" ".join(segments))

    return "\n".join(staff_lines)

def _pitch_to_abc(pitch_letter, octave, key_accidentals):
    base = pitch_letter[0].upper()
    accidental_char = pitch_letter[1:2]
    key_accidental = key_accidentals.get(base, "")

    accidental = ""
    if accidental_char == "#" and key_accidental != "#":
        accidental = "^"
    elif accidental_char == "b" and key_accidental != "b":
        accidental = "_"

    if octave >= 5:
        apostrophes = "'" * (octave - 5)
        return f"{accidental}{base.lower()}{apostrophes}"
    return f"{accidental}{base}{',' * max(0, 4 - octave)}"
```


#set text(font: "Times New Roman", size: 12pt)
=== `note_detection.py` - Notehead Detection & Classification
#set text(size: 6.5pt)
#set par(leading: 0.4em)
#set raw(tab-size: 2)
```python
def find_notes(mask: MatLike, staff: Staff, measure: Measure, measure_index: int):
    """Detect noteheads using morphology and connected component analysis."""
    notehead_mask = _extract_notehead_mask(mask, staff.spacing)
    components = cv.connectedComponentsWithStats(notehead_mask, connectivity=8)
    centers = _filter_notehead_candidates(components, staff.spacing)
    centers = _merge_nearby_centers(centers, merge_distance)
    notes = _resolve_notes(centers, mask, staff, measure, measure_index)
    return notes

def _extract_notehead_mask(mask: MatLike, spacing: float) -> MatLike:
    """Elliptical open removes anything smaller than a notehead (stems, beams)."""
    diameter = max(3, int(round(spacing * 0.45)))
    if diameter % 2 == 0:
        diameter += 1
    kernel = cv.getStructuringElement(cv.MORPH_ELLIPSE, (diameter, diameter))
    opened = cv.morphologyEx(mask, cv.MORPH_OPEN, kernel)
    cleanup_kernel = cv.getStructuringElement(cv.MORPH_ELLIPSE, (3, 3))
    return cv.morphologyEx(opened, cv.MORPH_CLOSE, cleanup_kernel)

def _filter_notehead_candidates(components, spacing: float):
    """Filter by area, size, and aspect ratio to isolate noteheads."""
    count, _, stats, centroids = components
    min_area = spacing * spacing * 0.15
    max_area = spacing * spacing * 1.2
    min_size = int(round(spacing * 0.3))
    max_size = int(round(spacing * 1.1))

    centers = []
    for i in range(1, count):
        w, h = int(stats[i, cv.CC_STAT_WIDTH]), int(stats[i, cv.CC_STAT_HEIGHT])
        area = float(stats[i, cv.CC_STAT_AREA])
        aspect = w / float(h) if h > 0 else float("inf")
        cx = int(round(centroids[i][0]))
        cy = int(round(centroids[i][1]))
        
        if (min_area <= area <= max_area and 
            min_size <= w <= max_size and min_size <= h <= max_size and
            0.6 <= aspect <= 1.7):
            centers.append((cx, cy))
    return centers

def _classify_duration(mask: MatLike, cx: int, cy: int, spacing: float):
    """Classify note duration from fill ratio, stem presence, and beam detection."""
    filled = _detect_fill(mask, cx, cy, spacing)
    has_stem = _detect_stem(mask, cx, cy, spacing)

    if not filled and not has_stem:
        return "whole"
    if not filled and has_stem:
        return "half"
    if filled and has_stem:
        return "quarter"
    if filled and not has_stem:
        return _resolve_ambiguous_filled_note(mask, cx, cy, spacing)
    return None

def resolve_pitches(notes: list[Note], clef: Clef | None) -> None:
    """Convert step positions to pitch letters and octaves based on clef."""
    if clef is None or clef.kind not in CLEF_BASE_POSITIONS:
        return
    base_letter, base_octave = CLEF_BASE_POSITIONS[clef.kind]
    key_accidentals = _get_key_accidentals(clef.key_signature)

    for note in notes:
        letter, octave = _step_to_pitch(base_letter, base_octave, note.step)
        note.pitch_letter = f"{letter}{key_accidentals.get(letter, '')}"
        note.octave = octave
```


#set text(font: "Times New Roman", size: 12pt)
=== `staff_detection.py` - Staff Line Detection & Removal
#set text(size: 6.5pt)
#set par(leading: 0.4em)
#set raw(tab-size: 2)
```python
def find_staffs(image: MatLike) -> tuple[list[Staff], MatLike, MatLike]:
    """Pipeline: grayscale -> binarize -> extract horizontal lines -> group into staffs."""
    gray = to_gray(image)
    binary = binarize(gray)
    line_mask = extract_horizontal_lines(binary)
    centers = find_line_centers(line_mask)
    staffs = group_into_staffs(centers, line_mask, binary.shape)
    return staffs, binary, line_mask

def extract_horizontal_lines(binary: MatLike) -> MatLike:
    """Wide horizontal kernel removes anything shorter than staff lines."""
    image_width = binary.shape[1]
    kernel_width = max(25, int(image_width / 12))
    kernel = cv.getStructuringElement(cv.MORPH_RECT, (kernel_width, 1))
    return cv.morphologyEx(binary, cv.MORPH_OPEN, kernel)

def find_line_centers(line_mask: MatLike) -> list[int]:
    """Count pixels per row; staff lines have much higher counts than gaps."""
    row_strength = np.sum(line_mask > 0, axis=1).astype(np.float32)
    peak = float(np.max(row_strength))
    candidate_rows = np.flatnonzero(row_strength >= peak * 0.35)
    return _cluster_rows(candidate_rows)

def group_into_staffs(line_centers: list[int], shape: tuple) -> list[Staff]:
    """Group 5 lines into staffs; reject candidates with uneven spacing."""
    staffs = []
    gap_count = 4  # 5 lines -> 4 gaps
    i = 0
    
    while i + 5 <= len(line_centers):
        candidate = line_centers[i:i + 5]
        gaps = [candidate[j + 1] - candidate[j] for j in range(gap_count)]
        mean_gap = sum(gaps) / gap_count

        tolerance = max(3, mean_gap * 0.25)
        if not all(abs(g - mean_gap) <= tolerance for g in gaps):
            i += 1
            continue

        lines = [StaffLine(y=y, x_start=x0, x_end=x1) for y, (x0, x1) in ...]
        pad = 0.5 * mean_gap
        top = max(0, int(candidate[0] - pad))
        bottom = min(shape[0] - 1, int(candidate[-1] + pad))
        staffs.append(Staff(lines=lines, spacing=mean_gap, top=top, bottom=bottom))
        i += 5
    return staffs

def erase_staff_for_notes(gray: MatLike, staffs: list[Staff]) -> tuple[MatLike, MatLike]:
    """Adaptive threshold, then subtract staff lines only within narrow bands."""
    inverted = cv.bitwise_not(gray)
    bw = cv.adaptiveThreshold(inverted, 255, cv.ADAPTIVE_THRESH_MEAN_C, 
                               cv.THRESH_BINARY, 15, -2)
    
    kernel_width = max(1, bw.shape[1] // 30)
    structure = cv.getStructuringElement(cv.MORPH_RECT, (kernel_width, 1))
    staff_reconstruction = cv.dilate(cv.erode(bw, structure), structure)

    allowed = _staff_removal_band_mask(bw.shape, staffs)  # +-0.2*spacing around lines
    result = cv.subtract(bw, cv.bitwise_and(staff_reconstruction, allowed))
    processed = _repair_slits(result, staffs)
    return bw, processed

def _repair_slits(ink: MatLike, staffs: list[Staff]) -> MatLike:
    """Close vertical gaps left when stems cross staff lines using 1xN vertical kernel."""
    h, w = ink.shape[:2]
    repair_mask = np.zeros((h, w), dtype=np.uint8)
    
    for staff in staffs:
        band = max(1, int(round(staff.spacing * 0.3)))
        for line in staff.lines:
            y0, y1 = max(0, line.y - band), min(h, line.y + band + 1)
            repair_mask[y0:y1, line.x_start:line.x_end] = 255
    
    kernel = cv.getStructuringElement(cv.MORPH_RECT, (1, 3))
    repaired = cv.morphologyEx(ink, cv.MORPH_CLOSE, kernel)
    return np.where(repair_mask > 0, repaired, ink).astype(ink.dtype)
```


#set text(font: "Times New Roman", size: 12pt)
=== `main.py` - Entry Point
#set text(size: 6.5pt)
#set par(leading: 0.4em)
#set raw(tab-size: 2)
```python
def main():
    """Process all sheet music images in the music_sheets/ directory."""
    music_dir = Path("./music_sheets")
    image_paths = sorted(music_dir.glob("*.png"))
    for image_path in image_paths:
        run_pipeline(str(image_path), show_windows=False)
```


#set text(font: "Times New Roman", size: 12pt)
=== `schema.py` - Score Tree Schema
#set text(size: 6.5pt)
#set par(leading: 0.4em)
#set raw(tab-size: 2)
```python
@dataclass
class Measure:
    x_start: int
    x_end: int
    y_top: int
    y_bottom: int
    staff_index: int
    notes: list["Note"] = field(default_factory=list)
    closing_bar: "BarLine | None" = None
    crop: MatLike | None = None

@dataclass
class Clef:
    staff_index: int
    kind: ClefKind | None
    x_start: int
    x_end: int
    y_top: int
    y_bottom: int
    key_signature: KeySignature
    time_signature: TimeSignature
    key_header_glyphs: list["Accidental"] = field(default_factory=list)

@dataclass
class Note:
    kind: NoteKind
    staff_index: int
    measure_index: int
    center_x: int
    center_y: int
    step: int
    step_confidence: StepConfidence | None = None
    pitch_letter: str | None = None
    octave: int | None = None
    duration_class: DurationClass | None = None

@dataclass
class Score:
    image_path: str
    sheet_image: MatLike
    staffs: list[Staff]
    measures: list[Measure]
    bars: list[BarLine]
    notes: list[Note]
    clefs: dict[int, Clef]
    clef_detections: dict[int, ClefDetection]

    def get_measures_for_staff(self, staff_index: int) -> list[Measure]:
        return [m for m in self.measures if m.staff_index == staff_index]

def build_score(*, image_path: str, sheet_image: MatLike, staffs: list[Staff],
                bars: list[BarLine], clefs_by_staff: dict[int, Clef],
                clef_detections: dict[int, ClefDetection],
                measures_map: dict[int, list[Measure]],
                measure_crops: dict[int, list[MatLike]]) -> Score:
    # attaches crops + closing bars, then returns one flat Score object
    ...
```


#set text(font: "Times New Roman", size: 12pt)
=== `pipeline.py` - Orchestration
#set text(size: 6.5pt)
#set par(leading: 0.4em)
#set raw(tab-size: 2)
```python
def run_pipeline(image_path: str, show_windows: bool = False) -> Score:
    """Main OMR pipeline: staff detection → bar/clef detection → 
        measure splitting → note detection → ABC export."""
    raw_bgr = cv.imread(filename=image_path)
    
    # 1. Detect staff lines and compute spacing S
    staffs, binary, line_mask = find_staffs(raw_bgr)
    gray = to_gray(raw_bgr)
    
    # 2. Create masks: adaptive threshold for notes, global binary for bars
    notes_mask_raw, notes_mask = erase_staff_for_notes(gray, staffs)
    bars_mask = erase_staff_for_bars(binary, staffs)
    
    # 3. Detect bar lines and clefs
    bars = find_bars(image=bars_mask, staffs=staffs)
    clefs_by_staff = extract_clef_regions(staffs)
    for staff_index, crop in crop_clef_regions(clefs_by_staff, raw_bgr, notes_mask).items():
        clefs_by_staff[staff_index].kind = detect_clef(crop).clef
    
    # 4. Analyze key/time signatures from first staff header
    header = _analyze_first_staff_header(clef_crops, raw_crops, clefs_by_staff, staffs, bars)
    
    # 5. Split into measures
    measures_map = split_measures(bars=bars, staffs=staffs)
    measure_crops = crop_measures(measures_map, notes_mask)
    
    # 6. Build score and detect notes per measure
    score = build_score(image_path, raw_bgr, staffs, bars, clefs_by_staff, measures_map, measure_crops)
    _populate_notes(score)  # detects notes, refines rhythms, resolves pitches
    
    # 7. Export to ABC
    write_abc_file(score, output_path, title=..., meter=..., key=...)
    return score

def _populate_notes(score: Score) -> dict:
    """Detect notes in each measure, classify durations, resolve pitches."""
    for staff_index, staff in enumerate(score.staffs):
        clef = score.clefs.get(staff_index)
        for measure_index, measure in enumerate(score.get_measures_for_staff(staff_index)):
            if measure.crop is None:
                continue
            detected_notes, _ = find_notes(mask=measure.crop, staff=staff, 
                                            measure=measure, measure_index=measure_index)
            detected_notes = refine_beamed_durations(mask=measure.crop, 
                                                      notes=detected_notes, staff=staff)
            resolve_pitches(detected_notes, clef)
            measure.notes = detected_notes
            score.notes.extend(detected_notes)
```


#set text(font: "Times New Roman", size: 12pt)
=== `bar_detection.py` - Bar Line Detection
#set text(size: 6.5pt)
#set par(leading: 0.4em)
#set raw(tab-size: 2)
```python
def find_bars(image: MatLike, staffs: list[Staff]) -> list[BarLine]:
    """Detect bar lines in all staffs using vertical morphology."""
    all_bars = []
    for staff_idx, staff in enumerate(staffs):
        all_bars.extend(_find_staff_bars(image, staff, staff_idx))
    return sorted(all_bars, key=lambda b: (b.staff_index, b.x))

def _find_staff_bars(image: MatLike, staff: Staff, staff_idx: int) -> list[BarLine]:
    """Vertical close reconnects bar segments broken by staff erasure."""
    roi = image[staff.top:staff.bottom+1, :]
    left_skip = int(round(const.BAR_SEARCH_LEFT_SKIP_FRAC * staff.spacing))
    work = roi[:, left_skip:]
    
    # Vertical close kernel: reconnects vertical bar strokes
    kernel_h = int(round(const.BAR_CLOSE_KERNEL_HEIGHT_FRAC * staff.spacing))
    kernel = cv.getStructuringElement(cv.MORPH_RECT, (1, kernel_h))
    joined = cv.morphologyEx(work, cv.MORPH_CLOSE, kernel)
    
    contours, _ = cv.findContours(joined, cv.RETR_EXTERNAL, cv.CHAIN_APPROX_SIMPLE)
    bars = _contours_to_bars(contours, staff, left_skip, staff_idx)
    bars = _merge_and_classify_pairs(bars, staff, staff_idx)
    _classify_repeat_markers(roi=roi, bars=bars, staff=staff, y0=staff.top)
    return bars

def _classify_repeat_markers(roi: MatLike, bars: list[BarLine], staff: Staff, y0: int):
    """Detect repeat dots around double bars (|: or :|)."""
    pair_gap = int(round(const.BAR_PAIR_GAP_FRAC * staff.spacing))
    for i in range(len(bars) - 1):
        left, right = bars[i], bars[i+1]
        if left.kind == "double_left" and right.kind == "double_right" \
           and right.x - left.x <= pair_gap:
            has_left = _has_repeat_dots_on_side(roi, staff, y0, left.x, "left")
            has_right = _has_repeat_dots_on_side(roi, staff, y0, right.x, "right")
            if has_right and not has_left:
                left.repeat = right.repeat = "begin"  # |:
            elif has_left and not has_right:
                left.repeat = right.repeat = "end"    # :|
```


#set text(font: "Times New Roman", size: 12pt)
=== `clef_detection.py` - Template Matching
#set text(size: 6.5pt)
#set par(leading: 0.4em)
#set raw(tab-size: 2)
```python
def detect_clef(clef_key_crop: MatLike) -> ClefDetection:
    """Classify clef as treble or bass using template matching."""
    treble_template, bass_template = _load_templates()
    roi = _prepare_roi(clef_key_crop)  # crop left portion, invert to ink-on-black
    
    # Letterbox match: scale template to fill ROI, center on white canvas
    treble_score, treble_rect = _letterbox_match(roi, treble_template)
    bass_score, bass_rect = _letterbox_match(roi, bass_template)
    
    # Multi-scale match: try template at several heights for robustness
    slide_treble = _multi_scale_match(roi, treble_template)
    slide_bass = _multi_scale_match(roi, bass_template)
    
    clef, confidence = _select_clef(treble_score, bass_score)
    return ClefDetection(clef=clef, confidence=confidence, ...)

def _letterbox_match(roi: MatLike, template: MatLike) -> tuple[float, tuple]:
    """Scale template to fill ROI while preserving aspect, return match score."""
    roi_h, roi_w = roi.shape[:2]
    th, tw = template.shape[:2]
    scale = max(min((roi_w - 1) / tw, (roi_h - 1) / th) * 0.99, 1e-6)
    new_w, new_h = max(1, int(round(tw * scale))), max(1, int(round(th * scale)))
    
    interp = cv.INTER_AREA if scale < 1.0 else cv.INTER_CUBIC
    resized = cv.resize(template, (new_w, new_h), interpolation=interp)
    
    # Center resized template on white canvas (same size as ROI)
    canvas = np.full((roi_h, roi_w), 255, dtype=np.uint8)
    y0 = max(0, (roi_h - new_h) // 2)
    canvas[y0:y0+new_h, 0:new_w] = resized[:min(new_h, roi_h-y0), :min(new_w, roi_w)]
    
    result = cv.matchTemplate(roi, canvas, cv.TM_CCOEFF_NORMED)
    return float(result[0, 0]), (0, y0, new_w, new_h)

def _multi_scale_match(roi: MatLike, template: MatLike) -> float:
    """Try template at multiple scales, return best match score."""
    roi_h, roi_w = roi.shape[:2]
    best_score = 0.0
    for scale_frac in (0.4, 0.6, 0.8, 1.0, 1.2):
        target_h = max(12, min(roi_h - 1, int(round(roi_h * scale_frac))))
        scaled = resize_to_height(template, target_h)
        scaled = fit_to_roi(scaled, roi_h, roi_w)
        if scaled.shape[0] < 4 or scaled.shape[1] < 4:
            continue
        result = cv.matchTemplate(roi, scaled, cv.TM_CCOEFF_NORMED)
        _, max_val, _, _ = cv.minMaxLoc(result)
        best_score = max(best_score, float(max_val))
    return best_score
```


#set text(font: "Times New Roman", size: 12pt)
=== `accidental_detection.py` - Key Signature Detection
#set text(size: 6.5pt)
#set par(leading: 0.4em)
#set raw(tab-size: 2)
```python
def detect_key_signature_accidentals(clef_key_crop: MatLike, staff, staff_index: int,
                                      x_start: int, x_end: int) -> list[Accidental]:
    """Detect sharps/flats in key signature region between clef and first bar."""
    key_roi = clef_key_crop[:, x_start:x_end]
    use_geometric = staff.spacing <= 8.5 or key_roi.shape[1] <= 24
    
    if use_geometric:
        matches = _detect_header_accidentals_geometric(key_roi, staff.spacing)
    else:
        matches = _match_templates_in_roi(cv.bitwise_not(key_roi), staff.spacing)
    
    return [Accidental(kind=kind, center_x=x_start+cx, center_y=cy, ...) 
            for score, cx, cy, kind in matches]

def _match_templates_in_roi(roi: MatLike, spacing: float) -> list[tuple]:
    """Multi-scale template matching for sharp and flat symbols."""
    sharp_tmpl, flat_tmpl = _load_templates()
    min_dist = max(4, int(round(spacing * 0.55)))
    candidates = []
    
    for kind, template in [("sharp", sharp_tmpl), ("flat", flat_tmpl)]:
        for frac in (0.35, 0.5, 0.65, 0.8):
            target_h = max(4, int(round(spacing * frac)))
            scaled = resize_to_height(template, target_h)
            scaled = fit_to_roi(scaled, roi.shape[0], roi.shape[1])
            result = cv.matchTemplate(roi, scaled, cv.TM_CCOEFF_NORMED)
            for score, cx, cy in _gather_peaks(result, threshold=0.5, min_dist=min_dist):
                candidates.append((score, cx, cy, kind))
    
    return _nms(candidates, min_dist)  # non-maximum suppression

def _detect_header_accidentals_geometric(roi: MatLike, spacing: float) -> list[tuple]:
    """Fallback: classify by counting tall vertical stroke clusters.
    Sharps have 2 clusters; flats have 1."""
    count, labels, stats, _ = cv.connectedComponentsWithStats(roi, connectivity=8)
    matches = []
    
    for i in range(1, count):
        area = int(stats[i, cv.CC_STAT_AREA])
        h = int(stats[i, cv.CC_STAT_HEIGHT])
        if area < spacing*spacing*0.18 or h < spacing*1.8:
            continue
        
        comp = (labels[...] == i).astype(np.uint8)
        tall_threshold = int(h * 0.75)
        tall_cols = [idx for idx, v in enumerate(np.sum(comp, axis=0)) if v >= tall_threshold]
        tall_clusters = _count_index_clusters(tall_cols)
        
        cx = int(stats[i, cv.CC_STAT_LEFT]) + int(stats[i, cv.CC_STAT_WIDTH])//2
        cy = int(stats[i, cv.CC_STAT_TOP]) + int(stats[i, cv.CC_STAT_HEIGHT])//2
        
        if tall_clusters >= 2:
            matches.append((0.9, cx, cy, "sharp"))
        elif tall_clusters == 1:
            matches.append((0.85, cx, cy, "flat"))
    return matches
```


#set text(font: "Times New Roman", size: 12pt)
=== `measure_splitting.py` - Measure Segmentation
#set text(size: 6.5pt)
#set par(leading: 0.4em)
#set raw(tab-size: 2)
```python
def split_measures(bars: list, staffs: list, *, 
                   left_header_spacings: float = 7.0) -> dict[int, list[Measure]]:
    """Divide each staff into measures using bar line positions."""
    barlines_by_staff = _group_barlines_by_staff(bars, len(staffs))
    measures_map = {}
    
    for staff_index, staff in enumerate(staffs):
        measures_map[staff_index] = _split_staff(
            staff=staff, staff_index=staff_index,
            staff_bars=barlines_by_staff[staff_index],
            left_header_spacings=left_header_spacings
        )
    return measures_map

def _split_staff(staff, staff_index: int, staff_bars: list,
                 left_header_spacings: float) -> list[Measure]:
    """Create measures between successive bar lines, skipping header region."""
    staff_right = max(line.x_end for line in staff.lines) + 1
    content_start_x = _staff_left(staff) + int(round(left_header_spacings * staff.spacing))
    
    usable_bars = [b for b in staff_bars if content_start_x < b.x < staff_right]
    if not usable_bars:
        return [Measure(x_start=content_start_x, x_end=staff_right, ...)]
    
    measures = []
    trim = max(1, int(round(0.25 * staff.spacing)))
    current_start = content_start_x
    
    for bar in usable_bars:
        if bar.kind != "double_left":  # skip left stroke of double bar
            measure = Measure(x_start=current_start, x_end=bar.x - trim, 
                              y_top=staff.top, y_bottom=staff.bottom, staff_index=staff_index)
            measures.append(measure)
        current_start = bar.x + trim
    
    # Final measure to staff end
    measures.append(Measure(x_start=current_start, x_end=staff_right, ...))
    return measures
```


#set text(font: "Times New Roman", size: 12pt)
=== `rhythm_detection.py` - Beam Detection
#set text(size: 6.5pt)
#set par(leading: 0.4em)
#set raw(tab-size: 2)
```python
def refine_beamed_durations(mask: MatLike, notes: list[Note], staff: Staff) -> list[Note]:
    """Update duration_class for notes connected by beams (eighth/sixteenth)."""
    if len(notes) < 2:
        return notes
    
    _, labels, _, _ = cv.connectedComponentsWithStats(mask, connectivity=8)
    
    # Group notes by connected component - beamed notes share component via stems
    note_components: dict[int, list[tuple[int, Note]]] = {}
    for i, note in enumerate(notes):
        comp_id = int(labels[note.center_y, note.center_x])
        if comp_id > 0:
            note_components.setdefault(comp_id, []).append((i, note))
    
    for comp_id, note_group in note_components.items():
        if len(note_group) < 2:
            continue
        beam_count = _detect_beam_count(mask, note_group, staff)
        if beam_count > 0:
            for idx, note in note_group:
                if note.duration_class == "quarter":
                    notes[idx].duration_class = "eighth" if beam_count == 1 else "sixteenth"
    return notes

def _detect_beam_count(mask: MatLike, note_group: list, staff: Staff) -> int:
    """Count horizontal beams connecting a group of notes."""
    notes = [n for _, n in note_group]
    spacing = staff.spacing
    
    # Determine majority stem direction
    stem_dirs = [_estimate_stem_direction(mask, n, spacing) for n in notes]
    beam_dir = "up" if sum(1 for d in stem_dirs if d == "up") > len(notes)//2 else "down"
    
    min_x = min(n.center_x for n in notes)
    max_x = max(n.center_x for n in notes)
    if max_x - min_x < spacing * 0.5:
        return 0  # Too close - likely duplicate detection
    
    # Find beam tips and scan horizontal density
    beam_y = min(_find_stem_endpoint(mask, n, spacing, beam_dir) for n in notes)
    padding = int(spacing * 0.5)
    band = mask[beam_y-padding:beam_y+padding, min_x:max_x]
    
    horizontal_density = np.sum(band > 0, axis=1)
    threshold = max(2, spacing * 0.15)
    peaks = _find_peaks(horizontal_density, threshold)
    
    beam_count = 0
    for local_y, _ in peaks:
        row_mask = mask[beam_y-padding+local_y, min_x:max_x] > 0
        runs = _find_ink_runs(row_mask)
        longest_run = max((length for _, length in runs), default=0)
        if longest_run > spacing * 0.8:  # Real beam covers most of span
            beam_count += 1
            if beam_count >= 2:
                break
    return min(beam_count, 2)
```


#set text(font: "Times New Roman", size: 12pt)
=== `utils.py` - Helper Functions
#set text(size: 6.5pt)
#set par(leading: 0.4em)
#set raw(tab-size: 2)
```python
def to_gray(image: MatLike) -> MatLike:
    """Convert BGR to grayscale if needed."""
    if len(image.shape) == 2:
        return image.copy()
    return cv.cvtColor(image, cv.COLOR_BGR2GRAY)

def resize_to_height(template: MatLike, target_h: int) -> MatLike:
    """Resize template preserving aspect ratio to target height."""
    th, tw = template.shape[:2]
    if th < 1 or target_h < 1:
        return template
    scale = target_h / th
    new_w = max(1, int(round(tw * scale)))
    new_h = max(1, int(round(th * scale)))
    interp = cv.INTER_AREA if scale < 1.0 else cv.INTER_CUBIC
    return cv.resize(template, (new_w, new_h), interpolation=interp)

def fit_to_roi(template: MatLike, roi_h: int, roi_w: int) -> MatLike:
    """Scale template to fit within ROI bounds."""
    th, tw = template.shape[:2]
    if th <= roi_h and tw <= roi_w:
        return template
    scale = max(min((roi_h - 1) / th, (roi_w - 1) / tw) * 0.99, 1e-3)
    new_w = max(1, int(round(tw * scale)))
    new_h = max(1, int(round(th * scale)))
    return cv.resize(template, (new_w, new_h), interpolation=cv.INTER_AREA)

def group_notes_into_events(notes: list[Note], x_tol: int) -> list[list[Note]]:
    """Group notes with similar x-positions into chord events."""
    ordered = sorted(notes, key=lambda n: (n.center_x, n.center_y))
    events: list[list[Note]] = []
    for note in ordered:
        if not events or abs(note.center_x - events[-1][0].center_x) > x_tol:
            events.append([note])
        else:
            events[-1].append(note)
    return events
```

