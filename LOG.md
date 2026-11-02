# Overnight log

Goal: parity with homr on Partsong's corpus, stage by stage. Decisions and
results as they happen; newest at the bottom.

## Setup
- Corpus: Partsong's testdata page images (~70 music pages + app screenshots
  that contain scores) and the 10 Christ Is King pages.
- Each stage is checked against dumps of plain homr (commit 863eb3c) on the
  same pages: same inputs in, same outputs out.
- Speed is measured in retired instructions, by hand, not in CI.

## Stage 1: page analysis (segmentation masks -> staffs with symbols)
- Decision: test OpenCV ports directly against cv2 on every mask of every
  page (e.g. findContours), not only through homr's outputs: more cases, and
  a failure points at the routine rather than at a whole stage.
- findContours (RETR_TREE, CHAIN_APPROX_SIMPLE) ported from OpenCV's
  contours_new.cpp: identical to cv2 on 33 synthetic masks (holes, nesting,
  noise) and 336 real masks from 55 corpus pages (every contour, every point,
  same order).
- **Finding: Partsong runs OpenCV 4.11** (Pyodide's opencv-python 4.11.0.86),
  not the 5.0.0 that pip installs today. minAreaRect differs between them (box
  convention, and 4.11 converts the angle with a float32 multiply), so the
  default build follows 4.11 and the `opencv5` feature follows 5.0 (still
  151/151 on the 5.0 staff cases).
- **Finding: native OpenCV 4.11 on arm64 is compiled with fused
  multiply-add** (RotatedRect::points: all 1221 boxes of a page match FMA
  rounding), while Pyodide's (WebAssembly, no FMA) matches plain float32
  rounding, as does native 5.0. One staff case (fix-six-1) differed from native
  4.11 for that reason alone; Pyodide agrees with the port.
- **Decision: the reference is homr in Pyodide**, i.e. what Partsong actually
  runs. tools/pyodide_ref.mjs runs homr's detect_staffs_in_image in Pyodide on
  the masks dumped natively (masks come from the model, so they're inputs) and
  writes the same case files as tools/dump_staff.py.
- Decision: fitEllipse uses its own least-squares solver (QR) instead of a
  port of OpenCV's Jacobi SVD; homr uses the fit only through integer-truncated
  outlines and size filters. Checked by homr's outputs, not by the fit itself.
- Ported OpenCV's intersectConvexConvex (O'Rourke), pointPolygonTest,
  contourArea and ellipse2Poly (with OpenCV's sine table) for homr's polygon
  overlap test, and homr's create_rotated_bounding_boxes,
  create_bounding_ellipses, box merging, break_wide_fragments, the bar line
  dilation, combine_noteheads_with_stems and the bar line filter.
- Tooling bug found: serde_json's default float parsing is occasionally 1 ulp
  off; now parses exactly (float_roundtrip).
- Masks -> staff detection inputs (fragments with contours, clefs, bars):
  exact on the first 2 Pyodide-reference pages; full corpus running.
- **Masks -> staffs: exact on all 53 corpus pages** against homr in Pyodide
  (symbol detection: every fragment's box and contour, clefs, bar lines; then
  staff detection's grids). 3 more pages fail in homr itself ("No noteheads
  found").
- Decision: skip add_notes_to_staffs. It attaches notes to staffs, but
  nothing downstream reads them except debug images (staff_parsing only uses
  them under `if debug.debug`), so they can't change homr's output.
- Ported braces and grand staffs (brace_dot_detection.py, MultiStaff,
  Staff.merge), with OpenCV's elliptical structuring elements, erode/dilate,
  and fillPoly (Bresenham edges + scanline edge collection) for
  _trim_symbol_to_core_span. homr's "connections at bar lines / clefs" never
  fire in this pipeline (staffs only ever get notes attached), so only the
  "at lines" connections are ported.
- **Finding: last-bit differences came from the maths library.** homr in
  Pyodide uses musl's libm (tan, atan2...), native Rust used Apple's. Decision:
  use the `libm` crate (a Rust port of musl) everywhere; it's also what Rust
  uses on WebAssembly, so native tests and the browser agree.
- Page analysis end to end (masks -> MultiStaffs, grids compared bit for bit):
  first 56 pages: 48 exact before the libm fix, the remaining 8 were all last
  bit differences; full 4.11 corpus being checked.
- Speed: morphology was naive (16G instructions a page); now sliding windows,
  with prefix sums for binary masks: 3.4G for a whole page's analysis, of which
  1.6G is the brace image (5x35 elliptical dilation). TODO later: bit-packed
  rows.

### Stage 1 result
**Masks -> MultiStaffs: 158 of 158 corpus pages identical to homr in Pyodide**
(151 analysed, grids compared bit for bit, grand staffs and systems the same;
7 fail in homr - no noteheads / no staffs - and fail the same way in Rust).
Same on the narrower checks: 151/151 for symbol detection (every fragment box
and contour, clefs, bar lines) and 151/151 for staff detection.

## Stage 3: staff images (page image + staff -> the model's input canvas)
- Ported OpenCV 4.11's resize (uint8 bilinear, fixed point; the exact-2x
  area path too), getAffineTransform (6x6 LU as cv::solve), warpAffine
  (fixed-point remap tables, BORDER_CONSTANT), Subdiv2D (incremental
  Delaunay: quad-edges, locate, insert with flips, getTriangleList order),
  clipLine, the clipped Bresenham LineIterator and fillConvexPoly; and homr's
  prepare_staff_image, dewarping (span/optimal points, piecewise affine warp),
  remove_black_contours_at_edges_of_image and center_image_on_canvas.
- Checked: OpenCV's vectorised remap/resize paths give the same integers as
  the scalar code for uint8, so Pyodide's SIMD settings don't matter there.
- tools/pyodide_staffs.mjs records, in Pyodide, every staff image homr
  prepares (model call stubbed out), with the image before/after dewarping.
- **58 of 58 staff canvases identical** on the first 8 dev pages (9 pages,
  one has no staffs). Full corpus next.

## Stage 2: page image -> segmentation masks (around the model)
- Ported homr's autocrop (cvtColor BGR2GRAY fixed point, calcHist,
  threshold, morphologyEx close/erode, findContours RETR_EXTERNAL - added to
  the contour port and checked against cv2 on 40 random masks - contourArea,
  boundingRect), resize_image (**Pillow 12.2's bicubic resample**, Pyodide's
  version: fixed-point coefficients, horizontal then vertical pass), CLAHE
  (clip 1.0, 8x8 tiles, reflect-101 padding, float32 bilinear blend of tile
  lookup tables), segnet's tiling (white-padded 320px tiles, step 320),
  merge_patches (float32 average of class numbers, truncated), the noise
  filter (filter2D reflect-101, the uint8 wrap of the noise level) and
  make_lines_stronger.
- tools/dump_segnet.py runs segnet natively (the model only runs there) and
  keeps each tile's class map; tools/pyodide_pre.mjs makes the references in
  Pyodide from the decoded image and those class maps.
- **5 of 5 pages identical**: resized page, CLAHE image, merged class map,
  all five masks after filtering, and the filtered page image.
- Finding: native OpenCV's CLAHE differs from Pyodide's in 5-102 pixels a page
  (fused multiply-add again); the port matches Pyodide.
- Decision: the input is the decoded image (BGR). Partsong hands homr a PNG
  it made itself (lossless), so decoding isn't part of parity.

## Stage 4: transformer output -> MusicXML
- Ported homr's EncodedSymbol and durations (exact fractions), the voice
  clean-up (remove_duplicated_symbols: chord grouping, over-eager tuplet
  removal, lower staff without a clef, duplicate pitches - including homr's
  quirk that never replaces one - redundant clefs/keys/time signatures),
  parse_staffs (_ensure_same_number_of_staffs with its periodic layout search,
  voices, the upper/lower filter for non-grand staffs), the mapping of each
  symbol's position back to the input image (canvas offset, resizes, crops,
  inverse dewarp: point-in-triangle in float32 as numpy does it,
  invertAffineTransform), and the whole MusicXML writer (measures, chords,
  backups, voices rebalanced per measure, ties from slurs, tuplets, repeats,
  voltas) with a minimal element tree that serialises like ElementTree.
- tools/dump_decoder.py records natively what the transformer returned for
  each staff (the models only run there); tools/pyodide_replay.mjs replays it
  in Pyodide through homr's whole pipeline to its MusicXML.
- **6 of 6 pages: MusicXML byte-identical** (masks + preprocessed page ->
  page analysis -> staff images -> replayed symbols -> MusicXML, imgpos
  comments included). 2 more dev pages (fix-1, fix-six-1) crash in upstream
  homr itself with an AssertionError (a backup of zero duration) - a homr bug
  Partsong already works around (_build_note_chord_without_zero_rests).
- **Stage 3 on the whole corpus: 627 of 627 staff images identical**
  (88 music pages; the other 70 have no staffs or noteheads in homr itself).

## Speed (retired instructions, browser engine: Node/V8)
- Whole page from masks to MusicXML (page analysis, staff images, replayed
  transformer output, MusicXML), 6 pages: homr in Pyodide 38.1G a page; the
  port as WebAssembly first 5.9G (6.5x fewer), then after three
  optimisations **2.2G (17x fewer)**, still identical everywhere:
  - binary morphology on bit-packed rows (brace image 1.6G -> 0.3G a page);
  - staff images resize only the part of the page they crop (homr resizes the
    whole page for every staff): ~320M -> ~100M a staff;
  - contour scanning skips equal runs 8 bytes at a time.
- After these: stage 1 158/158 pages, stage 3 627/627 staff images, contours
  948/948 corpus masks + 40/40 RETR_EXTERNAL - unchanged.
- bench/py_page.mjs and bench/rs_page.mjs reproduce the comparison.
- Stage 2 (page image -> segnet tiles; segnet output -> masks), 5 pages, as
  WebAssembly: first 36.4G against Pyodide's 32.7G (slower: numpy/OpenCV/Pillow
  are native C there). After optimising the hot spots found natively with
  profile_pre/profile_crop (Pillow resize with whole-row loops; autocrop's
  morphology without per-shift allocations and with 8-at-a-time bit
  packing/unpacking; the noise filter's 3x3 without per-pixel reflection):
  **22.0G (1.5x fewer than Pyodide)**, still 5/5 identical; stage 1 still
  158/158. bench/py_pre.mjs and bench/rs_pre.mjs reproduce it.

## Whole corpus, stages 2 and 4
- 158 corpus pages; 88 have music (the other 70 have no staffs or noteheads
  in homr itself).
- **Stage 2: 88 of 88 pages identical** (segnet input tiles, and masks from
  the recorded segnet output, against homr in Pyodide).
- **Stage 4: 88 of 88 pages agree.** 75 give byte-identical MusicXML. On the
  other 13 (app-2, cam_0_exif1, files-1, fix-1, fix-six-1, gpufail-1, live-2,
  openpdf-1, pdf-1, scar-1, scar2-1, scar3-1, shot-2) upstream homr itself
  fails: `assert group_duration > Fraction(0)` in build_note_chord
  (music_xml_generator.py:795). The port fails at the same place on the same
  transformer output. dump_decoder.py now records these pages (decoder.json
  "error"), and check_xml expects the failure. Partsong avoids this crash with
  its own patch; under "homr-rs follows homr" the fix belongs upstream.
- So on our corpus the port matches homr at every stage, apart from the
  models, which stay in onnxruntime.

## Extension points for apps (for Partsong)
homr's behaviour is unchanged (stage 4 still 75 identical + 13 failing alike).
These let an app make the changes it used to make by patching homr's Python:
- `Staff.halves`: a grand staff keeps the two staffs merged into it.
- `Symbol.extra`: an app's own data, carried wherever the symbol is copied.
- `musicxml::generate_xml_with(voices, title, XmlOptions)`: `drop_zero_rests`
  (leave pitchless zero-length symbols out of a chord instead of homr's
  assertion) and `note_comments` (extra comments after each note's image
  position).
- `score::{Row, sorted_row, find_periodic_core, ensure_same_number_of_staffs,
  symbol_center}` and `PageMapping::apply` are public.

## Possible follow-ups (not started)
- Run the models from Rust as well with `ort` (pykeio/ort; retto uses it in
  WASM): would let the browser worker shrink to message passing, if it can
  use WebGPU and gives onnxruntime-web's results.
- Python bindings (PyO3) with a homr-shaped API: homr's results, faster, for
  existing Python users.
