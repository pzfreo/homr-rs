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
