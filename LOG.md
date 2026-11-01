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
