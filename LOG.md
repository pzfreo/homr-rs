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

## Native transformer decoder (2026-11-02)

### Implementation and decisions

- Added `decoder::{Decoder, Session, Step}`. `from_onnx` reads the downloaded
  upstream 465 decoder bytes at runtime. It supports this graph's architecture,
  with explicit vocabulary/matrix dimension checks, rather than pretending to
  be a general ONNX runtime. A checked protobuf reader handles initializers and
  node wiring; external tensor files and other architectures are rejected.
  No Rust dependencies were added; the model is never embedded or converted.
- Eight blocks implement quantized embeddings, self-attention, cross-attention,
  gated attention outputs, GEGLU feed-forward, norms, six logits, and homr's
  16-by-80 attention centroid. `Session::step` exposes logits for diagnosis;
  `generate` greedily feeds its own choices, includes EOS for checking, and
  stops at rhythm EOS or 608 steps. `Step::symbol` bridges to the existing
  vocabulary and coordinate/MusicXML pipeline.
- The reference is native arm64 ONNX Runtime 1.31.0, with default session
  options, running homr's real `ScoreDecoder.generate`. This is the environment
  that reproduces the existing `dec_all` recordings. Pin OpenCV 4.11.0.86 too.
  Recorded environment: numpy 2.5.4, Pillow 13.0.0. Other CPU architectures have
  a portable integer path but have not been checked against their local ORT
  default kernels; the executor retains the verified arm64 numerical profile.
- A misleading preliminary reference used the saved **Pyodide** staff canvases.
  Its 627 staffs / 38,637 steps all matched Rust's six greedy choices, but those
  canvases do not reproduce native homr's encoder inputs. That preliminary
  dataset is not the end-to-end acceptance reference. An initial diagnosis
  blaming ORT 1.31.1 was wrong: on the same first-page canvases, 1.31.0 and
  1.31.1 produced identical encoder outputs and transformer results. The final
  recorder runs full native pages and does not substitute saved canvases.
- Integer kernels: uint8 activation times int8 weights, exact i32 accumulation.
  Standard DynamicQuantizeLinear includes zero in min/max, divides by the
  float32 scale, rounds ties to even **before** adding the integer zero point,
  and saturates. Context projections and the six shared-quantization heads use
  those semantics. The fused DynamicQuantizeMatMul kernels in this CPU build
  use KleidiAI: signed asymmetric quantization, multiplier `255/(max-min)`,
  stored scale `1/multiplier`, an error-selected signed zero point, and
  ties-away-from-zero for activation quantization. Applying ordinary DQL to
  every matrix was wrong; early experiments diverged at app-2 staff 0 step 7.
  The two quantizers are tested separately. Integer scaling multiplies the
  two scales first, then scales the accumulated integer, then adds bias.
- Float details matter too: native Welford layer normalization, MLAS sigmoid,
  Erf/GELU and softmax polynomials, FMA and reduction order. The centroid
  averages layers before heads and uses Eigen's two four-lane accumulators for
  sums. WASM widens multiply/add to float64 and narrows once to approximate
  float32 FMA without a slow software `fmaf`; its corpus comparison measures
  the resulting error rather than assuming bit identity.
- Full context K/V are projected once per staff. The first context position is
  also projected with its **own** dynamic quantization range, and those 512 K/V
  values are appended at every later step, reproducing homr's context quirk.
  Preallocated position-major caches never copy previous positions. Attention
  probabilities are shared with the centroid calculation (the exported graph
  recomputes the same scores with Einsum). There is no per-token JS cache
  transfer. WASM SIMD128 and native NEON/dotprod process four weight columns
  together; the portable fallback retains exact integer arithmetic.
- `homr_rs.wasm` exports a small allocation/model/decode/free C ABI, without
  wasm-bindgen. Results are 32 bytes per step. The SIMD128 decoder-only cdylib
  is about 94 KiB; the 47 MB weights are supplied separately. These are library
  and benchmark interfaces; Partsong and upstream homr were not edited.
- `tools/record_decoder.py` copies images to scratch, runs homr natively,
  intercepts the original greedy loop without changing it, and records encoder
  outputs, all logits, tokens, attention and native symbols. It resumes whole
  completed pages. `check_decoder` independently generates Rust tokens and
  writes its own `decoder.json` for `check_xml`; any differing step reports
  page/staff/step/head and the winning margins from both implementations.
- Required formatting/Clippy checks failed on the pre-existing source. A
  separate commit formats it and fixes style lints without changing numerical
  operations or NaN behavior. OpenCV's intentionally rounded sine lookup table
  retains its original values (a narrowly scoped `approx_constant` allowance).
  The original MusicXML replay still checks 75 identical + 13 failing alike.

Kernel sources checked: [ORT dynamic quantized matmul](https://github.com/microsoft/onnxruntime/blob/main/onnxruntime/contrib_ops/cpu/quantization/dynamic_quantize_matmul.cc),
[MLAS quantization](https://github.com/microsoft/onnxruntime/blob/main/onnxruntime/core/mlas/lib/quantize.cpp),
[KleidiAI LHS packing](https://github.com/ARM-software/kleidiai/blob/main/kai/ukernels/matmul/pack/kai_lhs_quant_pack_qai8dxp_f32.c),
[MLAS float kernels](https://github.com/microsoft/onnxruntime/tree/main/onnxruntime/core/mlas/lib),
[ORT layer normalization](https://github.com/microsoft/onnxruntime/blob/main/onnxruntime/core/providers/cpu/nn/layer_norm_impl.cc),
and [ORT reductions](https://github.com/microsoft/onnxruntime/blob/main/onnxruntime/core/providers/cpu/reduction/reduction_ops.h).
MLAS's MIT notice is retained in `src/decoder/NOTICE.md`.

### Retired instructions

Benchmark: all four native-reference staffs of angelus_0, **123 steps including
EOS**, with each engine running the same contexts and fixed step counts.
Includes first-step context projections, all heads and coordinates. ORT's
fixed-length loop feeds chord after EOS, as the supplied ortbench does.
ONNX Runtime Web is **1.30.0**, SIMD, single thread; native ORT **1.31.0** also
uses one thread for an equal-thread benchmark. Reference recording still uses
default options. An initial default-thread benchmark counted substantial ORT
thread-pool spinning, so those numbers are not used here.

Three paired runs; table uses the median of decode minus that engine's
load-only instructions. This includes JIT/GC required for decoding. No wall-time
measurements are used. `/usr/bin/time -l` was attempted, but the sandbox denies
its `sysctl kern.clockrate` call before it can print instruction counts. The
harness's `--libproc` fallback reads the same macOS RUSAGE_INFO_V4 cumulative
retired-instruction counter from each live child at `BENCH_READY`, then lets it
exit. It does not estimate instructions from elapsed time.

| engine | net instructions (4 staffs) | per staff | per step |
|---|---:|---:|---:|
| Rust native | 8.900 G | 2.225 G | 72.35 M |
| ORT native, one thread | 5.699 G | 1.425 G | 46.33 M |
| Rust WASM SIMD128, Node | 40.994 G | 10.249 G | 333.29 M |
| ORT-web WASM SIMD, Node | 58.989 G | 14.747 G | 479.58 M |

Rust WASM uses **1.44x fewer** instructions on this page. Rust native uses
**1.56x more** than ORT native. This is an improvement for the browser, not the
13x int8-kernel speedup suggested by the original native-vs-web measurements.
The Rust executor still has substantial room for float attention and batched
context-projection optimization; no faster result is claimed.

Reproduce (point Python at the pinned environment, and ORT_MODULE at the
existing 1.30.0 package):

```sh
cargo build --release --bin bench_decoder
RUSTFLAGS='-C target-feature=+simd128' cargo build --release --target wasm32-unknown-unknown
python bench/measure_decoder.py --libproc --reps 3 \
  --python <reference-python> --ort-module <ort.node.min.mjs> \
  <decoder.onnx> <reference-dir> angelus_0/0 angelus_0/1 angelus_0/2 angelus_0/3
```

### Corpus acceptance results

Full native page recording reproduces the existing `dec_all` transformer
recordings **exactly on all 88 pages**, including coordinates and the 13 error
flags. All 627 staffs are present. The preliminary Pyodide-canvas recordings
mentioned above are not used for these figures.

- **Native Rust vs default native ORT: 627 staffs, 38,076 steps, all six tokens
  identical on every step** (228,456 token choices, including EOS steps).
  Zero differing steps, zero generation-length differences, and no remaining
  differing-step logit margins to explain. Checks run each implementation's
  actual greedy trajectory, without teacher forcing.
- Native attention: **max absolute error 0.0003662109375 pixels**, mean absolute
  error **0.0000035342254462343814 pixels**, across both coordinates of every
  compared step, including EOS and staffs that reach the 608-step cap.
- `check_xml` with `decoder.json` generated from Rust's own tokens and attention:
  **75 pages byte-identical, 13 failing alike in homr and Rust, zero different**.
  Coordinates are included in MusicXML's image-position comments; these were
  not stripped or given a tolerance.
- The separate ORT-web 1.30.0 vs default native ORT check on the benchmark's
  four staffs / 123 steps has identical tokens, but attention max difference
  **0.65496826171875 pixels**, mean **0.02671858934852166 pixels**. Rust's float
  differences are substantially smaller than this observed cross-build spread.
- `cargo fmt --check`, `cargo clippy --all-targets -- -D warnings`, and
  `cargo test --release` pass (five focused tests for malformed protobuf and
  integer quantization/SIMD arithmetic). The requested full release build for
  `wasm32-unknown-unknown`, with `-C target-feature=+simd128`, passes as well.

The independent WASM corpus check is still running; its completed figures will
be added below before this work is closed.
