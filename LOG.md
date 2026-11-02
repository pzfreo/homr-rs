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

- **WASM SIMD128 in Node/V8: all 88 pages, 627 staffs, 38,076 steps, and
  228,456 token choices also agree**, with zero differing steps and zero length
  differences. Max/mean absolute attention errors are exactly the same as
  native Rust: **0.0003662109375 / 0.0000035342254462343814 pixels**. Each page's
  max and mean error statistics agree with native Rust's as well. This check
  ran the actual `wasm32-unknown-unknown` cdylib, with its own greedy trajectory,
  through `bench/decoder.mjs rust check`, not a native stand-in.

All requested implementation and corpus checks completed on the supplied
arm64 machine. The strict `/usr/bin/time -l` invocation remains blocked by the
sandbox's `kern.clockrate` restriction; the reported measurements use the
retired-instruction counter directly, as explained above. Other CPU targets
and browser engines remain unverified; no cross-platform parity is inferred
from the Node/V8 run. All data and intermediate reports remain in scratch.
Changes were committed on `cx/decoder`; nothing was pushed or posted.

## Decoder round 2: profile before optimization

Built detached `origin/main` (d6617e8) in scratch `decoder-main`, with separate
native and WASI targets. All requested stage checks were run on main and HEAD;
correctness output agrees after removing only live instruction counts. TREE
contours: 948/948. EXTERNAL against the supplied TREE references: the same 641
matching / 307 differing masks on both builds; no reference was replaced.
Detailed final counts and paired pipeline benchmarks are recorded below.

Added optional `decoder-profile` instrumentation and a benchmark-only Node
N-API counter bridge (`bench/counter.c`). The native build reads
`proc_pid_rusage` directly; the WASM build synchronously imports that same
counter from Node at phase boundaries. Allocation/free accounting uses a
feature-gated allocator wrapper; nested phases subtract child time. An empty
guard loop calibrates counter/boundary overhead (about 6,845 native / 10,002
WASM retired instructions per span). Instrumentation is absent in normal
builds. Allocations were below the roughly 0.3M/step calibration resolution;
their raw spans were 2.87M native / 3.74M WASM, mostly counter overhead.

Initial profile on all 123 steps of the four staffs in angelus_0, including
staff setup, M retired instructions per step (exclusive calibrated spans):

| component | native | WASM |
|---|---:|---:|
| cross K/V staff setup | 41.94 | 162.10 |
| self QKV matmul | 1.52 | 5.77 |
| self output matmul | 1.01 | 3.86 |
| cross Q matmul | 0.51 | 1.92 |
| cross output matmul | 1.02 | 3.84 |
| FF input matmul | 4.05 | 15.39 |
| FF output matmul | 1.80 | 7.38 |
| heads matmul | 0.05 | 0.19 |
| activation quantization | 0.55 | 3.13 |
| attention scores | 12.31 | 65.00 |
| softmax/exp | 0.92 | 7.13 |
| weighted V | 4.33 | 52.04 |
| norms | 0.49 | 0.74 |
| sigmoid + GELU/Erf | 1.06 | 3.55 |
| centroid (Einsum scores already shared) | 0.62 | 0.76 |
| allocations/free | <0.3 | <0.3 |
| copies | 0.48 | 0.26 |

Sum excluding the residual/instrumentation bucket: 72.90M native / 333.07M
WASM, close to the uninstrumented paired 72.35M / 333.29M measurements. The
instrumentation bucket adds 6.19M / 8.22M and is not decoder work. There is no
separate attention-output Einsum: its identical probabilities are shared.

The largest cost is the initial 16 cross K/V matrices over 1,280 positions:
5.37 billion integer MACs per staff, or 174.6M MACs per benchmark step when
amortized over this short page. This is additional to steady-step matmuls.
WASM uses widening i16 dot products, not a hardware byte-dot instruction.
First optimization priority is this batch computation; next are the 117M
WASM instructions in attention scores and weighted sums. No float optimization
has been applied yet.

### Integer projection optimization

The measured first bottleneck was reduced with 4-row by 4-column register
blocking for context projections. Each quantized input block and weight block
is reused for sixteen output accumulators. WASM now multiplies signed bytes
with `i16x8.extmul` and widens pairwise sums instead of first extending both
operands to i16 and using a general i16 dot. Where weights exclude -128,
combining the low/high products in i16 is safe: each combined magnitude is at
most 32,512. Models with -128 weights take the non-combining exact path.
Native NEON uses a matching 4x4 byte-dot tile. Tail rows/columns retain the
scalar/SIMD fallback; synthetic tests cover tails and the -128 overflow case.
No floating-point operation or reduction order changed.

Three paired load-subtracted runs, the same four staffs / 123 steps:

| | before M/step | after M/step | after G/staff |
|---|---:|---:|---:|
| Rust native | 72.35 | 57.37 | 1.764 |
| ORT native, one thread | 46.33 | 46.23 | 1.422 |
| Rust WASM | 333.29 | 213.56 | 6.567 |
| ORT-web WASM | 479.58 | 479.23 | 14.736 |

The first page's native and WASM checks retain all 123 exact token steps and
exactly the previous attention error statistics. Next, the measured attention
bottleneck is addressed without changing each dot product's addition order.

### Pipeline neutrality of f00d474

Detached main and HEAD used independent native/WASI release builds. Every
correctness line agrees; instruction counters were the only fields removed
when comparing output text (they necessarily vary from run to run).

| check | main | HEAD |
|---|---|---|
| check_page p411 ref411 | 158 same, 0 different | same |
| check_symbols p411 ref411 | 151 same, 0 different | same |
| check_staffs p411b sref_all | 627 same, 0 different | same |
| check_contours p411 | 948 same, 0 different | same |
| EXTERNAL=1 check_contours p411 | 641 same, 307 different | same |
| check_pre seg_all preref_all | 88 same, 0 different | same |
| check_xml p411b dec_all replay_all | 75 identical + 13 matching failures | same |
| compare ref411 | 151 same, 0 different | same |

The EXTERNAL differences are the existing mismatch between RETR_EXTERNAL and
these RETR_TREE references, identical down to each diagnostic on both builds.
No correctness expectation or reference changed, and no lint fix required
reversion on this corpus.

`bench/compare_pipeline.py` runs the original `rs_page.mjs` and `rs_pre.mjs`
on angelus_0 through angelus_5. Three paired (3-repetitions minus 1-repetition)/2
runs, measured by the same macOS instruction counter through Node N-API:

| six-page pass | main median | HEAD median | HEAD/main |
|---|---:|---:|---:|
| rs_page | 15.947835 G | 15.953024 G | 1.000325 |
| rs_pre | 27.081108 G | 26.800900 G | 0.989653 |

Page run ranges were 15.943–15.975 G main and 15.937–15.970 G HEAD; the +0.033%
median change is inside that spread. Stage 2 decreased by 1.035%. There is no
measured regression beyond noise.

### Attention layout, first exact-precision change

Keys are now channel-major with fixed stride, allowing four independent dot
products in SIMD lanes without changing the order of any accumulation. V stays
position-major; eight weighted output channels are held in registers while
visiting positions in their original order. WASM still performs its float64
multiply/add followed by float32 rounding for every FMA. Native uses NEON FMA.

Full corpus rerun on the frozen native and WASM artifacts after this change:
88 pages / 627 staffs / 38,076 steps / 228,456 choices per build, zero token or
length differences. Attention max/mean remain 0.0003662109375 /
0.0000035342254462343814 pixels. Both native and WASM-generated decoder.json
pass check_xml: 75 byte-identical + 13 failing alike, zero different.

Three paired runs: Rust native 56.128 M/step (1.726 G/staff), Rust WASM
200.912 M/step (6.178 G/staff); ORT native 46.347 M/step and ORT-web
480.219 M/step. This change is correct but only a modest further improvement.
The remaining WASM FMA promotion/narrowing overhead is being measured next.

### Strict SIMD128: widen cached values once

The second float-related change preserves the same per-term rounding. WASM
stores each already-float32 K/V value exactly as float64 once, then holds pairs
of rounded accumulators in float64 registers. Each multiply/add is still
narrowed to float32 and re-promoted immediately; no summation-order or precision
relaxation is introduced. Queries are promoted/broadcast once per head.

This cuts repeated conversions/shuffles in the hot loops, at the cost of
**doubling the cache element storage** in the strict WASM artifact (maximum
reserved K/V capacity approximately 164 MB instead of 82 MB). Native retains
float32 caches. This is a memory/performance tradeoff, not model quantization.

Full native and strict-WASM reruns after this change each cover 88 pages,
627 staffs, 38,076 steps and 228,456 token choices, with zero token or length
differences. Max/mean attention errors are unchanged at 0.0003662109375 /
0.0000035342254462343814 pixels. Both symbol sets pass check_xml: 75 identical
and 13 failing alike. Synthetic attention tile/tail tests also compare every
float bit against the scalar ordered oracle; the integer tests cover -128
weights and partial matrix tiles. All seven tests pass natively and in WASI.

Three paired runs measured 182.303M strict-WASM instructions/step (5.606 G per
staff) against 480.396M ORT-web, a 2.635x reduction. This does **not** meet the
3x target for strict SIMD128. The final measurements below repeat this test
against a fresh build of the round-1 commit as well.

### Optional fused SIMD artifact, and the strict target limitation

A separate build with `-C target-feature=+simd128,+relaxed-simd` uses explicit
`f32x4_relaxed_madd` for the ordered attention kernels. It retains float32 K/V
storage (about 82 MB maximum reserved capacity) and directly uses fused SIMD on
the validated ARM64 V8 host. The ordinary `+simd128` build remains the strict
float64-emulation fallback, not a silently relaxed build.

This is a conditional deployment choice. The [WebAssembly specification](https://www.w3.org/TR/wasm-core/)
allows relaxed multiply-add to be fused or unfused. A load-time arithmetic
probe rejects an obviously unfused host and the exported
`decoder_fused_simd_supported()` lets a host check before copying weights.
Passing that probe does not prove every possible engine's code generation:
validate the actual engine with the corpus, and otherwise select the strict
artifact. Other browsers/CPUs have not been certified by this Node/V8 run.
`bench/build_decoder_variants.py` builds separately labelled artifacts; the
benchmark's `--wasm` argument records the exact selected file.

Full native and optional-fused-WASM validation after the final change each
covers all 88 pages / 627 staffs / 38,076 steps / 228,456 token choices, with
zero differing tokens or lengths. Attention max/mean are unchanged:
0.0003662109375 / 0.0000035342254462343814 pixels. Native median error is zero,
p95 0.0000152587890625 pixels. Each retained float optimization was checked on
its frozen native and WASM artifacts separately; all their independently
produced symbol sets pass check_xml with 75 byte-identical + 13 matching
failures, zero different. Seven synthetic tests pass natively and under WASI
in both strict SIMD128 and fused relaxed-SIMD configurations.

The 3x speed target is reached on this fused artifact, **not** on strict
SIMD128. Required FMA emulation is the measured remaining blocker in the strict
build. It is not correct to replace it with ordinary float32 multiply/add.
That alternative was built only in a separate scratch worktree and **rejected**:
full native and WASM runs each produced 4,177 differing overlapping steps and
7 length differences (37,454 generated steps, versus the reference's 38,076).
The first divergence is app-2 / staff 3 / step 123 / pitch: 44 instead of 41;
winning margins 0.015042782 in the experiment, 0.009525299 in ORT. This is an
arithmetic perturbation followed by quantization-boundary changes and greedy
history divergence. Error pages were included, not exempted. The unfused code
is not part of the branch and no acceptance criterion was relaxed.

### Final component profile and paired measurements

The same calibrated exclusive counters, four staffs / 123 steps, M retired
instructions per step. Native is unchanged between the two WASM artifact
configurations. The residual instrumentation bucket is excluded; sums remain
within approximately 1% of the ordinary, uninstrumented paired benchmark.
Allocation figures close to zero are below the calibration resolution.

| component | native final | strict SIMD128 final | fused SIMD final |
|---|---:|---:|---:|
| cross K/V staff setup | 27.13 | 60.64 | 60.68 |
| self QKV matmul | 1.51 | 3.07 | 3.09 |
| self output matmul | 1.01 | 2.07 | 2.05 |
| cross Q matmul | 0.51 | 1.02 | 1.02 |
| cross output matmul | 1.01 | 2.05 | 2.11 |
| FF input matmul | 4.04 | 8.20 | 8.25 |
| FF output matmul | 1.78 | 3.73 | 3.80 |
| heads matmul | 0.05 | 0.10 | 0.10 |
| activation quantization | 0.54 | 3.19 | 3.52 |
| attention scores | 9.79 | 41.42 | 11.67 |
| softmax/exp | 0.96 | 7.17 | 7.69 |
| weighted V | 4.73 | 41.19 | 8.65 |
| norms | 0.49 | 0.76 | 0.74 |
| sigmoid + GELU/Erf | 1.06 | 3.54 | 3.59 |
| centroid (no repeated Einsum) | 0.62 | 0.76 | 0.76 |
| allocation/free | 0.32 | 0.31 | <0.3 |
| copies/cache conversion | 1.14 | 2.91 | 1.54 |

Strict attention still costs 82.61M per step, versus 20.31M for the explicit
fused path. The largest remaining item in the fast artifact is staff setup,
60.68M when amortized over this short page. Allocation/copy removal alone
could not close the strict build's gap to 3x.

For the final before/after table, freshly built round-1 commit d00b5f1 in
another detached scratch worktree, then rebuilt HEAD. All measurements use
the same four native-reference angelus_0 staffs and all 123 steps. Each entry
is the median of **three paired decode-minus-load runs**, including first-step
K/V preparation. ORT-web 1.30.0 and native ORT 1.31.0 use one thread. The
counter is proc_pid_rusage RUSAGE_INFO_V4, as before; time -l's sandbox failure
is unchanged. Nothing is estimated from wall time.

| engine/artifact | before M/step | after M/step | before G/staff | after G/staff |
|---|---:|---:|---:|---:|
| Rust native | 72.250 | 56.251 | 2.221677 | 1.729733 |
| ORT native | 45.723 | 46.376 | 1.405987 | 1.426056 |
| Rust WASM, strict SIMD128 | 332.978 | 182.320 | 10.239076 | 5.606345 |
| ORT-web, strict comparison batch | 478.177 | 479.056 | 14.703955 | 14.730976 |
| Rust WASM, optional fused SIMD | 332.978 | 119.545 | 10.239076 | 3.676007 |
| ORT-web, fused comparison batch | 478.177 | 481.482 | 14.703955 | 14.805579 |

Thus strict SIMD128 uses **2.628x fewer** instructions than its paired ORT-web
baseline, and optional fused SIMD uses **4.028x fewer**. Native Rust improved
22.1% versus round 1 but still uses 1.21x as many as native ORT. External ORT
baseline variation between batches is reported rather than attributed to the
Rust changes. The comparison is not a claim about every phone's wall time.

No models, corpora, benchmark dumps or survey output were committed. Profile
and benchmark harnesses live under bench; the original pipeline checks and
all existing reference expectations remain intact.

### Final verification and rejected experiment details

The rejected unfused experiment was also replayed to MusicXML on both native
and WASM results: 21 pages identical, 13 matching failures, **54 different**.
This further rules out replacing fused arithmetic merely because most early
steps look close. All 88 pages and all 627 staffs were visited in both rejected
runs; their different step totals resulted from greedy divergence, not skipped
data. The retained implementations still have zero differences.

For scale, the actual gated 465 decoder has 42.153M steady-step integer MACs:
8 blocks x 5.243M, plus 0.210M in the heads. Its gated attention outputs have
1,024 columns and the GEGLU input projection has 4,096 columns. Initial context
projections add the 174.6M amortized MACs/step on this benchmark noted above.
The profile therefore explains the measured total without attributing it to
unmeasured allocation costs.

Final required checks pass: cargo fmt --check, cargo clippy --all-targets --
-D warnings, cargo test --release (seven tests), and the full release build
for wasm32-unknown-unknown with +simd128. Additional Clippy checks passed with
all features, with strict WASM, and with optional relaxed SIMD; the same seven
synthetic tests pass in both WASI SIMD configurations. All retained float
versions independently retained the full corpus and the 75 + 13 XML result.
No lint expectation was weakened, no pipeline fix was reverted, and nothing
was pushed or posted.

## Segnet and encoder native-parity survey (cx/models)

Work in progress: record the 88 pages through plain homr's default CPU path
(arm64 ORT 1.31.0, OpenCV 4.11), preserving batches of eight segnet tiles and
all 627 native staff canvases. Reuse the completed decoder-native context
fixtures and replay their symbols while collecting encoder inputs. Fixtures,
class maps, contexts, graph dumps and per-page/per-staff JSON stay in scratch.
Reusable recording/comparison tools are under tools/ and bench/.

Headless Chrome with the requested flags aborted with SIGABRT before CDP
became ready, both in a direct DOM probe and in bench/models_chrome.mjs.
No WebGPU measurements are inferred from WASM. The Chrome harness serves local
fixtures and ORT 1.30.0, requests an actual GPU adapter, and uses the same
comparison loop as Node. Nothing is pushed or posted.

Unattended execution decision: the full WASM corpus uses four threads, matching
Partsong's cross-origin-isolated setting (min(4, hardwareConcurrency)). Preserve
the initial one-thread four-page sample separately; do not merge its instruction
counts into four-thread results. Interrupt the one-thread process through its
own tool session before continuing, keeping CPU parallelism modest. Native
reference options remain homr's defaults. No answers or permission requested.

### Arithmetic evidence and scope (measured samples, not corpus totals)

Both supplied models are float32; neither has int8/quantization, Softmax,
attention, or ReduceMean nodes. The encoder is a ConvNeXt-style convolution/MLP
encoder, not another attention transformer. Inferred static shapes resolve all
Conv/MatMul sizes:

| model | ops | learned parameters | all initializer bytes | dense MACs/input | dense FLOPs/input |
|---|---|---:|---:|---:|---:|
| segnet | Conv 31, Relu 27, Add 8, Resize 5, MaxPool 1, Sub/Div 1 each, Shape 5, Concat 9 | 14,323,142 | 57,292,672 | 8,508,211,200/tile | 17,016,422,400/tile |
| encoder | Conv 18, MatMul 31, LayerNormalization 18, Erf/Div 15 each, Mul 45, Add 62, Transpose 37, Reshape 1 | 12,572,128 | 52,789,668 | 24,337,612,800/staff | 48,675,225,600/staff |

Encoder totals include 1.437G convolution MACs and 22.901G matrix MACs. Its
655,360-element folded positional tensor contains repeated learned height/width
embeddings: 16 x 170 + 80 x 342 = 30,080 independent parameters, included in
the learned count. Scalar/shape constants are excluded. Segnet excludes normalization and shape constants. FLOPs
are mathematical direct dense counts (2/MAC), not executed instructions or an
assumption that native convolution uses that exact algorithm. Nonlinear ops,
normalization and memory movement are additional work.

Identical-native-input, isolated-operator probes on angelus_0 (segnet batch 8;
encoder staff 0) give max absolute errors:

| operator | segnet | encoder |
|---|---:|---:|
| Sub, Div | 0, 0 | Div 0 |
| Conv | 3.8146973e-6 | 2.3841858e-6 |
| depthwise Conv | absent | 3.7252903e-8 |
| MatMul | absent | 1.4305115e-6 |
| LayerNormalization | absent | 4.7916546e-7 |
| Erf | absent | 5.9604645e-8 |
| nearest Resize | 0 | absent |

These probes expose intermediate values and may inhibit whole-graph fusions;
they isolate arithmetic, not an additive attribution of final error. They do
not justify claiming every node of an operator family differs equally.

The pinned [ORT 1.30 SIMD helper](https://github.com/microsoft/onnxruntime/blob/v1.30.0/onnxruntime/core/mlas/lib/mlasi.h)
uses wasm_f32x4_add(wasm_f32x4_mul(...)) versus ARM64 vfmaq_f32. The
[WASM SGEMM kernel](https://github.com/microsoft/onnxruntime/blob/v1.30.0/onnxruntime/core/mlas/lib/wasm_simd/SgemmKernelWasmSimd.cpp)
uses this helper. Erf polynomial evaluation uses it too. The CPU layer norm
fallback uses ordered Welford updates; contraction of M2 += delta*delta2 and
output scale/bias also matters. WebGPU has different shader reductions/kernel
schedules, so no bit parity is inferred from its float32 storage.

The supplied machine reports NEON=true, SME=true, SME2=true, SVE=false through
Rust feature detection (scratch diagnostic). ORT's
[1.31 platform dispatch](https://github.com/microsoft/onnxruntime/blob/v1.31.0/onnxruntime/core/mlas/lib/platform.cpp)
selects KleidiAI SGEMM/convolution overrides when SME is available. Therefore
“copy NEON FMA” alone is not a sufficient plan: record the selected kernel,
packing/blocking and accumulation order, including SME, before porting. The
native wheel reports commit 402979a. The reference is this actual default
native path, not a newly selected portable CPU algorithm.

Fresh default-native encoder runs on all four angelus_0 staffs match the
reused contexts bit for bit, in three runs each. Native 1.30.0 also matches
those four contexts bit for bit. Native 1.31 one-thread samples match too.
Thus this sample isolates backend arithmetic rather than merely the version
number or native thread count. WASM 1.30 encoder with optimizations disabled
still differs (max 7.6293945e-6, mean 4.2591845e-7 over the four staffs), versus
all optimizations (max 6.1988831e-6, mean 4.2019768e-7). No matching stock setting
has been found. npm rejects onnxruntime-web@1.31.0 with ETARGET; no published
1.31 package was available to test. A Node import of the browser JSEP bundle
failed loading its WASM via fetch; that attempt produced no comparison data.

The four all-optimized one-thread WASM encoder sample contexts cause zero
changed tokens/steps/lengths in the Rust decoder (123 steps). This is only a
sample; the full greedy corpus comparison is still required.

Default-native optimized graphs are also recorded in scratch: segnet fuses 27
Conv+Relu pairs into FusedConv, retaining four Conv; encoder has 31 Gemm, 15
Gelu, one NhwcFusedConv and 17 Conv, plus 18 LayerNormalization. Reproducing
that fused arithmetic schedule matters; an unfused ONNX-op transcription is
not automatically the default CPU program. A diagnostic sequential float32
FMA recurrence matches **all 24,576** sampled native first-MLP matrix outputs
(64 rows x 384 columns); separate multiply/add differs in **15,607**, max
4.7683716e-7. This supports the FMA diagnosis even with the SME dispatch caveat.

Speed sampling uses only retired process instructions from proc_pid_rusage.
Default native sessions' worker spinning makes their counts large and variable
on this loaded machine, so they are retained as raw capture metadata, not a
model speed claim. For a meaningful kernel baseline, fresh one-thread native
1.31 sessions were warmed and measured three times: median 23.350628368G per
8-tile segnet batch (2.918828546G/tile) and approximately 3.782G/encoder staff
(four angelus_0 staff medians 3.789476112, 3.781003135, 3.782403137,
3.782195698G). Both one-thread samples preserve the default native reference
outputs exactly (segnet argmax, encoder all float32 values). No option changed
the target recording. The actual machine's SME2 support matters to this count;
SIMD128 on a phone cannot be assumed to have native SME's instruction width.

The exported encoder's expanded positional initializer is not a nonlearned
sinusoid: the actual bytes split into 170 channels constant across width and
342 constant across height, with no channels constant across both. This agrees
with homr's [factorized learned position embeddings](https://github.com/liebharc/homr/blob/main/training/architecture/transformer/encoder.py).
The inventory tool now checks these exact repetitions before counting 30,080
independent positional parameters. A future port can losslessly store the two
arrays (120,320 bytes) instead of the expanded 2,621,440-byte tensor; assembly
is copying, so it introduces no arithmetic drift. Total inferred learned
inference parameters: **12,572,128 encoder**, **14,323,142 segnet**.

### Planning instruction budget (not measured full ports)

All 88 native pages completed: 8,076 tiles, 627 staffs, 75 matching successful
native XML files and 13 matching native AssertionErrors. Direct dense arithmetic
averages **1,561.643 GFLOPs/page segnet** and **346.811 GFLOPs/page encoder**;
page sizes and staff counts vary (30–438 tiles/page).

Warmed one-thread ORT-web 1.30 samples, three counter runs each, have median
**71.805326636G per 8-tile batch = 8.975665830G/tile** and
**28.407695541G/encoder staff**. Every repeat's output bytes equal the warmed
output. The native one-thread baselines above use identical tensors. Four-thread
capture counters include worker overhead and are not substituted for these
one-thread arithmetic baselines.

A reusable stand-alone 2x16, K=96 ordered GEMM cost probe checks its 32 outputs
bit for bit against native first-MLP results. Three counter runs (100,000
repetitions = 307.2M MACs) have median **6.8541 instructions/MAC strict SIMD128**
versus **0.8012 fused relaxed SIMD** on this host. A 2x8 strict variant is worse,
7.2953/MAC. First samples include V8 tier-up; medians of three are reported.
This is a microkernel cost calibration, not a completed model port or a claim
that every native Conv/Gemm has the same accumulation schedule.

Multiplying those costs by the model inventory gives raw arithmetic budgets
of roughly 58G/tile + 167G/staff strict, or 6.8G/tile + 19.5G/staff fused.
For planning, allow packing, boundaries, normalization, GELU and copies:

| future exact Rust artifact | expected G instructions/tile segnet | expected G instructions/staff encoder | vs current one-thread ORT-web |
|---|---:|---:|---|
| strict SIMD128 with FMA emulation | 60–90 | 180–270 | about 6–10x more instructions |
| validated fused relaxed SIMD | 8–12 | 24–35 | roughly 0.85–1.35x current work |

These deliberately broad engineering estimates require actual ports and full
validation. The strict probe uses the decoder-style widened recurrence; an
exact implementation must also handle any f64-to-f32 double-rounding corner
cases, underflow and special values it admits. Fused SIMD must probe actual
fused behavior and fall back when absent: the [WASM specification](https://www.w3.org/TR/wasm-core/)
permits fused or unfused relaxed multiply-add. Its availability alone proves
no model parity. No decoder-style 4x speedup is promised for these already
optimized dense float32 models. GPU work is not measured by host retired
instructions and cannot be ranked by this CPU counter.

### Reproducing the WebGPU part outside this sandbox

From this checkout, with the supplied sibling scratch directories, run:

```sh
node bench/models_chrome.mjs ../models_native ../models_files ../models_webgpu ../music_files.txt ../ortbench/node_modules/onnxruntime-web/dist
```

This uses --headless=new --enable-unsafe-webgpu, a fresh profile and local
HTTP serving; all data stays local. It requires an adapter and actual GPU
queue submissions during **each model/page**, recording adapter and Chrome
identity. The EP list is WebGPU+WASM as in Partsong: unsupported nodes may run
on WASM, but a whole-model CPU fallback is rejected. It does not fabricate a
GPU speed estimate from CPU submission instructions. Then run:

```sh
python3 bench/decode_models.py ../plain/src/homr/transformer/decoder_pytorch_model_465-597144cab54c8f6d0f6c9619df5c5312694eadd6.onnx ../decoder_native_ref ../models_webgpu ../models_webgpu_tokens ../music_files.txt
python3 bench/compare_model_xml.py ../p411b ../models_webgpu_tokens ../replay_all ../models_webgpu_xml
```

The WebGPU harness could only be syntax-checked here; the two real Chrome
launch attempts both aborted before an adapter/session existed. No WebGPU
class/context/token/XML numbers are available, and no Chrome GPU parity is
claimed.

### Recommended route, ordered by calendar time and phone work

1. **If a service is acceptable, use the already exact native pipeline as the
   canonical result**, on the pinned reference runtime/hardware. Wrapping and
   integrating that existing pipeline is a days-scale engineering task; it
   removes model inference instructions from the phone, at the cost of upload,
   connectivity and service operation. Do not assume another ARM64 CPU's
   default dispatch is identical just because its architecture name matches.
   No service, upload or integration was implemented in this stream.
2. **For offline browser parity, prototype a targeted ORT arithmetic build
   before committing to two ports.** Start from the pinned native version and
   optimized graph semantics, replace WASM multiply/add with a correct fused
   primitive, mirror native Welford/GELU and Conv/Gemm accumulation/bias order,
   and use a validated relaxed-SIMD fused path plus a strict fallback. Budget
   roughly 1–2 engineer-weeks for an operator-level feasibility prototype,
   then several more for both full models, devices and packaging. This retains
   ORT's graph execution, weight packing and memory management. No existing
   stock setting/build tested here matches native; a version bump, global
   fast-math, more threads or enabling relaxed SIMD alone is not a parity fix.
3. **A fixed-graph Rust port is sound and gives ownership of the numerical
   program**, but budget approximately 2–4 weeks for the encoder and 3–6 for
   segnet, plus device validation (engineering estimates, not delivery dates).
   Start with the encoder: its isolated probes identify all required arithmetic
   families, and the decoder's erf/FMA experience helps. Its large products
   are float32, so the decoder's int8 GEMM kernels are not directly reusable.
   Segnet is the larger arithmetic cost per corpus page. Preserve native
   convolution padding/layout/block schedules, Conv+Relu/bias fusion, pooling
   and nearest resize details; validate every model boundary, then the full
   greedy trajectory and XML. Both models must be covered for exact parity.
   Use the measured microkernel-derived budgets above; do not promise the
   decoder's 4x speedup. Portable strict SIMD likely makes phones substantially
   slower, while validated fused SIMD may stay near current WASM work.
4. **GPU remains the fast approximate option until separately certified.**
   Stock WebGPU float32 has no portable native-bit contract: [WGSL evaluation
   rules](https://www.w3.org/TR/WGSL/#floating-point-evaluation) permit rounding,
   contraction and reassociation choices. A carefully ordered, device-specific
   shader implementation could be checked on each supported stack; universal
   exactness would need deterministic integer/software float arithmetic for
   the sensitive operations, including division/sqrt/erf. That is a separate,
   larger project (many weeks/months), with unmeasured instruction/battery
   costs. GPU preview plus canonical native/WASM completion is sound; confidence
   margins, pixel thresholds or “only recheck uncertain tokens” do not prove
   exact parity through quantized greedy decoding and cannot replace checks.

This recommendation does not alter homr's target, add a model approximation,
implement a large port, or change Partsong. The current scope is measurement,
reusable local harnesses, and the exact XML checker's optional scratch export.

### Completed WASM segnet result

All 88 pages / 8,076 tiles / 826,982,400 tile pixels were run at batch 8
(last batch preserved), ORT-web 1.30.0, four WASM threads, all graph optimizations.
**14 class pixels differ on 11 pages: fraction 1.69290180783e-08 (1.69290180783e-06%).**
77 pages have no differing class pixels. Worst fraction: fix-six-1,
3 / 8,601,600 = 3.487723214e-7. Counts include tile overlap/padding, as the
model emits it, not just unique page pixels.

| differing page | pixels different | tile pixels | fraction |
|---|---:|---:|---:|
| angelus_1 | 1 | 5529600 | 1.808449074e-07 |
| app-3 | 2 | 22118400 | 9.04224537e-08 |
| app-4 | 1 | 44851200 | 2.229594749e-08 |
| fix-six-1 | 3 | 8601600 | 3.487723214e-07 |
| follow-3 | 1 | 8601600 | 1.162574405e-07 |
| manual-1 | 1 | 8601600 | 1.162574405e-07 |
| t-follow-3 | 1 | 8601600 | 1.162574405e-07 |
| t-manual-1 | 1 | 8601600 | 1.162574405e-07 |
| t-views-1 | 1 | 8601600 | 1.162574405e-07 |
| views-1 | 1 | 8601600 | 1.162574405e-07 |
| zoom-1 | 1 | 8601600 | 1.162574405e-07 |

All 11 differing-class pages were replayed through native homr with the
native decoder symbols, checking every staff canvas against the recorded
native canvas. **Zero changed canvases, call counts, XML files or error
outcomes**; the other 77 pages have identical class maps at the model boundary.
This is a recorded-symbol segmentation replay, not a claim that class maps
are exact. A combined replay with the newly decoded WASM encoder symbols is
also queued for those 11 pages.
