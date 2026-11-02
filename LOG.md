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

The one-/four-thread cross-check also completed: all four angelus_0 encoder
contexts are byte-identical across thread counts, and all seven batches on
each of the first four segnet pages (28 batches total) have byte-identical
class maps. This is a setting sample, not an unrun one-thread full corpus.

A native-context baseline run through the XML diagnostic confirms the old
acceptance result unchanged: **75 byte-identical + 13 matching failures, zero
different**, both with and without comments. On the four-staff WASM encoder
sample, zero tokens change but one XML file changes by an attention-derived
imgpos comment (y=658 vs 659); removing comments makes it identical. The final
corpus report retains byte differences and supplies comment-stripped counts
as an additional diagnostic, never as a relaxed pass criterion.

### Final encoder, greedy decoder and combined XML results

The full four-thread ORT-web WASM 1.30.0 run completed without skipping any
of the **88 pages / 627 staffs**, including all staffs on the 13 native failure
pages. Each staff uses the native [1,1,256,1280] input and outputs all
[1,1280,512] float32 values. All 627 contexts differ. Across 410,910,720 values,
387,918,136 (94.40448%) differ numerically. Global max absolute error is
**1.1920928955078125e-5** (fix-1, staff 10); the value-weighted mean is
**4.36081726078844e-7**. Per-staff maxima range 4.291534423828125e-6 to the
maximum above; per-staff means range 3.728967971738939e-7 to
5.998186930256111e-7 (largest mean: pdf-1 staff 8). Full per-staff max/mean
measurements remain in scratch models_wasm4/<page>/metrics.json, alongside
all saved context bytes. Native input canvases and normalized images are in
models_native/<page>/<staff>/; model hashes and native package versions are
recorded per page.

| backend | segnet differing tile pixels | worst-page fraction | encoder max abs | encoder mean abs |
|---|---:|---:|---:|---:|
| WASM 1.30.0, 4 threads | 14 / 826,982,400 | 3.487723214e-7 (fix-six-1) | 1.192092896e-5 | 4.360817261e-7 |
| Chrome WebGPU + WASM fallback (external Chrome 156 / Apple Metal) | 24 / 826,982,400 (16 pages) | 2.325148810e-7 (fix-six-1, 2 pixels) | 1.86e-5 | 1.32e-6 |

The WebGPU row was supplied by the maintainer after the original sandbox
survey, measured outside this sandbox on all 88 pages / 627 staffs. Worst
encoder case is pdf-1 staff 8. Its downstream token/XML results were not
supplied; the original local Chrome SIGABRT limitation remains.

The exact Rust decoder follows its own greedy trajectory on every context,
including EOS, all six heads, and each error-page staff. It never stops at the
first mismatch or teacher-forces native tokens.

| encoder backend -> Rust decoder | staffs with token/length changes | differing overlapping steps | differing head choices | staffs with changed length | extra steps | missing steps |
|---|---:|---:|---:|---:|---:|---:|
| WASM | 47 / 627 | 3,924 / 36,166 | 15,500 / 216,996 | 17 | 109 | 1,910 |
| WebGPU | not run | not run | not run | not run | not run | not run |

Native emits 38,076 steps; WASM-context decoding emits 36,275. The overlapping
step count is 36,166, with 109 extra and 1,910 missing. If missing/extra steps
are counted as unmatched events, there are 5,943 differing/unmatched events;
they are separated above so no tail is hidden in a denominator. First token
mismatch: app-2/staff 0/step 2, position head 3 versus native 1. Other first
mismatches quickly affect rhythms/pitches (app-2/staff 3/step 15 rhythm 100
versus 201; staff 6/step 1 rhythm 30 versus 28). The observed amplification is consistent with
quantization-boundary changes in this int8 decoder followed by greedy-history
divergence; this stream did not trace every changed quantized element. Isolated
first-step closeness is not a parity criterion.

Whole-page XML replay using native masks and these newly decoded symbols has
**74 byte-different results**, **10 different after removing comments**, and
**13 matching native AssertionError outcomes**. Of the 75 successful native
pages, only one remains byte-identical. The optional comment-stripped count
supplements the exact comparison; it is not a new acceptance criterion.

For the combined result, all 11 differing-segnet pages were replayed again
through native homr, this time with the WASM-context/Rust-decoder symbols.
All canvases and call counts still match; ten successful XML files match the
encoder-only Rust replay byte for byte, and fix-six-1 still has the same
native failure. The other 77 pages have identical segnet model boundaries.
Thus the combined boundary replay has the same **74 byte differences, 10
comment-stripped differences, 13 matching failures**. Segnet alone caused zero
XML changes; the measured final changes come from the encoder perturbations.
This is real model inference plus full greedy decoding and page/XML replay,
not an unbroken live browser app run; Chrome and actual phone GPU behavior
were not executed.

A second Node browser-bundle diagnostic with explicit JSEP bytes selected an
incompatible asyncify module and failed during initialization; it produced no
metrics and is not a tested setting. The full results above use the normal
pinned Node ORT-web WASM artifact, as in the existing ortbench tooling.

### Per-page model and decoder measurements

Encoder errors below summarize every staff on each page; the mean is the
mean over all its context values. Per-staff raw metrics and step trajectories
remain in scratch. Steps/heads columns compare overlapping greedy positions;
extra/missing counts are in the full scratch comparison and the table above.

| page | seg diff pixels | seg fraction | staffs | encoder max abs | encoder mean abs | token-different staffs | differing steps | differing heads |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| angelus_0 | 0 | 0 | 4 | 6.1988831e-06 | 4.2019768e-07 | 0 | 0 | 0 |
| angelus_1 | 1 | 1.80845e-07 | 12 | 6.6757202e-06 | 4.2671997e-07 | 0 | 0 | 0 |
| angelus_2 | 0 | 0 | 12 | 8.1062317e-06 | 4.2544037e-07 | 0 | 0 | 0 |
| angelus_3 | 0 | 0 | 4 | 6.1988831e-06 | 4.2699849e-07 | 0 | 0 | 0 |
| angelus_4 | 0 | 0 | 12 | 6.6757202e-06 | 4.2298719e-07 | 0 | 0 | 0 |
| angelus_5 | 0 | 0 | 12 | 7.6293945e-06 | 4.2921244e-07 | 0 | 0 | 0 |
| app-2 | 0 | 0 | 13 | 8.9406967e-06 | 4.6399938e-07 | 3 | 395 | 1267 |
| app-3 | 2 | 9.04225e-08 | 14 | 6.6757202e-06 | 4.3724488e-07 | 0 | 0 | 0 |
| app-4 | 1 | 2.22959e-08 | 28 | 7.0929527e-06 | 4.147907e-07 | 0 | 0 | 0 |
| cam_0_exif1 | 0 | 0 | 1 | 5.7220459e-06 | 4.166498e-07 | 1 | 42 | 150 |
| cam_0_exif6 | 0 | 0 | 4 | 5.2452087e-06 | 4.2579375e-07 | 0 | 0 | 0 |
| cam_1_exif6 | 0 | 0 | 6 | 6.6757202e-06 | 4.2801731e-07 | 0 | 0 | 0 |
| cam_2_exif3 | 0 | 0 | 1 | 5.7220459e-06 | 4.0176366e-07 | 0 | 0 | 0 |
| cam_2_exif6 | 0 | 0 | 4 | 5.7220459e-06 | 4.3062394e-07 | 0 | 0 | 0 |
| cam_3_exif6 | 0 | 0 | 5 | 6.4373016e-06 | 4.4279215e-07 | 0 | 0 | 0 |
| fail-1 | 0 | 0 | 2 | 6.5565109e-06 | 4.5509091e-07 | 1 | 2 | 2 |
| files-1 | 0 | 0 | 13 | 9.894371e-06 | 4.5698969e-07 | 4 | 225 | 525 |
| fix-1 | 0 | 0 | 13 | 1.1920929e-05 | 4.6158315e-07 | 1 | 78 | 327 |
| fix-six-1 | 3 | 3.48772e-07 | 12 | 7.1525574e-06 | 4.696005e-07 | 5 | 765 | 3303 |
| follow-1 | 0 | 0 | 8 | 6.6757202e-06 | 4.3438539e-07 | 0 | 0 | 0 |
| follow-2 | 0 | 0 | 8 | 6.1988831e-06 | 4.3292165e-07 | 0 | 0 | 0 |
| follow-3 | 1 | 1.16257e-07 | 1 | 4.7683716e-06 | 4.3333805e-07 | 0 | 0 | 0 |
| gpufail-1 | 0 | 0 | 13 | 7.6293945e-06 | 4.5842421e-07 | 6 | 425 | 1988 |
| huron_0 | 0 | 0 | 4 | 5.7220459e-06 | 4.3071412e-07 | 1 | 1 | 1 |
| huron_1 | 0 | 0 | 6 | 6.6757202e-06 | 4.2725036e-07 | 0 | 0 | 0 |
| huron_2 | 0 | 0 | 4 | 5.2452087e-06 | 4.3091443e-07 | 0 | 0 | 0 |
| huron_3 | 0 | 0 | 5 | 5.8412552e-06 | 4.4487032e-07 | 0 | 0 | 0 |
| img-000 | 0 | 0 | 6 | 6.6757202e-06 | 4.3650799e-07 | 0 | 0 | 0 |
| img-001 | 0 | 0 | 4 | 6.1988831e-06 | 4.3309483e-07 | 0 | 0 | 0 |
| img-002 | 0 | 0 | 6 | 8.1062317e-06 | 4.3695772e-07 | 0 | 0 | 0 |
| img-003 | 0 | 0 | 8 | 7.390976e-06 | 4.3187531e-07 | 0 | 0 | 0 |
| img-004 | 0 | 0 | 11 | 8.1062317e-06 | 4.3881892e-07 | 0 | 0 | 0 |
| img-005 | 0 | 0 | 4 | 6.1988831e-06 | 4.2504383e-07 | 0 | 0 | 0 |
| img-006 | 0 | 0 | 6 | 6.1988831e-06 | 4.2869283e-07 | 0 | 0 | 0 |
| img-007 | 0 | 0 | 7 | 6.6757202e-06 | 4.2950473e-07 | 0 | 0 | 0 |
| img-008 | 0 | 0 | 7 | 5.7220459e-06 | 4.3265743e-07 | 0 | 0 | 0 |
| img-009 | 0 | 0 | 6 | 6.6757202e-06 | 4.2976081e-07 | 0 | 0 | 0 |
| lib-2 | 0 | 0 | 4 | 6.1988831e-06 | 4.3276624e-07 | 1 | 71 | 210 |
| live-2 | 0 | 0 | 13 | 8.9406967e-06 | 4.6399938e-07 | 3 | 395 | 1267 |
| live-3 | 0 | 0 | 14 | 6.4373016e-06 | 4.3835991e-07 | 0 | 0 | 0 |
| live-4 | 0 | 0 | 28 | 7.1525574e-06 | 4.1378453e-07 | 2 | 2 | 2 |
| manual-1 | 1 | 1.16257e-07 | 5 | 6.6757202e-06 | 4.3595583e-07 | 0 | 0 | 0 |
| markfirst-1 | 0 | 0 | 12 | 7.6293945e-06 | 4.5606392e-07 | 0 | 0 | 0 |
| openpdf-1 | 0 | 0 | 13 | 8.5830688e-06 | 4.5580808e-07 | 6 | 295 | 1149 |
| pdf-1 | 0 | 0 | 11 | 9.9539757e-06 | 4.5468899e-07 | 4 | 303 | 1620 |
| readf-1 | 0 | 0 | 4 | 5.7220459e-06 | 4.2359397e-07 | 0 | 0 | 0 |
| round-1 | 0 | 0 | 2 | 7.5101852e-06 | 4.8988504e-07 | 1 | 501 | 2386 |
| rt-follow-1 | 0 | 0 | 8 | 6.6757202e-06 | 4.310852e-07 | 0 | 0 | 0 |
| rt-follow-2 | 0 | 0 | 8 | 6.1988831e-06 | 4.3161467e-07 | 0 | 0 | 0 |
| rt-follow-3 | 0 | 0 | 1 | 4.7683716e-06 | 4.3649179e-07 | 0 | 0 | 0 |
| rt-follow2-1 | 0 | 0 | 8 | 6.6757202e-06 | 4.3074714e-07 | 0 | 0 | 0 |
| rt-follow2-2 | 0 | 0 | 8 | 6.6757202e-06 | 4.3254176e-07 | 0 | 0 | 0 |
| rt-follow2-3 | 0 | 0 | 1 | 7.5101852e-06 | 4.3590305e-07 | 0 | 0 | 0 |
| scanner-1 | 0 | 0 | 1 | 7.6293945e-06 | 4.7446163e-07 | 0 | 0 | 0 |
| scar-1 | 0 | 0 | 2 | 7.1525574e-06 | 4.7546131e-07 | 0 | 0 | 0 |
| scar2-1 | 0 | 0 | 3 | 7.6293945e-06 | 4.6869376e-07 | 0 | 0 | 0 |
| scar3-1 | 0 | 0 | 3 | 7.6293945e-06 | 4.6869376e-07 | 0 | 0 | 0 |
| sharepos-1 | 0 | 0 | 4 | 6.6757202e-06 | 4.3202478e-07 | 0 | 0 | 0 |
| shot-2 | 0 | 0 | 13 | 8.9406967e-06 | 4.6399938e-07 | 3 | 395 | 1267 |
| shot-3 | 0 | 0 | 14 | 6.4373016e-06 | 4.3835991e-07 | 0 | 0 | 0 |
| shot-4 | 0 | 0 | 28 | 6.6757202e-06 | 4.142077e-07 | 1 | 3 | 3 |
| t-follow-1 | 0 | 0 | 8 | 6.6757202e-06 | 4.3330344e-07 | 0 | 0 | 0 |
| t-follow-2 | 0 | 0 | 8 | 6.6757202e-06 | 4.3215109e-07 | 0 | 0 | 0 |
| t-follow-3 | 1 | 1.16257e-07 | 1 | 4.7683716e-06 | 4.3333805e-07 | 0 | 0 | 0 |
| t-library-2 | 0 | 0 | 4 | 6.6757202e-06 | 4.3614253e-07 | 0 | 0 | 0 |
| t-manual-1 | 1 | 1.16257e-07 | 5 | 6.6757202e-06 | 4.3595583e-07 | 0 | 0 | 0 |
| t-views-1 | 1 | 1.16257e-07 | 4 | 5.7220459e-06 | 4.116707e-07 | 1 | 1 | 1 |
| t-views-2 | 0 | 0 | 4 | 5.7220459e-06 | 4.0962989e-07 | 1 | 1 | 1 |
| try-2 | 0 | 0 | 1 | 6.1988831e-06 | 5.5299385e-07 | 1 | 23 | 30 |
| vid-1 | 0 | 0 | 5 | 7.1525574e-06 | 4.352329e-07 | 0 | 0 | 0 |
| vid-10 | 0 | 0 | 5 | 7.1525574e-06 | 4.352329e-07 | 0 | 0 | 0 |
| vid-11 | 0 | 0 | 5 | 7.1525574e-06 | 4.352329e-07 | 0 | 0 | 0 |
| vid-12 | 0 | 0 | 5 | 7.1525574e-06 | 4.352329e-07 | 0 | 0 | 0 |
| vid-13 | 0 | 0 | 5 | 7.1525574e-06 | 4.352329e-07 | 0 | 0 | 0 |
| vid-14 | 0 | 0 | 5 | 7.1525574e-06 | 4.352329e-07 | 0 | 0 | 0 |
| vid-15 | 0 | 0 | 5 | 7.1525574e-06 | 4.352329e-07 | 0 | 0 | 0 |
| vid-16 | 0 | 0 | 5 | 7.1525574e-06 | 4.352329e-07 | 0 | 0 | 0 |
| vid-2 | 0 | 0 | 5 | 7.1525574e-06 | 4.352329e-07 | 0 | 0 | 0 |
| vid-3 | 0 | 0 | 5 | 7.1525574e-06 | 4.352329e-07 | 0 | 0 | 0 |
| vid-4 | 0 | 0 | 5 | 7.1525574e-06 | 4.352329e-07 | 0 | 0 | 0 |
| vid-5 | 0 | 0 | 5 | 7.1525574e-06 | 4.352329e-07 | 0 | 0 | 0 |
| vid-6 | 0 | 0 | 5 | 7.1525574e-06 | 4.352329e-07 | 0 | 0 | 0 |
| vid-7 | 0 | 0 | 5 | 7.1525574e-06 | 4.352329e-07 | 0 | 0 | 0 |
| vid-8 | 0 | 0 | 5 | 7.1525574e-06 | 4.352329e-07 | 0 | 0 | 0 |
| vid-9 | 0 | 0 | 5 | 7.1525574e-06 | 4.352329e-07 | 0 | 0 | 0 |
| views-1 | 1 | 1.16257e-07 | 4 | 5.7220459e-06 | 4.116707e-07 | 1 | 1 | 1 |
| views-2 | 0 | 0 | 4 | 5.7220459e-06 | 4.0984815e-07 | 0 | 0 | 0 |
| zoom-1 | 1 | 1.16257e-07 | 10 | 6.6757202e-06 | 4.217838e-07 | 0 | 0 | 0 |

No data, ONNX files, contexts, class maps or generated reports were added to
Git. Numbers are recorded here; reusable harnesses are under bench/ and tools/.
The existing check_xml gained only optional scratch artifact export; its
byte-exact comparison and failure/call-consumption checks remain intact.

### Final verification and unattended decisions

Final required commands pass: **cargo fmt --check**, **cargo clippy
--all-targets -- -D warnings**, and **cargo test --release** (seven library
tests, all binary/doc test targets successful). Python harnesses compile and
all added JavaScript modules pass node --check; the standalone SIMD cost
probe is rustfmt-clean and both strict/fused artifacts matched its native
sample bytes before measuring instructions. The baseline XML diagnostic
preserves 75 exact results + 13 matching native failures.

All background measurement, token, segmentation replay, combined replay and
XML jobs exited successfully. Full native and WASM measurements ran; WebGPU
was scripted and launch-attempted only. No actual phone, Chrome GPU inference,
new ORT build, large Rust model port or native service was run/implemented.
Blockers are the headless Chrome SIGABRT before CDP and unavailable published
ORT-web 1.31.0 package. GPU numeric/speed results remain explicitly unknown.

Unattended choices were to preserve default native options, reuse existing
native encoder outputs after fresh sample validation, store lossless tile
inputs as uint8, use Partsong's four-thread setting for the full WASM corpus,
and use separate warmed one-thread runs for speed budgets. Input duplicates
were not used to skip browser inference: all 8,076 tiles and all 627 staffs
ran. Segnet-only and combined replay reuse symbols deliberately, with canvas
and call-count checks; they are labelled as boundary replays rather than live
browser runs. No user answers were needed. All work is committed on cx/models;
Partsong and the plain homr checkout were not edited. Nothing was pushed or
posted, and generated data/reports remain only in scratch.

## Exact Rust encoder port (cx/encoder)

The maintainer supplied external Chrome 156 / Apple Metal measurements for all
88 pages / 627 staffs: WebGPU segnet differs at **24 / 826,982,400 pixels on
16 pages**, worst fix-six-1 with **2 pixels**; encoder max absolute difference
**1.86e-5**, mean **1.32e-6**, worst pdf-1 staff 8. These supplement the WASM
measurements above; downstream GPU tokens/XML have not been supplied and are
not inferred. The encoder port targets native default arm64 ORT 1.31.0.
Segnet remains in ORT-web as requested. No large segmentation port is planned.

### Stage 1: native scalar operator checks

Shared the ONNX protobuf reader at src/onnx.rs, retaining decoder parsing and
adding typed node attributes. Factored the existing MLAS erf polynomial out of
the decoder GELU without changing its arithmetic. Initial native single-node
checks (angelus_0 staff 0) are bit-identical for stem Conv (1,966,080 outputs),
LayerNorm (1,966,080), depthwise 7x7 Conv (1,966,080), MatMul (7,864,320), Div
(7,864,320), and Erf (7,864,320). The stem Conv must initialize accumulators
with bias; summing from zero and adding bias later differed at 1,423,236
outputs. Depthwise uses the opposite schedule: zero accumulation, bias after.
No fixture hashes or approximate tolerances are used for these checks.
Seven release library tests and all existing binary/doc test targets pass.

### Stage 2: optimized native subgraphs and first whole-model agreement

Added NEON/SIMD128 ordered 4x16 float GEMM. All previous single-node outputs
remain bit-identical; native optimized Gemm+bias and Gelu each match all
7,864,320 probe values. A diagnostic graph retains native optimized fusions
and disables re-optimization; its final context is byte-identical to the
ordinary default native session before its 114 intermediate outputs are used.

The first assembled model initially diverged at the first downsample Conv.
Native MLAS segments its contraction into **independent 128-term fused sums**,
then adds those partials and finally bias. Replaying one uninterrupted FMA
chain or initializing bias early is wrong for these two convolutions. The
stem instead initializes bias early. With the segmented schedule, every
observed intermediate and all **655,360 output float32 bits** on angelus_0
staff 0 match native exactly. This is whole-model sample agreement, not yet a
corpus claim. Runtime weight loading uses homr's original encoder ONNX.

### Stage 3: vector activations and depthwise

Vectorized MLAS erf/GELU and padded NHWC depthwise convolution. All native
single-node checks, optimized Gemm/Gelu probes and observed whole-model
intermediates remain bit-identical on angelus_0 staff 0. The native corpus
runner now executes every staff, compares float32 bits directly, and records
retired instructions plus an operation breakdown. The first full native
corpus run is underway; no corpus result is claimed yet.

### Stage 4: WASM ABI and double-rounding correction

Added encoder_load/drop/run and encoder_decode exports beside the unchanged
decoder ABI. encoder_run writes to caller linear memory which decoder_run can
read directly; encoder_decode passes its internal context Vec to the decoder
without a context copy. Optional probe/trace/profile features are benchmark
only and absent from the normal artifacts.

All initial isolated operators and optimized Gemm/Gelu probes pass on both
WASM variants. The first fused four-staff sample has 4/4 bit-identical contexts,
using roughly 20.57G retired instructions per warmed staff versus the prior
28.41G ORT-web single-thread baseline. Full-corpus results are still pending.

The original widened strict recurrence passed simple probes but failed the
whole model. Trace showed its first observed one-bit discrepancy late at
stage 2/block 4's activation, followed by 161,290 changed context values.
Float64 multiply/add followed by float32 conversion is not always a correctly
rounded fused float32 operation. Added a halfway/subnormal/special-value
check in SIMD recurrence, calling libm's correctly-rounded fmaf only on rare
lanes; scalar strict Welford also uses fmaf. A numeric regression case uses
(1+2^-23)*(1-2^-23)+16777218: naive double rounding gives 16777220 instead of
16777218. With correction, every observed first-staff stage and context bit
matches native. Decoder arithmetic was not changed by this fix.

### Stage 5: native corpus and strict fallback engineering

The native NEON encoder completed **627/627 bit-identical contexts**, followed
by the rebuilt Rust decoder: **38,076 steps, zero token/step/length changes**.
Native encoder+decoder XML remains **75 byte-identical + 13 matching failures**.

Strict SIMD's fully general corrected recurrence was accurate but expensive:
the initial explicit halfway/special checks used approximately 729G retired
instructions/staff; outlining its rare scalar correction reduced this to
roughly 669G. A branchless TwoSum + round-to-odd recurrence preserves exact
fused rounding, including subnormal cases, and measured about 542G. These
are superseded experiments, not the final strict performance claim.

The retained strict GEMM adds a proven range fast path: if all nonzero
operands/biases lie in [2^-40,2^40] and K<=65536, their products have a 2^-126
grid. Every rounded partial sum remains on that grid, so it is zero or normal;
K and the upper bound also prevent overflow. Only halfway double-rounding
cases require scalar repair. Inputs outside these bounds execute the general
round-to-odd path. No model input, staff or operation is skipped and this is
not a fixture/hash gate. The native-like first full bounded-path staff
(follow-3) matches every context bit and measures about **440G instructions**,
still much more than ORT-web; this cost is not hidden.

For the complete strict corpus use two independent, single-threaded Node
workers over disjoint page lists (modest CPU parallelism), both executing
normal, non-profile final artifacts and their own Rust decoder in the same
linear memory. Fused and native runs are also complete/in progress as noted.

Strict engineering update: the range fast path batches halfway checks over a
2x8 tile and uses an inline vector round-to-odd repair, avoiding scalar calls
inside GEMM. Its full follow-3 context is bit-identical, at approximately
273.7G retired instructions including profiling, versus 440G for the earlier
range-checked scalar repair. The final normal strict artifact is frozen at
this implementation. The complete strict corpus will use four balanced,
disjoint single-threaded Node shards, at most four CPU workers total (the
fourth strict worker starts after the fused corpus exits). Previous slow
prefix runs are not merged into the final corpus or final speed table.

### Fused WASM corpus complete

The normal fused artifact completed all **627 bit-identical encoder contexts**
and its own WASM Rust decoder on the same context pointer: **38,076 steps,
zero differing heads/steps/lengths**, including all native failure-page staffs.
No JS context copy is required for the decoder. Full fused XML replay is being
checked separately. Four balanced strict workers now execute the retained
normal strict artifact; no diagnostic/profile imports are used in those runs.

### API validation and final build checks

Fused XML replay completed: **75 byte-identical results and 13 matching native
AssertionErrors**, with zero byte-different pages. The single-call
encoder_decode ABI also matches the native 121-step follow-3 staff, with a
separately checked bit-identical context. Existing decoder exports remain
present. No changes were made to decoder arithmetic or its stopping rules.

Added checked dimension products before unsafe encoder SIMD entry points,
including a release-mode overflow regression test. These integer guards and
a Clippy iterator rewrite leave valid-image arithmetic and accumulation order
unchanged. Full strict corpus runs use the frozen normal batch-kernel artifact;
fresh guarded-source artifacts are built separately for final checks and speed.
The native optimized graph records 114 outputs; **70 named computational
boundaries** are observed and compared by the Rust trace (the recording also
contains reshape-shape constants and other fused-graph outputs). The trace
harness now fails its process on any bit difference or an empty observation.

Native release tests pass all **11** library tests and all binary/doc targets.
Strict and fused WASI runs each pass **10** tests; WASI's aborting panic runtime
ignores the should-panic dimension-overflow test, which passes natively.
Native all-target/all-feature and both WASM all-feature Clippy checks pass
with -D warnings. No user answers or new authorization were needed.

### Threading assessment and scope

The encoder currently executes one staff on one thread. Independent output
rows/pixels, layer-normalization rows, and pointwise activations can be divided
among workers without changing any contraction or Welford reduction order.
Splitting K or reducing partial sums in a new order would break exactness.
An intra-staff browser implementation needs shared WebAssembly.Memory,
atomics/bulk-memory, a compatible thread-safe Rust std/allocator build,
COOP/COEP isolation, worker stacks/initialization, and a persistent worker pool
with stage barriers. The C ABI can be kept; wasm-bindgen is not required.

Unmeasured planning estimate: two workers could provide approximately 1.5–1.8x
parallel throughput, four approximately 2–3x. A 95%-parallel Amdahl model gives
a four-worker ceiling near 3.48x before barriers, memory traffic and phone
big/LITTLE effects. Total retired instructions would stay similar or increase
(budget 5–20% for dispatch/synchronization; busy spinning could be worse).
These are estimates, not phone or wall-time benchmark results. ORT-web's four
threads and WebGPU can still have a latency advantage despite a smaller Rust
single-thread instruction count. A worker pool is intentionally not built in
this round: it requires a distinct shared-memory runtime and another exactness
validation matrix. No Partsong integration or Android measurement was run.

The final benchmark preflight found that the isolated optimized GELU fixture
flattened its input to 2D while retaining a declared 4D output. Native ORT 1.31
accepted this inconsistent rank, whereas ORT-web 1.30 rejected it. The recorder
now follows the optimized graph's actual constant Reshape shape for both Gemm
and Gelu. Correcting the scratch GELU input declaration to [1,64,320,384]
preserves every native output byte and makes the browser probe load. This is
a diagnostic-fixture correction, not an encoder arithmetic change.

### Complete three-build encoder agreement

All four retained strict shards exited successfully. Their disjoint merge
contains exactly 88 pages and 627 staffs; every staff actually executes both
models. Strict results are **627 bit-identical contexts, 38,076 steps, zero
differing token heads/steps, zero extra or missing steps**. Its byte-exact XML
replay gives **75 identical files and 13 matching native AssertionErrors**.
No staff input, inference, decoder trajectory or failed native page was skipped.

| encoder + Rust decoder | context bit-identical staffs | decoder steps | changed heads / steps / lengths | XML identical / matching failures |
|---|---:|---:|---:|---:|
| native NEON fused | 627/627 | 38,076 | 0 / 0 / 0 | 75 / 13 |
| WASM SIMD128 + relaxed SIMD, fused probe passes | 627/627 | 38,076 | 0 / 0 / 0 | 75 / 13 |
| WASM strict SIMD128, emulated fused | 627/627 | 38,076 | 0 / 0 / 0 | 75 / 13 |

The scratch evidence is encoder_native/comparison.json,
encoder_native_tokens/comparison.json, encoder_fused/comparison.json,
encoder_strict_final/comparison.json and each corresponding *_xml/comparison.json.
The strict merger rejects duplicate/missing pages and mixed artifact names.
Whole contexts are compared as float32 bits, including signed zero, with no
hashes or numeric tolerance. XML replay uses recorded native page/staff inputs
and each build's newly generated symbols/attention; it is not a live browser
segmentation/whole-page integration test. Segnet remains on ORT-web.

Fresh guarded-source strict and fused artifacts also pass every isolated
operator probe and the combined encoder_decode follow-3 check (121 matching
native steps each, plus separate context-bit checks). Native guarded source
passes the observed stage trace. The frozen corpus kernels and these final
builds have identical floating arithmetic; late edits add dimension validation
and an equivalent iterator spelling, as documented above. Sequential final
benchmarks use the fresh normal artifacts, separately from profile artifacts.

### Final encoder instructions (not wall time)

Warmed angelus_0 staff 0, one warm-up then three measured executions per
engine, median below. All models/inputs are loaded before the counter. Each
Rust result is checked against native context bits; ORT-web repeats are checked
for byte determinism, not falsely claimed exact against native. Every native
ORT repeat matches the native reference. All measurement jobs run sequentially
after the corpus workers exit. Counters are process-wide proc_pid_rusage
RUSAGE_INFO_V4, including ORT worker threads. No elapsed times are used.

| encoder engine/build | G retired instructions/staff |
|---|---:|
| native Rust NEON, fused, one thread | 12.341592 |
| native ORT 1.31.0, homr default CPU options | 7.800154 |
| native ORT 1.31.0, one thread | 3.790445 |
| Rust WASM SIMD128 + relaxed SIMD, fused | 20.634634 |
| Rust WASM strict SIMD128, emulated fused | 267.354596 |
| ORT-web 1.30.0 WASM, one thread | 28.423430 |
| ORT-web 1.30.0 WASM, four threads | 30.838291 |

The requested fused target is met: **27.4027% fewer instructions** than
ORT-web single-threaded (1.377x instruction ratio). Strict costs **12.9566x
fused Rust**, or **9.4061x ORT-web one-thread**. Native NEON is 3.256x the
controlled native one-thread ORT count; native ORT uses its SME2 kernels here.
The native default-options count includes the default worker pool's extra
work. Four-thread ORT-web uses 8.496% more process instructions than one-thread;
this says nothing by itself about its elapsed latency, particularly on phones.

Scratch reports: encoder_bench_final/*.json. The fresh normal WASM paths are
encoder-fused-final.wasm and encoder-strict-final.wasm; profile variants have
separate names. Normal WASM counts do not include profile imports. Native
phase callbacks add a negligible amount of counter overhead to its total.
Reproduce with bench/measure_encoder.py --help; the driver now includes both
default and controlled one-thread native baselines (the default baseline above
was invoked separately with that same models_native.py command).

Whole Rust encoder phase medians, **G instructions per staff**. Profiling
callbacks exclude their own counter reads from phase intervals; allocation,
layout conversion and operation setup are charged to the following operation.
The sum of component medians need not equal the median total. Profile totals
are 12.341592G native, 20.646609G fused WASM and 267.270415G strict WASM,
consistent with the separately measured normal artifacts.

| operation | native NEON | WASM fused | WASM strict |
|---|---:|---:|---:|
| stem Conv | 0.023762 | 0.034486 | 0.319027 |
| downsample Conv | 0.400796 | 0.637326 | 7.191821 |
| depthwise 7x7 Conv | 1.203037 | 1.692840 | 17.397260 |
| LayerNorm | 0.237912 | 0.584870 | 1.973784 |
| MLP fc1 Gemm | 4.689187 | 7.815699 | 105.510139 |
| GELU (including erf) | 0.889347 | 1.819010 | 26.469227 |
| MLP fc2 Gemm | 4.623426 | 7.664214 | 105.830966 |
| layer scale Mul | 0.159394 | 0.192838 | 0.192803 |
| residual Add | 0.010104 | 0.028528 | 0.028569 |
| projection Gemm | 0.103291 | 0.171635 | 2.345138 |
| positional Add | 0.000498 | 0.001402 | 0.001402 |

Representative **isolated ORT operator** medians below are M instructions per
invocation, not a decomposition of the full ORT graph. Conv/LayerNorm/depthwise
have 1,966,080 output values; MatMul/Gemm/Div/Erf/Gelu have 7,864,320. These
use recorded native operands and the actual stem/first-block shapes, one warm
and three measurements. Native columns use one thread. Optimized Gemm/Gelu
probes and unfused nodes are deliberately distinguished. These numbers cannot
be summed or multiplied by the op inventory to recover full-model totals.

| isolated node | native ORT | ORT-web 1 thread | ORT-web 4 threads |
|---|---:|---:|---:|
| Conv | 8.625 | 48.909 | 78.853 |
| LayerNormalization | 29.002 | 57.679 | 161.844 |
| depthwise Conv | 259.708 | 304.023 | 301.305 |
| MatMul | 6.772 | 807.303 | 999.894 |
| Div | 11.872 | 89.518 | 378.447 |
| Erf | 126.280 | 370.790 | 685.869 |
| optimized Gemm+bias | 29.948 | 804.938 | 1078.184 |
| optimized Gelu | 141.325 | 427.505 | 727.071 |

ORT's public JS interface has no per-node retired-instruction callback. Its
standard profiler reports elapsed time, which is not used here. A full
in-session ORT instruction decomposition would need an instrumented runtime;
that build was not made. The isolated-node comparisons and full Rust phase
breakdown above are what actually ran.

### Final outcome and remaining limits

No encoder operator prevented bit identity. Conv/bias scheduling, segmented
128-term downsample sums, ordered fused GEMM, Welford LayerNorm and MLAS
erf/GELU all pass native and both WASM single-node/subgraph checks. The strict
double-rounding failure was fixed rather than hidden behind a token-only
tolerance. All three complete model/corpus checks then pass exactly. There
are no new attention/softmax/int8 encoder kernels: this graph is float ConvNeXt.

The fused artifact is the practical exact phone path on hosts that pass the
load-time fused probe. Keep strict as an expensive compatibility fallback;
it is not a speed improvement over ORT-web. Browser/Android device validation,
live Partsong integration, GPU encoder execution, a shared-memory Rust worker
pool and an instrumented ORT build were not run/implemented. Device access and
the separate browser threading runtime are concrete remaining requirements,
not unresolved numeric mismatches. Supplied external WebGPU figures are logged
beside the prior WASM survey; segnet is unchanged.

Final required checks pass again: cargo fmt --check; cargo clippy --all-targets
--all-features -- -D warnings; cargo test --release (11 library tests, all
binary/doc targets); both normal wasm32-unknown-unknown release builds.
Earlier final-source strict/fused WASI tests and WASM Clippy checks also pass
as detailed above. The existing decoder ABI and corpus agreement are preserved.
All artifacts, raw counts, contexts and XML stay in scratch. Work is committed
on cx/encoder; nothing was pushed or posted, and neither Partsong nor homr was
edited. Unattended decisions were the fixed graph executor, the rigorously
bounded strict fast path, four disjoint corpus workers, and deferring a browser
threading runtime until its own validation round.

## Exact encoder threads and GPU prototype (cx/threads)

### Shared-memory worker implementation and first verified stage

Selected a host-managed synchronous worker pool, without std::thread or
wasm-bindgen. The coordinator owns graph execution, allocation and convolution
lowering. Kernels publish a stack-resident descriptor, dispatch disjoint output
ranges, compute rank zero, then join all workers before the descriptor/buffers
can be released. GEMM splits whole four-row tiles; depthwise splits spatial
pixels; activations split whole SIMD vectors; LayerNorm splits complete rows.
No contraction K or Welford channel reduction is split or reordered.

tools/encoder_pool.mjs and encoder_worker.mjs work in Node and browser Workers.
Workers share one memory/model, with private stacks, per-instance stack globals
and initialized TLS blocks. They enter only allocation-free encoder_work.
Atomics publish jobs and join; Atomics.wait/notify sleep instead of spinning.
The coordinator must be a Web Worker in the browser because Window forbids
blocking Atomics.wait. A worker failure poisons the pool; terminate workers
before discarding memory rather than reusing an abandoned Rust stack job.

The stock precompiled Rust std cannot link --shared-memory (confirmed by a
linker rejection). bench/build_encoder_threads.py rebuilds std,panic_abort
with atomics/bulk-memory and imports shared memory, exporting stack/TLS setup.
The installed compiler is nightly 1.101.0 (3e92e1a2d, 2026-11-01). Existing
strict/fused nonshared builds remain available with encoder-threads disabled.
The shared build uses the existing encoder/decoder C ABI plus encoder_work.
No model weights, inputs, output data or fixture hashes are embedded.

All isolated Conv, depthwise Conv, LayerNorm, MatMul/Gemm, Div, Erf and Gelu
checks pass with **zero differing bits**, in both shared WASM variants at
**1, 2, 4 and 6 threads**. The strict four-thread follow-3 sample has an exact
context and **121 matching decoder steps**, with **89 joined jobs per staff**.
Native release tests and all-target/all-feature Clippy -D warnings pass after
the allocation-free kernel refactor. Corpus checks are the next stage.

The first exploratory scaling harness put all cases in one Node process, so
its process-wide instruction counter could include idle engines' GC/JIT work.
For example fused-one-thread samples were 22.086G, 20.970G and 22.323G despite
other thread counts clustering near 20.98G. It is retained in scratch as
threads_scaling_shared_process.json, not used as the final comparison. The
final harness isolates each engine in its own persistent process and samples
that process, including all its kernel workers. Cases still run sequentially,
with one warm-up and three interleaved measured rounds, preserving every raw
sample and the predeclared rotation/reversal order. This corrects attribution,
not arithmetic or the chosen workload. The machine remains shared and loaded.

### Node scaling: final isolated-process measurements

Host reports Apple M5 Max, 18 logical CPUs, ARM64 Node 26.9.0. Every case has
one coordinator plus N-1 allocation-free kernel workers; only one case runs
at once. Other resident cases are separate idle processes. One warm-up per
case, then three interleaved rounds with predeclared rotation/reversal. Model
loading, pool creation and input-file reads are outside the measurement.
Each measured Rust call checks native context bits; ORT repeats check their
own warmed bytes (ORT is not claimed exact). Host dispatch, the final WASM
context copy, output checks and coordinator work are included. Instruction
counts cover the case process and all its workers, not the benchmark driver's
IPC. Wall times include the measured call's IPC round trip; engine-only values
are retained separately. No sample is dropped or replaced by a fastest run.

Median of three, G retired instructions and seconds per staff:

| engine | total threads | G instructions | seconds |
|---|---:|---:|---:|
| Rust fused shared WASM | 1 | 20.975702 | 1.884906 |
| Rust fused shared WASM | 2 | 20.976307 | 0.847230 |
| Rust fused shared WASM | 4 | 20.988078 | 0.563445 |
| Rust fused shared WASM | 6 | 20.996975 | 0.255151 |
| Rust strict shared WASM | 1 | 267.678843 | 38.058830 |
| Rust strict shared WASM | 2 | 267.877712 | 22.316589 |
| Rust strict shared WASM | 4 | 268.276638 | 6.411590 |
| Rust strict shared WASM | 6 | 268.666513 | 4.035695 |
| ORT-web 1.30 WASM | 1 | 28.407107 | 1.456688 |
| ORT-web 1.30 WASM | 4 | 30.753257 | 0.387993 |

Raw wall samples in seconds (round order), to show the loaded machine's range:

| engine / threads | trial 1 | trial 2 | trial 3 |
|---|---:|---:|---:|
| fused / 1 | 1.659992 | 2.285288 | 1.884906 |
| fused / 2 | 0.661052 | 1.310578 | 0.847230 |
| fused / 4 | 0.563445 | 0.429210 | 0.635169 |
| fused / 6 | 0.243802 | 0.255151 | 0.347490 |
| strict / 1 | 23.206173 | 44.961189 | 38.058830 |
| strict / 2 | 12.103092 | 36.750369 | 22.316589 |
| strict / 4 | 5.847406 | 6.411590 | 14.292391 |
| strict / 6 | 4.035695 | 4.017108 | 6.816574 |
| ORT / 1 | 1.456688 | 1.403349 | 2.316602 |
| ORT / 4 | 0.386605 | 0.387993 | 0.769672 |

Strict instruction overhead relative to its one-thread shared path is 0.223%
at four threads and 0.369% at six; fused is 0.059% and 0.101%. Sleeping waits
avoid spin amplification. Instruction counts remain stable except ORT's last
four-thread sample (41.542313G, retained, versus 30.546214G/30.753257G), where
runtime work/spinning is included rather than subtracted. Superlinear wall
ratios reflect load/core-placement differences on this heterogeneous machine;
they are not claims of algorithmic speedup or portable phone scaling.
Raw final evidence is threads_scaling_final.json; the exploratory shared-PID
run is archived separately. Four-thread corpus work starts after timing exits.

### WebGPU prototype and Chrome launch blocker

The prototype is fc1 [20480,96] x [96,384], **7,864,320 output values**. Each
invocation owns four adjacent outputs, executes ascending K with explicit WGSL
fma, then applies bias separately. No parallel K reduction, fp16 or dot/subgroup
reassociation is used. Input/native tensors come from the original optimized
native single-node probe. The Rust output is independently generated and
matches every native bit before the fixture is accepted. No model data is
committed; tools/prepare_gpu_fc1.py writes all operands/references to scratch.

The load-time known-answer design tests six storage-fed fma triples (fusion,
double-rounding, subnormals, signed zero and ties), then selected cells from
the same fc1 shader generator to distinguish K order and bias-after behavior.
All **10 known answers** were independently verified against native fmaf and
ORT 1.31.0. Both generated shaders parse and validate with Naga 27.0.3 in a
scratch validator. These are analytic numeric answers, not model-output hashes.
Production must reject the GPU path on any failure/device loss, rerun tests
after device recreation, and still validate every model operator/full corpus
on the target engine. Fma tests alone cannot certify LayerNorm division/sqrt
or GELU, nor every input permitted by WGSL's floating-point rules.

The GPU harness records Apple adapter information, refuses software/other
adapters, compares every output against both native and Rust, warms once and
times three runs. It reports compute-pass GPU timestamps when available,
input-upload + dispatch/wait wall time, and output readback separately. Even a
failed self-test is followed by the diagnostic fc1 comparison, but exact=true
requires both tests and every full-output bit comparison to pass.

Both the GPU and CPU Chrome scripts were launch-attempted with headless=new
and enable-unsafe-webgpu. Chrome exits **SIGABRT before CDP** in this sandbox;
no GPU shader ran and no Chrome speed/bit comparison is claimed. The actual
fc1 GPU differences and timings are **unmeasured**, not zero. The CPU browser
harness includes COOP/COEP and runs one engine at a time, warming/setup outside
counters. Its new instructionsForPid addon reads the renderer's total retired
instructions via proc_pid_rusage; it refuses stale addons that could silently
read the Node driver's PID. Renderer PID changes invalidate the measurement.

One-line GPU command from this clone, using the prepared scratch fixtures:

    node bench/gpu_fc1_chrome.mjs ../threads_gpu_fixture ../threads_gpu_chrome.json

CPU browser command (after building current bench/counter.c as threads_counter.node):

    COUNTER_ADDON=../threads_counter.node node bench/threads_chrome.mjs ../threads_build ../models_files/encoder_pytorch_model_465-597144cab54c8f6d0f6c9619df5c5312694eadd6.onnx ../models_native/angelus_0/0 ../ortbench/node_modules/onnxruntime-web/dist ../threads_cpu_chrome.json

The shaders, harnesses and their launch-failure evidence are ready for a host
where Chrome can start. No workaround changes the sandbox or substitutes a
software GPU for Apple Metal.

### Threaded fused corpus verified

Four-thread fused WASM completed **627/627 bit-identical contexts**, **38,076
native decoder steps**, zero differing heads/steps/lengths, and XML **75
byte-identical files + 13 matching native failures**. Every staff records
**[89,89,89,89]** completed jobs across coordinator and workers: 55,803 joined
jobs, 223,212 rank executions. The strict four-thread corpus is now running.

### Phone recommendation and explicit estimate assumptions

Rank current evidence by exactness first: (1) the shared CPU encoder, with
the passing fused artifact on Android and strict on iPhone; (2) a future GPU
encoder only after its arithmetic gate, every operator and the full corpus
pass on the actual browser/device; (3) the ordinary nonshared strict/fused
artifact where shared memory or the worker pool cannot be used. The pool
improves elapsed CPU latency without materially increasing total work. It
does not remove strict fused-emulation's roughly 267G instruction budget.

No physical iPhone/Android CPU run or exact GPU timing was available. For a
concrete planning scenario, assume a phone takes **2–4 times this loaded
Mac's measured elapsed latency at the same worker count**. This assumption is
not a measured hardware ratio: core placement, thermal limits, Safari versus
V8 code generation and background load can move results outside the ranges.
In particular the Mac's raw samples vary substantially and are not a device
performance calibration. Do not use these estimates as a shipping promise.

| exact CPU option | 6-core iPhone, strict (scenario) | 4-core Android, fused (scenario) | remaining work |
|---|---:|---:|---|
| shared pool, 6 iPhone / 4 Android threads | about 8–16 s/staff | about 1.1–2.3 s/staff | integrate host assets, validate full corpus and worker lifecycle on devices, choose thermal/UI worker cap |
| nonshared fallback, 1 thread | about 76–152 s/staff | about 3.8–7.5 s/staff | device validation; no worker/isolation requirement |

These fallback estimates use the measured one-thread shared path as an
approximation; normal nonshared artifacts are regression-checked but were
not separately wall-time benchmarked. The estimates intentionally expose that
uncertainty rather than substituting the earlier faster exploratory run.

The exact GPU whole-encoder phone time is **not estimable from this run**:
Chrome never started, and neither this fc1 kernel nor a complete exact GPU
encoder ran. The supplied Android **0.35 s/staff** is the current nonexact
ORT-web GPU reference, not a measured time or promised target for our shader.
Matching that order of latency is the reason to pursue the GPU experiment.
Remaining work is to run the prepared Metal probe, then implement/verify
depthwise and downsample schedules, ordered Welford LayerNorm with native
division/sqrt rounding, MLAS erf/GELU, pointwise operations and device-resident
buffers. Validate all 627 contexts/tokens on Chrome Metal and Safari/WebKit,
plus Android backends, before selecting it as an exact path. A small fma
self-test alone cannot establish whole-model exactness or portable WGSL
floating-point guarantees. No full GPU encoder is implemented in this round.

### Regression and build verification

Both ordinary nonshared artifacts and both shared artifacts pass the combined
encoder_decode regression: a separate bit-exact follow-3 context and **121
matching native token steps** each. Shared samples use the one-rank dispatch
path, while full corpus uses four ranks and all 1/2/4/6 operator probes pass.
The existing C ABI signatures are retained; the shared build only adds its
worker entry/imports and setup globals.

Required checks pass: cargo fmt --check; cargo clippy --all-targets
--all-features -- -D warnings; cargo test --release (**11 library tests**, all
binary/doc targets); strict/fused shared WASM Clippy -D warnings using rebuilt
std; strict/fused nonshared WASI tests (**10 pass, 1 should-panic test ignored
by the aborting WASI runner**, with that test passing natively). Both normal
WASM builds, both shared WASM builds and both shared diagnostic probe builds
are rebuilt successfully. The two shared normal modules reproduce the frozen
corpus modules **byte-for-byte** after rebuilding, establishing artifact
provenance; this is not a substitute for output comparisons or a hash test.
All actual outputs are still compared element by element against native.

### Final threaded corpus result and unattended report

The strict run and both XML replays exited successfully. Both normal shared
artifacts execute the complete original corpus through **four total ranks**,
including the coordinator. Every staff records **[89,89,89,89]** joined job
executions: no single-thread fallback or missing rank is hidden in these runs.
All 410,910,720 context float32 values per build are bit-compared directly.

| build, 4 total threads | exact contexts | decoder steps | changed heads / steps / lengths | XML byte-identical / matching native failures |
|---|---:|---:|---:|---:|
| shared fused SIMD128 + relaxed SIMD | 627/627 | 38,076 | 0 / 0 / 0 | 75 / 13 |
| shared strict SIMD128, fused emulation | 627/627 | 38,076 | 0 / 0 / 0 | 75 / 13 |

All 1/2/4/6-thread native operator comparisons and benchmark staff outputs
also pass; full-corpus runs use four threads, not every thread count. The
ordinary single-thread artifacts retain every previous C ABI function with
**zero imports**; shared artifacts retain them and add env.memory and
encoder_threads.dispatch. Encoder/decoder combined API regression passes on
both ordinary and both shared artifacts. Corpus uses the direct encoder_run
output pointer as decoder_run input; neither stage's context crosses JS.

Scratch evidence: threads_fused/comparison.json, threads_strict/comparison.json,
threads_fused_xml/comparison.json, threads_strict_xml/comparison.json,
threads_scaling_final.json and threads_final_checks.log. Input recordings and
the unchanged native reference are reused; every staff actually runs the new
encoder and its own Rust decoder. XML is a recorded-boundary replay, not a live
Partsong/browser page run. Neither Partsong nor the homr checkout was edited.

Unattended decisions were a host-managed allocation-free pool with rebuilt
atomic std/private TLS, four-thread corpus runs to limit concurrent CPU use,
isolated-process counters rather than the contaminated exploratory harness,
and keeping the GPU result unknown after the observed Chrome SIGABRT. Phone
ranges are explicitly uncalibrated planning assumptions; no device timing or
exact GPU whole-model performance is invented. The next iPhone performance
decision needs the prepared Metal probe and real-device validation. The
remaining GPU work is the full graph, including exact division/sqrt/GELU and
Chrome/Safari/device corpus checks; the prototype alone is not a GPU encoder.

All source/tool changes and result numbers are local commits on cx/threads.
Model weights, recordings, generated outputs, timing samples, XML and browser
profiles stay in scratch. Nothing was pushed or posted.

### WebGPU fc1 probe, run by the maintainer's side (Chrome 156, Apple Metal 3)

`node bench/gpu_fc1_chrome.mjs`: **0 of 7,864,320** values differ from native
(and from Rust) in each of three trials; median GPU time 0.40 ms. The
known-answer gate still **fails**: the subnormal fma case returns 0 instead of
the subnormal 2 (Metal flushes subnormal results to zero), and schedule cell 51
(a subnormal answer) does likewise. fc1's corpus inputs happen to produce no
subnormals, so the outputs agree, but bit-identity is not guaranteed on Metal for
inputs that reach the subnormal range. An exact GPU encoder would need either
proof that no subnormals arise, or detection and CPU fallback for affected tiles.

## Dynamic encoder chunks for heterogeneous cores (cx/chunks)

The maintainer supplied real Android 10 / Chrome 156 combined encoder+decoder
observations: about **2.8 s/staff with one thread**, **2.1–3.1 s with four**.
The eight-core phone is likely heterogeneous, but slow-rank dominance is an
inference, not a measured cause. The supplied iPhone *simulator* observation is
**111 -> 30 s per six-staff page**, strict/four threads, with MusicXML identical
to Chrome. This is not a physical iPhone measurement. No device timings are
invented or inferred from those observations in this round.

### Verified implementation stage

Every job now owns an atomic, saturating next-unit cursor. Every rank, including
the coordinator, claims another chunk until exhausted. The CAS loop prevents
wrap at exhaustion, including usize::MAX in its numeric regression test. The
cursor is 128-byte padded away from immutable metadata. Relaxed ordering is
sufficient for ownership; existing host publication/join atomics still provide
the input/output happens-before edges. The descriptor stays on the coordinator
stack until all ranks join. Workers construct mutable slices only for the
range they claimed; inputs and weights remain immutable.

GEMM chunks contain whole four-row tiles. Depthwise chunks contain whole
pixels, activations whole SIMD vectors, and LayerNorm whole rows. No K/channel
reduction is split or reassociated. Downsample's independent 128-term sums and
their ordered combination, bias schedules and all floating-point kernels are
unchanged. The strict guard now checks immutable B/bias/K once per job and A
once per claimed slice. This avoids rescanning a full weight matrix for every
small chunk; the same proven bounded-arithmetic conditions are enforced.
Slices failing the guard still execute general correctly rounded arithmetic.

The C ABI, dispatch signature and existing two host files are unchanged:
tools/encoder_pool.mjs and encoder_worker.mjs have **no changes**. Jobs are
opaque to JS and every worker receives the same compiled module. Hosts only
need to replace the WASM artifact; existing thread options continue to work.
Ordinary nonshared builds keep their previous implementation and exports.

The selected-source isolated Conv, depthwise, LayerNorm, MatMul/Gemm, Div, Erf
and Gelu checks pass at **1/2/4/6 threads** on both builds, with zero differing
bits. Native release library tests pass **13 tests**, including concurrent
exactly-once ticket coverage/tails and overflow-safe exhaustion. All-target,
all-feature native Clippy -D warnings passes. Full corpus follows final timing.

### Chunk-size evidence and selected geometry

The scratch diagnostic feature encoder-chunk-bench supports static scheduling,
chunk overrides, per-rank work statistics and WASM atomic-wait *sleeping*.
None of these hooks, delays, statistics or static scheduling paths is compiled
into production modules. There are no new imports. The normal shipped host
files also contain no slowdown option. The benchmark overrides only dispatch
in its own process, using diagnostic exports before publication/after joining.

A single-rank calibration warms once and records three full encodes. The
median of each job's elapsed time per unit sets an additional sleeping budget
of **3x calibrated work time** on ranks 2 and 3. This models slower service
without burning dummy instructions on the loaded machine. It is not a CPU
clock throttle or a claim that those ranks run at exactly one-quarter speed:
OS wait granularity and descheduling can increase the effective delay. Static
and dynamic use the same calibration, input, arithmetic and sleeping rule.
Dynamic sleepers release the remaining tickets for faster ranks to claim.

Sweep: four threads; one warm-up and three interleaved measured trials for
static, quarter/base/four-times-base dynamic chunks, separately per build.
Each run bit-compares the full context and verifies **33 native decoder steps**
using the same context pointer. Encoder-only and combined wall times are kept
separately because the decoder remains serial. This differs from comparing
phone combined times with an encoder-only Mac speedup. Process counters include
all case workers; different cases reside in separate idle processes. Only one
case executes at a time. Raw samples and work shares stay in scratch.

| diagnostic schedule / size | fused encoder seconds | fused combined seconds | strict encoder seconds | strict combined seconds |
|---|---:|---:|---:|---:|
| static equal shares | 1.8173 | 2.1218 | 27.3558 | 28.2057 |
| dynamic quarter size | 0.6200 | 1.0782 | 16.4161 | 17.1818 |
| dynamic base size | 0.7153 | 1.3004 | 14.8857 | 15.9238 |
| dynamic four-times size | 1.0812 | 1.3789 | 17.5724 | 18.6186 |

The op-group medians (GEMM, activation, depthwise, norm) agree with the whole
encode choice: quarter wins each fused group; base wins each strict group.
For example fused GEMM drops from 1262.69 ms static to 431.87 ms at quarter,
versus 473.15 ms base and 714.08 ms coarse. Strict GEMM is 12172.71/11108.32/
13211.65 ms for quarter/base/coarse. These are diagnostic elapsed intervals,
not claimed production instruction overhead or universal phone optima.

Selected base units: GEMM floor(1,000,000 / (4*K*N)), clamped to 1..64
four-row tiles; activation 2048 four-float vectors; depthwise floor(250,000 /
(C*kernel^2)) pixels, at least one; norm floor(4096/C) rows, at least one.
Strict uses these base sizes. Fused quarters the resulting integer unit count,
with a minimum of one. In the actual graph this gives:

| operation | fused chunk | strict chunk |
|---|---|---|
| MLP GEMM / projection | 4 rows | 4 or 24 MLP rows; projection 4 |
| stem Conv lowered GEMM | 64 rows | 256 rows |
| downsample partial GEMM, stages 1/2 | 8 / 4 rows | 40 / 20 rows |
| GELU/Erf | 2048 float32 values | 8192 float32 values |
| depthwise C=96/192/384 | 13 / 6 / 3 pixels | 53 / 26 / 13 pixels |
| LayerNorm C=96/192/384 | 10 / 5 / 2 rows | 42 / 21 / 10 rows |

There are many chunks per rank even in late stages (e.g. 320 four-row MLP
tiles), while expensive reductions stay intact. Frozen PR #4 production
artifacts in chunks_static are the independent instruction baseline. The new
static diagnostic schedule shares the hoisted guard with dynamic, isolating
scheduling for the slowdown experiment. Production instruction comparisons
use uninstrumented original-static versus uninstrumented selected-dynamic
modules, not the diagnostic feature. Counter overhead, balancing and 8-rank
recommendation measurements are still running at this stage.

### Independent instruction-budget check

Uninstrumented PR #4 static modules versus the selected uninstrumented dynamic
modules: one warm-up per case and three interleaved trials, separately isolated
case processes. Every output context and the paired 33-step decoder trajectory
match native. Process-wide proc_pid_rusage includes all ranks. No sleeping,
diagnostic counters or tuning callbacks are present in these production runs.

| total ranks | fused static G | fused dynamic G | dynamic change | strict static G | strict dynamic G | dynamic change |
|---|---:|---:|---:|---:|---:|---:|
| 1 | 20.976078 | 20.959912 | -0.077% | 267.661999 | 267.688295 | +0.010% |
| 4 | 20.988019 | 20.969987 | -0.086% | 268.300881 | 267.847858 | -0.169% |
| 6 | 20.988791 | 20.970397 | -0.088% | 268.651249 | 267.636136 | -0.378% |
| 8 | 21.024552 | 20.977367 | -0.224% | 268.983984 | 267.620287 | -0.507% |

The ~1% budget passes at every requested count. Small net reductions are not
claimed as an intrinsic cost-free queue: strict removes redundant immutable
weight scans, and code layout/loop setup also differ from the older module.
The operation and element work is not skipped. Input-slice guards still run.
Raw samples, encoder/decoder times, modules and trial order are retained in
chunks_overhead.json. The loaded machine makes wall-time ratios less reliable
than these retired-instruction totals.

### Final uneven-rank comparison and thread-count recommendation

These independent follow-up trials use the selected default chunks, the same
fixed calibration as the sweep, and additional sleeping on ranks 2..N-1. Thus
two ranks have no artificial delay; all other ranks add 3x calibrated unit
time. It is a controlled synthetic service-rate imbalance, not a measurement
of the Android phone's cores. Counts in the diagnostic module verify every
job's total claimed units, and native context/tokens are checked on every run.
The case is still angelus_0 staff 0, **33 decoder steps**. Three interleaved
trials per configuration, one warm-up, medians below. No fastest sample is used.

| build / total ranks | static encoder s | dynamic encoder s | static combined s | dynamic combined s |
|---|---:|---:|---:|---:|
| fused / 2 | 0.491401 | 0.456949 | 0.694371 | 0.661569 |
| fused / 4 | 1.405956 | 0.401338 | 1.614135 | 0.599002 |
| fused / 6 | 0.983674 | 0.359595 | 1.195267 | 0.561079 |
| fused / 8 | 0.760735 | 0.334056 | 0.971430 | 0.546372 |
| strict / 2 | 8.165176 | 8.286689 | 8.587021 | 8.706171 |
| strict / 4 | 25.302966 | 7.905961 | 25.787282 | 8.358725 |
| strict / 6 | 17.301198 | 6.609772 | 17.745162 | 7.033454 |
| strict / 8 | 13.615893 | 5.809145 | 14.025706 | 6.208149 |

The two-rank case has no artificially slow ranks. Dynamic and static strict
are within about 1.5% there; this experiment is about imbalance, not universal
speedup. With four ranks, the two undelayed ranks perform median **86.40% of
fused GEMM MACs / 86.71% strict**, versus **50%** under static ownership. This
work-weighted share counts MACs, not incomparable totals of vectors and pixels.
The observed effective slowdown is affected by OS wait granularity/load; the
requested sleeping budget and individual rank units/chunks remain in the raw
report so it is not misrepresented as an exact hardware frequency ratio.

Also delaying coordinator rank 0 and helper rank 3 at four total ranks gives:

| build | static encoder s | dynamic encoder s | static combined s | dynamic combined s |
|---|---:|---:|---:|---:|
| fused | 1.345755 | 0.404503 | 1.547690 | 0.603064 |
| strict | 25.416481 | 7.776948 | 25.826240 | 8.189261 |

Every rank really pulls, including the coordinator. The host still joins all
ranks before releasing a job; dynamic claiming removes computational equal-
share tails, not arbitrary OS suspension or the serial decoder. This does
not establish that the phone's original poor scaling was caused by core
heterogeneity; on-phone per-rank profiling is still needed to confirm that.

Recommend the caller start with **min(hardwareConcurrency || 4, 8)**, retaining
an override and validating UI/thermal behavior on-device. Use more ranks than
the number of fast cores: in this synthetic case, dynamic 8 versus dynamic 4
reduces combined latency by **8.79% fused / 25.73% strict**, and both beat the
two-fast-rank-only combined case. Six ranks also help; the extra benefit from
6 to 8 is smaller for fused, but all three fused trial pairs improve. The
production eight-rank instruction comparison also remains within budget.

The pool's existing API default remains 1 for backward compatibility. Partsong
already passes an explicit count; no host JS/C ABI change is needed for dynamic
claiming at its current count. Selecting the recommended larger count is an
optional change to that existing threads setting, not a protocol migration.
Partsong and homr were not edited. Raw evidence is chunks_calibration.json,
chunks_sweep.json, chunks_overhead.json and chunks_uneven.json in scratch.

### Freeze before full corpus

Production/diagnostic strict and fused WASM Clippy -D warnings pass. After a
style-only lint annotation, all six rebuilt modules are byte-for-byte identical
to the measured modules; full outputs are still validated directly, never by
hash. Selected-source op checks are repeated at 1/2/4/6 ranks and all pass.
The frozen production modules (no diagnostic exports/static path/delays) now
run all 627 staffs with four actual host ranks per build. Existing per-rank
host counts measure calls to encoder_work, **not chunks**: a rank may correctly
find no remaining work on a cheap job. Actual work shares are measured only
in the separate diagnostic experiment above.

### Fused dynamic corpus verified

The frozen production fused module completes **627/627 bit-identical contexts**
and its Rust decoder completes **38,076 native steps**, with zero differing
heads, steps or lengths. XML replay gives **75 byte-identical files and 13
matching native failures**. Production exports contain no encoder_bench_ entry
points, and imports remain only env.memory and encoder_threads.dispatch.
The strict production corpus is now running. Host pool/worker sources remain
byte-unchanged from main; no Partsong or homr edits, push or posting occurred.

### Final build and compatibility checks

`cargo fmt --check`, all-target/all-feature Clippy `-D warnings`, and
`cargo test --release -j1` pass (13 library tests, including concurrent
exactly-once/tail chunk coverage and exhaustion without integer wrap). Both
WASI SIMD variants also pass their applicable tests. Both nonshared WASM
builds pass combined encoder/decoder smoke checks; both shared builds pass
the same checks with a one-rank pool. These four checks retain 121 native
token steps and bit-identical contexts. The original single-thread API and
the threaded one-rank path remain working.

Rebuilt every shared configuration with `bench/build_encoder_threads.py`:
strict/fused production, probes, chunk diagnostics, and probes plus chunk
diagnostics. The six frozen production/probe/diagnostic artifacts reproduce
the measured modules byte for byte; the two combined probe/diagnostic
artifacts also build successfully. This is artifact provenance only: op/corpus comparisons use
actual float32 output bytes and token sequences, not hashes. WASM Clippy
passes for strict/fused production and diagnostic features.

Decision during validation: a fresh nonshared build made no visible dependency
progress for several minutes with the globally configured shared sccache
wrapper. Reran the checks with per-command `RUSTC_WRAPPER=''` and a separate
scratch target directory; they completed. Left global configuration and the
shared cache server alone. The cause of the cache stall was not diagnosed.
