# homr-rs

> **Experimental.** This port is new: it is not yet supported, and it has not
> yet been tested in real use. It matches homr on the test pages described
> below, but expect rough edges, and don't rely on it yet.
>
> **Written with AI.** Most of the code, tests and notes here were written by
> Claude (Anthropic's AI model, working in Claude Code) and Codex (OpenAI),
> directed by Paul Fremantle. Its correctness rests on the comparisons against
> homr described below (LOG.md has the detail), not on line-by-line human review.

A Rust port of [homr](https://github.com/liebharc/homr), Christian Liebhardt's
optical music recognition, aimed at running in the browser as WebAssembly
without Python. homr is AGPL-3.0, and so is this port (see `LICENSE`).

**Status: the pipeline, encoder and transformer decoder are ported.** Page,
staff and MusicXML processing is checked against native homr (OpenCV 4.11,
Pillow 12, numpy 2). Both Rust models target native arm64 ONNX Runtime
1.31.0's default CPU results. Segmentation (homr's segnet, a ResNet-18 U-Net) also runs in Rust (`segnet.rs`, loading the original ONNX file; NEON on arm64, SIMD128 in WebAssembly). The encoder loads its original
float32 ONNX weights and produces the context in Rust. The decoder
loads homr's original quantized ONNX file at runtime and runs in Rust, including
WebAssembly with SIMD128. Its caches stay inside Rust across symbols.
See LOG.md for numerical checks, reference versions, and measured limits.

| stage | from | to | checked on the corpus |
|---|---|---|---|
| 2 | page image | segnet tiles; masks from its output | 88/88 music pages identical |
| 1 | masks | staffs, systems (MultiStaffs) | 158/158 pages identical |
| 3 | page + staff | the transformer's canvas | 627/627 staff images identical |
| encoder | staff image | context [1280,512] | native NEON, fused WASM and strict WASM: bit-identical across the corpus; see LOG.md |
| decoder | encoder output | six tokens + coordinates per step | native + WASM: 627/627 staffs, 38,076 steps each, all six tokens identical; attention max error 0.000366 pixels |
| 4 | transformer output | MusicXML | 88/88 music pages: 75 byte-identical, 13 fail in homr itself (assert) and in the port alike |

## Using it

A page goes through segmentation, the encoder, and the Rust decoder.

    preprocess::preprocess(&bgr)                  // autocrop, resize to 1920 wide, CLAHE
    preprocess::tile_origins / tile_input         // -> segnet input tiles
      [segnet] -> per-tile argmax class maps
    preprocess::merge_tiles, masks, postprocess   // -> masks (noise filtered, staff strengthened)
    page::analyse_page(&masks)                    // -> staffs and systems (MultiStaffs)
    staff_image::prepare_staff(&staff, &gray, ..) // -> the transformer's canvas + mapping back
    staff_image::to_model_input(&canvas)          // -> encoder input
    encoder.encode(&image)?                     // -> context: 1280 * 512 f32 values
    decoder.generate(&context)          // -> six tokens + coordinates per step
    tokens::Vocab::symbol(tokens, coordinates)    // -> homr's symbols
    score::page_xml(&multi, &gray, &mapping, ..)  // -> homr's MusicXML (with a predict callback)

`score::page_xml` follows homr's own voice assembly; an app with its own layout
(Partsong) calls `prepare_staff` per staff and the symbol/MusicXML parts itself.

## Rust encoder

```rust,ignore
use homr_rs::{encoder::Encoder, decoder::Decoder};
let encoder = Encoder::from_onnx(&encoder_onnx_bytes)?;
let decoder = Decoder::from_onnx(&decoder_onnx_bytes)?;
let context = encoder.encode(&normalized_staff_image)?;
let steps = decoder.generate(&context)?;
```

The input is 256*1280 normalized float32 values from
`staff_image::to_model_input`, in [1,1,256,1280] order; the output is a
[1280,512] context. This fixed executor supports homr's
`encoder_pytorch_model_509-2293bba14bfea47d9d28411339362af15fce4c31.onnx`
(the 465 encoder has the same graph and loads too), not arbitrary ONNX graphs. It shares the protobuf reader at `src/onnx.rs`.
Native uses NEON fused arithmetic. The optional fused WASM artifact uses the
same fused-host probe as the decoder. Strict SIMD128 emulates fused rounding,
including halfway double-rounding and subnormal cases; it is substantially
more expensive. LOG.md records actual corpus checks and instruction counts.
Each encoder build followed by its Rust decoder also matches every native
token step; XML replay preserves identical pages and matching native failures.
The full corpus runs on recorded staff images, not a live browser
segmentation integration. See LOG.md for artifact and final-build provenance.

Both models are included in the same `homr_rs.wasm`. All existing decoder
exports remain available. Allocate with `decoder_alloc`, then use:

```js
// Input and output pointers are in the same module's linear memory.
const context = e.decoder_alloc(1280 * 512 * 4);
if (!e.encoder_run(encoder, image, context)) throw Error("encoder failed");
const steps = e.decoder_run(decoder, context, tokens, 608, 0);
// Or perform both stages without copying the internal context at all:
const combinedSteps = e.encoder_decode(encoder, decoder, image, tokens, 608);
```

Load/drop encoder handles with `encoder_load` / `encoder_drop`; check
`encoder_fused_simd_supported()` before loading the optional fused artifact.
The strict artifact does not require a passing fused probe. The safety and
buffer contracts are documented in `src/encoder/wasm.rs`.
`bench/encoder.mjs` checks contexts and can run the decoder directly on the
same output buffer; `bench/encoder_ops.mjs` checks isolated native probes.
Optional `encoder-probes`, `encoder-trace`, and `encoder-profile` features
are diagnostic only; normal artifacts have no diagnostic imports.

### Shared-memory encoder workers

`encoder-threads` adds allocation-free worker jobs for GEMM/convolution,
depthwise convolution, GELU and LayerNorm. It preserves each output's original
reduction order. The coordinator owns allocation and joins every operation;
workers share the weights and have private stacks/TLS. Build shared-memory
strict and fused artifacts separately from the ordinary nonshared fallback:

```sh
rustup toolchain install nightly --profile minimal --component rust-src
python bench/build_encoder_threads.py <scratch-output-dir>
```

The builder recompiles the standard library with atomics and bulk memory.
It produces `encoder-threads-strict.wasm` and `encoder-threads-fused.wasm`.
The fused artifact still requires its fused-host probe to pass. In a browser,
COOP/COEP isolation and a Web Worker coordinator are required: the pool uses
blocking `Atomics.wait`, which is forbidden on Window. The host needs no
wasm-bindgen glue:

```js
import {EncoderPool} from './tools/encoder_pool.mjs';
const module = await WebAssembly.compile(await (await fetch(wasmUrl)).arrayBuffer());
const pool = await EncoderPool.create(module, {threads: 4});
const e = pool.exports;
// Existing decoder_alloc/load and encoder_load/run/decode signatures apply.
// Drop models and host buffers after inference, then release the pool:
await pool.close();
```

Each rank, including the coordinator, repeatedly claims whole output chunks
from an atomic counter. Faster cores therefore take more work; reductions
retain their original order. Chunk budgets differ between strict and fused
builds; LOG.md records the sweep, exactness checks and instruction overhead.
The descriptor is opaque to the host, so existing pool/worker files and C ABI
calls remain compatible: update the WASM asset to enable dynamic scheduling.

For a caller-selected worker count, the measured starting recommendation is
`Math.min(navigator.hardwareConcurrency || 4, 8)`, with a device-specific
override. This can use slower cores as well as fast ones. The pool API default
remains one thread for compatibility. Device thermal/UI behavior and combined
encoder/serial-decoder latency still need testing on the target phone.

`bench/chunk_bench.mjs` measures static/dynamic production instruction counts
and diagnostic uneven-rank timing. `--chunk-bench` on the shared builder enables
sleeping-rank simulation and work-share counters only in separately named
diagnostic artifacts; do not ship those artifacts.

Only the coordinator may call graph/allocator exports; worker instances call
`encoder_work` with disjoint output ranges. Treat a pool as unusable after a
worker failure and terminate it before discarding its memory. Ordinary
nonshared builds keep the existing exports and work without isolation.
LOG.md records exactness, interleaved scaling results and remaining device
validation. `bench/thread_scaling.mjs` measures total process instructions and
wall time; `bench/threads_chrome.mjs` is the isolated-page browser harness.

### WebGPU encoder candidate

`tools/encoder_gpu.mjs` implements the full ordered encoder with GPU buffers
and a CPU fallback. Copy it together with `encoder_gpu_math.mjs` and
`encoder_gpu_kernels.mjs` from the same revision. The existing pool/worker files
and CPU exports remain compatible. This candidate still requires the external
Chrome/Safari qualification runs below: LOG.md separates CPU/offline checks
from actual GPU evidence.

```js
import {EncoderGPU} from './tools/encoder_gpu.mjs';
// e is the ordinary WASM exports object or pool.exports; encoder is already loaded.
const gpu = await EncoderGPU.create({exports: e, encoder, onReport: console.log});
const result = await gpu.encode(image, context);
// result.backend says whether a flagged/unavailable GPU used the CPU fallback.
const steps = e.decoder_run(decoder, context, tokens, 608, 0);
gpu.destroy(); // before dropping the encoder handle or closing its pool
```

Serialize calls, keep the image unchanged until `encode()` resolves, and
retain the encoder handle until `destroy()`. Run shared
builds in a Web Worker, as above. Ordinary builds also work on Window. Rust
`encoder_gpu_plan_create` exports immutable JSON metadata and packed weights
through pointer/length getters; JavaScript never parses ONNX. The temporary
CPU plan is freed after upload unless diagnostics request `retainPlan: true`.
The final mapped context copies straight into WASM memory for the decoder.

The GPU requires only default limits: four storage bindings per pipeline, one uniform,
64 invocations/workgroup, less than 13 KiB workgroup memory, and bindings below
128 MiB. No fp16 or subgroup features. Timestamp queries are optional; reports
say `null` when unavailable. The self-test covers each production kernel,
fused rounding, division/sqrt, polynomial activations and underflow detection.
It reports full initialization, shader compilation and probe-only times.
Any failed requirement/test, arithmetic flag, validation/internal/OOM error or
device loss selects the existing CPU encoder. Flagged raw GPU values are
discarded. The guard uses raw integer bits and conservative zero-result checks;
normal division/sqrt use integer midpoint certificates after refinement.
An integer GPU pass checks all weights at load; another checks/sanitizes the
GPU image copy before float use on each staff. The original WASM image remains
intact for CPU fallback. Staff timestamps include the image scan.

WGSL alone does not guarantee RN-even fused arithmetic. The runtime integer
rounding barrier preserves separate operations, while known answers and corpus
runs qualify the actual add/mul/fma implementation. A passing finite self-test
is not a proof for every conforming GPU/compiler. Do not infer full encoder
qualification from the older `bench/gpu_fc1_chrome.mjs` operator probe.

From this clone, with `S` set to the supplied scratchpad path:

```sh
python3 bench/build_decoder_variants.py "$S/gpu_single"
python3 bench/build_encoder_threads.py "$S/threads_build"
# First run op checks and one staff with per-kernel timing (partial primitive audit):
node bench/gpu_encoder_chrome.mjs "$S" "$S/threads_build/encoder-threads-fused.wasm" "$S/gpu-chrome-one.json" --limit 1 --profile 3 --audit none --math-cases 65536
# Then the complete corpus, including every native arithmetic operand:
node bench/gpu_encoder_chrome.mjs "$S" "$S/threads_build/encoder-threads-fused.wasm" "$S/gpu-chrome.json"
# Open the printed local URL in Mac Safari; JSON is POSTed to this server's /report:
python3 bench/gpu_encoder_server.py "$S" "$S/threads_build/encoder-threads-strict.wasm" "$S/gpu-safari"
# Same page, one staff/per-kernel profiling in Safari (open the printed URL):
python3 bench/gpu_encoder_server.py "$S" "$S/threads_build/encoder-threads-strict.wasm" "$S/gpu-safari-one" --limit 1 --profile 3 --audit none --math-cases 65536
```

The static page runs the coordinator in a dedicated worker and supplies
COOP/COEP headers. If an engine exposes WebGPU only on Window, use the ordinary
strict WASM path with `--main-thread --threads 1`. Both runners first test large
random/edge primitive sets, then all intermediate operations on one staff,
then the corpus. Default `--audit full` streams every native div/sqrt and
GELU/Erf/inner-exp operand through GPU checks for every staff, keeping an
independent CPU state even when the GPU drifts or flags. It also compares all
GPU op outputs. This audit is expensive and separate from the timestamped
production pass. `--audit first` limits this extra audit to the first staff;
`--audit none` omits corpus primitive audits. These report partial coverage.
`--math-cases` and `--only page/staff` support focused debugging.

Reports distinguish raw GPU drift/tokens, effective results after fallback,
guard reasons/frequency, timestamp timings, and interleaved guarded/unguarded
timing controls. Unguarded controls never supply an encoder result. Only
`corpus_qualified: true` denotes the complete unflagged GPU/corpus/audit target.
`bench/check_gpu_shaders.py` performs offline WGSL validation/Metal translation;
it does not execute the GPU. `bench/check_gpu_host.mjs` tests fallback/control
logic and optional real-WASM ABI compatibility, also without GPU claims.

The production dense kernels cache 16x32 output tiles with 16 K values at a
time; each invocation owns two vec4 outputs. Both sums retain ascending K.
Conv preserves the stem's initial bias and downsampling's independent
128-term sums and subsequent bias. Depthwise caches a 4x4 spatial tile and
its halo, preserving all 49 FMAs including padded zeros. LayerNorm assigns
one whole row to each invocation instead of leaving 63 lanes waiting for the
Welford loop. Its divisions, sqrt and affine arithmetic are unchanged. GELU,
scale, residual addition, position addition, guards and scalar math are
unchanged. No cross-invocation float reduction or new arithmetic approximation.

`--profile 3` adds per-pass timestamp medians, totals by op kind, separate
image/weight scan timings, and a comparison with the previous guarded schedule.
Tiled guarded, tiled arithmetic-control and legacy guarded runs are interleaved
(one warm-up plus three trials each), and all final contexts are checked against
native. `profile.valid` must be true before using speedup/guard deltas. These
differential estimates can be negative under noise; they are not hardware
counters. Controls retain certificates and rounding barriers and never supply
an encoder result. The weight-scan metric is a warm scan of resident weights;
full initialization wall time is reported separately. Certified div/sqrt,
including their flags, remain part of arithmetic in the guard control: the
delta removes wrapper FTZ/domain checks. The quick profiling commands omit the expensive corpus
primitive streams and report that partial coverage; use the full command for
qualification. Missing timestamp-query reports profiling unavailable.

For a phone, the same server/page works via Android localhost forwarding
(`adb reverse tcp:8796 tcp:8796`, then open `http://localhost:8796/`) or a trusted
HTTPS LAN endpoint (`--bind 0.0.0.0 --cert CERT.pem --key KEY.pem`). Plain LAN
HTTP is not a WebGPU secure context. Use a certificate trusted by the phone;
the server does not install certificates or change system trust.

Optional Node/Dawn path, installed entirely into scratch:

```sh
npm install --prefix "$S/gpu_node" --cache "$S/npm-cache" webgpu
node bench/gpu_encoder_node.mjs "$S" "$S/gpu_single/decoder-fused.wasm" "$S/gpu-node.json" "$S/gpu_node/node_modules/webgpu/index.js" --limit 1 --profile 3 --audit none --math-cases 65536
python3 bench/check_gpu_shaders.py "$S/gpu_mobile_validate" --offline
RUSTC_WRAPPER='' cargo run --release --bin gpu_plan_check -- "$S"/models_files/encoder_pytorch_model_465-*.onnx "$S/models_native/angelus_0/0/input.f32" "$S/decoder_native_ref/angelus_0/0/context.bin" --simulate
```

The last command independently simulates tile/lane ownership and data movement
against the CPU encoder at every operation, and checks partial-tile self-tests.
It does not execute GPU instructions or measure GPU speed. The load-time
self-test includes the same partial-tile cases. ABI version and `EncoderGPU`
usage are unchanged; copy all three `tools/encoder_gpu*.mjs` files together.

## Rust decoder

```rust,ignore
use homr_rs::{decoder::Decoder, tokens::Vocab};
let decoder = Decoder::from_onnx(&downloaded_onnx_bytes)?;
let steps = decoder.generate(&encoder_output)?;
let vocab = Vocab::default();
let symbols: Vec<_> = steps.iter().filter_map(|s| s.symbol(&vocab)).collect();
```

For individual steps, create `decoder.session(&encoder_output)?`, then call
`session.step([rhythm, pitch, lift, articulation, slur])`; start with
`[1, 0, 0, 0, 0]`. Results use ONNX output order: rhythm, pitch, lift,
position, articulation, slur. `generate` includes the EOS step for checking;
`Step::symbol` removes it. Generation stops at rhythm EOS (2), or 608 steps.

This executor supports homr's `decoder_pytorch_model_509-2293bba14bfea47d9d28411339362af15fce4c31.onnx`
graph (the 465 one differs only in its vocabulary sizes, 260/62 instead of 278/89), not arbitrary ONNX networks. It follows the native arm64 ONNX Runtime
1.31.0 default CPU kernels, including KleidiAI's fused quantization semantics.
No runtime dependencies were added. Native int8 multiplication uses NEON/dotprod
where available; WASM uses SIMD128; other native targets have a portable path.
Cross-target validation details and limitations are in LOG.md.

```sh
RUSTFLAGS='-C target-feature=+simd128' cargo build --release --target wasm32-unknown-unknown
```

An optional faster artifact uses `+simd128,+relaxed-simd`. It issues explicit
fused float32 SIMD multiply-add, preserving the reference's accumulation order.
The WebAssembly specification permits relaxed multiply-add to be unfused on
some hosts: this artifact rejects model loading if its arithmetic probe is
unfused. A passing probe is not a substitute for validating the target engine.
It has passed the complete corpus on the supplied ARM64 Node/V8 host; other
engines should use the strict artifact until checked.

Build separately labelled artifacts in scratch (no weights are embedded):

```sh
python bench/build_decoder_variants.py <scratch-output-dir>
```

This produces `decoder-simd128.wasm` and `decoder-fused.wasm`. Hosts must first
check that the engine accepts the module (for example `WebAssembly.validate`),
then check `decoder_fused_simd_supported()` on the optional artifact before
loading weights. Keep the strict artifact as the fallback. Strict SIMD128
stores exact float32 K/V values widened to float64 to reduce inner-loop
conversions: maximum reserved cache capacity is about 164 MB, versus 82 MB
for native and the fused artifact. These figures exclude weights and scratch
buffers; neither artifact quantizes the cache values.

The resulting `homr_rs.wasm` also exports a minimal C ABI, documented in
`src/decoder/wasm.rs`: allocate buffers, load the model once, run one staff,
and free buffers/model. Each output step is six little-endian u32 tokens plus
two float32 coordinates (32 bytes). No JS round trip is needed per symbol.
`bench/decoder.mjs` demonstrates the host calls. Keep weights and encoder
outputs outside the library; neither is embedded in the binary.

## Faithful to homr

The port aims to give exactly homr's results, not approximately: same staffs,
same grid points. That includes the OpenCV routines homr relies on, ported from
OpenCV 5.0.0 step for step in float32 (`minAreaRect` with OpenCV's Sklansky
convex hull and rotating calipers, `rotatedRectangleIntersection` with its
centre shift, `boxPoints`), Python's round-half-to-even and numpy's pairwise
summation. Each of these changed results on real pages before it was matched.

Fixes or improvements to homr's behaviour belong upstream in homr first; this
port follows homr.

## Speed

Whole page from masks to MusicXML (the transformer's output replayed), in the
browser engine (Node/V8): homr in Pyodide 38.1G retired instructions a page,
this port as WebAssembly 2.2G (17x fewer). Staff detection alone: 41x fewer.
Stage 2 (page image to segnet input, its output to masks): Pyodide 6.5G a page,
the port 4.4G (here homr's work is mostly native numpy/OpenCV/Pillow already).

Decoder only, four staffs / 123 steps, three paired load-subtracted runs:
strict SIMD128 **182.32M** retired instructions/step versus ORT-web **479.06M**
(**2.63x fewer**); the optional fused relaxed-SIMD artifact uses **119.55M**
versus **481.48M** (**4.03x fewer**) on the validated ARM64 Node/V8 host.
Native Rust uses 56.25M versus single-threaded native ORT's 46.38M.
The 3x target is met by the optional fused artifact, **not** the portable
SIMD128 fallback. LOG.md contains the component profile, memory tradeoff,
failed unfused-arithmetic experiment, and before/after tables.

## Results for staff detection (the first stage ported)

Counted in retired instructions (`proc_pid_rusage`), not wall time.

| | pages | same as homr | Python | Rust |
|---|---|---|---|---|
| native, homr with Partsong's speed patches | 151 | 151 | 610 G | 22.6 G (27x fewer) |
| native, plain homr | 6 | 6 | 62.5 G | 1.7 G (38x fewer) |
| browser engine (Node/V8): Pyodide vs WebAssembly | 25 | 25 | 374 G | 9.1 G (41x fewer) |

## Checking against homr

You need a homr checkout with its model files (`homr/segmentation/*.onnx`).

    # dump homr's inputs and results per page
    uv run --no-project --with numpy --with opencv-python-headless --with scipy \
      --with scikit-image --with pillow --with musicxml --with typing_extensions \
      --with scikit-learn --with onnxruntime \
      python -I tools/dump_staff.py <homr dir> cases <page images...>

    # compare (and count instructions)
    cargo run --release --bin compare cases

To check whole pages end to end against native homr (needs the same pinned
reference versions as below, with Pillow in homr's range, `>=12.1.1,<13`: Pillow 13
resizes differently):

```sh
python -I tools/dump_segnet.py <homr dir> seg <page images...>     # image, segnet tiles, homr's masks
python -I tools/run_corpus.py <homr dir> refs <page images...>     # homr's MusicXML or its error
python -I tools/dump_canvases.py <homr dir> canv <page images...>  # homr's staff images
cargo run --release --bin check_pages -- seg refs canv <encoder.onnx> <decoder.onnx> [page names...]
```

`check_pages` redoes everything after segmentation in Rust (autocrop, CLAHE,
tile merging, masks, staff detection, system repair, staff images, the
transformer, MusicXML) and reports per page whether the masks and the XML are
identical, or whether both fail. Native
arm64 OpenCV/numpy fuse multiply-adds, so some values differ by one grey level:
the report counts CLAHE and staff-image differences, and the staff images
homr produced are fed to the transformer so those one-level differences
can't change the XML. On the 148-page Partsong corpus every page agrees.

To record and check the transformer decoder, pin the native reference versions
that reproduce the existing corpus (ONNX Runtime 1.31.0 and OpenCV 4.11):

```sh
uv run --no-project --with numpy==2.5.4 --with onnxruntime==1.31.0 \
  --with opencv-python-headless==4.11.0.86 --with scipy --with scikit-image \
  --with "pillow>=12.1.1,<13" --with musicxml --with typing_extensions --with scikit-learn \
  python tools/record_decoder.py <homr-dir> <scratch-reference-dir> <page-images...>
cargo run --release --bin check_decoder -- <decoder.onnx> <scratch-reference-dir> <scratch-rust-dir>
```

The recorder runs homr's actual greedy loop, saves encoder outputs, all logits,
chosen tokens and coordinates, and resumes completed pages. It copies source
images into its scratch output directory before running homr, so input image
directories are untouched. The Rust checker runs its own greedy trajectory,
reports every differing page/staff/step and both logit margins.
Keep the large recordings in scratch. `bench/decoder.mjs rust check ...` also
checks the WASM build against these recordings. To stream the whole corpus
without loading all encoder outputs into JS simultaneously, use `@all`:

```sh
DECODER_WASM=<decoder.wasm> REPORT=<scratch-report.json> \
  node bench/decoder.mjs rust check <decoder.onnx> <scratch-reference-dir> @all
```


Decoder instruction benchmarks are in `bench/measure_decoder.py` (use `--help`).
Select an artifact explicitly with `--wasm <file>`; the report records its path.
`--native-bin` permits benchmarking a separate checkout's native executable.
They compare identical staffs and fixed step counts, including first-step
context projections, with a separate load-only baseline for each engine.
Use `--ort-module` to point to onnxruntime-web 1.30.0's `ort.node.min.mjs`, and
`--python` for the pinned native reference environment. Both benchmark ORT
engines use one thread; recording preserves native default session options.
`/usr/bin/time -l` is the default counter; `--libproc` reads the macOS retired
instruction counter directly when sandbox restrictions prevent `time -l`.

For component profiling, compile with `--features decoder-profile`, then run
`bench_decoder profile ...` natively or `bench/profile_decoder.mjs` for WASM.
On macOS the latter imports a benchmark-only Node counter addon:

```sh
cc -bundle -undefined dynamic_lookup -I/opt/homebrew/include/node \
  bench/counter.c -o <scratch-dir>/counter.node
```

Set `COUNTER_ADDON` to that file and `PROFILE_WASM` to the instrumented WASM.
Use a separate `CARGO_TARGET_DIR` for profile builds. Normal decoder builds
contain no counter callbacks or allocation instrumentation. The synthetic
SIMD tests also run through `bench/run_wasi.mjs`; this is a test runner, not
a benchmark. All instruction benchmarking stays outside CI.

When a case differs, `tools/dump_mid.py` + `cargo run --release --bin mid cases/<page>`
compare the intermediate steps (anchors, raw staffs).

The benchmarks below are for measuring by hand, not for CI; CI (when added)
should only check that the Rust matches homr.

Browser engine (instructions for one pass = half the difference between 3 and 1 reps):

    (cd bench && npm i)
    cargo build --release --bin bench --target wasm32-wasip1
    /usr/bin/time -l node bench/py.mjs <homr dir> 1 cases/*    # and 3
    /usr/bin/time -l node --no-warnings bench/rs.mjs 1 cases/*  # and 3

## Plan

- Port homr stage by stage, each checked against homr on the same pages:
  staff detection (done), then the rest of the page analysis (symbols, notes,
  bar lines, braces), staff image preparation for the models, decoding the
  models' output, and MusicXML.
- Keep homr's stages callable separately, so an app can add its own layout
  logic between them (as Partsong does with choir systems).
- Segmentation stays in onnxruntime-web. The Rust encoder and decoder read
  the upstream ONNX weights at runtime without converted model files.
- Test pages: Partsong's test images and choral PDFs, and homr's benchmark sets
  (polish-scores, SMB) for comparing outputs.
