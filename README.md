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
staff and MusicXML processing is checked against homr as Partsong runs it
(Pyodide, OpenCV 4.11). Both Rust models target native arm64 ONNX Runtime
1.31.0's default CPU results. Segmentation stays with onnxruntime-web. The encoder loads its original
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
`encoder_pytorch_model_465-597144cab54c8f6d0f6c9619df5c5312694eadd6.onnx`,
not arbitrary ONNX graphs. It shares the protobuf reader at `src/onnx.rs`.
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

`bench/gpu_fc1_chrome.mjs` prototypes ordered WGSL fc1 arithmetic with a
load-time known-answer gate. This is an experimental operator probe, not a GPU
encoder. LOG.md distinguishes actual results from scripts blocked by Chrome
launch restrictions.

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

This executor supports homr's `decoder_pytorch_model_465-597144cab54c8f6d0f6c9619df5c5312694eadd6.onnx`
graph, not arbitrary ONNX networks. It follows the native arm64 ONNX Runtime
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

To record and check the transformer decoder, pin the native reference versions
that reproduce the existing corpus (ONNX Runtime 1.31.0 and OpenCV 4.11):

```sh
uv run --no-project --with numpy==2.5.4 --with onnxruntime==1.31.0 \
  --with opencv-python-headless==4.11.0.86 --with scipy --with scikit-image \
  --with pillow==13.0.0 --with musicxml --with typing_extensions --with scikit-learn \
  python tools/record_decoder.py <homr-dir> <scratch-reference-dir> <page-images...>
cargo run --release --bin check_decoder -- <decoder.onnx> <scratch-reference-dir> <scratch-rust-dir>
cargo run --release --bin check_xml -- <pages-dir> <scratch-rust-dir> <replay-dir>
```

The recorder runs homr's actual greedy loop, saves encoder outputs, all logits,
chosen tokens and coordinates, and resumes completed pages. It copies source
images into its scratch output directory before running homr, so input image
directories are untouched. The Rust checker runs its own greedy trajectory,
reports every differing page/staff/step and both logit margins, and writes
`decoder.json` from its own tokens and attention for `check_xml`.
Keep the large recordings in scratch. `bench/decoder.mjs rust check ...` also
checks the WASM build against these recordings. To stream the whole corpus
without loading all encoder outputs into JS simultaneously, use `@all`:

```sh
DECODER_WASM=<decoder.wasm> REPORT=<scratch-report.json> \
  node bench/decoder.mjs rust check <decoder.onnx> <scratch-reference-dir> @all
```

To also generate WASM `decoder.json` files for `check_xml`, set
`SAVE_DECODER=<scratch-output-dir>` and `VOCAB=<scratch-vocabulary.json>`;
create the latter with `cargo run --release --bin dump_vocab` redirected into
scratch. Every staff must complete before a page's output is written.

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
