# homr-rs

A Rust port of [homr](https://github.com/liebharc/homr), Christian Liebhardt's
optical music recognition, aimed at running in the browser as WebAssembly
without Python. homr is AGPL-3.0, and so is this port (see `LICENSE`).

**Status: prototype.** One stage is ported: staff detection
(`homr/staff_detection.py: detect_staff`, homr commit `863eb3c`). The rest of
homr, including running its models, is still Python.

## Faithful to homr

The port aims to give exactly homr's results, not approximately: same staffs,
same grid points. That includes the OpenCV routines homr relies on, ported from
OpenCV 5.0.0 step for step in float32 (`minAreaRect` with OpenCV's Sklansky
convex hull and rotating calipers, `rotatedRectangleIntersection` with its
centre shift, `boxPoints`), Python's round-half-to-even and numpy's pairwise
summation. Each of these changed results on real pages before it was matched.

Fixes or improvements to homr's behaviour belong upstream in homr first; this
port follows homr.

## Results so far (staff detection)

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
- Models stay ONNX, run by onnxruntime-web; Rust prepares their inputs and
  reads their outputs, so no part of the pipeline has to block on JavaScript.
- Test pages: Partsong's test images and choral PDFs, and homr's benchmark sets
  (polish-scores, SMB) for comparing outputs.
