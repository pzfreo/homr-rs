# homr-rs

A Rust port of [homr](https://github.com/liebharc/homr), Christian Liebhardt's
optical music recognition, aimed at running in the browser as WebAssembly
without Python. homr is AGPL-3.0, and so is this port (see `LICENSE`).

**Status: everything except the models is ported**, checked against homr as
Partsong runs it (in Pyodide, with OpenCV 4.11): from the page image to the
segmentation model's input, from its output to the staffs and systems, the
staff images the transformer reads, and from the transformer's output to
homr's MusicXML. The three ONNX models (segmentation, encoder, decoder) stay
with onnxruntime(-web); Rust prepares their inputs and reads their outputs, so
nothing has to block on JavaScript. See LOG.md for how each stage was checked.

| stage | from | to | checked on the corpus |
|---|---|---|---|
| 2 | page image | segnet tiles; masks from its output | 88/88 music pages identical |
| 1 | masks | staffs, systems (MultiStaffs) | 158/158 pages identical |
| 3 | page + staff | the transformer's canvas | 627/627 staff images identical |
| 4 | transformer output | MusicXML | 88/88 music pages: 75 byte-identical, 13 fail in homr itself (assert) and in the port alike |

## Using it

A page goes through the models three times; Rust does everything around them.

    preprocess::preprocess(&bgr)                  // autocrop, resize to 1920 wide, CLAHE
    preprocess::tile_origins / tile_input         // -> segnet input tiles
      [segnet] -> per-tile argmax class maps
    preprocess::merge_tiles, masks, postprocess   // -> masks (noise filtered, staff strengthened)
    page::analyse_page(&masks)                    // -> staffs and systems (MultiStaffs)
    staff_image::prepare_staff(&staff, &gray, ..) // -> the transformer's canvas + mapping back
    staff_image::to_model_input(&canvas)          // -> encoder input
      [encoder, decoder loop] -> token numbers + coordinates per step
    tokens::Vocab::symbol(tokens, coordinates)    // -> homr's symbols
    score::page_xml(&multi, &gray, &mapping, ..)  // -> homr's MusicXML (with a predict callback)

`score::page_xml` follows homr's own voice assembly; an app with its own layout
(Partsong) calls `prepare_staff` per staff and the symbol/MusicXML parts itself.

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
