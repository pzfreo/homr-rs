"""python tools/synth_staff_cases.py <homr dir> <cases dir> <count> <seed>
(needs record_decoder.py's Python dependencies; PAGES_DIR=<dir> also saves the masks as
tools/dump_page.py does, for tools/pyodide_ref.mjs; NONOTES=1 makes pages with only rests)
Synthetic segmentation masks (staff lines, clefs, noteheads, stems, bar lines) in the
situations homr's staff detection handles: staffs split by a gap with symbols in it,
staffs without a clef, short staffs of a neighbor page at the edge, the empty gutter
between the pages of a book, tilted lines. They stand in for the segmentation
model's output; homr's detect_staff then gives the reference, written like
tools/dump_staff.py does (compare them with the `compare` binary)."""
import json, os, random, sys
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import cv2
import numpy as np
from homr_setup import config, load

main = load(os.path.abspath(sys.argv[1]))
from homr.debug import Debug  # noqa: E402
from homr.model import InputPredictions  # noqa: E402
from homr.music_xml_generator import XmlGeneratorArguments  # noqa: E402

out_root, count, seed = sys.argv[2], int(sys.argv[3]), int(sys.argv[4])
H, W = 1700, 1200


def line(img, x0, y0, x1, y1, t=2):
    cv2.line(img, (int(x0), int(y0)), (int(x1), int(y1)), 255, t)


def make(rng):
    staff, symbols, clefs, heads, stems = (np.zeros((H, W), np.uint8) for _ in range(5))
    unit = rng.randint(10, 16)
    top = rng.randint(70, 140)
    pitch = 4 * unit + rng.randint(4 * unit, 8 * unit)
    n = rng.randint(2, 7)
    x0, x1 = rng.randint(50, 110), rng.randint(1070, 1150)
    book = rng.random() < 0.2
    for s in range(n):
        y0 = top + s * pitch
        if y0 + 4 * unit + 30 >= H:
            break
        slope = rng.uniform(-0.015, 0.015) if rng.random() < 0.5 else 0.0
        parts = [(x0, x1)]
        if book:
            mid = rng.randint(540, 660)
            gutter = rng.randint(50, 100)
            parts = [(x0, mid - gutter // 2), (mid + gutter // 2, x1)]
        elif rng.random() < 0.12:
            parts = [(x0, x0 + rng.randint(250, 450))]  # a short staff
        blurry = not book and rng.random() < 0.3  # clef and bar lines missed: only the lines show
        if blurry:
            parts = [(parts[0][0] + rng.randint(70, 330), min(W - 5, parts[0][1] + rng.randint(-60, 130)))]
        for pi, (a, b) in enumerate(parts):
            gaps = []
            if not book and rng.random() < 0.4:  # symbols cover the lines for a while
                g0 = rng.randint(a + 250, max(a + 260, b - 300))
                gaps.append((g0, g0 + rng.randint(40, 150)))
            if book and pi == 0 and rng.random() < 0.5:  # something in the gutter
                symbols[y0 : y0 + 4 * unit, b + 5 : b + 40] = 255
            jump = rng.randrange(5) if gaps and rng.random() < 0.3 else None  # one line jumps at the gap
            shift = rng.uniform(0.3, 1.1) * unit * rng.choice([-1, 1])
            for i in range(5):
                yy = y0 + i * unit
                if jump == i:
                    line(staff, a, yy + slope * a, gaps[0][0], yy + slope * gaps[0][0])
                    line(staff, gaps[0][1], yy + shift + slope * gaps[0][1], b, yy + shift + slope * b)
                else:
                    line(staff, a, yy + slope * a, b, yy + slope * b)
            for g0, g1 in gaps:
                staff[:, g0:g1][max(0, y0 - 6) : y0 + 4 * unit + 8, :] = 0
                symbols[y0 - 4 : y0 + 4 * unit + 4, g0:g1] = 255
                if rng.random() < 0.7:  # a clef or key change in it
                    clefs[y0 - unit : y0 + 5 * unit, g0 + 2 : g0 + 36] = 255
            if (rng.random() < 0.85 or pi > 0) and not blurry:  # the clef at the start (not missed)
                if rng.random() < 0.85:
                    cx = a + 4
                    staff[y0 - 6 : y0 + 4 * unit + 8, cx : cx + 34] = 0
                    clefs[y0 - unit : y0 + 5 * unit, cx : cx + 30] = 255
                    symbols[y0 - unit : y0 + 5 * unit, cx : cx + 30] = 255
            for _ in range(0 if (blurry or os.environ.get('NONOTES')) else rng.randint(2, 12)):  # notes (none: only rests)
                nx = rng.randint(a + 60, b - 20)
                ny = y0 + rng.randint(0, 8) * unit // 2
                cv2.ellipse(heads, (nx, int(ny + slope * nx)), (unit // 2 + 1, unit // 2 - 1), 0, 0, 360, 255, -1)
                cv2.ellipse(symbols, (nx, int(ny + slope * nx)), (unit // 2 + 1, unit // 2 - 1), 0, 0, 360, 255, -1)
                line(stems, nx + unit // 2, ny - 3 * unit, nx + unit // 2, ny, 2)
            for bx in ([] if blurry else range(a + 200, b - 50, rng.randint(150, 260))):  # bar lines
                line(stems, bx, y0 + slope * bx, bx, y0 + 4 * unit + slope * bx, 2)
    if rng.random() < 0.3:  # staffs of the neighbor page at the edge of the photo
        yy = top + rng.randint(0, 3) * pitch + rng.randint(-20, 20)
        left = rng.random() < 0.5
        a, b = (0, rng.randint(40, 100)) if left else (rng.randint(1110, 1160), W - 1)
        for i in range(5):
            line(staff, a, yy + i * unit, b, yy + i * unit)
    if rng.random() < 0.15:  # lines shining through from the other side of the paper
        yy = top + rng.randint(1, 3) * pitch + unit * 2
        for i in range(5):
            line(staff, x0, yy + i * unit, x1, yy + i * unit, 1)
    return staff, symbols, clefs, heads, stems


class Stop(Exception):
    pass


def box(b):
    (cx, cy), (w, h), a = b.box
    return [float(cx), float(cy), float(w), float(h), float(a)]


_orig = main.detect_staff
cur = {}
def spy(debug, image, fragments, clefs, bars, symbols=None):
    result = _orig(debug, image, fragments, clefs, bars, symbols)
    d = cur["dir"]; os.makedirs(d, exist_ok=True)
    np.ascontiguousarray(image, dtype=np.uint8).tofile(os.path.join(d, "staff.bin"))
    if symbols is not None:
        np.ascontiguousarray(symbols, dtype=np.uint8).tofile(os.path.join(d, "symbols.bin"))
    json.dump({
        "shape": list(image.shape),
        "fragments": [{"box": box(f), "contour": np.asarray(f.contours).reshape(-1, 2).tolist()} for f in fragments],
        "clefs": [box(c) for c in clefs], "bars": [box(c) for c in bars],
        "expected": [[[float(p.x), [float(y) for y in p.y], float(p.angle)] for p in s.grid] for s in result],
        "python_instructions": 0,
    }, open(os.path.join(d, "case.json"), "w"))
    raise Stop
main.detect_staff = spy


def predictions(staff, symbols, clefs, heads, stems):
    pre = np.where(staff > 0, 0, 255).astype(np.uint8)
    p = InputPredictions(pre, pre, heads, symbols, staff, clefs, stems)
    p.staff = main.make_lines_stronger(p.staff, (1, 2))
    if os.environ.get("PAGES_DIR"):  # the same masks as tools/dump_page.py writes, for tools/pyodide_ref.mjs
        d = os.path.join(os.environ["PAGES_DIR"], os.path.basename(cur["dir"])); os.makedirs(d, exist_ok=True)
        for name, m in (("notehead", p.notehead), ("symbols", p.symbols), ("staff", p.staff), ("clefs_keys", p.clefs_keys), ("stem_rests", p.stems_rest), ("preprocessed", p.preprocessed)):
            np.ascontiguousarray(m, dtype=np.uint8).tofile(os.path.join(d, f"{name}.bin"))
        json.dump({"shape": list(p.staff.shape), "preprocessed_shape": list(p.preprocessed.shape), "contours": {}}, open(os.path.join(d, "page.json"), "w"))
    return p, Debug(pre, "synthetic.png", False), None


rng = random.Random(seed)
made = 0
for n in range(count):
    layers = make(rng)
    cur["dir"] = os.path.join(out_root, f"synth-{seed}-{n}")
    main.load_and_preprocess_predictions = lambda *a, layers=layers: predictions(*layers)
    try:
        main.process_image("synthetic.png", config(main), XmlGeneratorArguments())
    except Stop:
        made += 1
    except Exception as e:  # homr itself fails on this one
        print("skip", n, type(e).__name__, e, flush=True)
print("cases", made, flush=True)
