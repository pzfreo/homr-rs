// Reference staff images from homr as Partsong runs it (Pyodide): for each
// page dumped by tools/dump_page.py, runs homr's page analysis and then
// parse_staffs with the model call stubbed out, recording for every staff what
// prepare_staff_image made of it: the canvas the model would get, the image
// before and after dewarping, and the dewarp's points and triangles.
//   node tools/pyodide_staffs.mjs <homr dir> <pages dir> <out dir> [pages...]
import { createRequire } from "node:module";
import fs from "node:fs";
import path from "node:path";
const here = path.dirname(new URL(import.meta.url).pathname);
const require = createRequire(path.join(here, "../bench/package.json"));
const { loadPyodide } = require("pyodide");
const [homrDir, pagesDir, outDir, ...only] = process.argv.slice(2);
const py = await loadPyodide();
await py.loadPackage(["numpy", "opencv-python", "pillow", "typing-extensions"], { messageCallback: () => {} });
const lib = "/home/pyodide/lib";
const copy = (from, to) => {
  py.FS.mkdirTree(to);
  for (const e of fs.readdirSync(from, { withFileTypes: true })) {
    if (e.isDirectory()) copy(path.join(from, e.name), `${to}/${e.name}`);
    else if (e.name.endsWith(".py")) py.FS.writeFile(`${to}/${e.name}`, fs.readFileSync(path.join(from, e.name)));
  }
};
copy(path.join(homrDir, "homr"), `${lib}/homr`);
py.FS.writeFile(`${lib}/homr_setup.py`, fs.readFileSync(path.join(here, "homr_setup.py")));
py.runPython(`
import sys, json, contextlib, io
sys.path.insert(0, "${lib}")
import numpy as np
from homr_setup import config, load
main = load("${lib}")
from homr.model import InputPredictions
from homr import staff_parsing
from homr.transformer.configs import Config

def grid(s):
    return [[float(p.x), [float(y) for y in p.y], float(p.angle)] for p in s.grid]

def run(page_dir, out_dir):
    meta = json.load(open(f"{page_dir}/page.json"))
    shape = tuple(meta["shape"])
    m = {n: np.fromfile(f"{page_dir}/{n}.bin", dtype=np.uint8).reshape(shape) for n in ["notehead", "symbols", "staff", "clefs_keys", "stem_rests"]}
    pre = np.fromfile(f"{page_dir}/preprocessed.bin", dtype=np.uint8).reshape(tuple(meta["preprocessed_shape"]))
    blank = np.zeros(shape + (3,), dtype=np.uint8)
    pred = InputPredictions(original=blank, preprocessed=pre, notehead=m["notehead"], symbols=m["symbols"],
                            staff=m["staff"], clefs_keys=m["clefs_keys"], stems_rest=m["stem_rests"])
    debug = main.Debug(blank, "/tmp/page.png", False)
    main.load_and_preprocess_predictions = lambda *a, **k: (pred, debug, None)
    calls = []
    current = {}
    orig_prepare = staff_parsing.prepare_staff_image
    orig_dewarp = staff_parsing.dewarp_staff_image
    def dewarp_spy(image, staff, index, dbg):
        d = orig_dewarp(image, staff, index, dbg)
        t = d.tform
        current["dewarp"] = None if t is None else {
            "src": t.src_points.tolist(), "dst": t.dst_points.tolist(),
            "simplices": t.triangulation.simplices.tolist() if t.triangulation is not None and t.triangulation.simplices is not None else None,
            "staff_in_image": grid(staff), "image_shape": list(image.shape)}
        np.ascontiguousarray(image).tofile(f"{out_dir}/predewarp_{len(calls)}.bin")
        orig_d = d.dewarp
        def dewarp_img(img, *a, **k):
            out = orig_d(img, *a, **k)
            np.ascontiguousarray(out).tofile(f"{out_dir}/postdewarp_{len(calls)}.bin")
            current["post_shape"] = list(out.shape)
            return out
        d.dewarp = dewarp_img
        return d
    def prepare_spy(dbg, index, staff, image, regions):
        current.clear()
        img, st, mapping = orig_prepare(dbg, index, staff, image, regions)
        np.ascontiguousarray(img).tofile(f"{out_dir}/canvas_{len(calls)}.bin")
        calls.append({"index": index, "grid": grid(staff), "grand": bool(staff.is_grandstaff), "centers": [list(map(float, c)) for c in regions.centers],
                      "canvas_shape": list(img.shape), **current})
        return img, st, mapping
    staff_parsing.prepare_staff_image = prepare_spy
    staff_parsing.dewarp_staff_image = dewarp_spy
    orig_tromr = staff_parsing.parse_staff_tromr
    staff_parsing.parse_staff_tromr = lambda **k: []
    try:
        with contextlib.redirect_stderr(io.StringIO()):
            multi = main.detect_staffs_in_image("/tmp/page.png", config(main))[0]
            staff_parsing.parse_staffs(debug, multi, pre, Config())
    finally:
        staff_parsing.prepare_staff_image = orig_prepare
        staff_parsing.dewarp_staff_image = orig_dewarp
        staff_parsing.parse_staff_tromr = orig_tromr
    json.dump(calls, open(f"{out_dir}/staffs.json", "w"))
`);
const pages = (only.length ? only : fs.readdirSync(pagesDir)).filter((p) => fs.existsSync(path.join(pagesDir, p, "preprocessed.bin")));
for (const p of pages) {
  const src = `/pages/${p}`, out = `/out/${p}`;
  py.FS.mkdirTree(src);
  py.FS.mkdirTree(out);
  for (const f of fs.readdirSync(path.join(pagesDir, p))) py.FS.writeFile(`${src}/${f}`, fs.readFileSync(path.join(pagesDir, p, f)));
  try {
    py.globals.get("run")(src, out);
    fs.mkdirSync(path.join(outDir, p), { recursive: true });
    for (const f of py.FS.readdir(out).filter((f) => !f.startsWith("."))) fs.writeFileSync(path.join(outDir, p, f), py.FS.readFile(`${out}/${f}`));
    console.log("staffs", p);
  } catch (e) {
    console.log("skip", p, String(e.message).split("\n").filter(Boolean).at(-1));
  }
  for (const d of [src, out]) {
    for (const f of py.FS.readdir(d).filter((f) => !f.startsWith("."))) py.FS.unlink(`${d}/${f}`);
    py.FS.rmdir(d);
  }
}
