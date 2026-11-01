// Reference results from homr as Partsong runs it: in Pyodide (OpenCV 4.11
// compiled to WebAssembly). Takes the segmentation masks dumped by
// tools/dump_page.py (natively; the masks come from the model) and runs homr's
// detect_staffs_in_image on them, writing case.json (staff detection's inputs
// and result, as tools/dump_staff.py) into <out>/<page>.
//   node tools/pyodide_ref.mjs <homr dir> <pages dir> <out dir> [pages...]
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

def box(b):
    (cx, cy), (w, h), a = b.box
    return [float(cx), float(cy), float(w), float(h), float(a)]

class Done(Exception): pass

def run(page_dir, out_dir):
    meta = json.load(open(f"{page_dir}/page.json"))
    shape = tuple(meta["shape"])
    m = {n: np.fromfile(f"{page_dir}/{n}.bin", dtype=np.uint8).reshape(shape) for n in ["notehead", "symbols", "staff", "clefs_keys", "stem_rests"]}
    blank = np.zeros(shape + (3,), dtype=np.uint8)
    pred = InputPredictions(original=blank, preprocessed=blank[:, :, 0].copy(), notehead=m["notehead"], symbols=m["symbols"],
                            staff=m["staff"], clefs_keys=m["clefs_keys"], stems_rest=m["stem_rests"])
    debug = main.Debug(blank, "/tmp/page.png", False)
    main.load_and_preprocess_predictions = lambda *a, **k: (pred, debug, None)
    case = {}
    orig = main.detect_staff
    def spy(dbg, image, fragments, clefs, bars):
        result = orig(dbg, image, fragments, clefs, bars)
        case.update({
            "shape": list(image.shape),
            "fragments": [{"box": box(f), "contour": np.asarray(f.contours).reshape(-1, 2).tolist()} for f in fragments],
            "clefs": [box(c) for c in clefs], "bars": [box(c) for c in bars],
            "expected": [[[float(p.x), [float(y) for y in p.y], float(p.angle)] for p in s.grid] for s in result],
            "python_instructions": 0,
        })
        np.ascontiguousarray(image, dtype=np.uint8).tofile(f"{out_dir}/staff.bin")
        raise Done
    main.detect_staff = spy
    try:
        with contextlib.redirect_stderr(io.StringIO()):
            main.detect_staffs_in_image("/tmp/page.png", config(main))
    except Done:
        pass
    finally:
        main.detect_staff = orig
    json.dump(case, open(f"{out_dir}/case.json", "w"))
`);
const pages = (only.length ? only : fs.readdirSync(pagesDir)).filter((p) => fs.existsSync(path.join(pagesDir, p, "page.json")));
for (const p of pages) {
  const src = `/pages/${p}`, out = `/out/${p}`;
  py.FS.mkdirTree(src);
  py.FS.mkdirTree(out);
  for (const f of fs.readdirSync(path.join(pagesDir, p))) py.FS.writeFile(`${src}/${f}`, fs.readFileSync(path.join(pagesDir, p, f)));
  try {
    py.globals.get("run")(src, out);
    fs.mkdirSync(path.join(outDir, p), { recursive: true });
    for (const f of py.FS.readdir(out).filter((f) => !f.startsWith("."))) fs.writeFileSync(path.join(outDir, p, f), py.FS.readFile(`${out}/${f}`));
    console.log("ref", p);
  } catch (e) {
    console.log("skip", p, String(e.message).split("\n").filter(Boolean).at(-1));
  }
  for (const d of [src, out]) {
    for (const f of py.FS.readdir(d).filter((f) => !f.startsWith("."))) py.FS.unlink(`${d}/${f}`);
    py.FS.rmdir(d);
  }
}
