// homr in Pyodide from masks to MusicXML (page analysis, staff images, the
// transformer's recorded output replayed, MusicXML), `reps` times per page.
//   node bench/py_page.mjs <homr dir> <reps> <page dir>:<decoder dir> ...
import { loadPyodide } from "pyodide";
import fs from "node:fs";
import path from "node:path";
const [homrDir, reps, ...pairs] = process.argv.slice(2);
const here = path.dirname(new URL(import.meta.url).pathname);
const py = await loadPyodide();
await py.loadPackage(["numpy", "opencv-python", "pillow", "typing-extensions"], { messageCallback: () => {} });
const lib = "/home/pyodide/lib";
const copy = (from, to) => {
  py.FS.mkdirTree(to);
  for (const e of fs.readdirSync(from, { withFileTypes: true })) {
    if (e.isDirectory()) copy(path.join(from, e.name), `${to}/${e.name}`);
    else if (e.name.endsWith(".py") || e.name.endsWith(".json")) py.FS.writeFile(`${to}/${e.name}`, fs.readFileSync(path.join(from, e.name)));
  }
};
copy(path.join(homrDir, "homr"), `${lib}/homr`);
py.FS.writeFile(`${lib}/homr_setup.py`, fs.readFileSync(path.join(here, "../tools/homr_setup.py")));
pairs.forEach((p, i) => {
  const [pd, dd] = p.split(":");
  py.FS.mkdirTree(`/p/${i}`);
  for (const f of fs.readdirSync(pd)) py.FS.writeFile(`/p/${i}/${f}`, fs.readFileSync(path.join(pd, f)));
  py.FS.writeFile(`/p/${i}/decoder.json`, fs.readFileSync(path.join(dd, "decoder.json")));
});
py.runPython(`
import sys, json, contextlib, io
sys.path.insert(0, "${lib}")
import numpy as np
from homr_setup import config, load
main = load("${lib}")
from homr.model import InputPredictions
from homr import staff_parsing_tromr
from homr.point_mapping import chain, undo_crop, undo_resize
from homr.transformer.vocabulary import EncodedSymbol
from homr.music_xml_generator import XmlGeneratorArguments
class Replay:
    def __init__(self, calls): self.calls = list(calls)
    def predict(self, image):
        return [EncodedSymbol(r, p, l, a, s, pos, coordinates=c) for r, p, l, a, s, pos, c in self.calls.pop(0)]
for i in range(${pairs.length}):
    d = f"/p/{i}"
    meta = json.load(open(f"{d}/page.json")); dec = json.load(open(f"{d}/decoder.json"))
    shape = tuple(meta["shape"])
    m = {n: np.fromfile(f"{d}/{n}.bin", dtype=np.uint8).reshape(shape) for n in ["notehead", "symbols", "staff", "clefs_keys", "stem_rests"]}
    pre = np.fromfile(f"{d}/preprocessed.bin", dtype=np.uint8).reshape(tuple(meta["preprocessed_shape"]))
    blank = np.zeros(shape + (3,), dtype=np.uint8)
    for _ in range(${Number(reps)}):
        pred = InputPredictions(original=blank, preprocessed=pre.copy(), notehead=m["notehead"], symbols=m["symbols"],
                                staff=m["staff"], clefs_keys=m["clefs_keys"], stems_rest=m["stem_rests"])
        debug = main.Debug(blank, "/tmp/page.png", False)
        mapping = chain(undo_resize(tuple(dec["crop"]["cropped_shape"]), pre.shape), undo_crop(*dec["crop"]["top_left"]))
        main.load_and_preprocess_predictions = lambda *a, **k: (pred, debug, mapping)
        staff_parsing_tromr.inference = Replay(dec["calls"])
        with contextlib.redirect_stderr(io.StringIO()):
            main.process_image("/tmp/page.png", config(main), XmlGeneratorArguments())
print("python: done")
`);
