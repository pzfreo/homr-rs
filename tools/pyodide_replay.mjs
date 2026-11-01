// Reference MusicXML from homr as Partsong runs it (Pyodide), replaying the
// transformer's output recorded natively by tools/dump_decoder.py: page
// analysis from the dumped masks (tools/dump_page.py), staff images, then the
// recorded symbols in place of the model, then homr's MusicXML. Also records
// what parse_staff_image returned for each staff (symbols with positions).
//   node tools/pyodide_replay.mjs <homr dir> <pages dir> <decoder dir> <out dir> [pages...]
import { createRequire } from "node:module";
import fs from "node:fs";
import path from "node:path";
const here = path.dirname(new URL(import.meta.url).pathname);
const require = createRequire(path.join(here, "../bench/package.json"));
const { loadPyodide } = require("pyodide");
const [homrDir, pagesDir, decDir, outDir, ...only] = process.argv.slice(2);
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
py.FS.writeFile(`${lib}/homr_setup.py`, fs.readFileSync(path.join(here, "homr_setup.py")));
py.runPython(`
import sys, json, contextlib, io
sys.path.insert(0, "${lib}")
import numpy as np
from homr_setup import config, load
main = load("${lib}")
from homr.model import InputPredictions
from homr import staff_parsing, staff_parsing_tromr
from homr.point_mapping import chain, undo_crop, undo_resize
from homr.transformer.vocabulary import EncodedSymbol
from homr.music_xml_generator import XmlGeneratorArguments

class Replay:
    def __init__(self, calls):
        self.calls = list(calls)
    def predict(self, image):
        if not self.calls:
            raise Exception("replay ran out of recorded staffs")
        return [EncodedSymbol(r, p, l, a, s, pos, coordinates=c) for r, p, l, a, s, pos, c in self.calls.pop(0)]

def run(page_dir, dec_dir, out_dir):
    meta = json.load(open(f"{page_dir}/page.json"))
    dec = json.load(open(f"{dec_dir}/decoder.json"))
    shape = tuple(meta["shape"])
    m = {n: np.fromfile(f"{page_dir}/{n}.bin", dtype=np.uint8).reshape(shape) for n in ["notehead", "symbols", "staff", "clefs_keys", "stem_rests"]}
    pre = np.fromfile(f"{page_dir}/preprocessed.bin", dtype=np.uint8).reshape(tuple(meta["preprocessed_shape"]))
    blank = np.zeros(shape + (3,), dtype=np.uint8)
    pred = InputPredictions(original=blank, preprocessed=pre, notehead=m["notehead"], symbols=m["symbols"],
                            staff=m["staff"], clefs_keys=m["clefs_keys"], stems_rest=m["stem_rests"])
    debug = main.Debug(blank, "/tmp/page.png", False)
    mapping = chain(undo_resize(tuple(dec["crop"]["cropped_shape"]), pre.shape), undo_crop(*dec["crop"]["top_left"]))
    main.load_and_preprocess_predictions = lambda *a, **k: (pred, debug, mapping)
    replay = Replay(dec["calls"])
    staff_parsing_tromr.inference = replay
    per_staff = []
    orig = staff_parsing.parse_staff_image
    def spy(*a, **k):
        res = orig(*a, **k)
        per_staff.append([[s.rhythm, s.pitch, s.lift, s.articulation, s.slur, s.position,
                           None if s.image_coordinates is None else [float(s.image_coordinates[0]), float(s.image_coordinates[1])]] for s in res])
        return res
    staff_parsing.parse_staff_image = spy
    try:
        with contextlib.redirect_stderr(io.StringIO()):
            xml_path = main.process_image("/tmp/page.png", config(main), XmlGeneratorArguments())
    finally:
        staff_parsing.parse_staff_image = orig
    if replay.calls:
        raise Exception(f"{len(replay.calls)} recorded staffs left over")
    with open(xml_path, encoding="utf-8") as f, open(f"{out_dir}/replay.musicxml", "w", encoding="utf-8") as g:
        g.write(f.read())
    json.dump(per_staff, open(f"{out_dir}/staff_symbols.json", "w"))
`);
const pages = (only.length ? only : fs.readdirSync(decDir)).filter((p) => fs.existsSync(path.join(decDir, p, "decoder.json")) && fs.existsSync(path.join(pagesDir, p, "preprocessed.bin")));
for (const p of pages) {
  const src = `/pages/${p}`, dsrc = `/dec/${p}`, out = `/out/${p}`;
  for (const d of [src, dsrc, out]) py.FS.mkdirTree(d);
  for (const f of fs.readdirSync(path.join(pagesDir, p))) py.FS.writeFile(`${src}/${f}`, fs.readFileSync(path.join(pagesDir, p, f)));
  py.FS.writeFile(`${dsrc}/decoder.json`, fs.readFileSync(path.join(decDir, p, "decoder.json")));
  try {
    py.globals.get("run")(src, dsrc, out);
    fs.mkdirSync(path.join(outDir, p), { recursive: true });
    for (const f of py.FS.readdir(out).filter((f) => !f.startsWith("."))) fs.writeFileSync(path.join(outDir, p, f), py.FS.readFile(`${out}/${f}`));
    console.log("replay", p);
  } catch (e) {
    console.log("skip", p, String(e.message).split("\n").filter(Boolean).at(-1));
  }
  for (const d of [src, dsrc, out]) {
    for (const f of py.FS.readdir(d).filter((f) => !f.startsWith("."))) py.FS.unlink(`${d}/${f}`);
    py.FS.rmdir(d);
  }
}
