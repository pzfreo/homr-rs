// Reference preprocessing and segmentation post-processing from homr in
// Pyodide, for pages dumped by tools/dump_segnet.py: from the decoded image,
// homr's autocrop, resize_image (Pillow) and apply_clahe; and from segnet's
// tile class maps, merge_patches, filter_predictions and make_lines_stronger.
//   node tools/pyodide_pre.mjs <homr dir> <segnet dir> <out dir> [pages...]
import { createRequire } from "node:module";
import fs from "node:fs";
import path from "node:path";
const here = path.dirname(new URL(import.meta.url).pathname);
const require = createRequire(path.join(here, "../bench/package.json"));
const { loadPyodide } = require("pyodide");
const [homrDir, segDir, outDir, ...only] = process.argv.slice(2);
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
from homr_setup import load
main = load("${lib}")
import cv2
from homr.autocrop import autocrop_with_offset
from homr.resize import resize_image
from homr import color_adjust
from homr.segmentation.inference_segnet import merge_patches
from homr.model import InputPredictions
from homr.noise_filtering import filter_predictions
from homr.staff_detection import make_lines_stronger

def run(d, out):
    meta = json.load(open(f"{d}/segnet.json"))
    img = np.fromfile(f"{d}/image.bin", dtype=np.uint8).reshape(meta["image_shape"])
    with contextlib.redirect_stderr(io.StringIO()):
        cropped, top_left = autocrop_with_offset(img)
        resized = resize_image(cropped)
        pre = color_adjust.apply_clahe(resized)
        np.ascontiguousarray(resized).tofile(f"{out}/resized.bin")
        np.ascontiguousarray(pre).tofile(f"{out}/preprocessed.bin")
        h, w = meta["merged_shape"]
        tiles = np.fromfile(f"{d}/tiles.bin", dtype=np.uint8).reshape(meta["tiles"], 320, 320).astype(np.int64)
        merged = merge_patches(list(tiles), (h, w), 320, 320)
        pred = InputPredictions(original=cv2.resize(resized, (w, h)), preprocessed=cv2.resize(pre, (w, h)),
                                notehead=(merged == 2).astype(np.uint8), symbols=(merged == 5).astype(np.uint8),
                                staff=(merged == 4).astype(np.uint8), clefs_keys=(merged == 3).astype(np.uint8),
                                stems_rest=(merged == 1).astype(np.uint8))
        pred = filter_predictions(pred, main.Debug(pred.original, "/tmp/x.png", False))
        pred.staff = make_lines_stronger(pred.staff, (1, 2))
    np.ascontiguousarray(merged.astype(np.uint8)).tofile(f"{out}/merged.bin")
    for n in ["staff", "symbols", "notehead", "clefs_keys", "stems_rest", "preprocessed"]:
        np.ascontiguousarray(getattr(pred, n)).tofile(f"{out}/final_{n}.bin")
    json.dump({"crop": [int(top_left[0]), int(top_left[1])], "resized_shape": list(resized.shape), "merged_shape": [h, w]}, open(f"{out}/pre.json", "w"))
`);
const pages = (only.length ? only : fs.readdirSync(segDir)).filter((p) => fs.existsSync(path.join(segDir, p, "segnet.json")));
for (const p of pages) {
  const src = `/seg/${p}`, out = `/out/${p}`;
  py.FS.mkdirTree(src);
  py.FS.mkdirTree(out);
  for (const f of fs.readdirSync(path.join(segDir, p))) py.FS.writeFile(`${src}/${f}`, fs.readFileSync(path.join(segDir, p, f)));
  try {
    py.globals.get("run")(src, out);
    fs.mkdirSync(path.join(outDir, p), { recursive: true });
    for (const f of py.FS.readdir(out).filter((f) => !f.startsWith("."))) fs.writeFileSync(path.join(outDir, p, f), py.FS.readFile(`${out}/${f}`));
    console.log("pre", p);
  } catch (e) {
    console.log("skip", p, String(e.message).split("\n").filter(Boolean).at(-1));
  }
  for (const d of [src, out]) {
    for (const f of py.FS.readdir(d).filter((f) => !f.startsWith("."))) py.FS.unlink(`${d}/${f}`);
    py.FS.rmdir(d);
  }
}
