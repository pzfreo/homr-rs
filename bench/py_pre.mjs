// homr's stage 2 in Pyodide (autocrop, resize_image, apply_clahe, merge_patches,
// filter_predictions, make_lines_stronger), `reps` times per page.
//   node bench/py_pre.mjs <homr dir> <reps> <segnet dirs...>
import { loadPyodide } from "pyodide";
import fs from "node:fs";
import path from "node:path";
const [homrDir, reps, ...dirs] = process.argv.slice(2);
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
dirs.forEach((d, i) => {
  py.FS.mkdirTree(`/s/${i}`);
  for (const f of ["segnet.json", "image.bin", "tiles.bin"]) py.FS.writeFile(`/s/${i}/${f}`, fs.readFileSync(path.join(d, f)));
});
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
for i in range(${dirs.length}):
    d = f"/s/{i}"
    meta = json.load(open(f"{d}/segnet.json"))
    img = np.fromfile(f"{d}/image.bin", dtype=np.uint8).reshape(meta["image_shape"])
    tiles = list(np.fromfile(f"{d}/tiles.bin", dtype=np.uint8).reshape(meta["tiles"], 320, 320).astype(np.int64))
    for _ in range(${Number(reps)}):
        with contextlib.redirect_stderr(io.StringIO()):
            cropped, tl = autocrop_with_offset(img)
            resized = resize_image(cropped)
            pre = color_adjust.apply_clahe(resized)
            h, w = pre.shape
            merged = merge_patches(tiles, (h, w), 320, 320)
            pred = InputPredictions(original=resized, preprocessed=pre, notehead=(merged == 2).astype(np.uint8), symbols=(merged == 5).astype(np.uint8),
                                    staff=(merged == 4).astype(np.uint8), clefs_keys=(merged == 3).astype(np.uint8), stems_rest=(merged == 1).astype(np.uint8))
            pred = filter_predictions(pred, main.Debug(resized, "/tmp/x.png", False))
            pred.staff = make_lines_stronger(pred.staff, (1, 2))
print("python: done")
`);
