// homr's detect_staff in Pyodide, `reps` times per case (instruction counts by difference).
//   node bench/py.mjs <homr dir> <reps> <case dirs...>     (npm i pyodide@314.0.7 first)
import { loadPyodide } from "pyodide";
import fs from "node:fs";
import path from "node:path";
const [homrDir, reps, ...cases] = process.argv.slice(2);
const here = path.dirname(new URL(import.meta.url).pathname);
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
py.FS.writeFile(`${lib}/homr_setup.py`, fs.readFileSync(path.join(here, "../tools/homr_setup.py")));
cases.forEach((c, i) => {
  py.FS.mkdirTree(`/cases/${i}`);
  for (const f of ["case.json", "staff.bin"]) py.FS.writeFile(`/cases/${i}/${f}`, fs.readFileSync(`${c}/${f}`));
});
py.runPython(`
import sys, json, contextlib, io
sys.path.insert(0, "${lib}")
import numpy as np
from homr_setup import load
main = load("${lib}")
from homr.bounding_boxes import RotatedBoundingBox
from homr.debug import Debug
ok = 0
for i in range(${cases.length}):
    c = json.load(open(f"/cases/{i}/case.json"))
    img = np.fromfile(f"/cases/{i}/staff.bin", dtype=np.uint8).reshape(c["shape"])
    rb = lambda b, cont=np.array([]), k=0: RotatedBoundingBox(((b[0], b[1]), (b[2], b[3]), b[4]), cont, k)
    frags = [rb(f["box"], np.array(f["contour"], dtype=np.int32).reshape(-1, 1, 2), k) for k, f in enumerate(c["fragments"])]
    clefs = [rb(b) for b in c["clefs"]]; bars = [rb(b) for b in c["bars"]]
    debug = Debug(img, "/tmp/x.png", False)
    with contextlib.redirect_stderr(io.StringIO()):
        for _ in range(${Number(reps)}):
            staffs = main.detect_staff(debug, img, frags, clefs, bars)
    exp = c["expected"]
    ok += len(staffs) == len(exp) and all(len(s.grid) == len(e) and all(abs(a - b) < 1e-9 for p, q in zip(s.grid, e) for a, b in zip(p.y, q[1])) for s, e in zip(staffs, exp))
print(f"python: {ok} of ${cases.length} cases same as the dump")
`);
