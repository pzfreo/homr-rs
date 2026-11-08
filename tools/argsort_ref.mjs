// np.argsort of arrays with many equal values, in Pyodide's numpy (the one homr runs
// on in the browser), for checking npmath::np_argsort with check_argsort.
//   node tools/argsort_ref.mjs <cases.json> <out.json>     cases: {"values": [[...], ...]}
import { createRequire } from "node:module";
import fs from "node:fs";
import path from "node:path";
const here = path.dirname(new URL(import.meta.url).pathname);
const require = createRequire(path.join(here, "../bench/package.json"));
const { loadPyodide } = require("pyodide");
const [casesFile, outFile] = process.argv.slice(2);
const py = await loadPyodide();
await py.loadPackage(["numpy"], { messageCallback: () => {} });
py.FS.writeFile("/cases.json", fs.readFileSync(casesFile));
py.runPython(`
import json, numpy as np
cases = json.load(open("/cases.json"))["values"]
out = [np.argsort(np.array(c, dtype=np.float64)).tolist() for c in cases]
json.dump({"numpy": np.__version__, "order": out}, open("/out.json", "w"))
`);
fs.writeFileSync(outFile, py.FS.readFile("/out.json"));
