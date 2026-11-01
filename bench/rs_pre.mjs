// Stage 2 as WebAssembly (WASI), `reps` times per page.
//   node bench/rs_pre.mjs <reps> <segnet dirs...>
import { WASI } from "node:wasi";
import fs from "node:fs";
import path from "node:path";
const [reps, ...dirs] = process.argv.slice(2);
const here = path.dirname(new URL(import.meta.url).pathname);
const wasi = new WASI({ version: "preview1", args: ["bench_pre", reps, ...dirs.map((d) => path.resolve(d))], preopens: { "/": "/" } });
const mod = await WebAssembly.compile(fs.readFileSync(path.join(here, "../target/wasm32-wasip1/release/bench_pre.wasm")));
wasi.start(await WebAssembly.instantiate(mod, wasi.getImportObject()));
