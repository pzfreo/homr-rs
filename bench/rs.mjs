// The Rust port as WebAssembly (WASI), `reps` times per case.
//   cargo build --release --bin bench --target wasm32-wasip1
//   node bench/rs.mjs <reps> <case dirs...>
import { WASI } from "node:wasi";
import fs from "node:fs";
import path from "node:path";
const [reps, ...cases] = process.argv.slice(2).map((a, i) => (i ? path.resolve(a) : a));
const here = path.dirname(new URL(import.meta.url).pathname);
const wasi = new WASI({ version: "preview1", args: ["bench", reps, ...cases], preopens: { "/": "/" } });
const mod = await WebAssembly.compile(fs.readFileSync(path.join(here, "../target/wasm32-wasip1/release/bench.wasm")));
wasi.start(await WebAssembly.instantiate(mod, wasi.getImportObject()));
