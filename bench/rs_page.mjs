// The Rust port as WebAssembly (WASI): a page from masks to MusicXML, `reps` times.
//   cargo build --release --bin bench_page --target wasm32-wasip1
//   node bench/rs_page.mjs <replay dir> <reps> <page dir>:<decoder dir> ...
import { WASI } from "node:wasi";
import fs from "node:fs";
import path from "node:path";
const [replays, reps, ...pairs] = process.argv.slice(2);
const here = path.dirname(new URL(import.meta.url).pathname);
const abs = pairs.map((p) => p.split(":").map((x) => path.resolve(x)).join(":"));
const wasi = new WASI({ version: "preview1", args: ["bench_page", reps, ...abs], env: { REPLAY_DIR: path.resolve(replays) }, preopens: { "/": "/" } });
const mod = await WebAssembly.compile(fs.readFileSync(path.join(here, "../target/wasm32-wasip1/release/bench_page.wasm")));
wasi.start(await WebAssembly.instantiate(mod, wasi.getImportObject()));
