// Runner for the synthetic Rust SIMD tests, not a benchmark.
// CARGO_TARGET_WASM32_WASIP1_RUNNER='node --no-warnings bench/run_wasi.mjs'
import {WASI} from 'node:wasi';
import fs from 'node:fs';
const [binary,...args]=process.argv.slice(2);
const wasi=new WASI({version:'preview1',args:[binary,...args],preopens:{'/':'/'}});
const module=await WebAssembly.compile(fs.readFileSync(binary));
wasi.start(await WebAssembly.instantiate(module,wasi.getImportObject()));
