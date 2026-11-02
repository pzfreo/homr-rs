// Warmed one-model retired-instruction sample. Native counterpart: models_native.py.
// ORT_MODULE=... COUNTER_ADDON=... node bench/model_instructions.mjs MODEL INPUT SHAPE OUTPUT [THREADS]
// INPUT: input.f32 or seg-N.input.u8; SHAPE: e.g. 1,1,256,1280 or 8,3,320,320.
import fs from 'node:fs';
import path from 'node:path';
import {createRequire} from 'node:module';
import {pathToFileURL} from 'node:url';
const [model, input, shape, output, threads = '1'] = process.argv.slice(2);
const ort = await import(pathToFileURL(path.resolve(process.env.ORT_MODULE)));
const counter = createRequire(import.meta.url)(path.resolve(process.env.COUNTER_ADDON));
ort.env.wasm.numThreads = Number(threads);
const session = await ort.InferenceSession.create(fs.readFileSync(model), {executionProviders: ['wasm'], graphOptimizationLevel: 'all'});
const bytes = fs.readFileSync(input);
const values = input.endsWith('.u8') ? Float32Array.from(bytes) : new Float32Array(bytes.buffer.slice(bytes.byteOffset, bytes.byteOffset + bytes.byteLength));
const tensor = new ort.Tensor('float32', values, shape.split(',').map(Number));
const run = async () => (await session.run({[session.inputNames[0]]: tensor}))[session.outputNames[0]];
const warm = await run();
const expected = new Uint8Array(warm.data.buffer, warm.data.byteOffset, warm.data.byteLength).slice();
warm.dispose();
const counts = [];
for (let i = 0; i < 3; i++) {
  const before = counter.instructions();
  const got = await run();
  counts.push(counter.instructions() - before);
  const data = new Uint8Array(got.data.buffer, got.data.byteOffset, got.data.byteLength);
  if (data.length !== expected.length || data.some((v, j) => v !== expected[j])) throw Error('nondeterministic output');
  got.dispose();
}
await session.release();tensor.dispose();
const result = {ort: ort.env.versions, threads: Number(threads), model, input, shape,
  counter: 'proc_pid_rusage RUSAGE_INFO_V4', instructions: counts, median: counts.toSorted((a,b) => a-b)[1]};
fs.writeFileSync(output, JSON.stringify(result, null, 2));console.log(result);
