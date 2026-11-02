// ORT_MODULE=/.../dist/ort.node.min.mjs node bench/model_probes.mjs PROBES OUTPUT
// Runs isolated operators on identical native tensors; this excludes upstream
// error propagation. These diagnostic graphs may disable production fusions.
import fs from 'node:fs';
import path from 'node:path';
import {pathToFileURL} from 'node:url';
const [root, output] = process.argv.slice(2);
const ort = await import(pathToFileURL(path.resolve(process.env.ORT_MODULE)));
ort.env.wasm.numThreads = 1;
const read = file => fs.readFileSync(path.join(root, file));
const f32 = bytes => new Float32Array(bytes.buffer.slice(bytes.byteOffset, bytes.byteOffset + bytes.byteLength));
const manifest = JSON.parse(read('probes.json'));
const rows = [];
for (const probe of manifest.probes) {
  const session = await ort.InferenceSession.create(read(probe.model), {executionProviders: ['wasm'], graphOptimizationLevel: 'all'});
  const feeds = Object.fromEntries(probe.inputs.map(t => {
    const bytes = read(t.file);
    const data = t.type === 'float32' ? f32(bytes) : t.type === 'int64' ? new BigInt64Array(bytes.buffer.slice(bytes.byteOffset, bytes.byteOffset + bytes.byteLength)) : null;
    if (!data) throw Error(`unsupported input type ${t.type}`);
    return [t.name, new ort.Tensor(t.type, data, t.shape)];
  }));
  const got = await session.run(feeds);
  for (const t of probe.outputs) {
    const ref = f32(read(t.file)), data = got[t.name].data;
    if (data.length !== ref.length) throw Error('output size mismatch');
    let max = 0, sum = 0, different = 0;
    for (let i = 0; i < ref.length; i++) {
      const e = Math.abs(data[i] - ref[i]);max = Math.max(max, e);sum += e;different += Number(e !== 0);
    }
    const row = {kind: probe.kind, node: probe.node, values: ref.length, different, max_abs: max, mean_abs: sum / ref.length};
    console.log(row);rows.push(row);
  }
  Object.values(got).forEach(t => t.dispose());Object.values(feeds).forEach(t => t.dispose());
  await session.release();
}
fs.writeFileSync(output, JSON.stringify(rows, null, 2));
