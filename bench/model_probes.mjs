// ORT_MODULE=/.../dist/ort.node.min.mjs node bench/model_probes.mjs PROBES OUTPUT
// Runs isolated operators on identical native tensors; this excludes upstream
// error propagation. These diagnostic graphs may disable production fusions.
import fs from 'node:fs';
import path from 'node:path';
import {pathToFileURL} from 'node:url';
import {createRequire} from 'node:module';
const [root, output] = process.argv.slice(2);
const ort = await import(pathToFileURL(path.resolve(process.env.ORT_MODULE)));
ort.env.wasm.numThreads = Number(process.env.ORT_THREADS || 1);
const counter=process.env.COUNTER_ADDON?createRequire(import.meta.url)(path.resolve(process.env.COUNTER_ADDON)):null;
const read = file => fs.readFileSync(path.join(root, file));
const f32 = bytes => new Float32Array(bytes.buffer.slice(bytes.byteOffset, bytes.byteOffset + bytes.byteLength));
const manifest = JSON.parse(read('probes.json'));
const rows = [];
for (const probe of manifest.probes) {
  const rowStart=rows.length;
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
  Object.values(got).forEach(t => t.dispose());
  if(counter){
    const counts=[];
    for(let i=0;i<3;i++){const start=counter.instructions();const output=await session.run(feeds);counts.push(counter.instructions()-start);Object.values(output).forEach(t=>t.dispose());}
    for(const row of rows.slice(rowStart))Object.assign(row,{threads:ort.env.wasm.numThreads,instructions:counts,median_instructions:counts.toSorted((a,b)=>a-b)[1]});
  }
  Object.values(feeds).forEach(t => t.dispose());
  await session.release();
}
fs.writeFileSync(output, JSON.stringify(rows, null, 2));
