// ORT_MODULE=/.../onnxruntime-web/dist/ort.node.min.mjs COUNTER_ADDON=/scratch/counter.node 
// node bench/models.mjs NATIVE_FIXTURES MODELS_DIR OUTPUT [CORPUS_TXT] [wasm threads optimization]
// Models dir contains segnet_308-*.onnx and encoder_pytorch_model_465-*.onnx.
// Optional MODEL_KIND=encoder|segnet selects a component for setting surveys.
import fs from 'node:fs';
import path from 'node:path';
import {createRequire} from 'node:module';
import {pathToFileURL} from 'node:url';
import {compareModels} from './models_core.mjs';
const [native, modelRoot, output, corpus, ep = 'wasm', threads = '1', optimization = 'all'] = process.argv.slice(2);
if (!output || ep !== 'wasm') throw Error('models.mjs NATIVE MODELS OUTPUT CORPUS_TXT wasm THREADS OPTIMIZATION');
const ort = await import(pathToFileURL(path.resolve(process.env.ORT_MODULE)));
const counter = process.env.COUNTER_ADDON ? createRequire(import.meta.url)(path.resolve(process.env.COUNTER_ADDON)) : null;
const roots = {native, models: modelRoot, output};
const resolve = name => {
  const [root, ...parts] = name.split('/');
  if (!roots[root] || parts.includes('..')) throw Error('bad fixture path');
  return path.join(roots[root], ...parts);
};
const pages = corpus ? fs.readFileSync(corpus, 'utf8').trim().split('\n').map(p => path.parse(p).name) : fs.readdirSync(native).sort();
const models = Object.fromEntries([['segnet', 'segnet_308-'], ['encoder', 'encoder_pytorch_model_465-']]
  .filter(([kind]) => !process.env.MODEL_KIND || kind === process.env.MODEL_KIND).map(([k, prefix]) =>
  [k, `models/${fs.readdirSync(modelRoot).find(n => n.startsWith(prefix) && n.endsWith('.onnx') && !n.includes('fp16'))}`]));
console.log(await compareModels({ort, pages, models, ep, threads: Number(threads), optimization,
  instructions: counter ? () => counter.instructions() : undefined,
  read: async name => {
    const p = resolve(name);
    // A recorder may still be visiting later pages; completion is atomic at
    // models.json. Never infer an empty page from an incomplete recording.
    for (let tries = 0; name.endsWith('/models.json') && !fs.existsSync(p); tries++) {
      if (tries >= 3600) throw Error(`missing completed page ${name}`);
      await new Promise(r => setTimeout(r, 1000));
    }
    return fs.readFileSync(p);
  },
  write: async (name, data) => {
    const p = resolve(name);fs.mkdirSync(path.dirname(p), {recursive: true});
    fs.writeFileSync(`${p}.tmp`, data);fs.renameSync(`${p}.tmp`, p);
  },
}));
