// One active Chrome benchmark engine, hosted in a Worker so Atomics.wait works.
import {EncoderPool} from '../tools/encoder_pool.mjs';
const read = async url => {const r = await fetch(url); if (!r.ok) throw Error(url); return r.arrayBuffer();};
let run, close, jobs = () => 0;
self.onmessage = async ({data: a}) => {
  try {
    if (a.command === 'init') {
      const input = await read('/staff/input.f32'), native = new Uint32Array(await read('/staff/context.bin'));
      if (a.kind === 'ort') {
        const ort = await import('/ort/ort.wasm.min.mjs'); ort.env.wasm.wasmPaths = '/ort/'; ort.env.wasm.numThreads = a.threads;
        const session = await ort.InferenceSession.create(await read('/model/encoder.onnx'), {executionProviders: ['wasm'], graphOptimizationLevel: 'all'});
        const tensor = new ort.Tensor('float32', new Float32Array(input), [1, 1, 256, 1280]); let expected;
        run = async () => {
          const got = await session.run({[session.inputNames[0]]: tensor}), t = got[session.outputNames[0]];
          const bits = new Uint32Array(t.data.buffer, t.data.byteOffset, t.data.length);
          if (expected && (bits.length !== expected.length || bits.some((v, i) => v !== expected[i]))) throw Error('ORT output changed');
          expected ??= bits.slice(); t.dispose();
        };
        close = async () => {await session.release(); tensor.dispose();};
      } else {
        const pool = await EncoderPool.create(await WebAssembly.compile(await read(`/wasm/encoder-threads-${a.kind}.wasm`)), {threads: a.threads});
        const e = pool.exports;
        const copy = bytes => {const p = e.decoder_alloc(bytes.byteLength); new Uint8Array(e.memory.buffer, p, bytes.byteLength).set(new Uint8Array(bytes)); return p;};
        const bytes = await read('/model/encoder.onnx'), wp = copy(bytes), model = e.encoder_load(wp, bytes.byteLength); e.decoder_free(wp, bytes.byteLength);
        if (!model) throw Error('encoder rejected');
        const ip = copy(input), op = e.decoder_alloc(native.byteLength);
        run = () => {if (e.encoder_run(model, ip, op) !== 1) throw Error('encoder failed'); const got = new Uint32Array(e.memory.buffer, op, native.length); if (got.some((v, i) => v !== native[i])) throw Error('native context bits differ');};
        jobs = () => pool.jobs;
        close = async () => {if (!pool.failed) {e.encoder_drop(model); e.decoder_free(ip, input.byteLength); e.decoder_free(op, native.byteLength);} await pool.close();};
      }
      await run(); self.postMessage({ready: true});
    } else if (a.command === 'run') {const start = performance.now(), before = jobs(); await run(); self.postMessage({ms: performance.now() - start, jobs: jobs() - before});}
    else if (a.command === 'close') {await close(); self.postMessage({closed: true});}
  } catch (error) {self.postMessage({error: String(error.stack || error)});}
};
