// One persistent benchmark process. Its counter includes all its kernel workers,
// while other cases' GC/JIT work is isolated in separate idle processes.
import fs from 'node:fs';
import path from 'node:path';
import {pathToFileURL} from 'node:url';
import {createRequire} from 'node:module';
import {EncoderPool} from '../tools/encoder_pool.mjs';
const a = JSON.parse(process.argv[2]);
const counter = createRequire(import.meta.url)(path.resolve(process.env.COUNTER_ADDON));
const parentPort = {postMessage: value => process.send(value), on: (_, callback) => process.on('message', callback), close: () => process.disconnect()};

let run, close, jobs = () => 0;
try {
  const input = fs.readFileSync(path.join(a.staff, 'input.f32'));
  const reference = fs.readFileSync(path.join(a.staff, 'context.bin'));
  if (a.kind === 'ort') {
    const ort = await import(pathToFileURL(path.resolve(a.ortModule)));
    ort.env.wasm.numThreads = a.threads;
    const session = await ort.InferenceSession.create(fs.readFileSync(a.model), {executionProviders: ['wasm'], graphOptimizationLevel: 'all'});
    const tensor = new ort.Tensor('float32', new Float32Array(input.buffer.slice(input.byteOffset, input.byteOffset + input.length)), [1, 1, 256, 1280]);
    let expected;
    run = async () => {
      const got = await session.run({[session.inputNames[0]]: tensor});
      const t = got[session.outputNames[0]];
      const bytes = Buffer.from(t.data.buffer, t.data.byteOffset, t.data.byteLength);
      if (expected && !bytes.equals(expected)) throw Error('ORT output changed across repetitions');
      expected ??= Buffer.from(bytes);
      t.dispose();
    };
    close = async () => {await session.release(); tensor.dispose();};
  } else {
    const module = await WebAssembly.compile(fs.readFileSync(a.wasm));
    const pool = await EncoderPool.create(module, {threads: a.threads});
    const e = pool.exports;
    const copy = bytes => {const p = e.decoder_alloc(bytes.length); new Uint8Array(e.memory.buffer, p, bytes.length).set(bytes); return p;};
    const bytes = fs.readFileSync(a.model), p = copy(bytes), model = e.encoder_load(p, bytes.length);
    e.decoder_free(p, bytes.length);
    if (!model) throw Error('encoder model rejected');
    const ip = copy(input), op = e.decoder_alloc(reference.length);
    run = () => {
      if (e.encoder_run(model, ip, op) !== 1) throw Error('encoder failed');
      if (!Buffer.from(e.memory.buffer, op, reference.length).equals(reference)) throw Error('context bits differ from native');
    };
    jobs = () => pool.jobs;
    close = async () => {if (!pool.failed) {e.encoder_drop(model); e.decoder_free(ip, input.length); e.decoder_free(op, reference.length);} await pool.close();};
  }
  parentPort.postMessage({ready: true});
  parentPort.on('message', async ({id, command}) => {
    try {
      if (command === 'close') {await close(); parentPort.postMessage({id, closed: true}); parentPort.close();}
      else {const before = jobs(), instructions = counter.instructions(), start = performance.now(); await run(); const ms = performance.now() - start, retired = counter.instructions() - instructions; parentPort.postMessage({id, ms, instructions: retired, jobs: jobs() - before});}
    } catch (error) {parentPort.postMessage({id, error: String(error.stack || error)});}
  });
} catch (error) {parentPort.postMessage({error: String(error.stack || error)});}
