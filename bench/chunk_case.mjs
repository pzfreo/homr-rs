// One persistent isolated process. Production measurements have no diagnostics;
// scheduler simulations use only encoder-chunk-bench WASM exports.
import fs from 'node:fs';
import path from 'node:path';
import {createRequire} from 'node:module';
import {EncoderPool} from '../tools/encoder_pool.mjs';
const a = JSON.parse(process.argv[2]);
const counter = createRequire(import.meta.url)(path.resolve(process.env.COUNTER_ADDON));
const send = value => process.send(value);
let pool;
try {
  pool = await EncoderPool.create(await WebAssembly.compile(fs.readFileSync(a.wasm)), {threads: a.threads});
  const e = pool.exports;
  const copy = bytes => {const p = e.decoder_alloc(bytes.length); new Uint8Array(e.memory.buffer, p, bytes.length).set(bytes); return p;};
  const modelBytes = fs.readFileSync(a.model), mp = copy(modelBytes), model = e.encoder_load(mp, modelBytes.length); e.decoder_free(mp, modelBytes.length);
  if (!model) throw Error('encoder rejected');
  const input = fs.readFileSync(path.join(a.staff, 'input.f32')), ref = fs.readFileSync(path.join(a.staff, 'context.bin'));
  const ip = copy(input), op = e.decoder_alloc(ref.length);
  let jobs = [];
  if (a.diagnostic) {
    if (!e.encoder_bench_prepare) throw Error('diagnostic artifact required');
    const calibration = a.calibration ? JSON.parse(fs.readFileSync(a.calibration))[a.kind].rows : null;
    const dispatch = pool.dispatch.bind(pool);
    pool.dispatch = (job, units) => {
      const info = Array.from({length: 9}, (_, i) => e.encoder_bench_info(job, i));
      const [kind, expectedUnits, base] = info, dims = info.slice(3), previous = calibration?.[jobs.length];
      if (expectedUnits !== units) throw Error('unit metadata mismatch');
      if (previous && (previous.op !== kind || previous.units !== units || JSON.stringify(previous.dims) !== JSON.stringify(dims))) throw Error('calibration belongs to a different job');
      const scale = a.scale ?? 1, chunk = Math.max(1, Math.floor(base * scale));
      const slowMask = (a.slowRanks || []).reduce((mask, r) => mask | (1 << r), 0);
      const ns = (previous?.ns_per_unit || 0) * ((a.slowdown || 1) - 1);
      e.encoder_bench_prepare(job, a.scheduler === 'static' ? 0 : 1, chunk, ns, slowMask);
      const start = performance.now(); dispatch(job, units); const ms = performance.now() - start;
      const ranks = Array.from({length: a.threads}, (_, rank) => ({units: e.encoder_bench_stat(job, rank, 0), chunks: e.encoder_bench_stat(job, rank, 1), requested_ns: e.encoder_bench_stat(job, rank, 2)}));
      if (ranks.reduce((n, r) => n + r.units, 0) !== units) throw Error('missing/duplicate claimed units');
      jobs.push({op: kind, units, dims, base_chunk: base, chunk, ms, ranks});
    };
  }
  let decoder = 0, tokens = 0, nativeSteps;
  if (a.decoderModel) {
    const bytes = fs.readFileSync(a.decoderModel), p = copy(bytes); decoder = e.decoder_load(p, bytes.length); e.decoder_free(p, bytes.length);
    if (!decoder) throw Error('decoder rejected'); tokens = e.decoder_alloc(608 * 32);
    const staff = path.basename(a.staff), page = path.basename(path.dirname(a.staff));
    nativeSteps = JSON.parse(fs.readFileSync(path.join(a.decoderRefs, page, staff, 'steps.json')));
  }
  send({ready: true});
  process.on('message', async ({id, command}) => {
    try {
      if (command === 'close') {
        if (!pool.failed) {if (decoder) {e.decoder_drop(decoder); e.decoder_free(tokens, 608 * 32);} e.encoder_drop(model); e.decoder_free(ip, input.length); e.decoder_free(op, ref.length);}
        await pool.close(); send({id, closed: true}); process.disconnect(); return;
      }
      jobs = [];
      const before = counter.instructions(), start = performance.now();
      if (e.encoder_run(model, ip, op) !== 1) throw Error('encode failed');
      if (!Buffer.from(e.memory.buffer, op, ref.length).equals(ref)) throw Error('native context bits differ');
      const ms = performance.now() - start, instructions = counter.instructions() - before;
      let decoderMs = 0, decoderInstructions = 0, steps = 0;
      if (decoder) {
        const b = counter.instructions(), t = performance.now(); steps = e.decoder_run(decoder, op, tokens, 608, 0);
        if (steps !== nativeSteps.length) throw Error('decoder length differs');
        for (let i = 0; i < steps; i++) {
          const got = new Uint32Array(e.memory.buffer, tokens + i * 32, 6);
          if (got.some((v, j) => v !== nativeSteps[i].tokens[j])) throw Error('decoder tokens differ');
        }
        decoderMs = performance.now() - t; decoderInstructions = counter.instructions() - b;
      }
      send({id, instructions, ms, decoder_ms: decoderMs, decoder_instructions: decoderInstructions, combined_ms: performance.now() - start, steps, jobs});
    } catch (error) {send({id, error: String(error.stack || error)});}
  });
} catch (error) {await pool?.close(); send({error: String(error.stack || error)}); process.disconnect();}
