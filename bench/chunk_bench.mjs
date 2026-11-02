// COUNTER_ADDON=... [DECODER_MODEL=... DECODER_REFS=...] node bench/chunk_bench.mjs
// calibrate|sweep|overhead|uneven STATIC_DIR DYNAMIC_DIR MODEL STAFF OUT [CALIBRATION]
// Case processes isolate counters. Warm once; three interleaved trials, no spins.
import fs from 'node:fs';
import path from 'node:path';
import os from 'node:os';
import {fork} from 'node:child_process';
const [mode, staticDir, dynamicDir, model, staff, output, calibration] = process.argv.slice(2);
if (!output) throw Error('chunk_bench.mjs MODE STATIC_DIR DYNAMIC_DIR MODEL STAFF OUT [CALIBRATION]');
const median = xs => xs.toSorted((a, b) => a - b)[1];
const artifact = (dir, kind, diagnostic = false) => path.join(dir, `encoder-threads-${kind}${diagnostic ? '-chunk-bench' : ''}.wasm`);
async function startCase(config) {
  const child = fork(new URL('./chunk_case.mjs', import.meta.url), [JSON.stringify({...config, model, staff, calibration, decoderModel: process.env.DECODER_MODEL, decoderRefs: process.env.DECODER_REFS})], {stdio: ['ignore', 'inherit', 'inherit', 'ipc']});
  const pending = new Map(); let next = 0;
  try {await new Promise((resolve, reject) => {child.once('error', reject); child.once('message', m => m.error ? reject(Error(m.error)) : resolve()); child.once('exit', code => reject(Error(`startup exit ${code}`)));});}
  catch (error) {child.kill(); throw error;}
  child.on('message', m => {const p = pending.get(m.id); if (p) {pending.delete(m.id); m.error ? p.reject(Error(m.error)) : p.resolve(m);}});
  child.on('exit', code => {for (const p of pending.values()) p.reject(Error(`case exit ${code}`)); pending.clear();});
  child.on('error', error => {for (const p of pending.values()) p.reject(error); pending.clear();});
  return {config, child, call(command) {return new Promise((resolve, reject) => {const id = ++next; pending.set(id, {resolve, reject}); child.send({id, command});});}, async close() {try {await this.call('close');} finally {child.kill();}}};
}
const records = [], order = [], groups = [], report = {mode, complete: false, model, staff, counter: 'proc_pid_rusage, case process including all workers', machine: {cpus: os.cpus().map(c => c.model), node: process.version}, records, order, groups};
const save = () => fs.writeFileSync(output, JSON.stringify(report, null, 2));
async function measure(configs) {
  const cases = [];
  try {
    for (const config of configs) {const c = await startCase(config); cases.push(c); console.log('warm', config); await c.call('run');}
    for (let trial = 0; trial < 3; trial++) {
      let indices = cases.map((_, i) => i); if (trial === 1) indices.reverse(); if (trial === 2 && indices.length > 2) indices = [...indices.slice(1), indices[0]];
      for (const i of indices) {
        const c = cases[i]; order.push({trial, label: c.config.label});
        const r = await c.call('run'); delete r.id;
        const row = {trial, ...c.config, ...r}; records.push(row); save();
        console.log(c.config.label, trial, 'G', r.instructions / 1e9, 's', r.ms / 1000, 'combined_s', r.combined_ms / 1000);
      }
    }
    for (const config of configs) {
      const rows = records.filter(r => r.label === config.label);
      groups.push({...config, instructions: median(rows.map(r => r.instructions)), ms: median(rows.map(r => r.ms)), combined_ms: median(rows.map(r => r.combined_ms)), decoder_ms: median(rows.map(r => r.decoder_ms))});
    }
    save();
  } finally {for (const c of cases) await c.close();}
}
if (mode === 'calibrate') {
  for (const kind of ['fused', 'strict']) {
    const label = `${kind}-calibrate`, config = {kind, threads: 1, diagnostic: true, scheduler: 'static', label, wasm: artifact(dynamicDir, kind, true)};
    await measure([config]); const samples = records.filter(r => r.label === label);
    report[kind] = {rows: samples[0].jobs.map((j, i) => {
      if (samples.some(s => s.jobs[i].units !== j.units || s.jobs[i].op !== j.op)) throw Error('calibration job order changed');
      return {op: j.op, units: j.units, dims: j.dims, ns_per_unit: median(samples.map(s => s.jobs[i].ms)) * 1e6 / j.units};
    })};
  }
} else if (mode === 'sweep') {
  if (!calibration) throw Error('calibration required');
  for (const kind of ['fused', 'strict']) {
    const base = {kind, threads: 4, diagnostic: true, wasm: artifact(dynamicDir, kind, true), slowRanks: [2, 3], slowdown: 4};
    await measure([{...base, scheduler: 'static', label: `${kind}-static-4`}, ...[0.25, 1, 4].map(scale => ({...base, scale, scheduler: 'dynamic', label: `${kind}-dynamic-scale${scale}`}))]);
  }
} else if (mode === 'overhead') {
  for (const kind of ['fused', 'strict']) for (const threads of [1, 4, 6, 8]) {
    await measure([{kind, threads, scheduler: 'static', wasm: artifact(staticDir, kind), label: `${kind}-static-${threads}`}, {kind, threads, scheduler: 'dynamic', wasm: artifact(dynamicDir, kind), label: `${kind}-dynamic-${threads}`}]);
  }
} else if (mode === 'uneven') {
  if (!calibration) throw Error('calibration required');
  for (const kind of ['fused', 'strict']) {
    for (const threads of [2, 4, 6, 8]) {
      const base = {kind, threads, diagnostic: true, wasm: artifact(dynamicDir, kind, true), slowRanks: Array.from({length: threads - 2}, (_, i) => i + 2), slowdown: 4};
      await measure([{...base, scheduler: 'static', label: `${kind}-static-${threads}`}, {...base, scheduler: 'dynamic', label: `${kind}-dynamic-${threads}`}]);
    }
    const base = {kind, threads: 4, diagnostic: true, wasm: artifact(dynamicDir, kind, true), slowRanks: [0, 3], slowdown: 4};
    await measure([{...base, scheduler: 'static', label: `${kind}-static-slow-coordinator`}, {...base, scheduler: 'dynamic', label: `${kind}-dynamic-slow-coordinator`}]);
  }
} else throw Error(`unknown mode ${mode}`);
report.complete = true; save(); console.log('COMPLETE', mode, groups);
