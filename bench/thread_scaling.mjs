// Interleaved CPU scaling, one warm-up + three measured runs per case.
// COUNTER_ADDON=... node bench/thread_scaling.mjs STRICT FUSED MODEL STAFF ORT_MODULE OUTPUT
import fs from 'node:fs';
import path from 'node:path';
import os from 'node:os';
import {fork} from 'node:child_process';
const [strict, fused, model, staff, ortModule, output] = process.argv.slice(2);
if (!output) throw Error('thread_scaling.mjs STRICT FUSED MODEL STAFF ORT_MODULE OUTPUT');
const configurations = [
  ...[1, 2, 4, 6].flatMap(threads => [{kind: 'fused', threads, wasm: fused}, {kind: 'strict', threads, wasm: strict}]),
  ...[1, 4].map(threads => ({kind: 'ort', threads})),
];
const cases = [], records = [], order = [];
const median = xs => xs.toSorted((a, b) => a - b)[1];
async function startCase(config) {
  const worker = fork(new URL('./thread_scaling_case.mjs', import.meta.url), [JSON.stringify({...config, model, staff, ortModule})], {stdio: ['ignore', 'inherit', 'inherit', 'ipc']});
  const pending = new Map(); let next = 0;
  await new Promise((resolve, reject) => {worker.once('error', reject); worker.once('message', m => m.error ? reject(Error(m.error)) : resolve());});
  worker.on('message', m => {if (m.id) {const p = pending.get(m.id); pending.delete(m.id); m.error ? p.reject(Error(m.error)) : p.resolve(m);}});
  worker.on('error', error => {for (const p of pending.values()) p.reject(error); pending.clear();});
  return {...config, worker, call(command) {return new Promise((resolve, reject) => {const id = ++next; pending.set(id, {resolve, reject}); worker.send({id, command});});}};
}
try {
  for (const config of configurations) {const c = await startCase(config); cases.push(c); console.log('warm', c.kind, c.threads); await c.call('run');}
  // Predeclared rotation/reversal; keep raw samples/order, never pick fastest.
  for (let round = 0; round < 3; round++) {
    const indices = cases.map((_, i) => i);
    if (round === 1) indices.reverse();
    if (round === 2) indices.push(...indices.splice(0, 3));
    for (const index of indices) {
      const c = cases[index]; order.push({round, kind: c.kind, threads: c.threads});
      const wall = performance.now();
      const result = await c.call('run');
      const instructions = result.instructions, ms = performance.now() - wall;
      const row = {round, kind: c.kind, threads: c.threads, instructions, ms, engine_ms: result.ms, jobs: result.jobs};
      records.push(row); console.log(row);
      fs.writeFileSync(output, JSON.stringify({complete: false, records, order}, null, 2));
    }
  }
  const summary = configurations.map(c => {
    const rows = records.filter(r => r.kind === c.kind && r.threads === c.threads);
    return {...c, instructions: median(rows.map(r => r.instructions)), ms: median(rows.map(r => r.ms)), engine_ms: median(rows.map(r => r.engine_ms))};
  });
  const report = {complete: true, counter: 'proc_pid_rusage RUSAGE_INFO_V4, case process including all its workers; other cases are separate idle processes',
    wall: 'performance.now, round trip includes correctness check and host dispatch',
    machine: {cpus: os.cpus().map(c => c.model), node: process.version, platform: process.platform, arch: process.arch},
    model, staff, summary, records, order};
  fs.writeFileSync(output, JSON.stringify(report, null, 2)); console.log(summary);
} finally {
  for (const c of cases) {try {await c.call('close');} finally {if (!c.worker.killed) c.worker.kill();}}
}
