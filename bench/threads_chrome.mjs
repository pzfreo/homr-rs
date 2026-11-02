// COUNTER_ADDON=... node bench/threads_chrome.mjs WASM_DIR MODEL STAFF ORT_DIST OUT.json
// Build bench/counter.c's current addon: Chrome uses instructionsForPid.
import fs from 'node:fs';
import path from 'node:path';
import {createRequire} from 'node:module';
import {runHeadless} from './headless_chrome.mjs';
const [wasm, model, staff, ort, output] = process.argv.slice(2);
if (!output) throw Error('threads_chrome.mjs WASM_DIR MODEL STAFF ORT_DIST OUT.json');
const counter = createRequire(import.meta.url)(path.resolve(process.env.COUNTER_ADDON));
if (!counter.instructionsForPid) throw Error('Rebuild bench/counter.c; refusing to use Node self-counters for Chrome');
// Use a scratch directory containing only the selected model under a stable URL.
const modelRoot = path.join(path.dirname(path.resolve(output)), 'threads-chrome-model'); fs.mkdirSync(modelRoot, {recursive: true});
const modelLink = path.join(modelRoot, 'encoder.onnx'); if (!fs.existsSync(modelLink)) fs.symlinkSync(path.resolve(model), modelLink);
if (fs.realpathSync(modelLink) !== fs.realpathSync(model)) throw Error('scratch model URL points to a different model');
const configurations = [...[1, 2, 4, 6].flatMap(threads => [{kind: 'fused', threads}, {kind: 'strict', threads}]), ...[1, 4].map(threads => ({kind: 'ort', threads}))];
const median = xs => xs.toSorted((a, b) => a - b)[1];
const report = await runHeadless({output, roots: {wasm: path.resolve(wasm), model: modelRoot, staff: path.resolve(staff), ort: path.resolve(ort), bench: import.meta.dirname, tools: path.resolve(import.meta.dirname, '../tools')},
  html: `<!doctype html><script>
let worker;
function call(message){return new Promise((resolve,reject)=>{worker.onmessage=({data})=>data.error?reject(Error(data.error)):resolve(data);worker.onerror=reject;worker.postMessage(message);});}
window.prepare=async config=>{worker=new Worker('/bench/thread_scaling_browser_case.mjs',{type:'module'});await call({...config,command:'init'});};
window.measure=()=>call({command:'run'});
window.release=async()=>{await call({command:'close'});worker.terminate();worker=null;};
</script>`,
  drive: async ({send, browserSend}) => {
    const evaluate = async expression => {const r = await send('Runtime.evaluate', {expression, awaitPromise: true, returnByValue: true}); if (r.exceptionDetails) throw Error(JSON.stringify(r.exceptionDetails)); return r.result.value;};
    for (let i = 0; !(await evaluate("typeof window.prepare==='function'")); i++) {if (i > 100) throw Error('page setup timeout'); await new Promise(r => setTimeout(r, 100));}
    const pids = async () => (await browserSend('SystemInfo.getProcessInfo')).processInfo.filter(p => p.type === 'renderer').map(p => Number(p.id)).toSorted((a, b) => a - b);
    const records = [], order = [];
    for (let round = 0; round < 3; round++) {
      const indices = configurations.map((_, i) => i); if (round === 1) indices.reverse(); if (round === 2) indices.push(...indices.splice(0, 3));
      for (const i of indices) {
        const config = configurations[i]; order.push({round, ...config});
        // One active engine; warm/setup and teardown outside counters. Other
        // engines' GC/JIT contexts are not retained in the renderer.
        await evaluate(`window.prepare(${JSON.stringify(config)})`);
        const beforePids = await pids(); if (!beforePids.length) throw Error('no renderer PID');
        const before = beforePids.map(pid => counter.instructionsForPid(pid));
        const got = await evaluate('window.measure()');
        const after = beforePids.map(pid => counter.instructionsForPid(pid));
        if (JSON.stringify(beforePids) !== JSON.stringify(await pids())) throw Error('renderer PID changed; refusing incomplete counters');
        const row = {round, ...config, ms: got.ms, instructions: after.reduce((n, v, i) => n + v - before[i], 0), renderer_pids: beforePids, jobs: got.jobs};
        records.push(row); console.log(row); await evaluate('window.release()');
      }
    }
    return {complete: true, userAgent: await evaluate('navigator.userAgent'), crossOriginIsolated: await evaluate('crossOriginIsolated'), counter: 'proc_pid_rusage, all renderer PIDs including WASM workers, CDP overhead included',
      wall: 'performance.now inside encoder worker, correctness check included', records, order,
      summary: configurations.map(c => {const r = records.filter(v => v.kind === c.kind && v.threads === c.threads); return {...c, ms: median(r.map(v => v.ms)), instructions: median(r.map(v => v.instructions))};})};
  }});
console.log(JSON.stringify(report));
