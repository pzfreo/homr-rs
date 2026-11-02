// node bench/gpu_fc1_chrome.mjs SCRATCH_FIXTURE_DIR SCRATCH_REPORT.json
import path from 'node:path';
import {runHeadless} from './headless_chrome.mjs';
const [fixture, output] = process.argv.slice(2);
if (!output) throw Error('gpu_fc1_chrome.mjs FIXTURE_DIR OUTPUT_JSON');
const result = await runHeadless({output, roots: {fixture: path.resolve(fixture), bench: import.meta.dirname},
  html: `<!doctype html><script type="module">
import {runFc1} from '/bench/gpu_fc1.mjs';
try {window.result = await runFc1();} catch(e) {window.result={error:String(e.stack||e)};}
</script>`});
console.log(JSON.stringify(result));
if (!result.exact) process.exitCode = 1;
