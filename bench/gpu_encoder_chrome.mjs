// node bench/gpu_encoder_chrome.mjs S WASM SCRATCH_REPORT.json [server options]
// Default: all 627 staffs, million random cases/op, all corpus primitive audits.
import fs from 'node:fs';
import path from 'node:path';
import {execFileSync} from 'node:child_process';
import {runHeadless} from './headless_chrome.mjs';
const [scratch,wasm,output,...options]=process.argv.slice(2);
if(!output)throw Error('gpu_encoder_chrome.mjs SCRATCH WASM OUTPUT_JSON [--limit 1 --audit full ...]');
const out=path.resolve(output);fs.mkdirSync(path.dirname(out),{recursive:true});
const config=fs.mkdtempSync(path.join(path.dirname(out),'gpu-config-'));
const {roots}=JSON.parse(execFileSync('python3',[path.join(import.meta.dirname,'gpu_encoder_server.py'),scratch,wasm,config,'--manifest-only',...options],{encoding:'utf8'}));
const result=await runHeadless({output:out,roots,html:fs.readFileSync(path.join(import.meta.dirname,'gpu_encoder.html'),'utf8'),timeoutMs:24*60*60*1000});
console.log(JSON.stringify(result.summary??result));
// A CPU fallback never silently counts as GPU parity. A successful qualified
// guarded implementation can have effective_exact=true with explicit fallbacks.
if(!result.effective_exact||!result.complete)process.exitCode=1;
