// Headless Chrome WebGPU comparison, using exactly models_core.mjs's WASM loop.
// node bench/models_chrome.mjs NATIVE MODELS OUTPUT CORPUS_TXT ORT_DIST
// Scratch output only. Chrome launch/adapter failures fail loudly; no WASM-only
// result is labelled WebGPU. GPU instruction counts are deliberately not claimed.
import {spawn} from 'node:child_process';
import fs from 'node:fs';
import http from 'node:http';
import os from 'node:os';
import path from 'node:path';
const [native, models, output, corpus, ortDist] = process.argv.slice(2);
if (!ortDist) throw Error('models_chrome.mjs NATIVE MODELS OUTPUT CORPUS_TXT ORT_DIST');
const pages = fs.readFileSync(corpus, 'utf8').trim().split('\n').map(p => path.parse(p).name);
const modelPaths = Object.fromEntries([['segnet', 'segnet_308-'], ['encoder', 'encoder_pytorch_model_465-']].map(([k, prefix]) =>
  [k, `models/${fs.readdirSync(models).find(n => n.startsWith(prefix) && n.endsWith('.onnx') && !n.includes('fp16'))}`]));
const roots = {native, models, output, ort: ortDist, bench: import.meta.dirname};
const html = `<!doctype html><script type="module">
import * as ort from '/ort/ort.webgpu.min.mjs';
import {compareModels} from '/bench/models_core.mjs';
try {
 const adapter = await navigator.gpu?.requestAdapter();
 if (!adapter) throw Error('No WebGPU adapter');
 const info = adapter.info;
 const adapterInfo = Object.fromEntries(['vendor','architecture','device','description','subgroupMinSize','subgroupMaxSize'].map(k=>[k,info[k]]));
 // Keep ORT's default device creation, but record actual GPU submissions so
 // an adapter plus an entirely CPU-fallback session cannot masquerade as GPU.
 ort.env.webgpu.adapter = adapter;
 const originalSubmit = GPUQueue.prototype.submit;
 let gpuSubmissions = 0;
 GPUQueue.prototype.submit = function(...args) { gpuSubmissions++; return originalSubmit.apply(this,args); };
 ort.env.wasm.wasmPaths = '/ort/';
 const summary = await compareModels({ort, ep:'webgpu', threads:1, gpuSubmissions:()=>gpuSubmissions,
  pages:${JSON.stringify(pages)}, models:${JSON.stringify(modelPaths)},
  read: async name => { const r=await fetch('/'+name); if(!r.ok)throw Error(name+': '+r.status);return new Uint8Array(await r.arrayBuffer()); },
  write: async (name,data) => { const r=await fetch('/'+name,{method:'POST',body:data});if(!r.ok)throw Error('write '+name); },
 });
 if(!gpuSubmissions) throw Error('No WebGPU queue submissions; refusing a CPU-only result');
 window.result={summary, adapterInfo, gpuSubmissions, userAgent:navigator.userAgent};
} catch(e) {window.result={error:String(e.stack||e)};}
</script>`;
const server = http.createServer(async (req, res) => {
  res.setHeader('Cross-Origin-Opener-Policy', 'same-origin');
  res.setHeader('Cross-Origin-Embedder-Policy', 'require-corp');
  try {
    if (req.url === '/') {res.setHeader('Content-Type', 'text/html');res.end(html);return;}
    const [root, ...parts] = decodeURIComponent(req.url.split('?')[0]).slice(1).split('/');
    if (!roots[root] || parts.includes('..')) throw Error('invalid path');
    const file = path.join(roots[root], ...parts);
    if (req.method === 'POST') {
      if (root !== 'output') throw Error('write outside output');
      fs.mkdirSync(path.dirname(file), {recursive:true});
      const stream = fs.createWriteStream(file);
      req.pipe(stream); await new Promise((resolve,reject) => {stream.on('finish',resolve);stream.on('error',reject);});
      res.end('ok'); return;
    }
    res.setHeader('Content-Type', file.endsWith('.wasm') ? 'application/wasm' : file.endsWith('.mjs') ? 'text/javascript' : 'application/octet-stream');
    const stream=fs.createReadStream(file);stream.on('error',()=>{res.statusCode=404;res.end();});stream.pipe(res);
  } catch(e) {res.statusCode=400;res.end(String(e));}
});
await new Promise(r => server.listen(0, '127.0.0.1', r));
const profile=fs.mkdtempSync(path.join(os.tmpdir(),'cx-models-chrome-'));
const chrome=spawn(process.env.CHROME || '/Applications/Google Chrome.app/Contents/MacOS/Google Chrome',[
  '--headless=new','--enable-unsafe-webgpu','--remote-debugging-port=0',`--user-data-dir=${profile}`,'about:blank'],
  {stdio:['ignore','ignore','pipe']});
let ws, done=false;
const deadline = setTimeout(()=>finish(new Error('Chrome comparison timed out')), 4*3600*1000);
function finish(error) {
  if(done)return;done=true;clearTimeout(deadline);ws?.close();chrome.kill();server.close();
  if(error){console.error(error);process.exitCode=1;}
}
chrome.on('error',finish);
chrome.on('exit',(code, signal)=>{if(!done)finish(new Error(`Chrome exited ${code} ${signal}`));});
try {
  let port;
  for(let i=0;i<100&&!done;i++) {
    try {port=fs.readFileSync(path.join(profile,'DevToolsActivePort'),'utf8').split('\n')[0];break;}catch{}
    await new Promise(r=>setTimeout(r,200));
  }
  if(done) throw Error('Chrome failed before CDP became ready');
  if(!port) throw Error('Chrome CDP unavailable');
  const tabs=await(await fetch(`http://127.0.0.1:${port}/json`)).json();
  ws=new WebSocket(tabs.find(t=>t.type==='page').webSocketDebuggerUrl);
  await new Promise((resolve,reject)=>{ws.onopen=resolve;ws.onerror=reject;});
  let id=0;const pending=new Map();
  const send=(method,params={})=>new Promise((resolve,reject)=>{const key=++id;pending.set(key,{resolve,reject});ws.send(JSON.stringify({id:key,method,params}));});
  ws.onmessage=({data})=>{
    const m=JSON.parse(data);
    if(m.id){const p=pending.get(m.id);pending.delete(m.id);if(m.error)p?.reject(Error(JSON.stringify(m.error)));else p?.resolve(m.result);}
    if(m.method==='Runtime.consoleAPICalled')console.log(m.params.args.map(a=>a.value??a.description).join(' '));
  };
  await send('Runtime.enable');
  await send('Page.navigate',{url:`http://127.0.0.1:${server.address().port}/`});
  while(!done) {
    await new Promise(r=>setTimeout(r,2000));
    const r=await send('Runtime.evaluate',{expression:'JSON.stringify(window.result??null)',returnByValue:true});
    const value=r?.result?.value;
    if(value&&value!=='null') {
      fs.mkdirSync(output,{recursive:true});fs.writeFileSync(path.join(output,'chrome.json'),value);
      const result=JSON.parse(value);if(result.error)throw Error(result.error);
      console.log(value);finish();
    }
  }
} catch(e) {finish(e);}
