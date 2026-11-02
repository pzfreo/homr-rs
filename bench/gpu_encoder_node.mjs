// Optional Dawn/Node execution. Install webgpu into scratch, not this project:
// npm install --prefix S/gpu_node --cache S/npm-cache webgpu
// node bench/gpu_encoder_node.mjs S WASM OUT.json S/gpu_node/node_modules/webgpu/index.js [server options]
import fs from 'node:fs';
import path from 'node:path';
import {pathToFileURL} from 'node:url';
import {execFileSync} from 'node:child_process';
const [scratch,wasm,output,packageFile,...options]=process.argv.slice(2);
if(!packageFile)throw Error('gpu_encoder_node.mjs S WASM OUT.json WEBGPU_INDEX.mjs [--limit 1 --profile 3 ...]');
const out=path.resolve(output);fs.mkdirSync(path.dirname(out),{recursive:true});
const config=fs.mkdtempSync(path.join(path.dirname(out),'gpu-node-config-'));
const {roots,manifest}=JSON.parse(execFileSync('python3',[path.join(import.meta.dirname,'gpu_encoder_server.py'),scratch,wasm,config,'--manifest-only',...options],{encoding:'utf8'}));
const {create,globals}=await import(pathToFileURL(path.resolve(packageFile)).href);
Object.assign(globalThis,globals);
const gpu=create(['backend=metal']);
let report;
try {
  const adapter=await gpu.requestAdapter({powerPreference:'high-performance'});
  if(!adapter){report={complete:false,blocker:'Dawn Node Metal requestAdapter returned null; no GPU measurements',backend:'metal'};}
  else {
    Object.defineProperty(globalThis,'navigator',{value:{gpu,userAgent:'Dawn Node Metal '+process.version},configurable:true});
    // Same browser runner and data paths; no local HTTP server is necessary.
    globalThis.fetch=async url=>{
      const [prefix,...parts]=String(url).replace(/^\//,'').split('/'),root=roots[prefix];
      if(!root)throw Error('Unknown manifest URL '+url);
      const file=path.resolve(root,...parts);if(!file.startsWith(path.resolve(root)+path.sep))throw Error('Invalid manifest URL '+url);
      try{return new Response(fs.readFileSync(file));}catch{return new Response('Missing file',{status:404});}
    };
    const {runGPUEncoder}=await import('./gpu_encoder_run.mjs');
    report=await runGPUEncoder(manifest,value=>console.log(JSON.stringify(value)));
  }
}catch(error){report={complete:false,error:String(error.stack||error)};}
fs.writeFileSync(out,JSON.stringify(report,null,2));
console.log(JSON.stringify(report.summary??report));
// Dawn owns background threads; explicit exit after awaited work/report write.
process.exit(report.complete&&report.all_gpu_bit_exact?0:1);
