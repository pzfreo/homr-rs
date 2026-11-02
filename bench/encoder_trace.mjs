// node bench/encoder_trace.mjs TRACE_WASM MODEL INPUT NATIVE_STAGES OUTPUT
import fs from 'node:fs';import path from 'node:path';
const [wasm,model,input,root,output]=process.argv.slice(2);
const refs=JSON.parse(fs.readFileSync(path.join(root,'stages.json'))),rows=[];let e;
const imports={encoder_trace:{trace(np,n,dp,len,h,w){
 const name=new TextDecoder().decode(new Uint8Array(e.memory.buffer,np,n));const row=refs.find(r=>r.name===name);if(!row)return;
 const bytes=fs.readFileSync(path.join(root,row.file));const ref=new Uint32Array(bytes.buffer.slice(bytes.byteOffset,bytes.byteOffset+bytes.length));if(ref.length!==len)throw Error(`shape differs: ${name}`);const got=new Uint32Array(e.memory.buffer,dp,len);const shape=row.shape;const nchw=shape.length===4&&shape[2]===h&&shape[3]===w;const c=len/(h*w);
 let different=0,first=null;
 for(let i=0;i<len;i++){const j=nchw?(i%c)*h*w+Math.floor(i/c):i;if(got[i]!==ref[j]){different++;first??={index:i,got:got[i],reference:ref[j]};}}
 const result={name,values:len,bit_differences:different,first};console.log(result);rows.push(result);
}}};
({instance:{exports:e}}=await WebAssembly.instantiate(fs.readFileSync(wasm),imports));
const copy=b=>{const p=e.decoder_alloc(b.length);new Uint8Array(e.memory.buffer,p,b.length).set(b);return p;};
const weights=fs.readFileSync(model),p=copy(weights);const handle=e.encoder_load(p,weights.length);e.decoder_free(p,weights.length);if(!handle)throw Error('model rejected');const image=fs.readFileSync(input),ip=copy(image),op=e.decoder_alloc(1280*512*4);
if(!e.encoder_run(handle,ip,op))throw Error('run failed');fs.writeFileSync(output,JSON.stringify(rows,null,2));e.encoder_drop(handle);e.decoder_free(ip,image.length);e.decoder_free(op,1280*512*4);
if(!rows.length||rows.some(r=>r.bit_differences))process.exitCode=1;
