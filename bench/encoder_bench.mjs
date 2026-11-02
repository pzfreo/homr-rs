// COUNTER_ADDON=... node bench/encoder_bench.mjs WASM MODEL NATIVE_STAFF OUTPUT
// Warm once, measure three complete staffs; compare all output bits each time.
import fs from 'node:fs';import path from 'node:path';import {createRequire} from 'node:module';
const [wasm,model,staff,output]=process.argv.slice(2);
const counter=createRequire(import.meta.url)(path.resolve(process.env.COUNTER_ADDON));
let previous=0,phases=Array(11).fill(0),active=false;
const {instance:{exports:e}}=await WebAssembly.instantiate(fs.readFileSync(wasm),{encoder_profile:{mark(id){if(active){const now=counter.instructions();phases[id]+=now-previous;previous=counter.instructions();}}}});
const copy=b=>{const p=e.decoder_alloc(b.length);new Uint8Array(e.memory.buffer,p,b.length).set(b);return p;};
const weights=fs.readFileSync(model),wp=copy(weights),handle=e.encoder_load(wp,weights.length);e.decoder_free(wp,weights.length);if(!handle)throw Error('model rejected');
const image=fs.readFileSync(path.join(staff,'input.f32')),ip=copy(image),op=e.decoder_alloc(1280*512*4),ref=fs.readFileSync(path.join(staff,'context.bin'));
e.encoder_run(handle,ip,op);const rows=[];
for(let i=0;i<3;i++){phases=Array(11).fill(0);const start=counter.instructions();previous=start;active=true;if(!e.encoder_run(handle,ip,op))throw Error('encode failed');const total=counter.instructions()-start;active=false;if(!Buffer.from(e.memory.buffer,op,ref.length).equals(ref))throw Error('context differs');rows.push({instructions:total,phases});}
const report={wasm,counter:'proc_pid_rusage RUSAGE_INFO_V4',runs:rows};fs.writeFileSync(output,JSON.stringify(report,null,2));console.log(report);e.encoder_drop(handle);e.decoder_free(ip,image.length);e.decoder_free(op,ref.length);
