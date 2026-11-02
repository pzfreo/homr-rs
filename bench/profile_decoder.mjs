// Profile build only: PROFILE_WASM=... COUNTER_ADDON=... node bench/profile_decoder.mjs MODEL REF PAGE/STAFF ...
import fs from 'node:fs';
import path from 'node:path';
import {createRequire} from 'node:module';
const require=createRequire(import.meta.url);
const counter=require(path.resolve(process.env.COUNTER_ADDON));
for(let i=0;i<10000;i++)counter.instructions();
const [model,root,...staffs]=process.argv.slice(2);
const data=staffs.map(name=>({context:fs.readFileSync(path.join(root,name,'context.bin')),steps:JSON.parse(fs.readFileSync(path.join(root,name,'steps.json'))).length}));
const {instance:{exports:e}}=await WebAssembly.instantiate(fs.readFileSync(process.env.PROFILE_WASM),{decoder_profile:{instructions:counter.instructions}});
const weights=fs.readFileSync(model),p=e.decoder_alloc(weights.length);
new Uint8Array(e.memory.buffer,p,weights.length).set(weights);
const handle=e.decoder_load(p,weights.length);e.decoder_free(p,weights.length);
if(!handle)throw Error('model rejected');
const inputs=data.map(d=>{const c=e.decoder_alloc(d.context.length),o=e.decoder_alloc(608*32);new Uint8Array(e.memory.buffer,c,d.context.length).set(d.context);return {...d,c,o};});
const overhead=e.decoder_profile_calibrate();e.decoder_profile_reset();
let steps=0;
for(const d of inputs){steps+=e.decoder_run(handle,d.c,d.o,d.steps,1);}
e.decoder_profile_stop();
const names=['other','context_kv','self_qkv','self_out','cross_q','cross_out','ff_in','ff_out','heads','quantize','scores','softmax','weighted_v','norm','sigmoid_gelu','centroid','alloc_free','copies','empty'];
const phases=names.map((phase,i)=>{const raw=e.decoder_profile_value(i,0),calls=e.decoder_profile_value(i,1);return {phase,raw,calls,corrected:Math.max(0,raw-calls*overhead)};});
console.log(JSON.stringify({steps,overhead,phases},null,2));
