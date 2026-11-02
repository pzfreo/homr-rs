// node bench/encoder_ops.mjs WASM PROBE_DIRECTORY OUTPUT_JSON
import fs from 'node:fs';
import path from 'node:path';
import {EncoderPool} from '../tools/encoder_pool.mjs';
const [wasm,root,output]=process.argv.slice(2);
const pool=process.env.ENCODER_THREADS?await EncoderPool.create(await WebAssembly.compile(fs.readFileSync(wasm)),{threads:Number(process.env.ENCODER_THREADS)}):null;
const e=pool?.exports??(await WebAssembly.instantiate(fs.readFileSync(wasm),{})).instance.exports;
const copy=bytes=>{const p=e.decoder_alloc(bytes.length);new Uint8Array(e.memory.buffer,p,bytes.length).set(bytes);return p;};
const manifest=JSON.parse(fs.readFileSync(path.join(root,'probes.json'))),rows=[];
for(const probe of manifest.probes){
 const model=fs.readFileSync(path.join(root,probe.model));const p=copy(model);const handle=e.encoder_probe_load(p,model.length);e.decoder_free(p,model.length);if(!handle)throw Error('probe rejected');
 const bytes=fs.readFileSync(path.join(root,probe.inputs[0].file));const input=copy(bytes);const ref=fs.readFileSync(path.join(root,probe.outputs[0].file));const out=e.decoder_alloc(ref.length);
 const shape=probe.inputs[0].shape;
 const n=e.encoder_probe_run(handle,input,bytes.length/4,out,shape.length===4?shape[2]:0,shape.length===4?shape[3]:0,shape.length===4?shape[1]:0);if(n*4!==ref.length)throw Error('output size');
 const got=new Uint32Array(e.memory.buffer,out,n),expected=new Uint32Array(ref.buffer.slice(ref.byteOffset,ref.byteOffset+ref.length));let different=0;let first=null;
 for(let i=0;i<n;i++)if(got[i]!==expected[i]){different++;first??={index:i,got:got[i],reference:expected[i]};}
 const row={kind:probe.kind,values:n,bit_differences:different,first};console.log(row);rows.push(row);
 e.decoder_free(input,bytes.length);e.decoder_free(out,ref.length);e.encoder_probe_drop(handle);
}
fs.writeFileSync(output,JSON.stringify(rows,null,2));if(rows.some(r=>r.bit_differences))process.exitCode=1;
await pool?.close();
