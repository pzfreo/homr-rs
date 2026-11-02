// COUNTER_ADDON=... node bench/encoder.mjs WASM MODEL NATIVE_INPUTS OUT CORPUS_TXT
// Full corpus, bit comparison, scratch contexts, instruction and optional phase
// counters. Each staff executes even if its input repeats another staff's.
import fs from 'node:fs';
import path from 'node:path';
import {createRequire} from 'node:module';
const [wasm,model,root,out,corpus]=process.argv.slice(2);
const counter=process.env.COUNTER_ADDON?createRequire(import.meta.url)(path.resolve(process.env.COUNTER_ADDON)):null;
let profiling=false,previous=0,phases=Array(11).fill(0);
const imports={encoder_profile:{mark(id){if(profiling){const now=counter.instructions();phases[id]+=now-previous;if(process.env.TRACE_PHASES)console.log('phase',id,now-previous);previous=counter.instructions();}}}};
const {instance:{exports:e}}=await WebAssembly.instantiate(fs.readFileSync(wasm),imports);
const copy=bytes=>{const p=e.decoder_alloc(bytes.length);new Uint8Array(e.memory.buffer,p,bytes.length).set(bytes);return p;};
const bytes=fs.readFileSync(model),p=copy(bytes),handle=e.encoder_load(p,bytes.length);e.decoder_free(p,bytes.length);if(!handle)throw Error('encoder rejected (including fused-host check)');
const ip=e.decoder_alloc(256*1280*4),op=e.decoder_alloc(1280*512*4);let staffs=0,differing=0;const pages=[];
let decoder=0,tokensOut=0;const vocab=process.env.VOCAB?JSON.parse(fs.readFileSync(process.env.VOCAB)):null;
if(process.env.DECODER_MODEL){const b=fs.readFileSync(process.env.DECODER_MODEL),p=copy(b);decoder=e.decoder_load(p,b.length);e.decoder_free(p,b.length);if(!decoder)throw Error('decoder rejected');tokensOut=e.decoder_alloc(608*32);}
for(const source of fs.readFileSync(corpus,'utf8').trim().split('\n')){
 const page=path.parse(source).name;const meta=JSON.parse(fs.readFileSync(path.join(root,page,'models.json')));const rows=[];
 const calls=[];
 for(let staff=0;staff<meta.staffs;staff++){
  const d=path.join(root,page,String(staff));const image=fs.readFileSync(path.join(d,'input.f32'));new Uint8Array(e.memory.buffer,ip,image.length).set(image);
  phases=Array(11).fill(0);profiling=!!counter;const start=counter?.instructions();previous=start;
  if(e.encoder_run(handle,ip,op)!==1)throw Error('encode failed');
  const instructions=counter?counter.instructions()-start:null;profiling=false;
  const ref=fs.readFileSync(path.join(d,'context.bin'));const expected=new Uint32Array(ref.buffer.slice(ref.byteOffset,ref.byteOffset+ref.length));const got=new Uint32Array(e.memory.buffer,op,expected.length);let diff=0,max=0,sum=0;
  const ef=new Float32Array(expected.buffer),gf=new Float32Array(e.memory.buffer,op,got.length);
  for(let i=0;i<got.length;i++){diff+=Number(got[i]!==expected[i]);const error=Math.abs(gf[i]-ef[i]);if(!Number.isFinite(error))throw Error('nonfinite output');max=Math.max(max,error);sum+=error;}
  const dst=path.join(out,page,String(staff));fs.mkdirSync(dst,{recursive:true});fs.writeFileSync(path.join(dst,'context.bin'),Buffer.from(e.memory.buffer,op,ref.length));
  const row={staff,values:got.length,bit_differences:diff,max_abs:max,mean_abs:sum/got.length,instructions,phases,memory_bytes:e.memory.buffer.byteLength};
  if(decoder){
   const refs=JSON.parse(fs.readFileSync(path.join(process.env.DECODER_REFS,page,String(staff),'steps.json')));const steps=process.env.COMBINED?e.encoder_decode(handle,decoder,ip,tokensOut,608):e.decoder_run(decoder,op,tokensOut,608,0);if(!steps)throw Error('decode failed');
   let changedSteps=0,changedHeads=0;const symbols=[];
   for(let i=0;i<steps;i++){
    const t=Array.from(new Uint32Array(e.memory.buffer,tokensOut+i*32,6)),attention=Array.from(new Float32Array(e.memory.buffer,tokensOut+i*32+24,2));
    if(refs[i]){const changed=t.filter((v,j)=>v!==refs[i].tokens[j]).length;changedHeads+=changed;changedSteps+=Number(changed!==0);}
    if(vocab&&t[0]>2)symbols.push([vocab.rhythm[t[0]],vocab.pitch[t[1]],vocab.lift[t[2]],vocab.articulation[t[4]],vocab.slur[t[5]],vocab.position[t[3]],attention]);
   }
   Object.assign(row,{steps,differing_steps:changedSteps,differing_tokens:changedHeads,missing_steps:Math.max(0,refs.length-steps),extra_steps:Math.max(0,steps-refs.length)});calls.push(symbols);
  }
  rows.push(row);staffs++;differing+=Number(diff!==0);console.log(page,staff,diff,instructions, row.steps??'');
 }
 const row={page,encoder:rows};pages.push(row);fs.writeFileSync(path.join(out,page,'metrics.json.tmp'),JSON.stringify(row));fs.renameSync(path.join(out,page,'metrics.json.tmp'),path.join(out,page,'metrics.json'));
 if(decoder&&vocab){const meta=JSON.parse(fs.readFileSync(path.join(process.env.DECODER_REFS,page,'decoder.json')));meta.calls=calls;fs.writeFileSync(path.join(out,page,'decoder.json'),JSON.stringify(meta));}
}
const report={wasm,staffs,differing_staffs:differing,pages};fs.writeFileSync(path.join(out,'comparison.json'),JSON.stringify(report,null,2));console.log({staffs,differing});e.encoder_drop(handle);e.decoder_free(ip,256*1280*4);e.decoder_free(op,1280*512*4);if(differing)process.exitCode=1;
if(decoder){e.decoder_drop(decoder);e.decoder_free(tokensOut,608*32);if(pages.some(p=>p.encoder.some(s=>s.differing_steps||s.extra_steps||s.missing_steps)))process.exitCode=1;}
