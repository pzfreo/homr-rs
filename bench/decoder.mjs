// /usr/bin/time -l node bench/decoder.mjs rust|ort load|decode MODEL REF STAFF...
// e.g. REF=/scratch/decoder_ref STAFF=angelus_0/0 angelus_0/1
// ORT_MODULE=/path/to/node_modules/onnxruntime-web/dist/ort.node.min.mjs
// Check mode uses real greedy EOS; decode mode always uses reference step counts.
import fs from 'node:fs';
import path from 'node:path';
import { pathToFileURL } from 'node:url';
const [engine,mode,model,root,...requested]=process.argv.slice(2);
const corpus=requested.length===1&&requested[0]==='@all';
const staffs=corpus?fs.readdirSync(root).sort().flatMap(page=>{const p=path.join(root,page,'decoder.json');if(!fs.existsSync(p))throw Error(`missing page ${page}`);const n=JSON.parse(fs.readFileSync(p)).calls.length;return Array.from({length:n},(_,i)=>`${page}/${i}`);}):requested;
if (!['rust','ort'].includes(engine)||!['load','decode','check'].includes(mode)||!staffs.length)
  throw Error('decoder.mjs rust|ort load|decode|check MODEL REF PAGE/STAFF ...');
const readStaff=name=>{
  const d=path.join(root,name), bytes=fs.readFileSync(path.join(d,'context.bin'));
  return {name,context:new Float32Array(bytes.buffer.slice(bytes.byteOffset,bytes.byteOffset+bytes.byteLength)),reference:JSON.parse(fs.readFileSync(path.join(d,'steps.json')))};
};
const data=corpus?staffs.map(name=>({name})):staffs.map(readStaff);
const weights=fs.readFileSync(model);
let run;
if(engine==='rust') {
  const wasm=process.env.DECODER_WASM||'target/wasm32-unknown-unknown/release/homr_rs.wasm';
  const {instance:{exports:e}}=await WebAssembly.instantiate(fs.readFileSync(wasm),{});
  const p=e.decoder_alloc(weights.length);
  new Uint8Array(e.memory.buffer,p,weights.length).set(weights);
  const handle=e.decoder_load(p,weights.length);
  e.decoder_free(p,weights.length);
  if(!handle) throw Error('model rejected');
  run=async ({context,reference})=>{
    const c=e.decoder_alloc(context.byteLength),out=e.decoder_alloc(608*32);
    new Float32Array(e.memory.buffer,c,context.length).set(context);
    const count=e.decoder_run(handle,c,out,mode==='check'?608:reference.length,mode==='check'?0:1);
    if(!count) throw Error('decode failed');
    const result=Array.from({length:count},(_,i)=>({tokens:Array.from(new Uint32Array(e.memory.buffer,out+i*32,6)),attention:Array.from(new Float32Array(e.memory.buffer,out+i*32+24,2))}));
    e.decoder_free(c,context.byteLength);e.decoder_free(out,608*32);
    return result;
  };
} else {
  const ort=process.env.ORT_MODULE?await import(pathToFileURL(process.env.ORT_MODULE)):await import('onnxruntime-web');
  ort.env.wasm.numThreads=1;
  const session=await ort.InferenceSession.create(weights,{executionProviders:['wasm'],graphOptimizationLevel:'all'});
  const heads=['out_rhythms','out_pitchs','out_lifts','out_positions','out_articulations','out_slurs'];
  const names=[...heads,'attention',...Array.from({length:32},(_,i)=>`cache_out${i}`)];
  const token=v=>new ort.Tensor('int64',BigInt64Array.of(BigInt(v)),[1,1]);
  run=async ({context,reference})=>{
    const full=new ort.Tensor('float32',context,[1,1280,512]);
    const reduced=new ort.Tensor('float32',context.slice(0,512),[1,1,512]);
    let cache=Array.from({length:32},()=>new ort.Tensor('float32',new Float32Array(0),[1,8,0,64]));
    let last=[1,0,0,0,0];const result=[];
    for(let i=0;i<(mode==='check'?608:reference.length);i++) {
      const feeds={context:i===0?full:reduced,cache_len:new ort.Tensor('int64',BigInt64Array.of(BigInt(i)),[1])};
      ['rhythms','pitchs','lifts','articulations','slurs'].forEach((n,j)=>feeds[n]=token(last[j]));
      cache.forEach((t,j)=>feeds[`cache_in${j}`]=t);
      const out=await session.run(feeds,names);
      cache=names.slice(7).map(n=>out[n]);
      const tokens=heads.map(n=>{ const a=out[n].data;let best=0;for(let j=1;j<a.length;j++)if(a[j]>a[best])best=j;return best; });
      result.push({tokens,attention:Array.from(out.attention.data)});
      if(mode==='check'&&tokens[0]===2)break;
      last=[tokens[0]===2?3:tokens[0],tokens[1],tokens[2],tokens[4],tokens[5]];
    }
    return result;
  };
}
if(mode!=='load') {
let steps=0,differingSteps=0,lengthDifferences=0,maxAbs=0,totalAbs=0,count=0;
const differences=[];
const saved=new Map();
const vocab=process.env.SAVE_DECODER?JSON.parse(fs.readFileSync(process.env.VOCAB)):null;
for(const source of data) {
  const staff=corpus?readStaff(source.name):source;
  const results=await run(staff);
  if(vocab) {
    const [page,index]=staff.name.split('/');
    if(!saved.has(page)) {const meta=JSON.parse(fs.readFileSync(path.join(root,page,'decoder.json')));meta.calls=new Array(meta.calls.length).fill(null);saved.set(page,meta);}
    const meta=saved.get(page);
    meta.calls[+index]=results.filter(r=>r.tokens[0]>2).map(r=>{const t=r.tokens;return [vocab.rhythm[t[0]],vocab.pitch[t[1]],vocab.lift[t[2]],vocab.articulation[t[4]],vocab.slur[t[5]],vocab.position[t[3]],r.attention];});
    if(meta.calls.every(c=>c!==null)) {const dir=path.join(process.env.SAVE_DECODER,page);fs.mkdirSync(dir,{recursive:true});fs.writeFileSync(path.join(dir,'decoder.json'),JSON.stringify(meta));saved.delete(page);}
  }
  if(corpus) console.error('checked',staff.name,results.length);
steps+=results.length;
  if(mode==='check') {
    if(results.length!==staff.reference.length)lengthDifferences++;
    results.forEach((r,i)=>{
      const ref=staff.reference[i];if(!ref)return;
      if(r.tokens.some((t,j)=>t!==ref.tokens[j])) {
        differingSteps++;const diff={staff:staff.name,step:i,rust:r.tokens,reference:ref.tokens};differences.push(diff);
        console.error(JSON.stringify(diff));
      }
      for(let j=0;j<2;j++){const a=Math.abs(r.attention[j]-ref.attention[j]);maxAbs=Math.max(maxAbs,a);totalAbs+=a;count++;}
    });
  }
}
const report={engine,staffs:staffs.length,steps,differingSteps,lengthDifferences,attentionMaxAbs:maxAbs,attentionMeanAbs:totalAbs/count,differences};
console.log(JSON.stringify(report));
if(process.env.REPORT)fs.writeFileSync(process.env.REPORT,JSON.stringify(report,null,2));
if(differingSteps||lengthDifferences)process.exitCode=1;

}
if(process.env.BENCH_WAIT) {
  console.log('BENCH_READY');
  process.stdin.resume();
  await new Promise(resolve=>process.stdin.once('data',resolve));
  process.stdin.pause();
}
