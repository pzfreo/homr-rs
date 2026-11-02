// Browser worker/window runner. All reports are counts/first differences, never
// output hashes. The audit's CPU buffers are separate from GPU activations:
// validation cannot repair a GPU intermediate or hide drift behind fallback.
import {EncoderGPU,bitDifference,readBuffer} from '../tools/encoder_gpu.mjs';
import {EncoderPool} from '../tools/encoder_pool.mjs';

const load=async url=>{const r=await fetch(url);if(!r.ok)throw Error(`${url}: ${r.status}`);return r.arrayBuffer();};
const json=async url=>JSON.parse(new TextDecoder().decode(await load(url)));
const f32bits=x=>new Uint32Array(new Float32Array([x]).buffer)[0];
const float=b=>new Float32Array(new Uint32Array([b]).buffer)[0];
const median=xs=>{const a=xs.filter(x=>x!==null).toSorted((a,b)=>a-b);return a.length?a[Math.floor(a.length/2)]:null;};
function randomRecords(kind,count,state) {
  const next=()=>{state.value^=state.value<<13;state.value^=state.value>>>17;state.value^=state.value<<5;return state.value>>>0;};
  const normal=(low=1,width=254,positive=false)=>((positive?0:next()&0x80000000)|(next()&0x7fffff)|((low+next()%width)<<23))>>>0;
  const values=new Uint32Array(count*4);
  const edges=[0,0x80000000,0x00800000,0x00800001,0x3f6bffff,0x3f6c0000,0x3f6c0001,0x407b3333,0x407b3334,0xbf6c0000,0xc07b3333];
  for(let i=0;i<count;i++){
    const j=i*4;values[j]=kind;
    if(kind===1){values[j+1]=normal();values[j+2]=normal();}
    else if(kind===2)values[j+1]=normal(1,254,true);
    else if(kind===3||kind===4)values[j+1]=i%32===0?edges[next()%edges.length]:f32bits(next()/2**32*16-8);
    else if(kind===5)values[j+1]=f32bits(-next()/2**32*88.5);
    else {values[j+1]=normal(107,41);values[j+2]=normal(107,41);if(kind===0)values[j+3]=normal(107,41);}
  }
  return values;
}
async function checkRandomMath(gpu,cases,progress) {
  const result=[],state={value:0x46559714};
  for(let kind=0;kind<9;kind++){
    const row={kind,cases:0,differences:0,raw_differences:0,guards:0,first:null};
    for(let start=0;start<cases;start+=65536){
      const records=randomRecords(kind,Math.min(65536,cases-start),state);
      const r=await gpu.probeMath(records,undefined,null,{allowGuards:true});
      for(const name of ['cases','differences','raw_differences','guards'])row[name]+=r[name];row.first??=r.first;
    }
    row.unguarded_cases=row.cases-row.guards;result.push(row);progress({stage:'math',...row});
  }
  return result;
}
function tokens(e,decoder,context,out,native){
  const n=e.decoder_run(decoder,context,out,608,0);if(!n)throw Error('Decoder returned no steps');
  let changedSteps=0,changedHeads=0;
  for(let i=0;i<Math.min(n,native.length);i++){
    const got=new Uint32Array(e.memory.buffer,out+i*32,6);let changed=0;
    for(let j=0;j<6;j++)changed+=Number(got[j]!==native[i].tokens[j]);changedHeads+=changed;changedSteps+=Number(changed!==0);
  }
  return {steps:n,native_steps:native.length,differing_steps:changedSteps,differing_heads:changedHeads,extra_steps:Math.max(0,n-native.length),missing_steps:Math.max(0,native.length-n)};
}
class Audit {
  constructor(gpu){
    this.gpu=gpu;this.e=gpu.e;this.allocations=[];
    this.slots=gpu.meta.buffers.map(n=>this.alloc(n*4));
    this.temp=this.alloc(Math.max(...gpu.meta.buffers)*4);
    this.capacity=3*65536;
    this.records=this.alloc(this.capacity*16);this.expected=this.alloc(this.capacity*4);
  }
  alloc(n){const p=this.e.decoder_alloc(n);this.allocations.push([p,n]);return p;}
  reset(image){new Uint32Array(this.e.memory.buffer,this.slots[0],256*1280).set(new Uint32Array(this.e.memory.buffer,image,256*1280));this.rows=[];this.math={cases:0,differences:0,raw_differences:0,guards:0,by_kind:{},first:null};}
  async primitives(op){
    const q=op.params;if(q[0]!==3&&q[0]!==4)return;
    const norm=q[0]===3,total=q[1]*q[3],step=norm?Math.floor(65536/q[3])*q[3]:65536;
    for(let start=0;start<total;start+=step){
      const count=Math.min(step,total-start),n=this.e.encoder_gpu_audit_records(norm?0:1,this.slots[op.input]+start*4,count,q[3],float(q[6]),this.records,this.expected);
      if(!n||n>this.capacity)throw Error('Arithmetic audit record shape/capacity');
      // All expected values come from the CPU state, including after any GPU
      // guard/drift. No auditing of already-corrupted GPU operands as "native".
      const records=new Uint32Array(this.e.memory.buffer,this.records,n*4).slice();
      const expected=new Uint32Array(this.e.memory.buffer,this.expected,n).slice();
      const r=await this.gpu.probeMath(records,expected,null,{allowGuards:true});
      for(const name of ['cases','differences','raw_differences','guards'])this.math[name]+=r[name];
      for(let i=0;i<n;i++){const kind=records[i*4];this.math.by_kind[kind]=(this.math.by_kind[kind]??0)+1;}
      this.math.first??=r.first&&{op:op.name,batch_start:start,...r.first};
    }
  }
  async observe(gpu,op,index,phase,primitiveAudit){
    const q=op.params,n=q[1]*q[3];
    if(phase==='before'){
      if(primitiveAudit)await this.primitives(op);
      const count=this.e.encoder_gpu_reference(gpu.plan,index,this.slots[op.input],this.slots[op.residual],this.temp);
      if(count!==n)throw Error(`CPU oracle shape ${op.name}`);
      new Uint32Array(this.e.memory.buffer,this.slots[op.output],n).set(new Uint32Array(this.e.memory.buffer,this.temp,n));
    } else {
      const got=await readBuffer(gpu.device,gpu.buffers[op.output],n*4);
      const diff=bitDifference(got,new Uint32Array(this.e.memory.buffer,this.slots[op.output],n));
      const flags=new Uint32Array(await readBuffer(gpu.device,gpu.guard,4))[0];
      this.rows.push({name:op.name,...diff,cumulative_guard:flags});
    }
  }
  destroy(){for(const [p,n]of this.allocations.reverse())this.e.decoder_free(p,n);}
}

export async function runGPUEncoder(config,progress=console.log){
  const report={schema:1,revision:config.revision,wasm_bytes:config.wasm_bytes,wasm_mtime_ns:config.wasm_mtime_ns,userAgent:globalThis.navigator?.userAgent,scope:{pages:config.pages,staffs:config.staffs.length,corpus_staffs:config.corpus_staffs,full_corpus:config.staffs.length===config.corpus_staffs},audit_mode:config.audit,random_math_cases_per_kind:config.math_cases,gpu_ms_includes_input_scan:true,rows:[],complete:false};
  let pool,e,encoder=0,decoder=0,gpu,audit;const allocations=[];
  const started=performance.now();
  try {
    const module=await WebAssembly.compile(await load(config.wasm));
    const shared=WebAssembly.Module.imports(module).some(x=>x.module==='encoder_threads');
    pool=shared?await EncoderPool.create(module,{threads:config.threads}):null;
    e=pool?.exports??(await WebAssembly.instantiate(module,{})).exports;
    for(const name of ['encoder_gpu_reference','encoder_gpu_audit_records'])if(!e[name])throw Error(`Rebuild WASM: missing ${name}`);
    report.wasm={shared,threads:shared?config.threads:1};
    const alloc=n=>{const p=e.decoder_alloc(n);allocations.push([p,n]);return p;};
    const model=async(url,fn)=>{const bytes=await load(url),p=e.decoder_alloc(bytes.byteLength);new Uint8Array(e.memory.buffer,p,bytes.byteLength).set(new Uint8Array(bytes));try{const handle=fn(p,bytes.byteLength);if(!handle)throw Error('Model load/fused probe rejected');return handle;}finally{e.decoder_free(p,bytes.byteLength);}};
    encoder=await model(config.encoder,e.encoder_load);decoder=await model(config.decoder,e.decoder_load);
    const image=alloc(256*1280*4),context=alloc(1280*512*4),rawContext=alloc(1280*512*4),out=alloc(608*32);
    gpu=await EncoderGPU.create({exports:e,encoder,timestamps:true,retainPlan:true});report.gpu_status=gpu.status;
    if(!gpu.enabled){report.blocker='GPU initialization/self-test failed; CPU fallback available but not counted as a GPU run';return report;}
    progress({stage:'self-test',...gpu.status});
    report.math=await checkRandomMath(gpu,config.math_cases,progress);
    if(report.math.some(r=>r.differences||!r.unguarded_cases)){report.blocker='Primitive qualification failed';return report;}
    audit=new Audit(gpu);
    // Op checks first, then one complete staff, then the measured corpus.
    const first=config.staffs[0],firstImage=await load(first.input),firstNative=await load(first.context);
    new Uint8Array(e.memory.buffer,image,firstImage.byteLength).set(new Uint8Array(firstImage));audit.reset(image);
    const single=await gpu.encode(image,context,{captureRaw:true,observe:(...args)=>audit.observe(...args,config.audit!=='none')});
    report.single_staff={page:first.page,staff:first.staff,backend:single.backend,guard:single.guard,raw:single.raw?bitDifference(single.raw,firstNative):null,effective:bitDifference(new Uint32Array(e.memory.buffer,context,655360),firstNative),ops:audit.rows,math:audit.math,cpu_plan_context:bitDifference(new Uint32Array(e.memory.buffer,audit.slots[6],655360),firstNative)};
    if(report.single_staff.cpu_plan_context.bit_differences||audit.math.differences||audit.rows.some(r=>r.bit_differences&&!r.cumulative_guard)) {report.blocker='Single-staff op/CPU-plan qualification failed';return report;}
    if(!gpu.enabled){report.blocker='Device disabled during single-staff qualification';return report;}
    report.guard_cost=single.backend==='gpu'&&!single.guard&&!report.single_staff.raw?.bit_differences?
      await gpu.measureGuardCost(image,{reference:firstNative}):{available:false,reason:'First staff flagged or drifted; omitting unsafe unguarded control'};
    for(let i=0;i<config.staffs.length;i++){
      const staff=config.staffs[i];const [input,native,steps]=await Promise.all([load(staff.input),load(staff.context),json(staff.steps)]);
      new Uint8Array(e.memory.buffer,image,input.byteLength).set(new Uint8Array(input));
      const result=await gpu.encode(image,context,{captureRaw:true});
      const row={page:staff.page,staff:staff.staff,backend:result.backend,reason:result.reason,guard:result.guard??null,gpu_ms:result.gpu_ms??null,wall_ms:result.wall_ms??null,
        cpu_fallback_ms:result.cpu_ms??null,
        raw:result.raw?bitDifference(result.raw,native):null,effective:bitDifference(new Uint32Array(e.memory.buffer,context,655360),native),tokens:tokens(e,decoder,context,out,steps)};
      if(result.raw&&result.backend==='cpu'){
        new Uint32Array(e.memory.buffer,rawContext,655360).set(result.raw);
        try{row.raw_tokens=tokens(e,decoder,rawContext,out,steps);}catch(error){row.raw_tokens={error:String(error)};}
      }else row.raw_tokens=row.raw?row.tokens:null;
      if(config.audit==='full'||(config.audit==='first'&&i===0)){
        audit.reset(image);const r=await gpu.encode(image,context,{captureRaw:true,observe:(...args)=>audit.observe(...args,true)});
        row.audit={backend:r.backend,guard:r.guard,ops:audit.rows,math:audit.math,cpu_plan_context:bitDifference(new Uint32Array(e.memory.buffer,audit.slots[6],655360),native)};
      }
      report.rows.push(row);progress({stage:'corpus',completed:i+1,total:config.staffs.length,page:row.page,staff:row.staff,backend:row.backend,guard:row.guard,raw_differences:row.raw?.bit_differences,gpu_ms:row.gpu_ms});
      if(!gpu.enabled){report.blocker='Device lost/disabled; stopping GPU qualification after safe CPU fallback';break;}
    }
    const rows=report.rows,tokenDiff=t=>!t||t.error||t.differing_steps||t.differing_heads||t.extra_steps||t.missing_steps;
    report.summary={staffs:rows.length,gpu_staffs:rows.filter(r=>r.backend==='gpu').length,guard_firings:rows.filter(r=>r.guard).length,
      subnormal_guard_firings:rows.filter(r=>r.guard&1).length,certificate_guard_firings:rows.filter(r=>r.guard&4).length,domain_guard_firings:rows.filter(r=>r.guard&8).length,
      raw_context_exact:rows.filter(r=>r.raw?.bit_differences===0).length,effective_context_exact:rows.filter(r=>!r.effective.bit_differences).length,
      native_steps:rows.reduce((n,r)=>n+r.tokens.native_steps,0),differing_steps:rows.reduce((n,r)=>n+r.tokens.differing_steps+r.tokens.extra_steps+r.tokens.missing_steps,0),differing_heads:rows.reduce((n,r)=>n+r.tokens.differing_heads,0),
      raw_token_staffs_differing:rows.filter(r=>tokenDiff(r.raw_tokens)).length,median_gpu_ms:median(rows.map(r=>r.gpu_ms)),median_wall_ms:median(rows.map(r=>r.wall_ms)),
      audited_staffs:rows.filter(r=>r.audit).length,primitive_audit_cases:rows.reduce((n,r)=>n+(r.audit?.math.cases??0),0),primitive_audit_raw_differences:rows.reduce((n,r)=>n+(r.audit?.math.raw_differences??0),0),primitive_audit_unguarded_differences:rows.reduce((n,r)=>n+(r.audit?.math.differences??0),0),
      audited_op_values:rows.reduce((n,r)=>n+(r.audit?.ops.reduce((s,o)=>s+o.values,0)??0),0),audited_op_bit_differences:rows.reduce((n,r)=>n+(r.audit?.ops.reduce((s,o)=>s+o.bit_differences,0)??0),0)};
    report.complete=rows.length===config.staffs.length&&report.summary.native_steps===config.native_steps&&!report.blocker;
    report.effective_exact=report.complete&&rows.every(r=>!r.effective.bit_differences&&!tokenDiff(r.tokens))&&!rows.some(r=>r.audit&&(r.audit.cpu_plan_context.bit_differences||r.audit.math.differences||r.audit.ops.some(o=>o.bit_differences&&!o.cumulative_guard)));
    report.all_gpu_bit_exact=report.effective_exact&&rows.every(r=>r.backend==='gpu'&&!r.guard&&r.raw?.bit_differences===0&&!tokenDiff(r.raw_tokens));
    report.corpus_qualified=report.all_gpu_bit_exact&&report.scope.full_corpus&&config.audit==='full'&&report.summary.audited_staffs===config.corpus_staffs;
    return report;
  }catch(error){report.error=String(error.stack||error);return report;}
  finally{
    report.elapsed_ms=performance.now()-started;
    try{audit?.destroy();gpu?.destroy();if(e){if(encoder)e.encoder_drop(encoder);if(decoder)e.decoder_drop(decoder);for(const[p,n]of allocations.reverse())e.decoder_free(p,n);}await pool?.close();}
    catch(error){report.cleanup_error=String(error.stack||error);}
  }
}
