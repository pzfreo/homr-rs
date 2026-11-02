// Host-control tests without a GPU; optional real WASM plan/fallback check:
// node bench/check_gpu_host.mjs [WASM MODEL INPUT.f32 NATIVE_CONTEXT [THREADS]]
import assert from 'node:assert/strict';
import fs from 'node:fs';
import {EncoderGPU, specialBits, bitDifference} from '../tools/encoder_gpu.mjs';
import {EncoderPool} from '../tools/encoder_pool.mjs';
globalThis.GPUMapMode??={READ:1}; // Host-control mocks only, never GPU execution.
await assert.rejects(()=>EncoderGPU.create({exports:{},encoder:0,gpu:null}),/Live Rust encoder/);
assert.equal(specialBits(new Uint32Array([0,0x80000000,0x3f800000])),0);
assert.equal(specialBits(new Uint32Array([1])),1);
assert.ok(specialBits(new Uint32Array([0x7f800000]))&2);
assert.equal(specialBits(new Uint32Array([0x5e800000])),8);
for(const shared of [false,true])for(const scenario of ['guard','loss','validation','gpu']){
  const memory=new WebAssembly.Memory({initial:32,maximum:32,shared}),context=1500000;
  new Uint32Array(memory.buffer,context,2).fill(0x12345678);
  let cpuCalls=0,unmaps=0,pops=0;
  const e={memory,encoder_run(_model,_image,dst){cpuCalls++;assert.deepEqual([...new Uint32Array(memory.buffer,dst,2)],[0x12345678,0x12345678]);new Float32Array(memory.buffer,dst,2).fill(7);return 1;}};
  const mapped=new ArrayBuffer(32);new Float32Array(mapped,0,2).fill(9);new Uint32Array(mapped,8,1)[0]=scenario==='guard'?1:0;
  const gpu=Object.assign(new EncoderGPU(),{e,encoder:1,enabled:true,busy:false,onReport(){},status:{},timestamp:false,program:{dispatch(){}},ops:[],buffers:Array(7).fill({}),meta:{output:6},guard:{},contextBytes:8,
    device:{pushErrorScope(){},async popErrorScope(){return scenario==='validation'&&pops++===0?{message:'simulated validation error'}:null;},queue:{writeBuffer(){},submit(){}},createCommandEncoder(){return {copyBufferToBuffer(){},finish(){return {};}};}},
    readback:{async mapAsync(){if(scenario==='loss')throw Error('simulated device loss');},getMappedRange(){return mapped;},unmap(){unmaps++;}}});
  const result=await gpu.encode(0,context,{captureRaw:true});
  assert.equal(result.backend,scenario==='gpu'?'gpu':'cpu');assert.equal(cpuCalls,scenario==='gpu'?0:1);assert.equal(unmaps,scenario==='loss'?0:1);
  assert.deepEqual([...new Float32Array(memory.buffer,context,2)],scenario==='gpu'?[9,9]:[7,7]);
  if(scenario==='guard'){assert.equal(result.guard,1);assert.deepEqual([...new Float32Array(result.raw.buffer)],[9,9]);}
  if(scenario==='loss'||scenario==='validation')assert.equal(gpu.enabled,false);
  gpu.busy=true;await assert.rejects(()=>gpu.encode(0,context),/Serialize/);gpu.busy=false;gpu.closed=true;await assert.rejects(()=>gpu.encode(0,context),/closed/);
}
let rejects=0;
const failed=Object.assign(new EncoderGPU(),{e:{encoder_run(){rejects++;return 0;}},enabled:false,status:{reason:'unavailable'}});
await assert.rejects(()=>failed.encode(0,0),/CPU encoder rejected/);assert.equal(rejects,1);
console.log('Host-only SAB/ordinary copy, guard discard, device loss, validation fallback, closed/serialization and CPU-rejection tests passed (no GPU execution)');

if(process.argv.length>2){
  const [wasm,model,image,reference,threads='1']=process.argv.slice(2);
  const module=await WebAssembly.compile(fs.readFileSync(wasm));
  const shared=WebAssembly.Module.imports(module).some(x=>x.module==='encoder_threads');
  const pool=shared?await EncoderPool.create(module,{threads:Number(threads)}):null;
  const e=pool?.exports??(await WebAssembly.instantiate(module,{})).exports;
  const copy=data=>{const p=e.decoder_alloc(data.byteLength);new Uint8Array(e.memory.buffer,p,data.byteLength).set(data);return p;};
  const bytes=fs.readFileSync(model),mp=copy(bytes),encoder=e.encoder_load(mp,bytes.length);e.decoder_free(mp,bytes.length);assert.ok(encoder);
  const plan=e.encoder_gpu_plan_create(encoder);assert.ok(plan);
  const meta=JSON.parse(new TextDecoder().decode(new Uint8Array(e.memory.buffer,e.encoder_gpu_plan_json(plan),e.encoder_gpu_plan_json_len(plan))));
  assert.equal(meta.abi,1);assert.equal(meta.buffers[0],256*1280);assert.equal(meta.buffers[meta.output],1280*512);
  assert.ok(meta.ops.every(op=>op.params[15]===0&&op.params.length===16));
  assert.ok(e.encoder_gpu_plan_weights_len(plan)*4<=128*1024*1024);
  const weightsFlags=specialBits(new Uint32Array(e.memory.buffer,e.encoder_gpu_plan_weights(plan),e.encoder_gpu_plan_weights_len(plan)));
  assert.equal(weightsFlags,0,'Canonical model weights outside GPU domain');
  const input=fs.readFileSync(image),ip=copy(input),expected=fs.readFileSync(reference),op=e.decoder_alloc(expected.length);
  const gpu=await EncoderGPU.create({exports:e,encoder,gpu:null});
  const result=await gpu.encode(ip,op);assert.equal(result.backend,'cpu');
  const diff=bitDifference(new Uint32Array(e.memory.buffer,op,expected.length/4),new Uint32Array(expected.buffer,expected.byteOffset,expected.length/4));assert.equal(diff.bit_differences,0);
  console.log(JSON.stringify({shared,threads:Number(threads),plan_ops:meta.ops.length,weights:e.encoder_gpu_plan_weights_len(plan),weights_flags:weightsFlags,fallback:result.backend,...diff}));
  gpu.destroy();e.encoder_gpu_plan_drop(plan);e.encoder_drop(encoder);e.decoder_free(ip,input.length);e.decoder_free(op,expected.length);await pool?.close();
}
