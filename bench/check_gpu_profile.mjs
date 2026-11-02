// Host-side timestamp bookkeeping checks only. No GPU arithmetic is simulated.
import assert from 'node:assert/strict';
import {EncoderGPU} from '../tools/encoder_gpu.mjs';
import {dispatchCount,entries} from '../tools/encoder_gpu_kernels.mjs';
globalThis.GPUBufferUsage??={MAP_READ:1,COPY_SRC:4,COPY_DST:8,UNIFORM:64,STORAGE:128,QUERY_RESOLVE:512};
globalThis.GPUShaderStage??={COMPUTE:4};globalThis.GPUMapMode??={READ:1};
const names=[...entries,'math_probe','weights_scan','image_scan'],dispatches=[];
let clock=0n,destroyed=0;
const buffer=size=>({data:new ArrayBuffer(size),getMappedRange(){return this.data;},unmap(){},async mapAsync(){},destroy(){destroyed++;}});
const d={pushErrorScope(){},async popErrorScope(){return null;},
  createBuffer({size}){assert.equal(size%16,0);return buffer(size);},
  createQuerySet({count}){return {times:new BigUint64Array(count),destroy(){destroyed++;}};},
  createShaderModule({code}){return {legacy:!code.includes('var<workgroup> tile_a'),guarded:code.includes('mag(r) == 0u'),async getCompilationInfo(){return {messages:[]};}};},
  createBindGroupLayout(x){return x;},createPipelineLayout(x){return x;},async createComputePipelineAsync(x){return x;},createBindGroup(x){return x;},
  queue:{writeBuffer(dst,offset,value){new Uint8Array(dst.data).set(new Uint8Array(value.buffer,value.byteOffset,value.byteLength),offset);},submit(){}},
  createCommandEncoder(){return {
    beginComputePass({timestampWrites:t}={}){let pipeline;return {
      setPipeline(p){pipeline=p;},setBindGroup(){},
      dispatchWorkgroups(x,y=1){
        const {module,entryPoint}=pipeline.compute;dispatches.push({legacy:module.legacy,guarded:module.guarded,name:entryPoint,x,y});
        const scale=module.legacy?4:module.guarded?2:1;
        if(t?.beginningOfPassWriteIndex!==undefined)t.querySet.times[t.beginningOfPassWriteIndex]=clock;
        clock+=BigInt(scale*1000);
        if(t?.endOfPassWriteIndex!==undefined)t.querySet.times[t.endOfPassWriteIndex]=clock;
      },end(){},
    };},
    resolveQuerySet(q,first,count,dst,offset){assert.equal(offset%256,0);new BigUint64Array(dst.data,offset,count).set(q.times.subarray(first,first+count));},
    copyBufferToBuffer(src,offset,dst,target,size){new Uint8Array(dst.data,target,size).set(new Uint8Array(src.data,offset,size));},finish(){return {};},
  };},
};
const program={group(){return {};},dispatch(cmd,kind,q,group,timestampWrites){
  const pass=cmd.beginComputePass({timestampWrites});
  pass.setPipeline({compute:{entryPoint:names[kind],module:{guarded:true,legacy:false}}});pass.setBindGroup(0,group);
  pass.dispatchWorkgroups(kind===9?Math.ceil(q[14]/256):kind===10?Math.ceil(q[14]/256):dispatchCount(q));pass.end();
}};
const q0=new Uint32Array(16);q0[0]=0;q0[1]=19;q0[2]=47;q0[3]=36;
const q1=new Uint32Array(16);q1[0]=3;q1[1]=67;q1[3]=96;
const scanParams=new Uint32Array(16);scanParams[14]=256*1280;
const memory=new WebAssembly.Memory({initial:32}),buffers=Array.from({length:7},()=>buffer(8));buffers[0]=buffer(scanParams[14]*4);
const expected=new Uint32Array([0x3f800000,0xbf800000]);new Uint32Array(buffers[6].data).set(expected);
const instance=Object.assign(new EncoderGPU(),{e:{memory},device:d,enabled:true,timestamp:true,busy:false,program,scanParams,scanQ:buffer(64),
  meta:{weights:4,output:6},weights:buffer(16),guard:buffer(4),buffers,contextBytes:8,
  ops:[q0,q1].map((params,i)=>({params,input:0,residual:0,output:6,q:buffer(64),group:{},name:'mock-'+i}))});
const r=await instance.profile(0,{trials:1,reference:expected});
assert.equal(r.valid,true);assert.equal(r.rows.length,3);assert.equal(r.kernels.length,2);
assert.equal(r.guarded_ms,0.006);assert.equal(r.legacy_guarded_ms,0.012);assert.equal(r.arithmetic_control_ms,0.002);
assert.equal(r.speedup,2);assert.equal(r.image_scan_ms,0.002);assert.equal(r.weight_scan_once_ms,0.002);
assert.ok(dispatches.some(x=>x.name==='norm'&&!x.legacy&&x.x===2));
assert.ok(dispatches.some(x=>x.name==='norm'&&x.legacy&&x.x===67));
assert.ok(dispatches.some(x=>x.name==='matmul'&&!x.legacy&&x.x===4));
assert.ok(dispatches.some(x=>x.name==='matmul'&&x.legacy&&x.x===3));
assert.equal(instance.busy,false);assert.ok(destroyed>0);
const bad=await instance.profile(0,{trials:1,reference:new Uint32Array([0,0])});
assert.equal(bad.valid,false);assert.equal(bad.speedup,null);assert.equal(bad.guard_and_image_delta_ms,null);
assert.ok(bad.rows.every(row=>row.bit_differences===2));
console.log('Host-only per-kernel timestamps, schedule dispatch, medians and invalid-control rejection passed (no GPU execution)');
