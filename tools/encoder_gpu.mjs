// No bundler/wasm-bindgen. Works with ordinary exports or EncoderPool.exports.
// const gpu = await EncoderGPU.create({exports: e, encoder: encoderHandle});
// await gpu.encode(imagePtr, contextPtr); e.decoder_run(decoder,contextPtr,...);
import {bindings, dispatchCount, entries, shaderSource} from './encoder_gpu_kernels.mjs';

const bytes = value => new Uint8Array(value.buffer || value, value.byteOffset || 0, value.byteLength);
const inPlace = kind => kind === 4 || kind === 5 || kind === 7;
export function bitDifference(got, expected) {
  const a = got instanceof Uint32Array ? got : new Uint32Array(got);
  const b = expected instanceof Uint32Array ? expected : new Uint32Array(expected);
  if (a.length !== b.length) throw Error('comparison shape differs');
  let count = 0, first = null;
  for (let i=0;i<a.length;i++) if(a[i]!==b[i]) {count++;first??={index:i,got:a[i],expected:b[i]};}
  return {values:a.length,bit_differences:count,first};
}
export function specialBits(value) {
  let flags=0;
  for(const word of value) {const m=word&0x7fffffff;if(m && m<0x800000) flags|=1;if(m>=0x7f800000) flags|=2;}
  return flags;
}
function storage(device, valueOrSize, usage = GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_SRC | GPUBufferUsage.COPY_DST) {
  const data=typeof valueOrSize === 'number' ? null : bytes(valueOrSize);
  const buffer=device.createBuffer({size:Math.max(4,data?.byteLength??valueOrSize),usage,mappedAtCreation:!!data});
  if(data) {new Uint8Array(buffer.getMappedRange()).set(data);buffer.unmap();}
  return buffer;
}
function uniform(device,q) {return storage(device,new Uint32Array(q),GPUBufferUsage.UNIFORM|GPUBufferUsage.COPY_DST);}
export async function readBuffer(device,source,size,offset=0) {
  const target=storage(device,size,GPUBufferUsage.COPY_DST|GPUBufferUsage.MAP_READ);
  try {const cmd=device.createCommandEncoder();cmd.copyBufferToBuffer(source,offset,target,0,size);device.queue.submit([cmd.finish()]);
    await target.mapAsync(GPUMapMode.READ);return target.getMappedRange().slice(0);
  } finally {target.destroy();}
}

class Program {
  static async create(device,diagnosticUnguarded=false) {
    const code=shaderSource({diagnosticUnguarded}), module=device.createShaderModule({code});
    const info=await module.getCompilationInfo();
    const errors=info.messages.filter(m=>m.type==='error');if(errors.length) throw Error(JSON.stringify(errors));
    const names=[...entries,'math_probe'];
    const pipelines=await Promise.all(names.map(entryPoint=>device.createComputePipelineAsync({layout:'auto',compute:{module,entryPoint}})));
    return Object.assign(new Program(),{device,pipelines,diagnosticUnguarded});
  }
  group(kind,resources) {
    const indices=kind===8?[0,3,4,5]:bindings[kind];
    return this.device.createBindGroup({layout:this.pipelines[kind].getBindGroupLayout(0),entries:indices.map(binding=>({binding,resource:{buffer:resources[binding]}}))});
  }
  dispatch(command,kind,q,group,timestampWrites) {
    const pass=command.beginComputePass(timestampWrites?{timestampWrites}:{});
    pass.setPipeline(this.pipelines[kind]);pass.setBindGroup(0,group);
    pass.dispatchWorkgroups(kind===8?Math.ceil(q[14]/64):dispatchCount(q));pass.end();
  }
}

export function mathKnownAnswers() {
  const cases=[];
  const add=(kind,a,b=0,c=0,guard=false)=>cases.push({record:[kind,a>>>0,b>>>0,c>>>0],guard});
  add(0,0x3f800001,0x3f7ffffe,0xbf800000); // fusion
  add(0,0x3f800001,0x3f7ffffe,0x4b800001); // f64 double-round trap
  add(0,0x80000000,0x3f800000,0x80000000);add(0,0x3f800000,0x3f800000,0xbf800000);
  add(0,0x1c800000,0x1c800000,0x00800000); // product underflows, fused result normal
  for(const x of [0,0x80000000,0x00800000,0x00800001,0x3effffff,0x3f000000,0x3f7fffff,0x3f800000,0x3f800001,0x3fffffff,0x40000000,0x7f7fffff]) {
    add(1,x,0x3f800000);add(1,x,0x3fc00000,0,(x&0x7fffffff)>0&&(x&0x7fffffff)<0xc00000);
    add(2,x);
  }
  for(const x of [0,0x80000000,0x3e800000,0x3f6bffff,0x3f6c0000,0x3f6c0001,0x3f800000,0x407b3333,0x40800000,0xbf800000,0xc0800000]) {add(3,x);add(4,x);}
  for(const x of [0,0xbf000000,0xbf800000,0xc0800000,0xc1800000]) add(5,x);
  add(6,0x80000000,0x80000000);add(6,0x3f800000,0xbf800000);
  add(7,0x3f800001,0x3f7ffffe);add(8,0x3f800001,0x3f800000);
  add(0,1,0x3f000000,1,true);add(7,0x17800000,0x17800000,0,true);
  add(0,0x3f800001,0x0c000001,0x8c000002,true); // normal operands -> minsub
  add(8,0x00800001,0x00800000,0,true);add(2,1,0,0,true);
  return cases;
}

export class EncoderGPU {
  static async create({exports:e,encoder,gpu=globalThis.navigator?.gpu,onReport=()=>{},timestamps=false}={}) {
    const self=Object.assign(new EncoderGPU(),{e,encoder,onReport,enabled:false,busy:false,resources:[],status:{backend:'cpu',self_test:null}});
    const start=performance.now();
    try {
      if(!gpu) throw Error('WebGPU unavailable');
      const adapter=await gpu.requestAdapter({powerPreference:'high-performance'});if(!adapter) throw Error('No WebGPU adapter');
      self.status.adapter=Object.fromEntries(['vendor','architecture','device','description'].map(k=>[k,adapter.info?.[k]??'']));
      if(adapter.isFallbackAdapter || /swiftshader|llvmpipe|software/i.test(JSON.stringify(self.status.adapter))) throw Error('Software/fallback adapter');
      const features=timestamps&&adapter.features.has('timestamp-query')?['timestamp-query']:[];
      self.device=await adapter.requestDevice({requiredFeatures:features});self.timestamp=features.length>0;
      self.device.lost.then(info=>self.disable(`Device lost: ${info.reason}: ${info.message}`));
      self.device.addEventListener('uncapturederror',event=>self.disable(`WebGPU error: ${event.error.message}`));
      for(const name of ['encoder_gpu_plan_create','encoder_gpu_plan_json','encoder_gpu_math_reference']) if(!e[name]) throw Error(`Missing ABI export ${name}`);
      self.plan=e.encoder_gpu_plan_create(encoder);if(!self.plan) throw Error('GPU plan rejected');
      const pointer=e.encoder_gpu_plan_json(self.plan),length=e.encoder_gpu_plan_json_len(self.plan);
      self.meta=JSON.parse(new TextDecoder().decode(new Uint8Array(e.memory.buffer,pointer,length)));
      self.validateLimits();
      const wp=e.encoder_gpu_plan_weights(self.plan),wn=e.encoder_gpu_plan_weights_len(self.plan);
      const weights=new Uint32Array(e.memory.buffer,wp,wn);
      const flags=specialBits(weights);if(flags) throw Error(`Model weights require CPU: flags ${flags}`);
      self.weights=storage(self.device,weights);self.resources.push(self.weights);
      const compile=performance.now();self.program=await Program.create(self.device);self.status.compile_ms=performance.now()-compile;
      self.status.self_test=await self.selfTest();
      if(!self.status.self_test.passed) throw Error('GPU known-answer self-test failed');
      self.buffers=self.meta.buffers.map(n=>storage(self.device,n*4));self.resources.push(...self.buffers);
      self.guard=storage(self.device,4);self.resources.push(self.guard);
      self.ops=self.meta.ops.map(op=>{const q=uniform(self.device,op.params);self.resources.push(q);return {...op,q,group:self.program.group(op.params[0],[self.buffers[op.input],self.weights,self.buffers[op.residual],self.buffers[op.output],q,self.guard])};});
      self.contextBytes=self.meta.buffers[self.meta.output]*4;
      self.readback=storage(self.device,self.contextBytes+24,GPUBufferUsage.COPY_DST|GPUBufferUsage.MAP_READ);self.resources.push(self.readback);
      if(self.timestamp){self.queries=self.device.createQuerySet({type:'timestamp',count:2});self.resolve=storage(self.device,16,GPUBufferUsage.QUERY_RESOLVE|GPUBufferUsage.COPY_SRC);self.resources.push(self.resolve);}
      if(self.broken) throw Error(self.status.reason);self.enabled=true;self.status.backend='gpu';
    } catch(error) {self.disable(String(error.stack||error));}
    self.status.initialization_ms=performance.now()-start;
    self.onReport({...self.status});return self;
  }
  validateLimits() {
    const m=this.meta,l=this.device.limits;
    if(m.abi!==1||m.buffers.length!==7||m.max_channels>384) throw Error('Unsupported GPU plan ABI');
    if(l.maxStorageBuffersPerShaderStage<5||l.maxComputeInvocationsPerWorkgroup<64||l.maxComputeWorkgroupSizeX<64||l.maxComputeWorkgroupStorageSize<1568) throw Error('Missing default compute limits');
    for(const n of [...m.buffers,m.weights]) if(n*4>l.maxStorageBufferBindingSize||n*4>l.maxBufferSize) throw Error(`Buffer exceeds device limit: ${n*4}`);
    for(const op of m.ops) {const q=op.params;if(q.length!==16||q[15]!==0||q[0]>7||q[3]%4||dispatchCount(q)>l.maxComputeWorkgroupsPerDimension) throw Error(`Unsupported op ${op.name}`);}
  }
  disable(reason) {this.broken=true;this.enabled=false;this.status.backend='cpu';this.status.reason=reason;this.onReport({...this.status});}
  mathReference(records) {
    const e=this.e,ip=e.decoder_alloc(records.byteLength),op=e.decoder_alloc(records.length);
    try {new Uint32Array(e.memory.buffer,ip,records.length).set(records);e.encoder_gpu_math_reference(ip,op,records.length/4);return new Uint32Array(e.memory.buffer,op,records.length/4).slice();}
    finally {e.decoder_free(ip,records.byteLength);e.decoder_free(op,records.length);}
  }
  async probeMath(records,expected=this.mathReference(records),requiredGuards=null) {
    const d=this.device,count=records.length/4,q=new Uint32Array(16);q[14]=count;
    const input=storage(d,records),output=storage(d,count*8),params=uniform(d,q),guard=storage(d,4);
    try {
      const cmd=d.createCommandEncoder();this.program.dispatch(cmd,8,q,this.program.group(8,[input,null,null,output,params,guard]));d.queue.submit([cmd.finish()]);
      const actual=new Uint32Array(await readBuffer(d,output,count*8));let differences=0,guards=0,missingGuards=0,first=null;
      for(let i=0;i<count;i++) {const f=actual[i*2+1];if(f) guards++;
        const required=requiredGuards?.[i]??false;
        if(required){if(!(f&1)){missingGuards++;first??={index:i,reason:'missing subnormal guard'};}}
        else if(f||actual[i*2]!==expected[i]) {differences++;first??={index:i,got:actual[i*2],expected:expected[i],flags:f};}
      }
      return {cases:count,differences,guards,missing_guards:missingGuards,first,passed:!differences&&!missingGuards};
    } finally {for(const b of [input,output,params,guard]) b.destroy();}
  }
  async selfTest() {
    const start=performance.now(),d=this.device,cases=this.meta.self_tests,live=[],rows=[];
    try {
      const cmd=d.createCommandEncoder();let bytesNeeded=0;
      for(const test of cases) {
        const q=test.params,kind=q[0],aa=new Uint32Array(test.a),expected=new Uint32Array(test.expected);
        const input=storage(d,aa),w=storage(d,new Uint32Array(test.weights)),r=storage(d,new Uint32Array(test.r));
        const out=storage(d,inPlace(kind)?aa:expected.byteLength),params=uniform(d,q),guard=storage(d,4);
        live.push(input,w,r,out,params,guard);
        this.program.dispatch(cmd,kind,q,this.program.group(kind,[input,w,r,out,params,guard]));
        rows.push({kind,expected,out,guard,offset:bytesNeeded});bytesNeeded+=expected.byteLength+4;
      }
      const read=storage(d,bytesNeeded,GPUBufferUsage.COPY_DST|GPUBufferUsage.MAP_READ);live.push(read);
      for(const row of rows){cmd.copyBufferToBuffer(row.out,0,read,row.offset,row.expected.byteLength);cmd.copyBufferToBuffer(row.guard,0,read,row.offset+row.expected.byteLength,4);}
      d.queue.submit([cmd.finish()]);await read.mapAsync(GPUMapMode.READ);const mapped=read.getMappedRange();
      const kernels=rows.map(row=>({kind:row.kind,...bitDifference(new Uint32Array(mapped,row.offset,row.expected.length),row.expected),guard:new Uint32Array(mapped,row.offset+row.expected.byteLength,1)[0]}));read.unmap();
      const known=mathKnownAnswers(),records=new Uint32Array(known.flatMap(c=>c.record));
      const math=await this.probeMath(records,this.mathReference(records),known.map(c=>c.guard));
      const ms=performance.now()-start;
      return {passed:math.passed&&kernels.every(k=>!k.bit_differences&&!k.guard),ms,within_100ms:ms<100,kernels,math,
        policy:'Qualifies observed RN-even fused arithmetic; WGSL alone does not promise it. Corpus qualification remains required.'};
    } finally {for(const b of live)b.destroy();}
  }
  cpu(image,context,reason,extra={}) {
    if(this.e.encoder_run(this.encoder,image,context)!==1) throw Error('CPU encoder rejected image');
    return {backend:'cpu',reason,...extra};
  }
  async encode(image,context,{captureRaw=false,observe=null}={}) {
    if(this.busy) throw Error('Serialize encoder calls and WASM memory access');this.busy=true;
    const start=performance.now();
    try {
      if(!this.enabled) return this.cpu(image,context,this.status.reason);
      const input=new Uint32Array(this.e.memory.buffer,image,256*1280),inputFlags=specialBits(input);
      if(inputFlags) return this.cpu(image,context,'Input requires CPU',{guard:inputFlags});
      // Fresh ordinary ArrayBuffer: works even where writeBuffer rejects SAB views.
      this.device.queue.writeBuffer(this.buffers[0],0,input.slice());this.device.queue.writeBuffer(this.guard,0,new Uint32Array(1));
      let command=this.device.createCommandEncoder();
      for(let i=0;i<this.ops.length;i++) {
        const op=this.ops[i],q=op.params;
        const stamp=this.timestamp&&!observe?{querySet:this.queries,...(i===0?{beginningOfPassWriteIndex:0}:{}),...(i===this.ops.length-1?{endOfPassWriteIndex:1}:{})}:null;
        // No empty timestampWrites descriptors on intermediate passes.
        this.program.dispatch(command,q[0],q,op.group,stamp&&(stamp.beginningOfPassWriteIndex!==undefined||stamp.endOfPassWriteIndex!==undefined)?stamp:null);
        if(observe){this.device.queue.submit([command.finish()]);await observe(this,op,i);command=this.device.createCommandEncoder();}
      }
      command.copyBufferToBuffer(this.buffers[this.meta.output],0,this.readback,0,this.contextBytes);
      command.copyBufferToBuffer(this.guard,0,this.readback,this.contextBytes,4);
      if(this.timestamp&&!observe){command.resolveQuerySet(this.queries,0,2,this.resolve,0);command.copyBufferToBuffer(this.resolve,0,this.readback,this.contextBytes+8,16);}
      this.device.queue.submit([command.finish()]);await this.readback.mapAsync(GPUMapMode.READ);
      let raw,guard,gpu_ms;
      try {
        const data=this.readback.getMappedRange();guard=new Uint32Array(data,this.contextBytes,1)[0];
        raw=captureRaw?new Uint32Array(data,0,this.contextBytes/4).slice():undefined;
        gpu_ms=this.timestamp&&!observe?Number(new BigUint64Array(data,this.contextBytes+8,2)[1]-new BigUint64Array(data,this.contextBytes+8,2)[0])/1e6:null;
        if(!guard&&this.enabled) new Uint8Array(this.e.memory.buffer,context,this.contextBytes).set(new Uint8Array(data,0,this.contextBytes));
      } finally {this.readback.unmap();}
      const stats={guard,gpu_ms,wall_ms:performance.now()-start,raw};
      if(guard||!this.enabled) return this.cpu(image,context,guard?'GPU arithmetic guard':this.status.reason,stats);
      return {backend:'gpu',...stats};
    } catch(error) {this.disable(String(error.stack||error));return this.cpu(image,context,this.status.reason,{wall_ms:performance.now()-start});}
    finally {this.busy=false;}
  }
  // Diagnostic timing control only; never writes a context or affects fallback.
  async measureGuardCost(image,{trials=3}={}) {
    if(!this.enabled||!this.timestamp||this.busy) return {available:false,reason:'Need idle qualified GPU with timestamp-query'};
    this.busy=true;
    try {
      const unguarded=await Program.create(this.device,true);
      const groups=this.ops.map(op=>unguarded.group(op.params[0],[this.buffers[op.input],this.weights,this.buffers[op.residual],this.buffers[op.output],op.q,this.guard]));
      const rows=[];this.device.queue.writeBuffer(this.buffers[0],0,new Uint32Array(this.e.memory.buffer,image,256*1280).slice());
      for(let trial=-1;trial<trials;trial++)for(const enabled of (trial%2?[false,true]:[true,false])){
        const program=enabled?this.program:unguarded,cmd=this.device.createCommandEncoder();
        this.device.queue.writeBuffer(this.guard,0,new Uint32Array(1));
        for(let i=0;i<this.ops.length;i++){const op=this.ops[i],timestampWrites=i===0?{querySet:this.queries,beginningOfPassWriteIndex:0}:i===this.ops.length-1?{querySet:this.queries,endOfPassWriteIndex:1}:null;
          program.dispatch(cmd,op.params[0],op.params,enabled?op.group:groups[i],timestampWrites);}
        cmd.resolveQuerySet(this.queries,0,2,this.resolve,0);this.device.queue.submit([cmd.finish()]);
        const times=new BigUint64Array(await readBuffer(this.device,this.resolve,16));
        const flags=new Uint32Array(await readBuffer(this.device,this.guard,4))[0];
        if(trial>=0)rows.push({trial,guarded:enabled,gpu_ms:Number(times[1]-times[0])/1e6,flags});
      }
      const median=xs=>xs.toSorted((a,b)=>a-b)[Math.floor(xs.length/2)];
      const on=median(rows.filter(r=>r.guarded).map(r=>r.gpu_ms)),off=median(rows.filter(r=>!r.guarded).map(r=>r.gpu_ms));
      return {available:true,rows,guarded_gpu_ms:on,unguarded_gpu_ms:off,overhead_percent:(on/off-1)*100,policy:'Unguarded control is never used as an encoder result.'};
    } finally {this.busy=false;}
  }
  destroy() {
    if(this.busy) throw Error('Wait for encode before destroy');
    this.enabled=false;for(const b of this.resources)b.destroy();this.resources=[];this.queries?.destroy();this.device?.destroy();
    if(this.plan){this.e.encoder_gpu_plan_drop(this.plan);this.plan=0;}
  }
}
