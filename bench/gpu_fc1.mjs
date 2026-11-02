// Browser WebGPU prototype: ascending-K fused accumulation, then separate bias.
// No fp16, dot(), subgroup reduction or reordered K tiles. Self-tests load
// operands from storage to prevent constant folding. A passing test is only a
// gate: validating the whole encoder/corpus on each engine is still required.
const read = async url => {const r = await fetch(url); if (!r.ok) throw Error(`${url}: ${r.status}`); return r.arrayBuffer();};
const median = xs => xs.toSorted((a, b) => a - b)[1];
function shader(m, k, n) {
  return `
@group(0) @binding(0) var<storage, read> a: array<f32>;
@group(0) @binding(1) var<storage, read> b: array<vec4<f32>>;
@group(0) @binding(2) var<storage, read> bias: array<vec4<f32>>;
@group(0) @binding(3) var<storage, read_write> result: array<vec4<f32>>;
@compute @workgroup_size(64) fn main(@builtin(global_invocation_id) id: vec3<u32>) {
  let i = id.x; if (i >= ${m * n / 4}u) { return; }
  let row = i / ${n / 4}u; let col = i % ${n / 4}u;
  var sum = vec4<f32>(0.0);
  for (var p = 0u; p < ${k}u; p++) {
    sum = fma(vec4<f32>(a[row * ${k}u + p]), b[p * ${n / 4}u + col], sum);
  }
  result[i] = sum + bias[col];
}`;
}
async function pipeline(device, code) {
  const module = device.createShaderModule({code});
  const info = await module.getCompilationInfo();
  if (info.messages.some(m => m.type === 'error')) throw Error(JSON.stringify(info.messages));
  return device.createComputePipelineAsync({layout: 'auto', compute: {module, entryPoint: 'main'}});
}
function buffer(device, bytes, usage) {
  const b = device.createBuffer({size: Math.max(4, bytes.byteLength), usage, mappedAtCreation: true});
  new Uint8Array(b.getMappedRange()).set(new Uint8Array(bytes.buffer || bytes, bytes.byteOffset || 0, bytes.byteLength));
  b.unmap(); return b;
}
async function download(device, source, bytes) {
  const target = device.createBuffer({size: bytes, usage: GPUBufferUsage.COPY_DST | GPUBufferUsage.MAP_READ});
  const command = device.createCommandEncoder(); command.copyBufferToBuffer(source, 0, target, 0, bytes);
  device.queue.submit([command.finish()]); await target.mapAsync(GPUMapMode.READ);
  const result = target.getMappedRange().slice(0); target.unmap(); target.destroy(); return result;
}
function compare(got, reference) {
  const g = new Uint32Array(got), r = new Uint32Array(reference);
  if (g.length !== r.length) throw Error('output shape mismatch');
  const gf = new Float32Array(got), rf = new Float32Array(reference);
  let differing = 0, max = 0, sum = 0, first = null;
  for (let i = 0; i < g.length; i++) {
    if (g[i] !== r[i]) {differing++; first ??= {index: i, got: g[i], reference: r[i]};}
    const error = Math.abs(gf[i] - rf[i]); if (!Number.isFinite(error)) throw Error('nonfinite result');
    max = Math.max(max, error); sum += error;
  }
  return {values: g.length, bit_differences: differing, max_abs: max, mean_abs: sum / g.length, first};
}
async function makeMatmul(device, m, k, n, x, weights, bias) {
  const p = await pipeline(device, shader(m, k, n));
  const inputs = [x, weights, bias].map(v => buffer(device, v, GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST));
  const output = device.createBuffer({size: m * n * 4, usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_SRC});
  const group = device.createBindGroup({layout: p.getBindGroupLayout(0), entries: [...inputs, output].map((b, binding) => ({binding, resource: {buffer: b}}))});
  return {output, inputs, encode(command, timestampWrites) {
    const pass = command.beginComputePass(timestampWrites ? {timestampWrites} : {});
    pass.setPipeline(p); pass.setBindGroup(0, group); pass.dispatchWorkgroups(Math.ceil(m * n / 4 / 64)); pass.end();
  }, destroy() {inputs.forEach(b => b.destroy()); output.destroy();}};
}

export async function selfTest(device) {
  const triples = new Uint32Array([
    0x3f800001, 0x3f7ffffe, 0xbf800000, 0,
    0x3f800001, 0x3f7ffffe, 0x4b800001, 0,
    1, 0x3f000000, 1, 0,
    0x80000000, 0x3f800000, 0x80000000, 0,
    0x4b800000, 0x3f800000, 0x3f800000, 0,
    0x3f000000, 0x3f000000, 0xbe800000, 0,
  ]);
  const expected = [0xa8800000, 0x4b800001, 2, 0x80000000, 0x4b800000, 0];
  const input = buffer(device, triples, GPUBufferUsage.STORAGE);
  const output = device.createBuffer({size: expected.length * 4, usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_SRC});
  const p = await pipeline(device, `
@group(0) @binding(0) var<storage, read> triples: array<vec4<f32>>;
@group(0) @binding(1) var<storage, read_write> result: array<u32>;
@compute @workgroup_size(1) fn main(@builtin(global_invocation_id) id: vec3<u32>) {
 let v = triples[id.x]; result[id.x] = bitcast<u32>(fma(v.x, v.y, v.z));
}`);
  const group = device.createBindGroup({layout: p.getBindGroupLayout(0), entries: [input, output].map((b, binding) => ({binding, resource: {buffer: b}}))});
  const command = device.createCommandEncoder(), pass = command.beginComputePass();
  pass.setPipeline(p); pass.setBindGroup(0, group); pass.dispatchWorkgroups(expected.length); pass.end(); device.queue.submit([command.finish()]);
  const actual = Array.from(new Uint32Array(await download(device, output, expected.length * 4)));
  input.destroy(); output.destroy();
  // Same shader generator as production fc1. These selected cells distinguish
  // native K order, fused rounding, bias-after scheduling, and denormals.
  const x = new Float32Array(4 * 96), b = new Float32Array(96 * 16), bias = new Float32Array(16);
  x[0] = 1 + 2 ** -23; b[0] = 1 - 2 ** -23; bias[0] = 16777218;
  x[96] = -1; x[97] = 1 + 2 ** -23; b[1] = 1; b[16 + 1] = 1 - 2 ** -23;
  x[192] = 2 ** 24; x[193] = 1; x[194] = -(2 ** 24); b[2] = b[16 + 2] = b[32 + 2] = 1;
  x[288] = 2 ** -149; b[3] = 0.5; bias[3] = 2 ** -149;
  const mm = await makeMatmul(device, 4, 96, 16, x, b, bias);
  const cmd = device.createCommandEncoder(); mm.encode(cmd); device.queue.submit([cmd.finish()]);
  const matrix = new Uint32Array(await download(device, mm.output, 4 * 16 * 4)); mm.destroy();
  const cells = [[0, 0x4b800002], [17, 0xa8800000], [34, 0], [51, 1]];
  return {passed: expected.every((v, i) => actual[i] === v) && cells.every(([i, v]) => matrix[i] === v),
    fma_expected: expected, fma_actual: actual, schedule_cells: cells.map(([i, expected]) => ({index: i, expected, actual: matrix[i]})),
    policy: 'Require every known answer; reject on device loss. This is a conservative gate, not proof of the whole model.'};
}

export async function runFc1(root = '/fixture') {
  const adapter = await navigator.gpu?.requestAdapter({powerPreference: 'high-performance'});
  if (!adapter) throw Error('No WebGPU adapter; refusing CPU fallback');
  const adapterInfo = Object.fromEntries(['vendor', 'architecture', 'device', 'description'].map(k => [k, adapter.info[k]]));
  if (/swiftshader|llvmpipe|software/i.test(JSON.stringify(adapterInfo))) throw Error('Software adapter is not Apple Metal');
  if (!/apple/i.test(JSON.stringify(adapterInfo))) throw Error(`Expected Apple GPU on Metal, got ${JSON.stringify(adapterInfo)}`);
  const timestamp = adapter.features.has('timestamp-query');
  const device = await adapter.requestDevice({requiredFeatures: timestamp ? ['timestamp-query'] : []});
  let lost; device.lost.then(value => {lost = value;});
  const self_test = await selfTest(device);
  const meta = await (await fetch(root + '/fc1.json')).json();
  const [x, b, bias, native, rust] = await Promise.all(['input.f32', 'weights.f32', 'bias.f32', 'native.f32', 'rust.f32'].map(n => read(root + '/' + n)));
  const mm = await makeMatmul(device, meta.m, meta.k, meta.n, x, b, bias);
  const querySet = timestamp ? device.createQuerySet({type: 'timestamp', count: 2}) : null;
  const queryResolve = timestamp ? device.createBuffer({size: 16, usage: GPUBufferUsage.QUERY_RESOLVE | GPUBufferUsage.COPY_SRC}) : null;
  const queries = timestamp ? device.createBuffer({size: 16, usage: GPUBufferUsage.COPY_DST | GPUBufferUsage.MAP_READ}) : null;
  const rows = [];
  for (let trial = -1; trial < 3; trial++) {
    // Model weights remain resident. Include this staff's input upload in the
    // dispatch/wait wall measurement; GPU timestamp is compute-pass only.
    const start = performance.now(); device.queue.writeBuffer(mm.inputs[0], 0, x);
    const command = device.createCommandEncoder();
    mm.encode(command, timestamp ? {querySet, beginningOfPassWriteIndex: 0, endOfPassWriteIndex: 1} : null);
    if (timestamp) {command.resolveQuerySet(querySet, 0, 2, queryResolve, 0); command.copyBufferToBuffer(queryResolve, 0, queries, 0, 16);}
    device.queue.submit([command.finish()]); await device.queue.onSubmittedWorkDone();
    const dispatch_wait_ms = performance.now() - start;
    let gpu_ms = null;
    if (timestamp) {await queries.mapAsync(GPUMapMode.READ); const t = new BigUint64Array(queries.getMappedRange()); gpu_ms = Number(t[1] - t[0]) / 1e6; queries.unmap();}
    const readStart = performance.now(), got = await download(device, mm.output, meta.m * meta.n * 4), readback_ms = performance.now() - readStart;
    const native_diff = compare(got, native), rust_diff = compare(got, rust);
    if (lost) throw Error(`Device lost: ${lost.message}`);
    if (trial >= 0) rows.push({trial, dispatch_wait_ms, gpu_ms, readback_ms, native: native_diff, rust: rust_diff});
  }
  const report = {adapterInfo, timestamp_query: timestamp, userAgent: navigator.userAgent, meta, self_test, rows,
    median_dispatch_wait_ms: median(rows.map(r => r.dispatch_wait_ms)),
    median_gpu_ms: timestamp ? median(rows.map(r => r.gpu_ms)) : null,
    median_readback_ms: median(rows.map(r => r.readback_ms)),
    exact: self_test.passed && rows.every(r => !r.native.bit_differences && !r.rust.bit_differences)};
  mm.destroy(); queries?.destroy(); queryResolve?.destroy(); querySet?.destroy(); device.destroy();
  return report;
}
