// COUNTER_ADDON=... node bench/float_kernel.mjs MODULE A.f32 B.f32 NATIVE.f32 OUTPUT
// A: [2,K], B: [K,16], reference: [2,16], from identical-input MatMul probe.
import fs from 'node:fs';
import path from 'node:path';
import {createRequire} from 'node:module';
const [module, aFile, bFile, reference, output] = process.argv.slice(2);
const {instance:{exports:e}} = await WebAssembly.instantiate(fs.readFileSync(module), {});
const counter = createRequire(import.meta.url)(path.resolve(process.env.COUNTER_ADDON));
const read = file => {const bytes=fs.readFileSync(file);return new Float32Array(bytes.buffer.slice(bytes.byteOffset,bytes.byteOffset+bytes.byteLength));};
const a=read(aFile), b=read(bFile), ref=fs.readFileSync(reference), k=a.length/2;
const columns=e.columns();
if(b.length!==k*16||ref.length!==32*4)throw Error('probe dimensions');
const ap=e.alloc(a.length), bp=e.alloc(b.length), op=e.alloc(32);
new Float32Array(e.memory.buffer,ap,a.length).set(a);new Float32Array(e.memory.buffer,bp,b.length).set(b);
e.kernel(ap,bp,op,k,1);
const expected=Buffer.concat([ref.subarray(0,columns*4),ref.subarray(64,64+columns*4)]);
if(!Buffer.from(e.memory.buffer,op,columns*8).equals(expected))throw Error('probe does not match native FMA outputs');
const reps=100000, counts=[];
for(let i=0;i<3;i++){const start=counter.instructions();e.kernel(ap,bp,op,k,reps);counts.push(counter.instructions()-start);}
const median=counts.toSorted((a,b)=>a-b)[1];
const result={module,k,columns,reps,macs:2*columns*k*reps,instructions:counts,median,instructions_per_mac:median/(2*columns*k*reps)};
fs.writeFileSync(output,JSON.stringify(result,null,2));console.log(result);
