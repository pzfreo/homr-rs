// Retired-instruction wrapper for existing Node benchmark scripts.
// COUNTER_ADDON=/scratch/counter.node node --no-warnings bench/count_process.mjs SCRIPT ARGS...
import {createRequire} from 'node:module';
import {pathToFileURL} from 'node:url';
import path from 'node:path';
const require=createRequire(import.meta.url);
const counter=require(path.resolve(process.env.COUNTER_ADDON));
const [script,...args]=process.argv.slice(2);
process.argv=[process.argv[0],path.resolve(script),...args];
const start=counter.instructions();
await import(pathToFileURL(path.resolve(script)));
console.log('RETIRED',counter.instructions()-start);
