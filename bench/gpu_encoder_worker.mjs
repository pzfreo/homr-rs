import {runGPUEncoder} from './gpu_encoder_run.mjs';
self.onmessage=async({data})=>{
  try{self.postMessage({result:await runGPUEncoder(data,progress=>self.postMessage({progress}))});}
  catch(error){self.postMessage({result:{error:`${error}\n${error.stack||""}`,complete:false}});}
};
