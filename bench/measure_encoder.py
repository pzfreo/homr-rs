"""Sequential warmed encoder instruction baselines; use only outside CI.

measure_encoder.py MODEL NATIVE_STAFF OUT --ort-module MODULE --counter ADDON
 --native-python PYTHON --fused WASM --strict WASM
 [--fused-profile WASM --strict-profile WASM --probes ROOT ...]
Models load before counters, each engine warms once and measures three runs.
Threaded ORT's process counter includes its workers. Never uses wall time.
"""
import argparse
import json
import os
from pathlib import Path
import subprocess
import sys

p=argparse.ArgumentParser(description=__doc__)
for name in ['model','staff','output']:p.add_argument(name)
for name in ['ort-module','counter','native-python','fused','strict']:p.add_argument('--'+name,required=True)
for name in ['fused-profile','strict-profile']:p.add_argument('--'+name)
p.add_argument('--probes',action='append',default=[])
p.add_argument('--native-bin',default='target/release/encoder_bench')
a=p.parse_args();out=Path(a.output);out.mkdir(parents=True,exist_ok=True)
env=dict(os.environ,ORT_MODULE=str(Path(a.ort_module).resolve()),COUNTER_ADDON=str(Path(a.counter).resolve()))
def run(name,cmd,extra=None):
    print(name,flush=True)
    with (out/(name+'.log')).open('w') as log:
        subprocess.run(cmd,env=dict(env,**(extra or {})),stdout=log,stderr=subprocess.STDOUT,check=True)
    return str(out/(name+'.json'))
reports={}
reports['rust_native']=run('rust_native',[a.native_bin,a.model,a.staff,str(out/'rust_native.json')])
for kind,wasm in [('rust_fused',a.fused),('rust_strict',a.strict),('rust_fused_profile',a.fused_profile),('rust_strict_profile',a.strict_profile)]:
    if wasm:reports[kind]=run(kind,['node','bench/encoder_bench.mjs',wasm,a.model,a.staff,str(out/(kind+'.json'))])
for threads in [1,4]:
    name=f'ort_wasm_{threads}t'
    reports[name]=run(name,['node','bench/model_instructions.mjs',a.model,str(Path(a.staff)/'input.f32'),'1,1,256,1280',str(out/(name+'.json')),str(threads)])
staff=Path(a.staff)
reports['ort_native']=run('ort_native',[a.native_python,'bench/models_native.py',a.model,str(staff.parent.parent),str(out/'ort_native.json'),str(Path(staff.parent.name)/staff.name)],{'NATIVE_THREADS':'1'})
for index,root in enumerate(a.probes):
    run(f'ops_native_{index}',[a.native_python,'bench/encoder_ops_native.py',root,str(out/f'ops_native_{index}.json')])
    for threads in [1,4]:
        run(f'ops_wasm_{threads}t_{index}',['node','bench/model_probes.mjs',root,str(out/f'ops_wasm_{threads}t_{index}.json')],{'ORT_THREADS':str(threads)})
(out/'reports.json').write_text(json.dumps(reports,indent=2))
print(json.dumps(reports),flush=True)
