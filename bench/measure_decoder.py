"""Paired retired-instruction benchmarks (never CI).

Normally use /usr/bin/time -l on commands printed by --commands. In a sandbox
where time's kern.clockrate query is denied, --libproc reads the same macOS
RUSAGE_INFO_V4 retired-instruction counter while the child waits at BENCH_READY.
No wall-clock substitute. Each mode includes identical input/model loading;
subtract the load-only count. Use a pinned reference Python environment.
"""
import argparse
import ctypes
import json
import os
from pathlib import Path
import statistics
import subprocess
import sys
sys.path.insert(0, str(Path(__file__).resolve().parents[1]/"tools"))
from instr import _lib, _RU
p=argparse.ArgumentParser()
p.add_argument('model');p.add_argument('references');p.add_argument('staffs',nargs='+')
p.add_argument('--python',default=sys.executable);p.add_argument('--ort-module',required=True)
p.add_argument('--reps',type=int,default=3);p.add_argument('--libproc',action='store_true')
p.add_argument('--output');p.add_argument('--commands',action='store_true')
p.add_argument('--wasm',help='Explicit Rust WASM artifact (recorded in the report)')
p.add_argument('--native-bin',default='target/release/bench_decoder')
a=p.parse_args()
commands={
 'Rust native':[a.native_bin],
 'ORT native':[a.python,'bench/decoder_native.py'],
 'Rust WASM':['node','bench/decoder.mjs','rust'],
 'ORT WASM':['node','bench/decoder.mjs','ort']}
env=dict(os.environ,ORT_MODULE=str(Path(a.ort_module).resolve()))
if a.wasm:env["DECODER_WASM"]=str(Path(a.wasm).resolve())
counts={name:{'load':[],'decode':[]} for name in commands}
for rep in range(a.reps):
 for name,base in commands.items():
  for mode in ['load','decode']:
   cmd=base+[mode,a.model,a.references]+a.staffs
   if a.commands:
    import shlex
    print('/usr/bin/time -l '+shlex.join(cmd));continue
   if a.libproc:
    proc=subprocess.Popen(cmd,env=dict(env,BENCH_WAIT='1'),stdin=subprocess.PIPE,stdout=subprocess.PIPE,stderr=subprocess.PIPE,text=True)
    while True:
     line=proc.stdout.readline()
     if line.strip()=='BENCH_READY':break
     if not line: raise RuntimeError((cmd,proc.wait(),proc.stderr.read()))
    usage=_RU();rc=_lib.proc_pid_rusage(proc.pid,4,ctypes.byref(usage))
    if rc:raise RuntimeError('proc_pid_rusage denied')
    n=usage.instructions
    proc.communicate('\n');assert proc.returncode==0
   else:
    run=subprocess.run(['/usr/bin/time','-l']+cmd,env=env,capture_output=True,text=True,check=True)
    matches=[int(line.strip().split()[0]) for line in run.stderr.splitlines() if 'instructions retired' in line]
    if len(matches)!=1:raise RuntimeError(run.stderr)
    n=matches[0]
   counts[name][mode].append(n);print(rep,name,mode,n,flush=True)
if not a.commands:
 steps=sum(len(json.loads((Path(a.references)/s/'steps.json').read_text())) for s in a.staffs)
 summary={name:{'load':r['load'],'decode':r['decode'],
 'net_instructions':statistics.median(d-l for d,l in zip(r['decode'],r['load'])),
 'instructions_per_staff':statistics.median(d-l for d,l in zip(r['decode'],r['load']))/len(a.staffs),
 'instructions_per_step':statistics.median(d-l for d,l in zip(r['decode'],r['load']))/steps} for name,r in counts.items()}
 report={'counter':'libproc RUSAGE_INFO_V4' if a.libproc else '/usr/bin/time -l','staffs':a.staffs,'steps':steps,'native_artifact':a.native_bin,'wasm_artifact':env.get('DECODER_WASM','target/wasm32-unknown-unknown/release/homr_rs.wasm'),'results':summary}
 print(json.dumps(report,indent=2))
 if a.output:Path(a.output).write_text(json.dumps(report,indent=2))
