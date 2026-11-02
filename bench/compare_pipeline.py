"""Compare the existing WASI page/pre benchmarks in two checkouts, three paired 3-vs-1 runs.
Usage: COUNTER_ADDON=/scratch/counter.node python bench/compare_pipeline.py MAIN HEAD SCRATCH OUTPUT
"""
import json,os,pathlib,statistics,subprocess,sys
main,head,scratch,out=map(pathlib.Path,sys.argv[1:]);main=main.resolve();head=head.resolve();scratch=scratch.resolve()
pages=['angelus_0','angelus_1','angelus_2','angelus_3','angelus_4','angelus_5']
result={}
for kind in ['page','pre']:
 runs={name:[] for name in ['main','head']}
 for rep in range(3):
  for name,root in [('main',main),('head',head)]:
   totals=[]
   for reps in [1,3]:
    if kind=='page':args=[str(scratch/'replay_all'),str(reps)]+[str(scratch/'p411b'/p)+':'+str(scratch/'dec_all'/p) for p in pages]
    else:args=[str(reps)]+[str(scratch/'seg_all'/p) for p in pages]
    run=subprocess.run(['node','--no-warnings',str(head/'bench/count_process.mjs'),str(root/f'bench/rs_{kind}.mjs')]+args,capture_output=True,text=True,check=True)
    n=int([l.split()[1] for l in run.stdout.splitlines() if l.startswith('RETIRED ')][0]);totals.append(n)
   runs[name].append((totals[1]-totals[0])/2);print(kind,rep,name,runs[name][-1],flush=True)
 result[kind]={'runs':runs,'main_median':statistics.median(runs['main']),'head_median':statistics.median(runs['head'])}
 result[kind]['ratio']=result[kind]['head_median']/result[kind]['main_median']
report={'pages':pages,'results':result};out.write_text(json.dumps(report,indent=2));print(json.dumps(report,indent=2))
