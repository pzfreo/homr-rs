"""Merge disjoint encoder corpus workers, rejecting missing/duplicate pages.

merge_encoder_runs.py CORPUS_TXT OUT WORKER_OUT...
Links scratch per-page artifacts; no arithmetic is substituted or skipped.
"""
import json
from pathlib import Path
import sys

corpus,out,*workers=map(Path,sys.argv[1:])
expected=[Path(p).stem for p in corpus.read_text().splitlines()]
pages={};artifacts=set()
for worker in workers:
    report=json.loads((worker/'comparison.json').read_text())
    artifacts.add(report['wasm'])
    for row in report['pages']:
        name=row['page']
        if name in pages:raise RuntimeError(f'duplicate page {name}')
        pages[name]=(row,worker/name)
if set(pages)!=set(expected):raise RuntimeError('missing or unexpected pages')
if len(artifacts)!=1:raise RuntimeError('workers used different named artifacts')
out.mkdir(parents=True,exist_ok=True)
for name in expected:
    target=out/name
    if target.exists() or target.is_symlink():raise RuntimeError(f'output already exists: {target}')
    target.symlink_to(pages[name][1].resolve(),target_is_directory=True)
rows=[pages[name][0] for name in expected];staffs=[s for p in rows for s in p['encoder']]
summary={'pages':len(rows),'staffs':len(staffs),'differing_staffs':sum(s['bit_differences']!=0 for s in staffs),
         **{k:sum(s.get(k,0) for s in staffs) for k in ['steps','differing_steps','differing_tokens','extra_steps','missing_steps']}}
(out/'comparison.json').write_text(json.dumps({'wasm':artifacts.pop(),'summary':summary,'pages':rows},indent=2))
print(summary)
