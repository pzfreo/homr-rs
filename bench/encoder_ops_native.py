"""Warmed, isolated native-ORT operator instruction samples (one thread).

encoder_ops_native.py PROBES OUTPUT_JSON; use pinned ORT 1.31.0 + numpy.
These support the whole-model totals; they are not a full graph profile.
"""
import json
from pathlib import Path
import statistics
import sys
import numpy as np
import onnxruntime as ort

sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'tools'))
from instr import instructions
root,out=map(Path,sys.argv[1:]);rows=[]
for p in json.loads((root/'probes.json').read_text())['probes']:
    options=ort.SessionOptions();options.intra_op_num_threads=1;options.inter_op_num_threads=1
    session=ort.InferenceSession(str(root/p['model']),sess_options=options)
    feed={t['name']:np.fromfile(root/t['file'],dtype=t['type']).reshape(t['shape']) for t in p['inputs']}
    warm=session.run(None,feed);counts=[]
    for t,v in zip(p['outputs'],warm):
        assert v.tobytes()==(root/t['file']).read_bytes()
    for _ in range(3):
        before=instructions();session.run(None,feed);counts.append(instructions()-before)
    rows.append({'kind':p['kind'],'instructions':counts,'median_instructions':statistics.median(counts)})
out.write_text(json.dumps(rows,indent=2));print(rows)
