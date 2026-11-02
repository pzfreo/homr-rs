"""/usr/bin/time -l python bench/decoder_native.py load|decode MODEL REF STAFF ...
Same fixed step counts, cache/context quirk, single-threaded native session (recording uses defaults).
"""
import atexit
import os
import json
from pathlib import Path
import sys
import numpy as np
import onnxruntime as ort
def wait():
    if os.environ.get("BENCH_WAIT"):
        print("BENCH_READY", flush=True)
        sys.stdin.readline()
atexit.register(wait)
mode, model, root, *staffs = sys.argv[1:]
data = [(np.fromfile(Path(root)/s/"context.bin", "<f4").reshape(1,1280,512),
         len(json.loads((Path(root)/s/"steps.json").read_text()))) for s in staffs]
options = ort.SessionOptions()
options.intra_op_num_threads = 1
options.inter_op_num_threads = 1
session = ort.InferenceSession(model, options)
if mode == "load":
    sys.exit(0)
assert mode == "decode"
heads = ["out_rhythms","out_pitchs","out_lifts","out_positions","out_articulations","out_slurs"]
names = heads+["attention"]+[f"cache_out{i}" for i in range(32)]
for context, steps in data:
    cache = [np.zeros((1,8,0,64),np.float32) for _ in range(32)]
    last = [1,0,0,0,0]
    for step in range(steps):
        feeds = dict(zip(["rhythms","pitchs","lifts","articulations","slurs"],
                         [np.array([[v]],np.int64) for v in last]))
        feeds.update(context=context if step==0 else context[:,:1],cache_len=np.array([step],np.int64))
        feeds.update({f"cache_in{i}":v for i,v in enumerate(cache)})
        outputs=session.run(names, feeds)
        tokens=[int(v.reshape(-1).argmax()) for v in outputs[:6]]
        cache=outputs[7:]
        last=[3 if tokens[0]==2 else tokens[0],tokens[1],tokens[2],tokens[4],tokens[5]]
print(sum(n for _,n in data),"steps")
