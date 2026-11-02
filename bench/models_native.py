"""Native encoder verification and retired-instruction samples.

models_native.py MODEL NATIVE_FIXTURES OUTPUT_JSON PAGE/STAFF...
For segnet, pass PAGE/seg-N instead of PAGE/STAFF.
NATIVE_THREADS optionally sets intra/inter op thread counts for an experiment;
unset preserves homr defaults. Warm once, report three runs' median and check
every result against the reused native context. No wall-clock measurements.
"""
import json
import os
from pathlib import Path
import statistics
import sys

import numpy as np
import onnxruntime as ort

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'tools'))
from instr import instructions

model, native, output, *staffs = sys.argv[1:]
options = ort.SessionOptions()
if 'NATIVE_THREADS' in os.environ:
    options.intra_op_num_threads = int(os.environ['NATIVE_THREADS'])
    options.inter_op_num_threads = int(os.environ['NATIVE_THREADS'])
session = ort.InferenceSession(model, sess_options=options)
rows = []
for staff in staffs:
    d = Path(native) / staff
    segmentation = d.name.startswith('seg-')
    if segmentation:
        index = int(d.name.split('-')[1])
        shape = json.loads((d.parent / 'models.json').read_text())['batches'][index]['shape']
        x = np.fromfile(str(d) + '.input.u8', dtype=np.uint8).astype(np.float32).reshape(shape)
        ref = np.fromfile(str(d) + '.native.u8', dtype=np.uint8)
    else:
        shape = json.loads((d / 'input.json').read_text())['shape']
        x = np.fromfile(d / 'input.f32', dtype='<f4').reshape(shape)
        ref = np.fromfile(d / 'context.bin', dtype='<f4')
    session.run(None, {session.get_inputs()[0].name: x})
    counts = []
    for _ in range(3):
        before = instructions()
        result = session.run(None, {session.get_inputs()[0].name: x})[0]
        counts.append(instructions() - before)
        result = result.argmax(axis=1).astype(np.uint8).reshape(-1) if segmentation else result.reshape(-1)
        error = np.abs(result.astype(np.float64) - ref)
        rows.append({'staff': staff, 'instructions': counts[-1], 'max_abs': float(error.max()),
                     'mean_abs': float(error.mean()), 'different': int(np.count_nonzero(error))})
    print(staff, 'median instructions', statistics.median(counts), flush=True)
report = {'ort': ort.__version__, 'numpy': np.__version__, 'threads': os.getenv('NATIVE_THREADS', 'default'), 'runs': rows}
Path(output).write_text(json.dumps(report, indent=2))
