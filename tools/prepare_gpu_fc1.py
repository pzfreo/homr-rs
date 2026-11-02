"""Prepare scratch fc1 GPU operands and independent native/Rust references.

prepare_gpu_fc1.py OPTIMIZED_PROBE_DIR RUST_GEMM.f32 OUT
Generate RUST_GEMM.f32 with PROBE_OUTPUT=... check_encoder_ops first.
Requires numpy and onnx. Does not run or modify homr.
"""
import json
from pathlib import Path
import sys

import numpy as np
import onnx
from onnx import helper, numpy_helper

root, rust, out = map(Path, sys.argv[1:])
out.mkdir(parents=True, exist_ok=True)
probe = next(p for p in json.loads((root / 'probes.json').read_text())['probes'] if p['kind'] == 'Gemm')
model = onnx.load(root / probe['model'])
node = model.graph.node[0]
attrs = {a.name: helper.get_attribute_value(a) for a in node.attribute}
assert node.op_type == 'Gemm'
assert attrs.get('alpha', 1) == 1 and attrs.get('beta', 1) == 1
assert attrs.get('transA', 0) == 0 and attrs.get('transB', 0) == 0
weights = {t.name: numpy_helper.to_array(t) for t in model.graph.initializer}
m, k = probe['inputs'][0]['shape']
assert weights[node.input[1]].shape[0] == k
n = weights[node.input[1]].shape[1]
assert n % 4 == 0
native = (root / probe['outputs'][0]['file']).read_bytes()
assert native == rust.read_bytes(), 'Rust output must match native before GPU testing'
(out / 'input.f32').write_bytes((root / probe['inputs'][0]['file']).read_bytes())
weights[node.input[1]].astype('<f4').tofile(out / 'weights.f32')
weights[node.input[2]].astype('<f4').tofile(out / 'bias.f32')
(out / 'native.f32').write_bytes(native)
(out / 'rust.f32').write_bytes(rust.read_bytes())
(out / 'fc1.json').write_text(json.dumps({'m': m, 'k': k, 'n': n, 'native_rust_bit_differences': 0}, indent=2))
print(m, k, n, 'native/Rust references identical')
