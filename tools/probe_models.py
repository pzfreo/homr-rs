"""Isolate arithmetic with identical native tensors at operator boundaries.

probe_models.py MODEL NATIVE_INPUT SHAPE_COMMA OUT
NATIVE_INPUT is input.f32 (encoder) or seg-0.input.u8 (segnet).
Diagnostic only: exposed intermediates can prevent production graph fusions.
Native default CPU session options are retained. Writes scratch data only.
"""
import json
from pathlib import Path
import sys

import numpy as np
import onnx
from onnx import helper, numpy_helper
import onnxruntime as ort

source, data, shape, dest = sys.argv[1:]
out = Path(dest)
out.mkdir(parents=True, exist_ok=True)
shape = [int(d) for d in shape.split(',')]
x = np.fromfile(data, dtype=np.uint8 if data.endswith('.u8') else '<f4').astype(np.float32).reshape(shape)
model = onnx.load(source)
for d, v in zip(model.graph.input[0].type.tensor_type.shape.dim, shape):
    d.dim_value = v
model = onnx.shape_inference.infer_shapes(model, data_prop=True)
initializers = {t.name: t for t in model.graph.initializer}
values = {v.name: v for v in [*model.graph.input, *model.graph.output, *model.graph.value_info]}
selected, seen = [], set()
for node in model.graph.node:
    kind = node.op_type
    if kind == 'Conv' and any(a.name == 'group' and a.i > 1 for a in node.attribute):
        kind = 'DepthwiseConv'
    if kind in {'Conv', 'DepthwiseConv', 'MatMul', 'LayerNormalization', 'Erf', 'Resize', 'Div', 'Sub'} and kind not in seen:
        selected.append((kind, node))
        seen.add(kind)
names = set(s for _, node in selected for s in [*node.input, *node.output] if s and s not in initializers)
existing = {v.name for v in model.graph.output}
for name in names - existing:
    model.graph.output.append(values[name])
session = ort.InferenceSession(model.SerializeToString())
actual = dict(zip(names, session.run(list(names), {model.graph.input[0].name: x})))
actual.update({k: numpy_helper.to_array(t) for k, t in initializers.items()})
probes = []
for index, (kind, node) in enumerate(selected):
    p = out / str(index)
    p.mkdir(exist_ok=True)
    inputs = [name for name in node.input if name and name not in initializers]
    weights = [initializers[name] for name in node.input if name in initializers]
    graph = helper.make_graph([node], 'isolated', [values[n] for n in inputs], [values[n] for n in node.output], weights)
    isolated = helper.make_model(graph, opset_imports=model.opset_import)
    isolated.ir_version = model.ir_version
    (p / 'model.onnx').write_bytes(isolated.SerializeToString())
    run = ort.InferenceSession(isolated.SerializeToString())
    feeds = {name: actual[name] for name in inputs}
    native = run.run(None, feeds)
    tensors = []
    for i, (name, arr) in enumerate(feeds.items()):
        arr.tofile(p / f'input-{i}.bin')
        tensors.append({'name': name, 'shape': list(arr.shape), 'type': str(arr.dtype), 'file': f'{index}/input-{i}.bin'})
    outputs = []
    for i, arr in enumerate(native):
        assert arr.dtype == np.float32
        arr.astype('<f4').tofile(p / f'output-{i}.bin')
        outputs.append({'name': node.output[i], 'shape': list(arr.shape), 'file': f'{index}/output-{i}.bin'})
    probes.append({'kind': kind, 'node': node.name, 'model': f'{index}/model.onnx', 'inputs': tensors, 'outputs': outputs})
(out / 'probes.json').write_text(json.dumps({'native_ort': ort.__version__, 'probes': probes}, indent=2))
print(json.dumps(probes, indent=2))
