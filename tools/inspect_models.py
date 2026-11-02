"""Dump ONNX operator/parameter inventories and dense arithmetic counts.

inspect_models.py OUTPUT_DIR MODEL...
Creates scratch graph.txt/JSON; never commit generated inventories. MAC counts
include Conv, ConvTranspose, MatMul, MatMulInteger, Gemm. Two FLOPs per MAC;
elementwise/normalization/quantization costs are additional. Constant model
buffers are counted separately from learned (weight/bias/scale/embedding) data.
"""
from collections import Counter
import json
from pathlib import Path
import sys

import numpy as np
import onnx
from onnx import helper, numpy_helper

out = Path(sys.argv[1])
out.mkdir(parents=True, exist_ok=True)
for source in sys.argv[2:]:
    model = onnx.load(source)
    for inp in model.graph.input:
        for i, d in enumerate(inp.type.tensor_type.shape.dim):
            if d.dim_param:
                # Both models have only dynamic batch; encoder spatial dims
                # below are the real staff canvas, never a reduced sequence.
                d.dim_value = 1
        if 'encoder' in Path(source).name:
            for d, v in zip(inp.type.tensor_type.shape.dim, [1, 1, 256, 1280]):
                d.dim_value = v
    model = onnx.shape_inference.infer_shapes(model, data_prop=True)
    shapes = {}
    for value in [*model.graph.input, *model.graph.output, *model.graph.value_info]:
        dims = value.type.tensor_type.shape.dim
        if all(d.HasField('dim_value') for d in dims):
            shapes[value.name] = [d.dim_value for d in dims]
    arrays = {t.name: numpy_helper.to_array(t) for t in model.graph.initializer}
    shapes.update({k: list(v.shape) for k, v in arrays.items()})
    inventory = Counter((n.domain + '::' if n.domain else '') + n.op_type for n in model.graph.node)
    lines, arithmetic, unknown = [], [], []
    for n in model.graph.node:
        attrs = {a.name: helper.get_attribute_value(a) for a in n.attribute}
        lines.append(f'{n.op_type} {n.name}\n  in: ' + ', '.join(f'{s}{shapes.get(s, "?")}' for s in n.input) +
                     '\n  out: ' + ', '.join(f'{s}{shapes.get(s, "?")}' for s in n.output) + '\n  ' + str(attrs))
        op = n.op_type
        if op not in {'Conv', 'ConvTranspose', 'MatMul', 'MatMulInteger', 'Gemm'}:
            continue
        try:
            x, w, y = shapes[n.input[0]], shapes[n.input[1]], shapes[n.output[0]]
            if op == 'Conv':
                mac = int(np.prod(y)) * int(np.prod(w[1:]))
            elif op == 'ConvTranspose':
                mac = int(np.prod(x)) * int(np.prod(w[1:]))
            else:
                k = x[-2] if op == 'Gemm' and attrs.get('transA') else x[-1]
                mac = int(np.prod(y)) * k
            arithmetic.append({'op': op, 'name': n.name, 'macs': mac, 'input': x, 'weight': w, 'output': y})
        except KeyError:
            unknown.append(n.name)
    # Initializer scalar/shape constants are not trainable parameters. Include
    # quantization scale/zp tensors but list each so counts are auditable.
    parameters = {k: {'shape': list(v.shape), 'dtype': str(v.dtype), 'elements': int(v.size), 'bytes': v.nbytes}
                  for k, v in arrays.items()}
    learned = {k: v for k, v in arrays.items() if any(s in k for s in ['weight', 'bias', 'MatMul_', 'Mul_'])}
    # This encoder folds learned, factorized height/width embeddings into an
    # expanded (1,H,W,D) constant. Recover the independent parameter count,
    # checking the repetition bit-for-bit rather than treating it as a buffer.
    positional_parameters = 0
    if '/Concat_output_0' in arrays:
        position = arrays['/Concat_output_0']
        if position.ndim == 4 and position.shape[0] == 1:
            position = position[0]
            same_x = np.all(position == position[:, 0:1, :], axis=(0, 1))
            same_y = np.all(position == position[0:1, :, :], axis=(0, 1))
            assert np.all(same_x ^ same_y), 'unrecognized positional embedding'
            positional_parameters = int(position.shape[0] * same_x.sum() + position.shape[1] * same_y.sum())
    report = {'model': str(source), 'opset': {o.domain: o.version for o in model.opset_import},
        'ops': dict(inventory), 'initializer_elements': sum(v.size for v in arrays.values()),
        'initializer_bytes': sum(v.nbytes for v in arrays.values()), 'initializers': parameters,
        'named_weight_parameter_elements': sum(v.size for v in learned.values()),
        'factorized_positional_parameters': positional_parameters,
        'learned_parameter_elements': sum(v.size for v in learned.values()) + positional_parameters,
        'dense_macs': sum(a['macs'] for a in arithmetic), 'dense_flops': 2 * sum(a['macs'] for a in arithmetic),
        'arithmetic': arithmetic, 'unresolved_arithmetic': unknown}
    name = Path(source).stem
    (out / f'{name}.graph.txt').write_text('\n'.join(lines))
    (out / f'{name}.json').write_text(json.dumps(report, indent=2))
    print(name, json.dumps({k: v for k, v in report.items() if k not in {'initializers', 'arithmetic'}}))
