"""Record the native optimized encoder graph without changing its fusions.

record_encoder_stages.py MODEL INPUT.f32 NATIVE_CONTEXT OUT
Use numpy, onnx and onnxruntime==1.31.0. All output belongs in scratch.
The optimized graph runs with re-optimization disabled; its final bytes must
match the default CPU encoder before any intermediate is accepted.
"""
import json
from pathlib import Path
import sys
import numpy as np
import onnx
from onnx import helper, numpy_helper
import onnxruntime as ort

model, image, reference, output = sys.argv[1:]
out=Path(output);out.mkdir(parents=True,exist_ok=True)
opts=ort.SessionOptions();opts.optimized_model_filepath=str(out/'optimized.onnx')
default=ort.InferenceSession(model,sess_options=opts)
x=np.fromfile(image,dtype='<f4').reshape(1,1,256,1280)
expected=np.fromfile(reference,dtype='<f4')
assert default.run(None,{'input':x})[0].tobytes()==expected.tobytes()
graph=onnx.load(out/'optimized.onnx')
values={v.name:v for v in [*graph.graph.input,*graph.graph.output,*graph.graph.value_info]}
names=[]
for node in graph.graph.node:
    if node.op_type not in {'Transpose','Reshape'}:
        for name in node.output:
            if name not in {v.name for v in graph.graph.output}:
                graph.graph.output.append(values.get(name,helper.make_tensor_value_info(name,onnx.TensorProto.FLOAT,None)))
            names.append(name)
names.append('output')
opts=ort.SessionOptions();opts.graph_optimization_level=ort.GraphOptimizationLevel.ORT_DISABLE_ALL
session=ort.InferenceSession(graph.SerializeToString(),sess_options=opts)
result=session.run(names,{'input':x})
assert result[-1].tobytes()==expected.tobytes(), 'diagnostic graph changed native result'
rows=[]
actual={'input':x}
for i,(name,value) in enumerate(zip(names,result)):
    value.astype('<f4').tofile(out/f'{i}.f32');actual[name]=value
    rows.append({'name':name,'shape':list(value.shape),'file':f'{i}.f32'})
(out/'stages.json').write_text(json.dumps(rows,indent=2))
# Additional isolated Gemm and optimized Gelu probes use actual native inputs.
weights={t.name:t for t in graph.graph.initializer}
seen=set();probes=[]
for node in graph.graph.node:
    if node.op_type not in {'Gemm','Gelu'} or node.op_type in seen:continue
    seen.add(node.op_type);idx=len(probes);d=out/str(idx);d.mkdir(exist_ok=True)
    # Reshape feeding Gemm has the same bytes as its unreshaped input.
    feeds={}
    for name in node.input:
        if name in weights:continue
        if name in actual:value=actual[name]
        else:
            producer=next(n for n in graph.graph.node if name in n.output)
            assert producer.op_type=='Reshape'
            value=actual[producer.input[0]].reshape(-1,actual[producer.input[0]].shape[-1])
        feeds[name]=value
    ins=[helper.make_tensor_value_info(n,onnx.TensorProto.FLOAT,v.shape) for n,v in feeds.items()]
    outs=[values.get(n,helper.make_tensor_value_info(n,onnx.TensorProto.FLOAT,None)) for n in node.output]
    sub=helper.make_model(helper.make_graph([node],'probe',ins,outs,[weights[n] for n in node.input if n in weights]),opset_imports=graph.opset_import);sub.ir_version=graph.ir_version
    (d/'model.onnx').write_bytes(sub.SerializeToString())
    native=ort.InferenceSession(sub.SerializeToString()).run(None,feeds)
    tensors=[]
    for i,(n,v) in enumerate(feeds.items()):
        v.astype('<f4').tofile(d/f'input-{i}.bin');tensors.append({'name':n,'shape':list(v.shape),'type':'float32','file':f'{idx}/input-{i}.bin'})
    outputs=[]
    for i,(n,v) in enumerate(zip(node.output,native)):
        v.astype('<f4').tofile(d/f'output-{i}.bin');outputs.append({'name':n,'shape':list(v.shape),'file':f'{idx}/output-{i}.bin'})
    probes.append({'kind':node.op_type,'model':f'{idx}/model.onnx','inputs':tensors,'outputs':outputs})
(out/'probes.json').write_text(json.dumps({'probes':probes},indent=2))
print('accepted',len(rows),'native stages, final context byte-identical')
