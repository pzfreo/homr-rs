use homr_rs::{encoder::kernels, onnx::Graph};
use serde_json::{json, Value};
use std::{fs, path::Path};
fn floats(path: &Path) -> Vec<f32> {
    fs::read(path)
        .unwrap()
        .as_chunks::<4>()
        .0
        .iter()
        .map(|v| f32::from_le_bytes(*v))
        .collect()
}
fn nhwc(x: &[f32], h: usize, w: usize, c: usize) -> Vec<f32> {
    (0..h * w * c).map(|i| x[(i % c) * h * w + i / c]).collect()
}
fn nchw(x: &[f32], h: usize, w: usize, c: usize) -> Vec<f32> {
    (0..h * w * c)
        .map(|i| x[(i % (h * w)) * c + i / (h * w)])
        .collect()
}
fn main() -> Result<(), Box<dyn std::error::Error>> {
    let args: Vec<_> = std::env::args().collect();
    let root = Path::new(&args[1]);
    let manifest: Value = serde_json::from_slice(&fs::read(root.join("probes.json"))?)?;
    let mut rows = Vec::new();
    for probe in manifest["probes"].as_array().unwrap() {
        let kind = probe["kind"].as_str().unwrap();
        let g = Graph::read(&fs::read(root.join(probe["model"].as_str().unwrap()))?)?;
        let node = &g.nodes[0];
        let input = &probe["inputs"][0];
        let shape: Vec<_> = input["shape"]
            .as_array()
            .unwrap()
            .iter()
            .map(|v| v.as_u64().unwrap() as usize)
            .collect();
        let x = floats(&root.join(input["file"].as_str().unwrap()));
        let got = match node.op.as_str() {
            "MatMul" | "Gemm" => {
                let b = g.tensor(&node.inputs[1])?;
                let n = b.dims[1];
                let k = *shape.last().unwrap();
                let bias = node
                    .inputs
                    .get(2)
                    .map(|s| g.tensor(s).unwrap().floats().unwrap());
                kernels::matmul(&x, &b.floats()?, x.len() / k, k, n, bias.as_deref())
            }
            "Conv" => {
                let (c, h, w) = (shape[1], shape[2], shape[3]);
                let t = g.tensor(&node.inputs[1])?;
                let b = g.tensor(&node.inputs[2])?.floats()?;
                let stride = node.ints["strides"][0] as usize;
                let pad = node.ints["pads"][0] as usize;
                let kernel = t.dims[2];
                let nh = nhwc(&x, h, w, c);
                let got = if node.ints.get("group").is_some_and(|v| v[0] > 1) {
                    kernels::depthwise(&nh, h, w, c, &t.floats()?, &b, kernel, pad)
                } else {
                    kernels::conv(
                        &nh,
                        h,
                        w,
                        c,
                        &t.floats()?,
                        &b,
                        t.dims[0],
                        kernel,
                        stride,
                        pad,
                    )
                };
                nchw(
                    &got,
                    (h + 2 * pad - kernel) / stride + 1,
                    (w + 2 * pad - kernel) / stride + 1,
                    t.dims[0],
                )
            }
            "LayerNormalization" => kernels::layer_norm(
                &x,
                *shape.last().unwrap(),
                &g.tensor(&node.inputs[1])?.floats()?,
                &g.tensor(&node.inputs[2])?.floats()?,
                node.floats["epsilon"],
            ),
            "Erf" => {
                let mut out = x.clone();
                kernels::erf_in_place(&mut out);
                out
            }
            "Gelu" => {
                let mut out = x.clone();
                kernels::gelu_in_place(&mut out);
                out
            }
            "Div" => {
                let b = g.tensor(&node.inputs[1])?.scalar()?;
                x.iter().map(|&v| v / b).collect()
            }
            op => return Err(format!("unsupported op {op}").into()),
        };
        let reference = floats(&root.join(probe["outputs"][0]["file"].as_str().unwrap()));
        assert_eq!(got.len(), reference.len());
        let mut count = 0;
        let mut max = 0.0f64;
        let mut sum = 0.0;
        let mut first = None;
        for (i, (&a, &b)) in got.iter().zip(&reference).enumerate() {
            let e = (a as f64 - b as f64).abs();
            max = max.max(e);
            sum += e;
            if a.to_bits() != b.to_bits() {
                count += 1;
                if first.is_none() {
                    first = Some(
                        json!({"index":i,"got":a,"reference":b,"got_bits":a.to_bits(),"reference_bits":b.to_bits()}),
                    );
                }
            }
        }
        let row = json!({"kind":kind,"values":got.len(),"bit_differences":count,"max_abs":max,"mean_abs":sum/got.len() as f64,"first":first});
        println!("{row}");
        rows.push(row);
    }
    fs::write(&args[2], serde_json::to_vec_pretty(&rows)?)?;
    Ok(())
}
