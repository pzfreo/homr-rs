//! C ABI using the same allocator and linear memory as the decoder.
use super::{Encoder, CONTEXT_VALUES, IMAGE_VALUES};

/// Load upstream encoder ONNX. Null indicates invalid model or unfused host.
/// # Safety
/// `data,len` must refer to readable model bytes for this call.
#[no_mangle]
pub unsafe extern "C" fn encoder_load(data: *const u8, len: usize) -> *mut Encoder {
    match Encoder::from_onnx(std::slice::from_raw_parts(data, len)) {
        Ok(m) => Box::into_raw(Box::new(m)),
        Err(_) => std::ptr::null_mut(),
    }
}
/// # Safety
/// `model` must be a live encoder_load handle, released exactly once.
#[no_mangle]
pub unsafe extern "C" fn encoder_drop(model: *mut Encoder) {
    drop(Box::from_raw(model));
}
/// Write [1280,512] context to caller memory. Pass it directly to decoder_run.
/// Returns 1 on success, 0 for invalid image. The final buffer copy stays in
/// WASM; no context floats need to cross the JS boundary.
/// # Safety
/// Handle must be live, image has IMAGE_VALUES aligned f32s, context has
/// CONTEXT_VALUES writable aligned f32s, and buffers do not alias the model.
#[no_mangle]
pub unsafe extern "C" fn encoder_run(
    model: *const Encoder,
    image: *const f32,
    context: *mut f32,
) -> u32 {
    let Ok(values) = (&*model).encode(std::slice::from_raw_parts(image, IMAGE_VALUES)) else {
        return 0;
    };
    std::ptr::copy_nonoverlapping(values.as_ptr(), context, CONTEXT_VALUES);
    1
}
/// Encode and greedily decode without copying the context at all.
/// # Safety
/// Handles must be live, image has IMAGE_VALUES aligned f32s, output has
/// steps*32 writable bytes; no buffers may alias either model.
#[no_mangle]
pub unsafe extern "C" fn encoder_decode(
    encoder: *const Encoder,
    decoder: *const crate::decoder::Decoder,
    image: *const f32,
    output: *mut u8,
    steps: usize,
) -> usize {
    let Ok(context) = (&*encoder).encode(std::slice::from_raw_parts(image, IMAGE_VALUES)) else {
        return 0;
    };
    crate::decoder::wasm::decoder_run(decoder, context.as_ptr(), output, steps, 0)
}
#[no_mangle]
pub extern "C" fn encoder_fused_simd_supported() -> u32 {
    crate::decoder::fused_simd_supported() as u32
}

/// Pack a GPU plan from a live, already validated encoder. Read-only to model.
/// # Safety
/// `model` is a live encoder_load handle; release the result once with plan_drop.
#[no_mangle]
pub unsafe extern "C" fn encoder_gpu_plan_create(model: *const Encoder) -> *mut super::gpu::Plan {
    Box::into_raw(Box::new(super::gpu::Plan::new(&*model)))
}
/// # Safety
/// A live encoder_gpu_plan_create handle, released exactly once.
#[no_mangle]
pub unsafe extern "C" fn encoder_gpu_plan_drop(plan: *mut super::gpu::Plan) {
    drop(Box::from_raw(plan));
}
/// # Safety
/// Live plan. Borrowed immutable bytes remain valid until plan_drop.
#[no_mangle]
pub unsafe extern "C" fn encoder_gpu_plan_json(plan: *const super::gpu::Plan) -> *const u8 {
    (*plan).json.as_ptr()
}
/// # Safety
/// Live plan.
#[no_mangle]
pub unsafe extern "C" fn encoder_gpu_plan_json_len(plan: *const super::gpu::Plan) -> usize {
    (*plan).json.len()
}
/// # Safety
/// Live plan. Borrowed immutable floats remain valid until plan_drop.
#[no_mangle]
pub unsafe extern "C" fn encoder_gpu_plan_weights(plan: *const super::gpu::Plan) -> *const f32 {
    (*plan).weights.as_ptr()
}
/// # Safety
/// Live plan. Number of float32 values (not bytes).
#[no_mangle]
pub unsafe extern "C" fn encoder_gpu_plan_weights_len(plan: *const super::gpu::Plan) -> usize {
    (*plan).weights.len()
}
/// Diagnostic per-node CPU oracle, deliberately using the existing CPU kernels.
/// # Safety
/// Live plan, valid op index, inputs/output have the shapes in its metadata.
/// Output must not alias the plan or inputs. Call only on the coordinator.
#[no_mangle]
pub unsafe extern "C" fn encoder_gpu_reference(
    plan: *const super::gpu::Plan,
    op: usize,
    input: *const f32,
    residual: *const f32,
    output: *mut f32,
) -> usize {
    let plan = &*plan;
    let Some(op) = plan.ops.get(op) else {
        return 0;
    };
    let q = &op.params;
    let n = match q[0] {
        super::gpu::MATMUL => (q[1] * q[2]) as usize,
        super::gpu::CONV | super::gpu::DEPTHWISE => (q[7] * q[8] * q[9]) as usize,
        _ => op.values(),
    };
    let r = if q[0] == super::gpu::ADD {
        std::slice::from_raw_parts(residual, op.values())
    } else {
        &[]
    };
    let values = plan.reference(op, std::slice::from_raw_parts(input, n), r);
    std::ptr::copy_nonoverlapping(values.as_ptr(), output, values.len());
    values.len()
}
/// Independent scalar/activation oracle for math probes. Records are four u32
/// words: opcode, a bits, b bits, c bits. Outputs are one u32 bit pattern each.
/// # Safety
/// Input has count*4 words, output count writable words, with no aliasing.
#[no_mangle]
pub unsafe extern "C" fn encoder_gpu_math_reference(
    input: *const u32,
    output: *mut u32,
    count: usize,
) {
    use super::kernels;
    let records = std::slice::from_raw_parts(input, count * 4);
    let out = std::slice::from_raw_parts_mut(output, count);
    // Use the encoder's actual SIMD activation, including its strict fused
    // emulation, rather than the decoder scalar's f64 approximation to fma.
    let mut erfs = Vec::new();
    let mut gelus = Vec::new();
    for r in records.chunks_exact(4) {
        match r[0] {
            3 => erfs.push(f32::from_bits(r[1])),
            4 => gelus.push(f32::from_bits(r[1])),
            _ => {}
        }
    }
    // Pad to SIMD width so the scalar tail cannot use a different fma helper.
    erfs.resize(erfs.len().next_multiple_of(4), 0.0);
    gelus.resize(gelus.len().next_multiple_of(4), 0.0);
    kernels::erf_in_place(&mut erfs);
    kernels::gelu_in_place(&mut gelus);
    let (mut ei, mut gi) = (0, 0);
    for (r, out) in records.chunks_exact(4).zip(out) {
        let a = f32::from_bits(r[1]);
        let b = f32::from_bits(r[2]);
        let c = f32::from_bits(r[3]);
        *out = match r[0] {
            0 => kernels::fma(a, b, c),
            1 => a / b,
            2 => a.sqrt(),
            3 => {
                let v = erfs[ei];
                ei += 1;
                v
            }
            4 => {
                let v = gelus[gi];
                gi += 1;
                v
            }
            5 => super::gpu::erf_exp(a),
            6 => a + b,
            7 => a * b,
            8 => a - b,
            _ => f32::NAN,
        }
        .to_bits();
    }
}

/// Stream every division/sqrt operand in CPU Welford/affine, or every
/// GELU/Erf/inner-exp operand. Benchmark oracle, not an encoder alternative.
/// kind=0: count floats in complete c-channel rows, (2*c+2)*count/c records.
/// kind=1: count activation floats, 3*count records. Each record has four u32
/// words (opcode,a,b,c); expected has one u32 IEEE result per record.
/// # Safety
/// Input has count readable floats; records/expected have the sizes above,
/// are writable, and alias neither input nor each other. Coordinator only.
#[no_mangle]
pub unsafe extern "C" fn encoder_gpu_audit_records(
    kind: u32,
    input: *const f32,
    count: usize,
    c: usize,
    epsilon: f32,
    records: *mut u32,
    expected: *mut u32,
) -> usize {
    use super::kernels;
    if kind > 1 || (kind == 0 && (c == 0 || !count.is_multiple_of(c))) {
        return 0;
    }
    let n = if kind == 0 {
        (2 * c + 2) * (count / c)
    } else {
        3 * count
    };
    let x = std::slice::from_raw_parts(input, count);
    let records = std::slice::from_raw_parts_mut(records, n * 4);
    let expected = std::slice::from_raw_parts_mut(expected, n);
    let mut index = 0;
    let mut record = |op: u32, a: f32, b: f32, v: f32| {
        records[index * 4..index * 4 + 4].copy_from_slice(&[op, a.to_bits(), b.to_bits(), 0]);
        expected[index] = v.to_bits();
        index += 1;
    };
    if kind == 0 {
        for row in x.chunks_exact(c) {
            let (mut mean, mut m2) = (0.0, 0.0);
            for (j, &v) in row.iter().enumerate() {
                let delta = v - mean;
                let divisor = (j + 1) as f32;
                let quotient = delta / divisor;
                record(1, delta, divisor, quotient);
                mean += quotient;
                m2 = kernels::fma(delta, v - mean, m2);
            }
            let var = m2 / c as f32;
            record(1, m2, c as f32, var);
            let var = var + epsilon;
            let std = var.sqrt();
            record(2, var, 0.0, std);
            for &v in row {
                let delta = v - mean;
                record(1, delta, std, delta / std);
            }
        }
    } else {
        let mut gelu = x.to_vec();
        gelu.resize(count.next_multiple_of(4), 0.0);
        kernels::gelu_in_place(&mut gelu);
        let mut erf: Vec<_> = x
            .iter()
            .map(|v| v * std::f32::consts::FRAC_1_SQRT_2)
            .collect();
        erf.resize(count.next_multiple_of(4), 0.0);
        kernels::erf_in_place(&mut erf);
        for (i, &v) in x.iter().enumerate() {
            let z = v * std::f32::consts::FRAC_1_SQRT_2;
            record(4, v, 0.0, gelu[i]);
            record(3, z, 0.0, erf[i]);
            let exp_input = super::gpu::erf_exp_input(z);
            record(5, exp_input, 0.0, super::gpu::erf_exp(exp_input));
        }
    }
    n
}

#[cfg(feature = "encoder-probes")]
mod probes {
    use crate::{encoder::kernels, onnx::Graph};
    /// # Safety
    /// Readable complete single-node probe ONNX, released with encoder_probe_drop.
    #[no_mangle]
    pub unsafe extern "C" fn encoder_probe_load(p: *const u8, n: usize) -> *mut Graph {
        match Graph::read(std::slice::from_raw_parts(p, n)) {
            Ok(g) if g.nodes.len() == 1 => Box::into_raw(Box::new(g)),
            _ => std::ptr::null_mut(),
        }
    }
    /// # Safety
    /// A live probe handle, released once.
    #[no_mangle]
    pub unsafe extern "C" fn encoder_probe_drop(p: *mut Graph) {
        drop(Box::from_raw(p));
    }
    /// # Safety
    /// Input/output buffers must have the sizes described by the probe, len and
    /// NCHW h,w,c dimensions; output must not alias input or the graph.
    #[no_mangle]
    pub unsafe extern "C" fn encoder_probe_run(
        g: *const Graph,
        input: *const f32,
        len: usize,
        output: *mut f32,
        h: usize,
        w: usize,
        c: usize,
    ) -> usize {
        let g = &*g;
        let n = &g.nodes[0];
        let x = std::slice::from_raw_parts(input, len);
        let result = match n.op.as_str() {
            "MatMul" | "Gemm" => {
                let b = g.tensor(&n.inputs[1]).unwrap();
                let k = b.dims[0];
                let nc = b.dims[1];
                let bias = n
                    .inputs
                    .get(2)
                    .map(|s| g.tensor(s).unwrap().floats().unwrap());
                kernels::matmul(x, &b.floats().unwrap(), len / k, k, nc, bias.as_deref())
            }
            "LayerNormalization" => {
                let scale = g.tensor(&n.inputs[1]).unwrap().floats().unwrap();
                kernels::layer_norm(
                    x,
                    scale.len(),
                    &scale,
                    &g.tensor(&n.inputs[2]).unwrap().floats().unwrap(),
                    n.floats["epsilon"],
                )
            }
            "Erf" | "Gelu" => {
                let mut v = x.to_vec();
                if n.op == "Erf" {
                    kernels::erf_in_place(&mut v);
                } else {
                    kernels::gelu_in_place(&mut v);
                }
                v
            }
            "Div" => {
                let d = g.tensor(&n.inputs[1]).unwrap().scalar().unwrap();
                x.iter().map(|v| v / d).collect()
            }
            "Conv" => {
                let wt = g.tensor(&n.inputs[1]).unwrap();
                let bias = g.tensor(&n.inputs[2]).unwrap().floats().unwrap();
                let k = wt.dims[2];
                let stride = n.ints["strides"][0] as usize;
                let pad = n.ints["pads"][0] as usize;
                let nh: Vec<_> = (0..len).map(|i| x[(i % c) * h * w + i / c]).collect();
                let out = if n.ints.get("group").is_some_and(|g| g[0] > 1) {
                    kernels::depthwise(&nh, h, w, c, &wt.floats().unwrap(), &bias, k, pad)
                } else {
                    kernels::conv(
                        &nh,
                        h,
                        w,
                        c,
                        &wt.floats().unwrap(),
                        &bias,
                        wt.dims[0],
                        k,
                        stride,
                        pad,
                    )
                };
                let plane = ((h + 2 * pad - k) / stride + 1) * ((w + 2 * pad - k) / stride + 1);
                (0..out.len())
                    .map(|i| out[(i % plane) * wt.dims[0] + i / plane])
                    .collect()
            }
            _ => return 0,
        };
        std::ptr::copy_nonoverlapping(result.as_ptr(), output, result.len());
        result.len()
    }
}
