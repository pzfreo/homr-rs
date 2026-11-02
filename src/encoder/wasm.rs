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
