//! Ordered float32 kernels. Arithmetic follows ORT/MLAS (MIT, Microsoft).
#![allow(clippy::too_many_arguments)]

#[inline]
pub fn fma(a: f32, b: f32, c: f32) -> f32 {
    #[cfg(all(target_arch = "wasm32", not(target_feature = "relaxed-simd")))]
    {
        return libm::fmaf(a, b, c);
    }
    #[cfg(all(target_arch = "wasm32", target_feature = "relaxed-simd"))]
    {
        use std::arch::wasm32::*;
        return f32x4_extract_lane::<0>(f32x4_relaxed_madd(
            f32x4_splat(a),
            f32x4_splat(b),
            f32x4_splat(c),
        ));
    }
    #[cfg(not(target_arch = "wasm32"))]
    crate::decoder::activation::fma(a, b, c)
}
pub fn gelu(x: f32) -> f32 {
    crate::decoder::activation::gelu(x)
}
pub fn erf(x: f32) -> f32 {
    crate::decoder::activation::erf(x)
}
pub fn gelu_in_place(x: &mut [f32]) {
    #[cfg(any(target_arch = "aarch64", target_arch = "wasm32"))]
    {
        unsafe { super::simd::activation(x, true) };
        return;
    }
    #[cfg(not(any(target_arch = "aarch64", target_arch = "wasm32")))]
    for v in x {
        *v = gelu(*v);
    }
}
pub fn erf_in_place(x: &mut [f32]) {
    #[cfg(any(target_arch = "aarch64", target_arch = "wasm32"))]
    {
        unsafe { super::simd::activation(x, false) };
        return;
    }
    #[cfg(not(any(target_arch = "aarch64", target_arch = "wasm32")))]
    for v in x {
        *v = erf(*v);
    }
}

/// Native Welford variance and fused affine transform; input/output are NHWC.
pub fn layer_norm(
    x: &[f32],
    channels: usize,
    scale: &[f32],
    bias: &[f32],
    epsilon: f32,
) -> Vec<f32> {
    let mut out = vec![0.0; x.len()];
    for (row, dst) in x.chunks_exact(channels).zip(out.chunks_exact_mut(channels)) {
        let (mut mean, mut m2) = (0.0, 0.0);
        for (i, &v) in row.iter().enumerate() {
            let delta = v - mean;
            mean += delta / (i + 1) as f32;
            m2 = fma(delta, v - mean, m2);
        }
        let std = (m2 / channels as f32 + epsilon).sqrt();
        for j in 0..channels {
            dst[j] = fma((row[j] - mean) / std, scale[j], bias[j]);
        }
    }
    out
}

/// B is row-major KxN. Products accumulate in ascending K, then bias is added.
pub fn matmul(
    a: &[f32],
    b: &[f32],
    m: usize,
    k: usize,
    n: usize,
    bias: Option<&[f32]>,
) -> Vec<f32> {
    product(a, b, m, k, n, bias, false)
}
fn product(
    a: &[f32],
    b: &[f32],
    m: usize,
    k: usize,
    n: usize,
    bias: Option<&[f32]>,
    bias_start: bool,
) -> Vec<f32> {
    assert_eq!(a.len(), m * k);
    assert_eq!(b.len(), k * n);
    if let Some(b) = bias {
        assert_eq!(b.len(), n);
    }
    let mut out = vec![0.0; m * n];
    #[cfg(any(target_arch = "wasm32", target_arch = "aarch64"))]
    if m.is_multiple_of(4) && n.is_multiple_of(16) {
        // SAFETY: lengths and tile divisibility are checked above.
        unsafe {
            super::simd::product(
                a,
                b,
                &mut out,
                m,
                k,
                n,
                if bias_start { bias } else { None },
            );
        }
        if !bias_start {
            if let Some(bias) = bias {
                for row in out.chunks_exact_mut(n) {
                    for (v, b) in row.iter_mut().zip(bias) {
                        *v += b;
                    }
                }
            }
        }
        return out;
    }
    if bias_start {
        if let Some(b) = bias {
            for row in out.chunks_exact_mut(n) {
                row.copy_from_slice(b);
            }
        }
    }
    for i in 0..m {
        for p in 0..k {
            let v = a[i * k + p];
            for j in 0..n {
                out[i * n + j] = fma(v, b[p * n + j], out[i * n + j]);
            }
        }
    }
    if !bias_start {
        if let Some(bias) = bias {
            for row in out.chunks_exact_mut(n) {
                for (v, b) in row.iter_mut().zip(bias) {
                    *v += b;
                }
            }
        }
    }
    out
}

/// Ordinary convolution with NCHW ONNX weights, NHWC activation storage.
pub fn conv(
    x: &[f32],
    h: usize,
    w: usize,
    c: usize,
    weights: &[f32],
    bias: &[f32],
    n: usize,
    kernel: usize,
    stride: usize,
    pad: usize,
) -> Vec<f32> {
    let oh = (h + 2 * pad - kernel) / stride + 1;
    let ow = (w + 2 * pad - kernel) / stride + 1;
    let k = c * kernel * kernel;
    let mut patches = vec![0.0; oh * ow * k];
    for y in 0..oh {
        for xx in 0..ow {
            for ch in 0..c {
                for ky in 0..kernel {
                    for kx in 0..kernel {
                        let iy = y * stride + ky;
                        let ix = xx * stride + kx;
                        if iy >= pad && iy < h + pad && ix >= pad && ix < w + pad {
                            patches[(y * ow + xx) * k + (ch * kernel + ky) * kernel + kx] =
                                x[((iy - pad) * w + ix - pad) * c + ch];
                        }
                    }
                }
            }
        }
    }
    let mut b = vec![0.0; k * n];
    for j in 0..n {
        for p in 0..k {
            b[p * n + j] = weights[j * k + p];
        }
    }
    if c == 1 {
        return product(&patches, &b, oh * ow, k, n, Some(bias), true);
    }
    // MLAS segmented convolution forms independent 128-term sums, then adds
    // them in order. It applies bias after the last segment (not before).
    let mut result = vec![0.0; oh * ow * n];
    for start in (0..k).step_by(128) {
        let count = (k - start).min(128);
        let mut a = Vec::with_capacity(oh * ow * count);
        for row in patches.chunks_exact(k) {
            a.extend_from_slice(&row[start..start + count]);
        }
        let partial = product(
            &a,
            &b[start * n..(start + count) * n],
            oh * ow,
            count,
            n,
            None,
            false,
        );
        for (out, v) in result.iter_mut().zip(partial) {
            *out += v;
        }
    }
    for row in result.chunks_exact_mut(n) {
        for (v, b) in row.iter_mut().zip(bias) {
            *v += b;
        }
    }
    result
}

pub fn depthwise(
    x: &[f32],
    h: usize,
    w: usize,
    c: usize,
    weights: &[f32],
    bias: &[f32],
    kernel: usize,
    pad: usize,
) -> Vec<f32> {
    assert_eq!(x.len(), h * w * c);
    assert_eq!(weights.len(), c * kernel * kernel);
    assert_eq!(bias.len(), c);
    #[cfg(any(target_arch = "aarch64", target_arch = "wasm32"))]
    if c.is_multiple_of(4) {
        return unsafe { super::simd::depthwise(x, h, w, c, weights, bias, kernel, pad) };
    }
    let mut out = vec![0.0; x.len()];
    for y in 0..h {
        for xx in 0..w {
            for ch in 0..c {
                let mut sum = 0.0;
                for ky in 0..kernel {
                    for kx in 0..kernel {
                        let iy = y + ky;
                        let ix = xx + kx;
                        let v = if iy >= pad && iy < h + pad && ix >= pad && ix < w + pad {
                            x[((iy - pad) * w + ix - pad) * c + ch]
                        } else {
                            0.0
                        };
                        sum = fma(v, weights[(ch * kernel + ky) * kernel + kx], sum);
                    }
                }
                out[(y * w + xx) * c + ch] = sum + bias[ch];
            }
        }
    }
    out
}

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn halfway_fma_is_not_double_rounded() {
        let a = f32::from_bits(0x3f800001);
        let b = f32::from_bits(0x3f7ffffe);
        let c = 16777218.0f32;
        assert_ne!(
            ((a as f64 * b as f64 + c as f64) as f32).to_bits(),
            c.to_bits()
        );
        assert_eq!(fma(a, b, c).to_bits(), c.to_bits());
        #[cfg(any(target_arch = "aarch64", target_arch = "wasm32"))]
        for v in super::super::simd::test_fma(a, b, c) {
            assert_eq!(v.to_bits(), c.to_bits());
        }
    }
    #[test]
    fn convolution_and_gemm_have_different_bias_schedules() {
        let a = f32::from_bits(0x3f800001);
        let b = f32::from_bits(0x3f7ffffe);
        let bias = [16777218.0f32; 16];
        let conv = conv(&[a; 4], 1, 4, 1, &[b; 16], &bias, 16, 1, 1, 0);
        assert!(conv.iter().all(|v| v.to_bits() == 16777218.0f32.to_bits()));
        let mm = matmul(&[a; 4], &[b; 16], 4, 1, 16, Some(&bias));
        assert!(mm.iter().all(|v| v.to_bits() == 16777220.0f32.to_bits()));
    }
}
