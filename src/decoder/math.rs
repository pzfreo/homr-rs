//! Exact integer dot products. Weights are transposed once at model load.
use super::profile::{enter, Phase};
use crate::onnx::Graph;

pub(super) struct Quantized {
    pub data: Vec<u8>,
    pub scale: f32,
    pub zero: i32,
}
impl Quantized {
    pub fn fused(x: &[f32]) -> Self {
        let _g = enter(Phase::Quantize);
        let (mut lo, mut hi) = (0.0f32, 0.0f32);
        for &v in x {
            lo = lo.min(v);
            hi = hi.max(v);
        }
        let multiplier = if lo == hi { 1.0 } else { 255.0 / (hi - lo) };
        let min = lo * multiplier;
        let max = hi * multiplier;
        let zero = if (-128.0 + min) + (127.0 + max) > 0.0 {
            -128.0 - min
        } else {
            127.0 - max
        };
        let zero = zero.clamp(-128.0, 127.0).round_ties_even() as i32;
        let data = x
            .iter()
            .map(|&v| ((v * multiplier).round() as i32 + zero).clamp(-128, 127) as i8 as u8 ^ 128)
            .collect();
        Self {
            data,
            scale: 1.0 / multiplier,
            zero: zero + 128,
        }
    }
    pub fn new(x: &[f32]) -> Self {
        let _g = enter(Phase::Quantize);
        let (mut lo, mut hi) = (0.0f32, 0.0f32);
        for &v in x {
            lo = lo.min(v);
            hi = hi.max(v);
        }
        let scale = if lo == hi { 1.0 } else { (hi - lo) / 255.0 };
        let zero = (-lo / scale).clamp(0.0, 255.0).round_ties_even() as i32;
        // MLAS divides by scale, rounds first, THEN adds the integral zero point.
        let data = x
            .iter()
            .map(|&v| ((v / scale).round_ties_even() as i32 + zero).clamp(0, 255) as u8)
            .collect();
        Self { data, scale, zero }
    }
}
pub(super) struct Linear {
    weights: Vec<i8>,
    sums: Vec<i32>,
    k: usize,
    n: usize,
    scale: f32,
    bias: Vec<f32>,
    kind: Phase,
    #[cfg(target_arch = "wasm32")]
    pair_safe: bool,
}
impl Linear {
    pub fn load(g: &Graph, weight: &str, bias: Option<&str>) -> Result<Self, String> {
        let t = g.tensor(weight)?;
        if t.kind != 3 || t.dims.len() != 2 {
            return Err(format!("expected int8 matrix: {weight}"));
        }
        let (k, n) = (t.dims[0], t.dims[1]);
        if k == 0 || n == 0 || k % 16 != 0 {
            return Err("matrix input dimension must be a multiple of 16".into());
        }
        let stem = weight
            .strip_suffix("_quantized")
            .ok_or("quantized weight name")?;
        if *g.tensor(&format!("{stem}_zero_point"))?.data != [0] {
            return Err("nonzero weight zero point unsupported".into());
        }
        let scale = g.tensor(&format!("{stem}_scale"))?.scalar()?;
        let mut weights = vec![0i8; k * n];
        let mut sums = vec![0; n];
        for j in 0..n {
            for i in 0..k {
                let v = t.data[i * n + j] as i8;
                weights[j * k + i] = v;
                sums[j] += v as i32;
            }
        }
        let bias = match bias {
            Some(s) => g.tensor(s)?.floats()?,
            None => vec![0.0; n],
        };
        if bias.len() != n {
            return Err("linear bias shape".into());
        }
        Ok(Self {
            weights,
            sums,
            k,
            n,
            scale,
            bias,
            kind: Phase::SelfQkv,
            #[cfg(target_arch = "wasm32")]
            pair_safe: !t.data.contains(&128),
        })
    }
    pub fn named(g: &Graph, name: &str, bias: Option<&str>) -> Result<Self, String> {
        let node = g
            .nodes
            .iter()
            .find(|n| n.op == "MatMulInteger" && n.name.starts_with(name))
            .ok_or_else(|| format!("matmul missing: {name}"))?;
        let mut linear = Self::load(g, node.inputs.get(1).ok_or("matmul weight input")?, bias)?;
        linear.kind = if name.contains("to_logits") {
            Phase::Heads
        } else if name.contains("ff/ff.0") {
            Phase::FfIn
        } else if name.contains("ff/ff.2") {
            Phase::FfOut
        } else if name.contains("to_k/") || name.contains("to_v/") {
            Phase::Context
        } else if name.contains("to_q/") {
            Phase::CrossQ
        } else {
            let layer: usize = name
                .split("layers.")
                .nth(1)
                .unwrap()
                .split('.')
                .next()
                .unwrap()
                .parse()
                .unwrap();
            if layer.is_multiple_of(3) {
                Phase::SelfOut
            } else {
                Phase::CrossOut
            }
        };
        Ok(linear)
    }
    pub fn shape(&self) -> (usize, usize) {
        (self.k, self.n)
    }
    pub fn run(&self, x: &[f32]) -> Vec<f32> {
        self.quantized(&Quantized::fused(x))
    }
    pub fn quantized(&self, q: &Quantized) -> Vec<f32> {
        let rows = q.data.len() / self.k;
        let mut out = vec![0.0; rows * self.n];
        let _g = enter(self.kind);
        let scale = q.scale * self.scale;
        #[cfg(any(
            target_arch = "aarch64",
            all(target_arch = "wasm32", target_feature = "simd128")
        ))]
        if rows >= 4 && self.n.is_multiple_of(4) {
            #[cfg(target_arch = "aarch64")]
            let available = std::arch::is_aarch64_feature_detected!("dotprod");
            #[cfg(target_arch = "wasm32")]
            let available = true;
            if available {
                let bulk = rows / 4 * 4;
                for r in (0..bulk).step_by(4) {
                    for c in (0..self.n).step_by(4) {
                        let a = &q.data[r * self.k..(r + 4) * self.k];
                        let b = &self.weights[c * self.k..(c + 4) * self.k];
                        // SAFETY: dimensions and the native feature were checked above.
                        #[cfg(target_arch = "aarch64")]
                        let dots = unsafe { neon_mat4(a, b, self.k) };
                        #[cfg(target_arch = "wasm32")]
                        let dots = unsafe {
                            if self.pair_safe {
                                wasm_mat4::<true>(a, b, self.k)
                            } else {
                                wasm_mat4::<false>(a, b, self.k)
                            }
                        };
                        for i in 0..4 {
                            for j in 0..4 {
                                let d = dots[i * 4 + j] + (128 - q.zero) * self.sums[c + j];
                                out[(r + i) * self.n + c + j] = d as f32 * scale + self.bias[c + j];
                            }
                        }
                    }
                }
                // Context has 1280 rows, but support partial batches as well.
                if bulk == rows {
                    return out;
                }
                let tail = Quantized {
                    data: q.data[bulk * self.k..].to_vec(),
                    scale: q.scale,
                    zero: q.zero,
                };
                out[bulk * self.n..].copy_from_slice(&self.quantized(&tail));
                return out;
            }
        }

        for (a, y) in q
            .data
            .chunks_exact(self.k)
            .zip(out.chunks_exact_mut(self.n))
        {
            #[cfg(target_arch = "aarch64")]
            if std::arch::is_aarch64_feature_detected!("dotprod") {
                let bulk = y.len() / 4 * 4;
                for (block, values) in y[..bulk].as_chunks_mut::<4>().0.iter_mut().enumerate() {
                    let col = block * 4;
                    // SAFETY: dotprod was detected, and the loader validates the dimensions.
                    let dots =
                        unsafe { neon_dot4(a, &self.weights[col * self.k..(col + 4) * self.k]) };
                    for (j, v) in values.iter_mut().enumerate() {
                        let d = dots[j] + (128 - q.zero) * self.sums[col + j];
                        *v = d as f32 * scale + self.bias[col + j];
                    }
                }
                for (j, v) in y[bulk..].iter_mut().enumerate() {
                    let j = bulk + j;
                    let d = dot(a, &self.weights[j * self.k..(j + 1) * self.k], self.sums[j])
                        - q.zero * self.sums[j];
                    *v = d as f32 * scale + self.bias[j];
                }
                continue;
            }
            #[cfg(all(target_arch = "wasm32", target_feature = "simd128"))]
            {
                let bulk = y.len() / 4 * 4;
                for (block, values) in y[..bulk].as_chunks_mut::<4>().0.iter_mut().enumerate() {
                    let col = block * 4;
                    // SAFETY: weights hold n columns of k bytes; k is divisible by 16.
                    let dots = unsafe {
                        if self.pair_safe {
                            wasm_mat1::<true>(a, &self.weights[col * self.k..(col + 4) * self.k])
                        } else {
                            wasm_mat1::<false>(a, &self.weights[col * self.k..(col + 4) * self.k])
                        }
                    };
                    for (j, v) in values.iter_mut().enumerate() {
                        *v = (dots[j] + (128 - q.zero) * self.sums[col + j]) as f32 * scale
                            + self.bias[col + j];
                    }
                }
                for (j, v) in y[bulk..].iter_mut().enumerate() {
                    let j = bulk + j;
                    let d = dot(a, &self.weights[j * self.k..(j + 1) * self.k], self.sums[j])
                        - q.zero * self.sums[j];
                    *v = d as f32 * scale + self.bias[j];
                }
            }
            #[cfg(not(all(target_arch = "wasm32", target_feature = "simd128")))]
            for (j, v) in y.iter_mut().enumerate() {
                let dot = dot(a, &self.weights[j * self.k..(j + 1) * self.k], self.sums[j])
                    - q.zero * self.sums[j];
                *v = dot as f32 * scale + self.bias[j];
            }
        }
        out
    }
}
#[inline]
fn dot(a: &[u8], b: &[i8], _sum: i32) -> i32 {
    #[cfg(all(target_arch = "wasm32", target_feature = "simd128"))]
    // SAFETY: the loader enforces multiples of 16 and both slices have equal length.
    unsafe {
        wasm_dot(a, b)
    }
    #[cfg(target_arch = "aarch64")]
    // SAFETY: NEON is baseline on aarch64; slices are checked by the loader.
    unsafe {
        if std::arch::is_aarch64_feature_detected!("dotprod") {
            return neon_dotprod(a, b) + 128 * _sum;
        }
        neon_dot(a, b)
    }
    #[cfg(not(any(
        target_arch = "aarch64",
        all(target_arch = "wasm32", target_feature = "simd128")
    )))]
    {
        a.iter().zip(b).map(|(&a, &b)| a as i32 * b as i32).sum()
    }
}
#[cfg(all(target_arch = "wasm32", target_feature = "simd128"))]
unsafe fn wasm_dot(a: &[u8], b: &[i8]) -> i32 {
    use core::arch::wasm32::*;
    let mut s = i32x4_splat(0);
    for i in (0..a.len()).step_by(16) {
        let av = v128_load(a.as_ptr().add(i).cast());
        let bv = v128_load(b.as_ptr().add(i).cast());
        s = i32x4_add(
            s,
            i32x4_dot_i16x8(u16x8_extend_low_u8x16(av), i16x8_extend_low_i8x16(bv)),
        );
        s = i32x4_add(
            s,
            i32x4_dot_i16x8(u16x8_extend_high_u8x16(av), i16x8_extend_high_i8x16(bv)),
        );
    }
    i32x4_extract_lane::<0>(s)
        + i32x4_extract_lane::<1>(s)
        + i32x4_extract_lane::<2>(s)
        + i32x4_extract_lane::<3>(s)
}
#[cfg(target_arch = "aarch64")]
unsafe fn neon_dot(a: &[u8], b: &[i8]) -> i32 {
    use core::arch::aarch64::*;
    let mut s = vdupq_n_s32(0);
    for i in (0..a.len()).step_by(16) {
        let av = vld1q_u8(a.as_ptr().add(i));
        let bv = vld1q_s8(b.as_ptr().add(i));
        let al = vreinterpretq_s16_u16(vmovl_u8(vget_low_u8(av)));
        let ah = vreinterpretq_s16_u16(vmovl_high_u8(av));
        let bl = vmovl_s8(vget_low_s8(bv));
        let bh = vmovl_high_s8(bv);
        s = vmlal_s16(s, vget_low_s16(al), vget_low_s16(bl));
        s = vmlal_high_s16(s, al, bl);
        s = vmlal_s16(s, vget_low_s16(ah), vget_low_s16(bh));
        s = vmlal_high_s16(s, ah, bh);
    }
    vaddvq_s32(s)
}

pub(super) fn norm(x: &[f32], gain: &[f32], bias: Option<&[f32]>) -> Vec<f32> {
    let _g = enter(Phase::Norm);
    let (mut mean, mut m2) = (0.0f32, 0.0f32);
    for (i, &v) in x.iter().enumerate() {
        let d = v - mean;
        mean += d / (i + 1) as f32;
        let d2 = v - mean;
        m2 = super::activation::fma(d, d2, m2);
    }
    let std = libm::sqrtf(m2 / x.len() as f32 + 1e-5);
    x.iter()
        .zip(gain)
        .enumerate()
        .map(|(i, (&x, &g))| {
            let v = (x - mean) / std;
            if let Some(b) = bias {
                super::activation::fma(v, g, b[i])
            } else {
                v * g
            }
        })
        .collect()
}
#[cfg(target_arch = "aarch64")]
#[target_feature(enable = "dotprod")]
unsafe fn neon_dotprod(a: &[u8], b: &[i8]) -> i32 {
    use core::arch::aarch64::*;
    let mut s = vdupq_n_s32(0);
    let shift = vdupq_n_u8(128);
    for i in (0..a.len()).step_by(16) {
        let av = vreinterpretq_s8_u8(veorq_u8(vld1q_u8(a.as_ptr().add(i)), shift));
        s = vdotq_s32(s, av, vld1q_s8(b.as_ptr().add(i)));
    }
    vaddvq_s32(s)
}
#[cfg(target_arch = "aarch64")]
#[target_feature(enable = "dotprod")]
unsafe fn neon_dot4(a: &[u8], b: &[i8]) -> [i32; 4] {
    use core::arch::aarch64::*;
    let mut sums = [vdupq_n_s32(0); 4];
    let shift = vdupq_n_u8(128);
    for i in (0..a.len()).step_by(16) {
        let av = vreinterpretq_s8_u8(veorq_u8(vld1q_u8(a.as_ptr().add(i)), shift));
        for (j, s) in sums.iter_mut().enumerate() {
            *s = vdotq_s32(*s, av, vld1q_s8(b.as_ptr().add(j * a.len() + i)));
        }
    }
    sums.map(|s| vaddvq_s32(s))
}

#[cfg(all(target_arch = "wasm32", target_feature = "simd128"))]
#[inline]
unsafe fn wasm_product<const SAFE: bool>(
    a: core::arch::wasm32::v128,
    b: core::arch::wasm32::v128,
) -> core::arch::wasm32::v128 {
    use core::arch::wasm32::*;
    let lo = i16x8_extmul_low_i8x16(a, b);
    let hi = i16x8_extmul_high_i8x16(a, b);
    if SAFE {
        i32x4_extadd_pairwise_i16x8(i16x8_add(lo, hi))
    } else {
        i32x4_add(
            i32x4_extadd_pairwise_i16x8(lo),
            i32x4_extadd_pairwise_i16x8(hi),
        )
    }
}
#[cfg(all(target_arch = "wasm32", target_feature = "simd128"))]
unsafe fn wasm_mat1<const SAFE: bool>(a: &[u8], b: &[i8]) -> [i32; 4] {
    use core::arch::wasm32::*;
    let shift = i8x16_splat(-128);
    let k = a.len();
    let mut c0 = i32x4_splat(0);
    let mut c1 = i32x4_splat(0);
    let mut c2 = i32x4_splat(0);
    let mut c3 = i32x4_splat(0);
    for t in (0..k).step_by(16) {
        let a0 = v128_xor(v128_load(a.as_ptr().add(t).cast()), shift);
        let b0 = v128_load(b.as_ptr().add(t).cast());
        let b1 = v128_load(b.as_ptr().add(k + t).cast());
        let b2 = v128_load(b.as_ptr().add(2 * k + t).cast());
        let b3 = v128_load(b.as_ptr().add(3 * k + t).cast());
        c0 = i32x4_add(c0, wasm_product::<SAFE>(a0, b0));
        c1 = i32x4_add(c1, wasm_product::<SAFE>(a0, b1));
        c2 = i32x4_add(c2, wasm_product::<SAFE>(a0, b2));
        c3 = i32x4_add(c3, wasm_product::<SAFE>(a0, b3));
    }
    [c0, c1, c2, c3].map(|s| {
        i32x4_extract_lane::<0>(s)
            + i32x4_extract_lane::<1>(s)
            + i32x4_extract_lane::<2>(s)
            + i32x4_extract_lane::<3>(s)
    })
}
#[cfg(all(target_arch = "wasm32", target_feature = "simd128"))]
unsafe fn wasm_mat4<const SAFE: bool>(a: &[u8], b: &[i8], k: usize) -> [i32; 16] {
    use core::arch::wasm32::*;
    let shift = i8x16_splat(-128);
    let mut c0 = i32x4_splat(0);
    let mut c1 = i32x4_splat(0);
    let mut c2 = i32x4_splat(0);
    let mut c3 = i32x4_splat(0);
    let mut c4 = i32x4_splat(0);
    let mut c5 = i32x4_splat(0);
    let mut c6 = i32x4_splat(0);
    let mut c7 = i32x4_splat(0);
    let mut c8 = i32x4_splat(0);
    let mut c9 = i32x4_splat(0);
    let mut c10 = i32x4_splat(0);
    let mut c11 = i32x4_splat(0);
    let mut c12 = i32x4_splat(0);
    let mut c13 = i32x4_splat(0);
    let mut c14 = i32x4_splat(0);
    let mut c15 = i32x4_splat(0);
    for t in (0..k).step_by(16) {
        let a0 = v128_xor(v128_load(a.as_ptr().add(t).cast()), shift);
        let a1 = v128_xor(v128_load(a.as_ptr().add(k + t).cast()), shift);
        let a2 = v128_xor(v128_load(a.as_ptr().add(2 * k + t).cast()), shift);
        let a3 = v128_xor(v128_load(a.as_ptr().add(3 * k + t).cast()), shift);
        let b0 = v128_load(b.as_ptr().add(t).cast());
        let b1 = v128_load(b.as_ptr().add(k + t).cast());
        let b2 = v128_load(b.as_ptr().add(2 * k + t).cast());
        let b3 = v128_load(b.as_ptr().add(3 * k + t).cast());
        c0 = i32x4_add(c0, wasm_product::<SAFE>(a0, b0));
        c1 = i32x4_add(c1, wasm_product::<SAFE>(a0, b1));
        c2 = i32x4_add(c2, wasm_product::<SAFE>(a0, b2));
        c3 = i32x4_add(c3, wasm_product::<SAFE>(a0, b3));
        c4 = i32x4_add(c4, wasm_product::<SAFE>(a1, b0));
        c5 = i32x4_add(c5, wasm_product::<SAFE>(a1, b1));
        c6 = i32x4_add(c6, wasm_product::<SAFE>(a1, b2));
        c7 = i32x4_add(c7, wasm_product::<SAFE>(a1, b3));
        c8 = i32x4_add(c8, wasm_product::<SAFE>(a2, b0));
        c9 = i32x4_add(c9, wasm_product::<SAFE>(a2, b1));
        c10 = i32x4_add(c10, wasm_product::<SAFE>(a2, b2));
        c11 = i32x4_add(c11, wasm_product::<SAFE>(a2, b3));
        c12 = i32x4_add(c12, wasm_product::<SAFE>(a3, b0));
        c13 = i32x4_add(c13, wasm_product::<SAFE>(a3, b1));
        c14 = i32x4_add(c14, wasm_product::<SAFE>(a3, b2));
        c15 = i32x4_add(c15, wasm_product::<SAFE>(a3, b3));
    }
    [
        c0, c1, c2, c3, c4, c5, c6, c7, c8, c9, c10, c11, c12, c13, c14, c15,
    ]
    .map(|s| {
        i32x4_extract_lane::<0>(s)
            + i32x4_extract_lane::<1>(s)
            + i32x4_extract_lane::<2>(s)
            + i32x4_extract_lane::<3>(s)
    })
}
#[cfg(target_arch = "aarch64")]
#[target_feature(enable = "dotprod")]
unsafe fn neon_mat4(a: &[u8], b: &[i8], k: usize) -> [i32; 16] {
    use core::arch::aarch64::*;
    let shift = vdupq_n_u8(128);
    let mut c0 = vdupq_n_s32(0);
    let mut c1 = vdupq_n_s32(0);
    let mut c2 = vdupq_n_s32(0);
    let mut c3 = vdupq_n_s32(0);
    let mut c4 = vdupq_n_s32(0);
    let mut c5 = vdupq_n_s32(0);
    let mut c6 = vdupq_n_s32(0);
    let mut c7 = vdupq_n_s32(0);
    let mut c8 = vdupq_n_s32(0);
    let mut c9 = vdupq_n_s32(0);
    let mut c10 = vdupq_n_s32(0);
    let mut c11 = vdupq_n_s32(0);
    let mut c12 = vdupq_n_s32(0);
    let mut c13 = vdupq_n_s32(0);
    let mut c14 = vdupq_n_s32(0);
    let mut c15 = vdupq_n_s32(0);
    for t in (0..k).step_by(16) {
        let a0 = vreinterpretq_s8_u8(veorq_u8(vld1q_u8(a.as_ptr().add(t)), shift));
        let a1 = vreinterpretq_s8_u8(veorq_u8(vld1q_u8(a.as_ptr().add(k + t)), shift));
        let a2 = vreinterpretq_s8_u8(veorq_u8(vld1q_u8(a.as_ptr().add(2 * k + t)), shift));
        let a3 = vreinterpretq_s8_u8(veorq_u8(vld1q_u8(a.as_ptr().add(3 * k + t)), shift));
        let b0 = vld1q_s8(b.as_ptr().add(t));
        let b1 = vld1q_s8(b.as_ptr().add(k + t));
        let b2 = vld1q_s8(b.as_ptr().add(2 * k + t));
        let b3 = vld1q_s8(b.as_ptr().add(3 * k + t));
        c0 = vdotq_s32(c0, a0, b0);
        c1 = vdotq_s32(c1, a0, b1);
        c2 = vdotq_s32(c2, a0, b2);
        c3 = vdotq_s32(c3, a0, b3);
        c4 = vdotq_s32(c4, a1, b0);
        c5 = vdotq_s32(c5, a1, b1);
        c6 = vdotq_s32(c6, a1, b2);
        c7 = vdotq_s32(c7, a1, b3);
        c8 = vdotq_s32(c8, a2, b0);
        c9 = vdotq_s32(c9, a2, b1);
        c10 = vdotq_s32(c10, a2, b2);
        c11 = vdotq_s32(c11, a2, b3);
        c12 = vdotq_s32(c12, a3, b0);
        c13 = vdotq_s32(c13, a3, b1);
        c14 = vdotq_s32(c14, a3, b2);
        c15 = vdotq_s32(c15, a3, b3);
    }
    [
        c0, c1, c2, c3, c4, c5, c6, c7, c8, c9, c10, c11, c12, c13, c14, c15,
    ]
    .map(|s| vaddvq_s32(s))
}
#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn batched_integer_tiles_and_tails_match_scalar() {
        for rows in [1, 3, 4, 5, 8] {
            for minimum in [-128i8, -127] {
                let (k, n) = (32, 8);
                let weights: Vec<i8> = (0..k * n)
                    .map(|i| if i % 3 == 0 { minimum } else { 127 })
                    .collect();
                let sums: Vec<i32> = weights
                    .chunks_exact(k)
                    .map(|col| col.iter().map(|&v| v as i32).sum())
                    .collect();
                let linear = Linear {
                    weights,
                    sums,
                    k,
                    n,
                    scale: 0.125,
                    bias: vec![0.5; n],
                    kind: Phase::SelfQkv,
                    #[cfg(target_arch = "wasm32")]
                    pair_safe: minimum >= -127,
                };
                let q = Quantized {
                    data: (0..rows * k)
                        .map(|i| if i % 2 == 0 { 0 } else { 255 })
                        .collect(),
                    scale: 0.25,
                    zero: 119,
                };
                let actual = linear.quantized(&q);
                for r in 0..rows {
                    for c in 0..n {
                        let expected: i32 = (0..k)
                            .map(|i| {
                                (q.data[r * k + i] as i32 - q.zero)
                                    * linear.weights[c * k + i] as i32
                            })
                            .sum();
                        assert_eq!(
                            actual[r * n + c],
                            expected as f32 * (q.scale * linear.scale) + linear.bias[c]
                        );
                    }
                }
            }
        }
    }
    #[test]
    fn exact_integer_dot() {
        let a: Vec<u8> = (0..512).map(|i| (i * 73) as u8).collect();
        let b: Vec<i8> = (0..512).map(|i| (i * 137) as i8).collect();
        assert_eq!(
            dot(&a, &b, b.iter().map(|&v| v as i32).sum()),
            a.iter()
                .zip(&b)
                .map(|(&a, &b)| a as i32 * b as i32)
                .sum::<i32>()
        );
    }
    #[test]
    fn four_column_accumulation_is_exact() {
        let a: Vec<u8> = (0..512).map(|i| (i * 71) as u8).collect();
        let b: Vec<i8> = (0..2048).map(|i| (i * 131 + 7) as i8).collect();
        let expected: [i32; 4] = std::array::from_fn(|j| {
            a.iter()
                .zip(&b[j * 512..(j + 1) * 512])
                .map(|(&a, &b)| a as i32 * b as i32)
                .sum()
        });
        #[cfg(target_arch = "aarch64")]
        if std::arch::is_aarch64_feature_detected!("dotprod") {
            // SAFETY: the feature is available and both dimensions are valid.
            let signed = unsafe { neon_dot4(&a, &b) };
            for j in 0..4 {
                assert_eq!(
                    signed[j]
                        + 128
                            * b[j * 512..(j + 1) * 512]
                                .iter()
                                .map(|&v| v as i32)
                                .sum::<i32>(),
                    expected[j]
                );
            }
        }
        #[cfg(all(target_arch = "wasm32", target_feature = "simd128"))]
        // SAFETY: a has 512 elements and b has four such columns.
        unsafe {
            let d = wasm_mat1::<false>(&a, &b);
            for j in 0..4 {
                assert_eq!(
                    d[j] + 128
                        * b[j * 512..(j + 1) * 512]
                            .iter()
                            .map(|&v| v as i32)
                            .sum::<i32>(),
                    expected[j]
                );
            }
        }
        std::hint::black_box(expected);
    }
    #[test]
    fn fused_kernel_rounding_is_distinct() {
        let x = [0.0, 255.0, 0.5, 1.5, 2.5];
        assert_eq!(Quantized::new(&x).data, [0, 255, 0, 2, 2]);
        assert_eq!(Quantized::fused(&x).data, [0, 255, 1, 2, 3]);
    }
    #[test]
    fn quantization_ties_and_zero() {
        let q = Quantized::new(&[0.0, 255.0, 0.5, 1.5, 2.5]);
        assert_eq!(q.data, [0, 255, 0, 2, 2]);
        let q = Quantized::new(&[0.0; 16]);
        assert_eq!((q.scale, q.zero), (1.0, 0));
    }
}
