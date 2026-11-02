//! Exact integer dot products. Weights are transposed once at model load.
use super::onnx::Graph;
use super::profile::{enter, Phase};

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
        if g.tensor(&format!("{stem}_zero_point"))?.data != [0] {
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
                    let dots =
                        unsafe { wasm_dot4(a, &self.weights[col * self.k..(col + 4) * self.k]) };
                    for (j, v) in values.iter_mut().enumerate() {
                        *v = (dots[j] - q.zero * self.sums[col + j]) as f32 * scale
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
#[cfg(all(target_arch = "wasm32", target_feature = "simd128"))]
unsafe fn wasm_dot4(a: &[u8], b: &[i8]) -> [i32; 4] {
    use core::arch::wasm32::*;
    let mut sums = [i32x4_splat(0); 4];
    for i in (0..a.len()).step_by(16) {
        let av = v128_load(a.as_ptr().add(i).cast());
        let lo = u16x8_extend_low_u8x16(av);
        let hi = u16x8_extend_high_u8x16(av);
        for (j, s) in sums.iter_mut().enumerate() {
            let bv = v128_load(b.as_ptr().add(j * a.len() + i).cast());
            *s = i32x4_add(*s, i32x4_dot_i16x8(lo, i16x8_extend_low_i8x16(bv)));
            *s = i32x4_add(*s, i32x4_dot_i16x8(hi, i16x8_extend_high_i8x16(bv)));
        }
    }
    sums.map(|s| {
        i32x4_extract_lane::<0>(s)
            + i32x4_extract_lane::<1>(s)
            + i32x4_extract_lane::<2>(s)
            + i32x4_extract_lane::<3>(s)
    })
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
#[cfg(test)]
mod tests {
    use super::*;
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
            assert_eq!(wasm_dot4(&a, &b), expected);
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
