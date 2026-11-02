//! Four rows by sixteen columns, preserving ascending-K fused accumulation.
#[cfg(target_arch = "aarch64")]
use std::arch::aarch64::*;
#[cfg(target_arch = "wasm32")]
use std::arch::wasm32::*;
#[cfg(target_arch = "aarch64")]
type V = float32x4_t;
#[cfg(target_arch = "wasm32")]
type V = v128;
#[inline(always)]
unsafe fn zero() -> V {
    #[cfg(target_arch = "aarch64")]
    {
        vdupq_n_f32(0.0)
    }
    #[cfg(target_arch = "wasm32")]
    {
        f32x4_splat(0.0)
    }
}
#[inline(always)]
unsafe fn load(p: *const f32) -> V {
    #[cfg(target_arch = "aarch64")]
    {
        vld1q_f32(p)
    }
    #[cfg(target_arch = "wasm32")]
    {
        v128_load(p.cast())
    }
}
#[inline(always)]
unsafe fn store(p: *mut f32, v: V) {
    #[cfg(target_arch = "aarch64")]
    {
        vst1q_f32(p, v)
    }
    #[cfg(target_arch = "wasm32")]
    {
        v128_store(p.cast(), v)
    }
}
#[inline(always)]
unsafe fn madd(a: f32, b: V, c: V) -> V {
    #[cfg(target_arch = "aarch64")]
    {
        vfmaq_n_f32(c, b, a)
    }
    #[cfg(all(target_arch = "wasm32", target_feature = "relaxed-simd"))]
    {
        f32x4_relaxed_madd(f32x4_splat(a), b, c)
    }
    #[cfg(all(target_arch = "wasm32", not(target_feature = "relaxed-simd")))]
    {
        let scale = f64x2_splat(a as f64);
        let lo = f64x2_add(
            f64x2_mul(scale, f64x2_promote_low_f32x4(b)),
            f64x2_promote_low_f32x4(c),
        );
        let hi = f64x2_add(
            f64x2_mul(
                scale,
                f64x2_promote_low_f32x4(i32x4_shuffle::<2, 3, 2, 3>(b, b)),
            ),
            f64x2_promote_low_f32x4(i32x4_shuffle::<2, 3, 2, 3>(c, c)),
        );
        i32x4_shuffle::<0, 1, 4, 5>(f32x4_demote_f64x2_zero(lo), f32x4_demote_f64x2_zero(hi))
    }
}
#[inline(always)]
unsafe fn splat(x: f32) -> V {
    #[cfg(target_arch = "aarch64")]
    {
        vdupq_n_f32(x)
    }
    #[cfg(target_arch = "wasm32")]
    {
        f32x4_splat(x)
    }
}
macro_rules! binary {
    ($name:ident,$neon:ident,$wasm:ident) => {
        #[inline(always)]
        unsafe fn $name(a: V, b: V) -> V {
            #[cfg(target_arch = "aarch64")]
            {
                $neon(a, b)
            }
            #[cfg(target_arch = "wasm32")]
            {
                $wasm(a, b)
            }
        }
    };
}
binary!(add, vaddq_f32, f32x4_add);
binary!(sub, vsubq_f32, f32x4_sub);
binary!(mul, vmulq_f32, f32x4_mul);
binary!(min, vminq_f32, f32x4_min);
binary!(max, vmaxq_f32, f32x4_max);
#[inline(always)]
unsafe fn maddv(a: V, b: V, c: V) -> V {
    #[cfg(target_arch = "aarch64")]
    {
        vfmaq_f32(c, a, b)
    }
    #[cfg(all(target_arch = "wasm32", target_feature = "relaxed-simd"))]
    {
        f32x4_relaxed_madd(a, b, c)
    }
    #[cfg(all(target_arch = "wasm32", not(target_feature = "relaxed-simd")))]
    {
        let high = |v| i32x4_shuffle::<2, 3, 2, 3>(v, v);
        let lo = f64x2_add(
            f64x2_mul(f64x2_promote_low_f32x4(a), f64x2_promote_low_f32x4(b)),
            f64x2_promote_low_f32x4(c),
        );
        let hi = f64x2_add(
            f64x2_mul(
                f64x2_promote_low_f32x4(high(a)),
                f64x2_promote_low_f32x4(high(b)),
            ),
            f64x2_promote_low_f32x4(high(c)),
        );
        i32x4_shuffle::<0, 1, 4, 5>(f32x4_demote_f64x2_zero(lo), f32x4_demote_f64x2_zero(hi))
    }
}
#[inline(always)]
unsafe fn horner(x: V, c: &[f32]) -> V {
    let mut p = splat(c[0]);
    for &v in &c[1..] {
        p = maddv(p, x, splat(v));
    }
    p
}
#[inline(always)]
unsafe fn erf_vec(v: V) -> V {
    #[cfg(target_arch = "aarch64")]
    let abs = vabsq_f32(v);
    #[cfg(target_arch = "wasm32")]
    let abs = f32x4_abs(v);
    let a = min(abs, splat(3.925));
    let small = maddv(
        horner(
            mul(a, a),
            &[
                -5.99104969e-4,
                4.99339588e-3,
                -2.67667342e-2,
                1.12818025e-1,
                -3.76124859e-1,
                1.28379151e-1,
            ],
        ),
        a,
        a,
    );
    let p = horner(
        a,
        &[
            1.72948930e-5,
            -3.83208680e-4,
            3.88393435e-3,
            -2.42545605e-2,
            1.06777847e-1,
            6.34846687e-1,
            1.28717512e-1,
        ],
    );
    let x = max(sub(zero(), maddv(p, a, a)), splat(-88.3762626647949));
    let biased = maddv(x, splat(std::f32::consts::LOG2_E), splat(12582912.0));
    let m = sub(biased, splat(12582912.0));
    let r = maddv(m, splat(-6.93145752e-1), x);
    let r = maddv(m, splat(-1.42860677e-6), r);
    #[cfg(target_arch = "aarch64")]
    let scale = vreinterpretq_f32_u32(vaddq_u32(
        vshlq_n_u32::<23>(vreinterpretq_u32_f32(biased)),
        vdupq_n_u32(0x3f800000),
    ));
    #[cfg(target_arch = "wasm32")]
    let scale = i32x4_add(i32x4_shl(biased, 23), i32x4_splat(0x3f800000));
    let exp = mul(
        horner(
            r,
            &[
                1.38319808e-3,
                8.37550033e-3,
                4.16689515e-2,
                1.66664466e-1,
                4.99999851e-1,
                1.0,
                1.0,
            ],
        ),
        scale,
    );
    let big = sub(splat(1.0), exp);
    #[cfg(target_arch = "aarch64")]
    {
        let result = vbslq_f32(vcgtq_f32(a, splat(0.921875)), big, small);
        vreinterpretq_f32_u32(vorrq_u32(
            vandq_u32(vreinterpretq_u32_f32(v), vdupq_n_u32(0x80000000)),
            vreinterpretq_u32_f32(result),
        ))
    }
    #[cfg(target_arch = "wasm32")]
    {
        let result = v128_bitselect(big, small, f32x4_gt(a, splat(0.921875)));
        v128_or(v128_and(v, i32x4_splat(i32::MIN)), result)
    }
}
pub(super) unsafe fn activation(x: &mut [f32], gelu: bool) {
    let n = x.len() / 4 * 4;
    for i in (0..n).step_by(4) {
        let value = load(x.as_ptr().add(i));
        let result = if gelu {
            mul(
                mul(splat(0.5), value),
                add(
                    erf_vec(mul(value, splat(std::f32::consts::FRAC_1_SQRT_2))),
                    splat(1.0),
                ),
            )
        } else {
            erf_vec(value)
        };
        store(x.as_mut_ptr().add(i), result);
    }
    for v in &mut x[n..] {
        *v = if gelu {
            super::kernels::gelu(*v)
        } else {
            super::kernels::erf(*v)
        };
    }
}

pub(super) unsafe fn depthwise(
    x: &[f32],
    h: usize,
    w: usize,
    c: usize,
    weights: &[f32],
    bias: &[f32],
    kernel: usize,
    pad: usize,
) -> Vec<f32> {
    let pw = w + 2 * pad;
    let mut padded = vec![0.0; (h + 2 * pad) * pw * c];
    for y in 0..h {
        padded[((y + pad) * pw + pad) * c..((y + pad) * pw + pad + w) * c]
            .copy_from_slice(&x[y * w * c..(y + 1) * w * c]);
    }
    let mut packed = vec![0.0; weights.len()];
    for ch in 0..c {
        for k in 0..kernel * kernel {
            packed[k * c + ch] = weights[ch * kernel * kernel + k];
        }
    }
    let mut out = vec![0.0; x.len()];
    for y in 0..h {
        for xx in 0..w {
            for ch in (0..c).step_by(4) {
                let mut sum = zero();
                for ky in 0..kernel {
                    for kx in 0..kernel {
                        sum = maddv(
                            load(padded.as_ptr().add(((y + ky) * pw + xx + kx) * c + ch)),
                            load(packed.as_ptr().add((ky * kernel + kx) * c + ch)),
                            sum,
                        );
                    }
                }
                store(
                    out.as_mut_ptr().add((y * w + xx) * c + ch),
                    add(sum, load(bias.as_ptr().add(ch))),
                );
            }
        }
    }
    out
}
pub(super) unsafe fn product(
    a: &[f32],
    b: &[f32],
    out: &mut [f32],
    m: usize,
    k: usize,
    n: usize,
    bias: Option<&[f32]>,
) {
    for row in (0..m).step_by(4) {
        for col in (0..n).step_by(16) {
            let mut acc = [zero(); 16];
            if let Some(bias) = bias {
                for r in 0..4 {
                    for j in 0..4 {
                        *acc.get_unchecked_mut(r * 4 + j) = load(bias.as_ptr().add(col + j * 4));
                    }
                }
            }
            for p in 0..k {
                for j in 0..4 {
                    let weights = load(b.as_ptr().add(p * n + col + j * 4));
                    for r in 0..4 {
                        let dst = acc.get_unchecked_mut(r * 4 + j);
                        *dst = madd(*a.get_unchecked((row + r) * k + p), weights, *dst);
                    }
                }
            }
            for r in 0..4 {
                for j in 0..4 {
                    store(
                        out.as_mut_ptr().add((row + r) * n + col + j * 4),
                        *acc.get_unchecked(r * 4 + j),
                    );
                }
            }
        }
    }
}
