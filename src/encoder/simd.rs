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
