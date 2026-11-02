//! Standalone cost probe, NOT an encoder/segnet implementation.
//! rustc --edition=2021 --crate-type=cdylib --target wasm32-unknown-unknown -O
//!   -C target-feature=+simd128 bench/float_kernel.rs -o /scratch/strict.wasm
//! Add --cfg fused and +relaxed-simd for the conditional hardware-FMA variant.
//! Two rows x sixteen columns, K ordered as native; no reassociation.
use std::arch::wasm32::*;

#[cfg(narrow)]
const COLUMNS: usize = 8;
#[cfg(not(narrow))]
const COLUMNS: usize = 16;

#[no_mangle]
pub extern "C" fn columns() -> usize {
    COLUMNS
}

#[no_mangle]
pub extern "C" fn alloc(n: usize) -> *mut f32 {
    let mut values = vec![0.0f32; n];
    let ptr = values.as_mut_ptr();
    std::mem::forget(values);
    ptr
}

#[cfg(not(fused))]
#[inline(always)]
unsafe fn round32(x: v128) -> v128 {
    f64x2_promote_low_f32x4(f32x4_demote_f64x2_zero(x))
}

#[no_mangle]
pub unsafe extern "C" fn kernel(
    a: *const f32,
    b: *const f32,
    out: *mut f32,
    k: usize,
    reps: usize,
) {
    for _ in 0..reps {
        #[cfg(fused)]
        let mut accum = [f32x4_splat(0.0); COLUMNS / 2];
        #[cfg(not(fused))]
        let mut accum = [[f64x2_splat(0.0); 2]; COLUMNS / 2];
        for i in 0..k {
            #[cfg(fused)]
            let scalars = [f32x4_splat(*a.add(i)), f32x4_splat(*a.add(k + i))];
            #[cfg(not(fused))]
            let scalars = [
                f64x2_splat(*a.add(i) as f64),
                f64x2_splat(*a.add(k + i) as f64),
            ];
            for j in 0..COLUMNS / 4 {
                let weights = v128_load(b.add(i * 16 + j * 4).cast());
                #[cfg(not(fused))]
                let widened = [
                    f64x2_promote_low_f32x4(weights),
                    f64x2_promote_low_f32x4(i32x4_shuffle::<2, 3, 2, 3>(weights, weights)),
                ];
                for row in 0..2 {
                    let dst = accum.get_unchecked_mut(row * (COLUMNS / 4) + j);
                    #[cfg(fused)]
                    {
                        *dst = f32x4_relaxed_madd(weights, scalars[row], *dst);
                    }
                    #[cfg(not(fused))]
                    for lane in 0..2 {
                        dst[lane] =
                            round32(f64x2_add(f64x2_mul(widened[lane], scalars[row]), dst[lane]));
                    }
                }
            }
        }
        for (j, value) in accum.iter().enumerate() {
            #[cfg(fused)]
            let packed = *value;
            #[cfg(not(fused))]
            let packed = i32x4_shuffle::<0, 1, 4, 5>(
                f32x4_demote_f64x2_zero(value[0]),
                f32x4_demote_f64x2_zero(value[1]),
            );
            v128_store(out.add(j * 4).cast(), packed);
        }
    }
}
