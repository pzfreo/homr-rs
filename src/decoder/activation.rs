//! MLAS's float32 polynomial approximations and reduction order. Using the
//! platform libm instead introduces quantization-boundary changes downstream.
//! Arithmetic and coefficients follow ONNX Runtime's MIT-licensed MLAS
//! compute.cpp, logistic.cpp, erf.cpp and gelu.cpp (Microsoft Corporation).
#![allow(clippy::excessive_precision)]
#[inline]
pub(super) fn fma(a: f32, b: f32, c: f32) -> f32 {
    // SIMD128 has no fused float32 multiply-add. Widening preserves the
    // float32 product, then narrows once; corpus checks bound any double rounding.
    #[cfg(target_arch = "wasm32")]
    {
        (a as f64 * b as f64 + c as f64) as f32
    }
    #[cfg(not(target_arch = "wasm32"))]
    {
        a.mul_add(b, c)
    }
}
fn horner(x: f32, coefficients: &[f32]) -> f32 {
    coefficients[1..]
        .iter()
        .fold(coefficients[0], |p, &c| fma(p, x, c))
}
pub(super) fn sigmoid(x: f32) -> f32 {
    let x = x.clamp(-18.0, 18.0);
    let x2 = x * x;
    let p = horner(
        x2,
        &[
            4.37031012579801e-11,
            1.15627324459942e-7,
            6.08574864600143e-5,
            8.51377133304701e-3,
            2.48287947061529e-1,
        ],
    ) * x;
    let q = horner(
        x2,
        &[
            6.10247389755681e-13,
            5.76102136993427e-9,
            6.29106785017040e-6,
            1.70198817374094e-3,
            1.16817656904453e-1,
            9.93151921023180e-1,
        ],
    );
    (p / q + 0.5).clamp(0.0, 1.0)
}
fn exp_reduced(x: f32, coefficients: &[f32]) -> f32 {
    let biased = fma(x, std::f32::consts::LOG2_E, 12582912.0);
    let m = biased - 12582912.0;
    let r = fma(m, -6.93145752e-1, x);
    let r = fma(m, -1.42860677e-6, r);
    let scale = f32::from_bits(biased.to_bits().wrapping_shl(23).wrapping_add(0x3f800000));
    horner(r, coefficients) * scale
}
fn exp(x: f32) -> f32 {
    exp_reduced(
        x.max(-88.3762626647949),
        &[
            0.0013780593872070312,
            0.008373124524950981,
            0.041669536381959915,
            0.166664719581604,
            0.49999985098838806,
            1.0,
            1.0,
        ],
    )
}
pub(super) fn gelu(x: f32) -> f32 {
    let v = x * std::f32::consts::FRAC_1_SQRT_2;
    let a = v.abs().min(3.925);
    let r = if a <= 0.921875 {
        let p = horner(
            a * a,
            &[
                -5.99104969e-4,
                4.99339588e-3,
                -2.67667342e-2,
                1.12818025e-1,
                -3.76124859e-1,
                1.28379151e-1,
            ],
        );
        fma(p, a, a)
    } else {
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
        let p = -fma(p, a, a);
        1.0 - exp_reduced(
            p.max(-88.3762626647949),
            &[
                1.38319808e-3,
                8.37550033e-3,
                4.16689515e-2,
                1.66664466e-1,
                4.99999851e-1,
                1.0,
                1.0,
            ],
        )
    };
    (0.5 * x) * (r.copysign(v) + 1.0)
}
pub(super) fn softmax(x: &mut [f32]) {
    let max = x.iter().copied().fold(f32::NEG_INFINITY, f32::max);
    let mut lanes = [0.0f32; 4];
    let bulk = x.len() / 4 * 4;
    for chunk in x[..bulk].as_chunks_mut::<4>().0 {
        for (s, v) in lanes.iter_mut().zip(chunk) {
            *v = exp(*v - max);
            *s += *v;
        }
    }
    let mut sum = (lanes[0] + lanes[1]) + (lanes[2] + lanes[3]);
    for v in &mut x[bulk..] {
        *v = exp(*v - max);
        sum += *v;
    }
    let inv = 1.0 / sum;
    for v in x {
        *v *= inv;
    }
}
