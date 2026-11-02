//! Native model of the WGSL refinement/certificate, checked against hardware
//! IEEE division/sqrt, rather than stored answers. Not an alternate CPU encoder.
use crate::encoder::kernels::fma;

fn correction(t: u64, lo: u64, hi: u64, m: u32) -> i32 {
    if t < lo || (t == lo && m & 1 != 0) {
        -1
    } else if t > hi || (t == hi && m & 1 != 0) {
        1
    } else {
        0
    }
}
fn assemble(mut m: u32, mut e: i32, sign: u32) -> Option<f32> {
    if m == 1 << 24 {
        m = 1 << 23;
        e += 1;
    }
    if !(1 << 23..1 << 24).contains(&m) || !(-126..=127).contains(&e) {
        return None;
    }
    Some(f32::from_bits(
        sign | ((e + 127) as u32) << 23 | (m & 0x7fffff),
    ))
}
fn div(a: f32, b: f32) -> Option<f32> {
    let (aa, bb) = (a.to_bits() & 0x7fffffff, b.to_bits() & 0x7fffffff);
    let sign = (a.to_bits() ^ b.to_bits()) & 0x80000000;
    if aa == 0 {
        return Some(f32::from_bits(sign));
    }
    let (ma, mb) = ((aa & 0x7fffff) | 0x800000, (bb & 0x7fffff) | 0x800000);
    let an = f32::from_bits(0x3f800000 | (aa & 0x7fffff));
    let bn = f32::from_bits(0x3f800000 | (bb & 0x7fffff));
    let mut r = f32::from_bits(0x7ef311c3 - bn.to_bits());
    for _ in 0..4 {
        r = fma(r, fma(-bn, r, 1.0), r);
    }
    let q0 = an * r;
    let q = fma(fma(-q0, bn, an), r, q0);
    let e = if ma >= mb { 0 } else { -1 };
    let qb = q
        .to_bits()
        .clamp(((127 + e) as u32) << 23, ((128 + e) as u32) << 23);
    let mut m = ((qb & 0x7fffff) | 0x800000) << ((qb >> 23) as i32 - 127 - e);
    let t = (ma as u64) << (24 - e);
    let delta = correction(
        t,
        (2 * m - 1) as u64 * mb as u64,
        (2 * m + 1) as u64 * mb as u64,
        m,
    );
    m = (m as i32 + delta) as u32;
    if correction(
        t,
        (2 * m - 1) as u64 * mb as u64,
        (2 * m + 1) as u64 * mb as u64,
        m,
    ) != 0
    {
        return None;
    }
    let exponent = e + (aa >> 23) as i32 - (bb >> 23) as i32;
    if exponent == -127 && t >= 0x1fffffeu64 * mb as u64 {
        return Some(f32::from_bits(sign | 0x800000));
    }
    assemble(m, exponent, sign)
}
fn sqrt(x: f32) -> Option<f32> {
    if x == 0.0 {
        return Some(x);
    }
    let b = x.to_bits();
    let e = (b >> 23) as i32 - 127;
    let parity = e & 1;
    let an = f32::from_bits(((127 + parity) as u32) << 23 | (b & 0x7fffff));
    let mut r = f32::from_bits(0x5f375a86 - (an.to_bits() >> 1));
    for _ in 0..4 {
        r *= fma(-0.5 * an, r * r, 1.5);
    }
    let q0 = an * r;
    let q = fma(fma(-q0, q0, an), 0.5 * r, q0);
    let qb = q.to_bits().clamp(0x3f800000, 0x40000000);
    let mut m = ((qb & 0x7fffff) | 0x800000) << ((qb >> 23) - 127);
    let t = (((b & 0x7fffff) | 0x800000) as u64) << (25 + parity);
    let square = |n: u32| n as u64 * n as u64;
    let delta = correction(t, square(2 * m - 1), square(2 * m + 1), m);
    m = (m as i32 + delta) as u32;
    if correction(t, square(2 * m - 1), square(2 * m + 1), m) != 0 {
        return None;
    }
    assemble(m, (e - parity) / 2, 0)
}
fn random(state: &mut u32) -> u32 {
    *state ^= *state << 13;
    *state ^= *state >> 17;
    *state ^= *state << 5;
    *state
}
#[test]
fn normal_refinement_certificates_match_ieee_on_random_and_bin_edges() {
    let mut seed = 0x46b8a4da;
    let mut divisions = 0;
    for _ in 0..1_000_000 {
        let a = f32::from_bits(
            (random(&mut seed) & 0x807fffff) | ((random(&mut seed) % 254 + 1) << 23),
        );
        let b = f32::from_bits(
            (random(&mut seed) & 0x807fffff) | ((random(&mut seed) % 254 + 1) << 23),
        );
        let expected = a / b;
        if expected.is_normal() {
            assert_eq!(
                div(a, b).map(f32::to_bits),
                Some(expected.to_bits()),
                "{a:?}/{b:?}"
            );
            divisions += 1;
        }
        let x = a.abs();
        assert_eq!(
            sqrt(x).map(f32::to_bits),
            Some(x.sqrt().to_bits()),
            "sqrt {x:?}"
        );
    }
    assert!(divisions > 700_000);
    // Powers of two, adjacent bin boundaries, halfway/even quotient cases,
    // every normal exponent, both signs, and integer Welford denominators.
    for exponent in 1..=254 {
        for fraction in [0, 1, 2, 0x3fffff, 0x400000, 0x7ffffe, 0x7fffff] {
            let a = f32::from_bits(exponent << 23 | fraction);
            assert_eq!(sqrt(a).map(f32::to_bits), Some(a.sqrt().to_bits()));
            for denominator in [
                1.0,
                2.0,
                3.0,
                7.0,
                96.0,
                192.0,
                384.0,
                f32::from_bits(0x3f800001),
            ] {
                for a in [a, -a] {
                    let expected = a / denominator;
                    if expected.is_normal() {
                        assert_eq!(
                            div(a, denominator).map(f32::to_bits),
                            Some(expected.to_bits())
                        );
                    }
                }
            }
        }
    }
    assert_eq!(sqrt(-0.0).unwrap().to_bits(), (-0.0f32).to_bits());
}
#[test]
fn wgsl_wide_multiply_matches_u64_without_overflow() {
    let mut seed = 0x465;
    for _ in 0..1_000_000 {
        let a = random(&mut seed);
        let b = random(&mut seed);
        let (a0, a1, b0, b1) = (a & 65535, a >> 16, b & 65535, b >> 16);
        let t0 = a0 * b0;
        let t1 = a1 * b0 + (t0 >> 16);
        let t2 = a0 * b1 + (t1 & 65535);
        let lo = (t2 << 16) | (t0 & 65535);
        let hi = a1 * b1 + (t1 >> 16) + (t2 >> 16);
        assert_eq!((hi as u64) << 32 | lo as u64, a as u64 * b as u64);
    }
}
