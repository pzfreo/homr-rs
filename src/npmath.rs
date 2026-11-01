//! Numeric helpers that reproduce Python/numpy results bit for bit, so the
//! port can be compared exactly against homr.

/// numpy's pairwise summation (numpy/core/src/umath/loops_utils.h.src).
pub fn np_sum(a: &[f64]) -> f64 {
    let n = a.len();
    if n < 8 {
        let mut res = 0.0;
        for &v in a {
            res += v;
        }
        res
    } else if n <= 128 {
        let mut r = [0.0f64; 8];
        r.copy_from_slice(&a[..8]);
        let mut i = 8;
        while i < n - (n % 8) {
            for j in 0..8 {
                r[j] += a[i + j];
            }
            i += 8;
        }
        let mut res = ((r[0] + r[1]) + (r[2] + r[3])) + ((r[4] + r[5]) + (r[6] + r[7]));
        while i < n {
            res += a[i];
            i += 1;
        }
        res
    } else {
        let mut n2 = n / 2;
        n2 -= n2 % 8;
        np_sum(&a[..n2]) + np_sum(&a[n2..])
    }
}

pub fn np_mean(a: &[f64]) -> f64 {
    np_sum(a) / a.len() as f64
}

/// np.std (population, ddof=0).
pub fn np_std(a: &[f64]) -> f64 {
    let m = np_mean(a);
    let sq: Vec<f64> = a.iter().map(|&v| (v - m) * (v - m)).collect();
    (np_sum(&sq) / a.len() as f64).sqrt()
}

/// Python's round(): halves go to the even neighbour.
pub fn py_round(v: f64) -> i64 {
    v.round_ties_even() as i64
}

/// Python's int(float): truncates toward zero.
pub fn py_int(v: f64) -> i64 {
    v.trunc() as i64
}

/// Python's float // int for a positive divisor.
pub fn py_floordiv(v: f64, d: f64) -> f64 {
    (v / d).floor()
}

/// Python's range(start, stop, step) for a positive step.
pub fn py_range(start: i64, stop: i64, step: i64) -> Vec<i64> {
    let mut v = Vec::new();
    let mut x = start;
    while x < stop {
        v.push(x);
        x += step;
    }
    v
}
