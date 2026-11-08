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

// np.argsort(a) (kind="quicksort", the default) of float64 values: numpy's
// introsort of indices (npysort/quicksort.cpp, aquicksort), not a stable sort:
// among equal values the order depends on how the partitioning moved them. That
// decides which of several equal peaks homr's find_peaks keeps. NaN sorts last.

fn less(a: f64, b: f64) -> bool {
    a < b || (b.is_nan() && !a.is_nan())
}

/// A range of more than 16 elements (pr - pl >= 16, both ends inclusive) is partitioned;
/// shorter ones are insertion sorted. Matches numpy 2.4 (Pyodide) and 2.5 on every case of
/// tools/argsort_ref.mjs.
const SMALL_QUICKSORT: usize = 15;

/// The heapsort numpy falls back to when the quicksort recursion gets too deep.
fn aheapsort(v: &[f64], tosort: &mut [usize]) {
    let n = tosort.len();
    // 1-based indexing as numpy's: a[k] is tosort[k - 1]
    let at = |t: &[usize], k: usize| t[k - 1];
    let mut n_ = n;
    let mut l = n_ >> 1;
    while l > 0 {
        let tmp = at(tosort, l);
        let (mut i, mut j) = (l, l << 1);
        while j <= n_ {
            if j < n_ && less(v[at(tosort, j)], v[at(tosort, j + 1)]) {
                j += 1;
            }
            if less(v[tmp], v[at(tosort, j)]) {
                tosort[i - 1] = at(tosort, j);
                i = j;
                j += j;
            } else {
                break;
            }
        }
        tosort[i - 1] = tmp;
        l -= 1;
    }
    while n_ > 1 {
        let tmp = at(tosort, n_);
        tosort[n_ - 1] = tosort[0];
        n_ -= 1;
        let (mut i, mut j) = (1, 2);
        while j <= n_ {
            if j < n_ && less(v[at(tosort, j)], v[at(tosort, j + 1)]) {
                j += 1;
            }
            if less(v[tmp], v[at(tosort, j)]) {
                tosort[i - 1] = at(tosort, j);
                i = j;
                j += j;
            } else {
                break;
            }
        }
        tosort[i - 1] = tmp;
    }
}

pub fn np_argsort(v: &[f64]) -> Vec<usize> {
    let num = v.len();
    let mut tosort: Vec<usize> = (0..num).collect();
    if num < 2 {
        return tosort;
    }
    let mut pl: usize = 0;
    let mut pr: usize = num - 1;
    let mut stack: Vec<(usize, usize, i32)> = Vec::new(); // (pl, pr, depth) of a pending partition
    let mut cdepth: i32 = (usize::BITS - 1 - num.leading_zeros()) as i32 * 2;
    loop {
        let mut heap_sorted = false;
        if cdepth < 0 {
            aheapsort(v, &mut tosort[pl..=pr]);
            heap_sorted = true;
        }
        if !heap_sorted {
            while pr - pl > SMALL_QUICKSORT {
                // quicksort partition
                let pm = pl + ((pr - pl) >> 1);
                if less(v[tosort[pm]], v[tosort[pl]]) {
                    tosort.swap(pm, pl);
                }
                if less(v[tosort[pr]], v[tosort[pm]]) {
                    tosort.swap(pr, pm);
                }
                if less(v[tosort[pm]], v[tosort[pl]]) {
                    tosort.swap(pm, pl);
                }
                let vp = v[tosort[pm]];
                let mut pi = pl;
                let mut pj = pr - 1;
                tosort.swap(pm, pj);
                loop {
                    loop {
                        pi += 1;
                        if !less(v[tosort[pi]], vp) {
                            break;
                        }
                    }
                    loop {
                        pj -= 1;
                        if !less(vp, v[tosort[pj]]) {
                            break;
                        }
                    }
                    if pi >= pj {
                        break;
                    }
                    tosort.swap(pi, pj);
                }
                let pk = pr - 1;
                tosort.swap(pi, pk);
                // push largest partition on stack
                cdepth -= 1;
                if pi - pl < pr - pi {
                    stack.push((pi + 1, pr, cdepth));
                    pr = pi - 1;
                } else {
                    stack.push((pl, pi - 1, cdepth));
                    pl = pi + 1;
                }
            }
            // insertion sort
            for pi in pl + 1..=pr {
                let vi = tosort[pi];
                let vp = v[vi];
                let mut pj = pi;
                while pj > pl && less(vp, v[tosort[pj - 1]]) {
                    tosort[pj] = tosort[pj - 1];
                    pj -= 1;
                }
                tosort[pj] = vi;
            }
        }
        match stack.pop() {
            None => break,
            Some((l, r, d)) => {
                pl = l;
                pr = r;
                cdepth = d;
            }
        }
    }
    tosort
}
