//! cv::fitEllipse (OpenCV 4.11, modules/imgproc/src/shapedescr.cpp).
//!
//! The structure, scaling, float32 steps and quirks follow OpenCV (e.g. the
//! angle stays 0 unless width > height). The linear algebra does not: OpenCV
//! solves its least-squares fits with its Jacobi SVD, here a Householder QR
//! gives the same solution up to rounding. homr only uses the fit through
//! integer-truncated outlines and size filters; whether that is exact enough
//! is checked on homr's outputs (see LOG.md).

const FLT_EPSILON: f64 = f32::EPSILON as f64;

/// Least squares min |A x - b| by Householder QR (A is rows x cols, row major).
fn lstsq(a: &[f64], rows: usize, cols: usize, b: &[f64]) -> Vec<f64> {
    let mut a = a.to_vec();
    let mut b = b.to_vec();
    for k in 0..cols {
        let norm = (k..rows).map(|i| a[i * cols + k].powi(2)).sum::<f64>().sqrt();
        if norm == 0.0 {
            continue;
        }
        let alpha = if a[k * cols + k] > 0.0 { -norm } else { norm };
        let mut v: Vec<f64> = (k..rows).map(|i| a[i * cols + k]).collect();
        v[0] -= alpha;
        let vnorm2: f64 = v.iter().map(|x| x * x).sum();
        if vnorm2 == 0.0 {
            continue;
        }
        for j in k..cols {
            let dot: f64 = (k..rows).map(|i| v[i - k] * a[i * cols + j]).sum();
            let f = 2.0 * dot / vnorm2;
            for i in k..rows {
                a[i * cols + j] -= f * v[i - k];
            }
        }
        let dot: f64 = (k..rows).map(|i| v[i - k] * b[i]).sum();
        let f = 2.0 * dot / vnorm2;
        for i in k..rows {
            b[i] -= f * v[i - k];
        }
    }
    let mut x = vec![0.0; cols];
    for k in (0..cols).rev() {
        let s: f64 = (k + 1..cols).map(|j| a[k * cols + j] * x[j]).sum();
        let d = a[k * cols + k];
        x[k] = if d.abs() > 1e-300 { (b[k] - s) / d } else { 0.0 };
    }
    x
}

/// Singular values of A (rows x 5), largest first: square roots of A^T A's eigenvalues (Jacobi).
fn singular_values(a: &[f64], rows: usize, cols: usize) -> Vec<f64> {
    let mut m = vec![0.0; cols * cols];
    for i in 0..cols {
        for j in 0..cols {
            m[i * cols + j] = (0..rows).map(|r| a[r * cols + i] * a[r * cols + j]).sum();
        }
    }
    for _ in 0..100 {
        let mut off = 0.0;
        for p in 0..cols {
            for q in p + 1..cols {
                off += m[p * cols + q].powi(2);
            }
        }
        if off < 1e-30 {
            break;
        }
        for p in 0..cols {
            for q in p + 1..cols {
                let apq = m[p * cols + q];
                if apq.abs() < 1e-300 {
                    continue;
                }
                let theta = (m[q * cols + q] - m[p * cols + p]) / (2.0 * apq);
                let t = theta.signum() / (theta.abs() + (theta * theta + 1.0).sqrt());
                let t = if theta == 0.0 { 1.0 } else { t };
                let c = 1.0 / (t * t + 1.0).sqrt();
                let s = t * c;
                for k in 0..cols {
                    let (mkp, mkq) = (m[k * cols + p], m[k * cols + q]);
                    m[k * cols + p] = c * mkp - s * mkq;
                    m[k * cols + q] = s * mkp + c * mkq;
                }
                for k in 0..cols {
                    let (mpk, mqk) = (m[p * cols + k], m[q * cols + k]);
                    m[p * cols + k] = c * mpk - s * mqk;
                    m[q * cols + k] = s * mpk + c * mqk;
                }
            }
        }
    }
    let mut sv: Vec<f64> = (0..cols).map(|i| m[i * cols + i].max(0.0).sqrt()).collect();
    sv.sort_by(|a, b| b.partial_cmp(a).unwrap());
    sv
}

fn ofs(i: usize, eps: f32) -> [f32; 2] {
    [((i & 1) as i32 * 2 - 1) as f32 * eps, ((i & 2) as i32 - 1) as f32 * eps]
}

/// (cx, cy, w, h, angle) as OpenCV returns them (float32), for a contour of >= 5 points.
pub fn fit_ellipse(points: &[[i32; 2]]) -> (f32, f32, f32, f32, f32) {
    if points.len() == 5 {
        if let Some(r) = fit_direct(points) {
            return r;
        }
    }
    fit_no_direct(points)
}

fn fit_no_direct(points: &[[i32; 2]]) -> (f32, f32, f32, f32, f32) {
    let n = points.len();
    let mut pts: Vec<[f32; 2]> = points.iter().map(|p| [p[0] as f32, p[1] as f32]).collect();
    let mut c = [0f32; 2];
    for p in &pts {
        c[0] += p[0];
        c[1] += p[1];
    }
    c[0] /= n as f32;
    c[1] /= n as f32;
    let mut s = 0f64;
    for p in &pts {
        s += ((p[0] - c[0]).abs()) as f64 + ((p[1] - c[1]).abs()) as f64;
    }
    let scale = 100.0 / if s > FLT_EPSILON { s } else { FLT_EPSILON };
    let design = |pts: &[[f32; 2]]| -> Vec<f64> {
        let mut a = vec![0.0; n * 5];
        for (i, p) in pts.iter().enumerate() {
            let px = (p[0] - c[0]) as f64 * scale;
            let py = (p[1] - c[1]) as f64 * scale;
            a[i * 5..i * 5 + 5].copy_from_slice(&[-px * px, -py * py, -px * py, px, py]);
        }
        a
    };
    let mut a = design(&pts);
    let sv = singular_values(&a, n, 5);
    if sv[0] * FLT_EPSILON > sv[4] {
        let eps = (s / (n * 2) as f64 * 1e-3) as f32;
        for (i, p) in pts.iter_mut().enumerate() {
            let o = ofs(i, eps);
            *p = [p[0] + o[0], p[1] + o[1]];
        }
        a = design(&pts);
    }
    let gfp = lstsq(&a, n, 5, &vec![10000.0; n]);
    // centre: [[2a, c], [c, 2b]] x = [d, e]
    let (m00, m01, m11) = (2.0 * gfp[0], gfp[2], 2.0 * gfp[1]);
    let det = m00 * m11 - m01 * m01;
    let rp0 = if det != 0.0 { (gfp[3] * m11 - m01 * gfp[4]) / det } else { 0.0 };
    let rp1 = if det != 0.0 { (m00 * gfp[4] - m01 * gfp[3]) / det } else { 0.0 };
    let mut a3 = vec![0.0; n * 3];
    for (i, p) in pts.iter().enumerate() {
        let px = (p[0] - c[0]) as f64 * scale;
        let py = (p[1] - c[1]) as f64 * scale;
        a3[i * 3..i * 3 + 3].copy_from_slice(&[(px - rp0) * (px - rp0), (py - rp1) * (py - rp1), (px - rp0) * (py - rp1)]);
    }
    let g = lstsq(&a3, n, 3, &vec![1.0; n]);
    let rp4 = -0.5 * g[2].atan2(g[1] - g[0]);
    let t = if g[2].abs() > 1e-8 { g[2] / (-2.0 * rp4).sin() } else { g[1] - g[0] };
    let mut rp2 = (g[0] + g[1] - t).abs();
    if rp2 > 1e-8 {
        rp2 = (2.0 / rp2).sqrt();
    }
    let mut rp3 = (g[0] + g[1] + t).abs();
    if rp3 > 1e-8 {
        rp3 = (2.0 / rp3).sqrt();
    }
    let cx = (rp0 / scale) as f32 + c[0];
    let cy = (rp1 / scale) as f32 + c[1];
    let mut w = (rp2 * 2.0 / scale) as f32;
    let mut h = (rp3 * 2.0 / scale) as f32;
    let mut angle = 0f32;
    if w > h {
        std::mem::swap(&mut w, &mut h);
        angle = (90.0 + rp4 * 180.0 / std::f64::consts::PI) as f32;
    }
    if angle < -180.0 {
        angle += 360.0;
    }
    if angle > 360.0 {
        angle -= 360.0;
    }
    (cx, cy, w, h, angle)
}

fn det3(m: &[[f64; 3]; 3]) -> f64 {
    m[0][0] * (m[1][1] * m[2][2] - m[1][2] * m[2][1]) - m[0][1] * (m[1][0] * m[2][2] - m[1][2] * m[2][0]) + m[0][2] * (m[1][0] * m[2][1] - m[1][1] * m[2][0])
}

/// Real eigenvectors of a 3x3 matrix (None if it has complex eigenvalues).
fn eigvecs3(m: &[[f64; 3]; 3]) -> Option<Vec<[f64; 3]>> {
    let tr = m[0][0] + m[1][1] + m[2][2];
    let c2 = m[0][0] * m[1][1] - m[0][1] * m[1][0] + m[0][0] * m[2][2] - m[0][2] * m[2][0] + m[1][1] * m[2][2] - m[1][2] * m[2][1];
    let d = det3(m);
    // l^3 - tr l^2 + c2 l - d = 0, via the trigonometric method
    let p = c2 - tr * tr / 3.0;
    let q = -2.0 * tr.powi(3) / 27.0 + tr * c2 / 3.0 - d;
    let disc = (q / 2.0).powi(2) + (p / 3.0).powi(3);
    if disc > 1e-12 * (1.0 + p.abs().powi(3)) {
        return None;
    }
    let r = (-p / 3.0).max(0.0).sqrt();
    let phi = if r == 0.0 { 0.0 } else { ((3.0 * q) / (2.0 * p * r)).clamp(-1.0, 1.0).acos() / 3.0 };
    let mut out = Vec::new();
    for k in 0..3 {
        let l = 2.0 * r * (phi - 2.0 * std::f64::consts::PI * k as f64 / 3.0).cos() + tr / 3.0;
        let a = [[m[0][0] - l, m[0][1], m[0][2]], [m[1][0], m[1][1] - l, m[1][2]], [m[2][0], m[2][1], m[2][2] - l]];
        let cross = |u: [f64; 3], v: [f64; 3]| [u[1] * v[2] - u[2] * v[1], u[2] * v[0] - u[0] * v[2], u[0] * v[1] - u[1] * v[0]];
        let cands = [cross(a[0], a[1]), cross(a[0], a[2]), cross(a[1], a[2])];
        let best = cands.iter().cloned().max_by(|x, y| (x[0] * x[0] + x[1] * x[1] + x[2] * x[2]).partial_cmp(&(y[0] * y[0] + y[1] * y[1] + y[2] * y[2])).unwrap()).unwrap();
        out.push(best);
    }
    Some(out)
}

fn fit_direct(points: &[[i32; 2]]) -> Option<(f32, f32, f32, f32, f32)> {
    let n = points.len();
    let mut c = [0f64; 2];
    for p in points {
        c[0] += p[0] as f32 as f64;
        c[1] += p[1] as f32 as f64;
    }
    c[0] /= n as f64;
    c[1] /= n as f64;
    let mut s = 0f64;
    for p in points {
        s += (p[0] as f64 - c[0]).abs() + (p[1] as f64 - c[1]).abs();
    }
    let scale = 100.0 / if s > FLT_EPSILON { s } else { FLT_EPSILON };
    let mut eps = 0f32;
    for _iter in 0..2 {
        let mut dm = [[0f64; 6]; 6];
        let rows: Vec<[f64; 6]> = points
            .iter()
            .enumerate()
            .map(|(i, p)| {
                let o = ofs(i, eps);
                let px = ((p[0] as f32 + o[0]) as f64 - c[0]) * scale;
                let py = ((p[1] as f32 + o[1]) as f64 - c[1]) * scale;
                [px * px, px * py, py * py, px, py, 1.0]
            })
            .collect();
        for i in 0..6 {
            for j in 0..6 {
                dm[i][j] = rows.iter().map(|r| r[i] * r[j]).sum::<f64>() / n as f64;
            }
        }
        let d = |i: usize, j: usize| dm[i][j];
        let mut tm = [[0f64; 3]; 3];
        for k in 0..3 {
            tm[0][k] = d(k, 5) * d(3, 5) * d(4, 4) - d(k, 5) * d(3, 4) * d(4, 5) - d(k, 4) * d(3, 5) * d(5, 4) + d(k, 3) * d(4, 5) * d(5, 4) + d(k, 4) * d(3, 4) * d(5, 5) - d(k, 3) * d(4, 4) * d(5, 5);
            tm[1][k] = d(k, 5) * d(3, 3) * d(4, 5) - d(k, 5) * d(3, 5) * d(4, 3) + d(k, 4) * d(3, 5) * d(5, 3) - d(k, 3) * d(4, 5) * d(5, 3) - d(k, 4) * d(3, 3) * d(5, 5) + d(k, 3) * d(4, 3) * d(5, 5);
            tm[2][k] = d(k, 5) * d(3, 4) * d(4, 3) - d(k, 5) * d(3, 3) * d(4, 4) - d(k, 4) * d(3, 4) * d(5, 3) + d(k, 3) * d(4, 4) * d(5, 3) + d(k, 4) * d(3, 3) * d(5, 4) - d(k, 3) * d(4, 3) * d(5, 4);
        }
        let ts = -(d(3, 5) * d(4, 4) * d(5, 3)) + d(3, 4) * d(4, 5) * d(5, 3) + d(3, 5) * d(4, 3) * d(5, 4) - d(3, 3) * d(4, 5) * d(5, 4) - d(3, 4) * d(4, 3) * d(5, 5) + d(3, 3) * d(4, 4) * d(5, 5);
        let mut m = [[0f64; 3]; 3];
        for k in 0..3 {
            m[0][k] = (d(2, k) + (d(2, 3) * tm[0][k] + d(2, 4) * tm[1][k] + d(2, 5) * tm[2][k]) / ts) / 2.0;
            m[1][k] = -d(1, k) - (d(1, 3) * tm[0][k] + d(1, 4) * tm[1][k] + d(1, 5) * tm[2][k]) / ts;
            m[2][k] = (d(0, k) + (d(0, 3) * tm[0][k] + d(0, 4) * tm[1][k] + d(0, 5) * tm[2][k]) / ts) / 2.0;
        }
        if det3(&m).abs() <= 1.0e-10 {
            eps = (s / (n * 2) as f64 * 1e-2) as f32;
            continue;
        }
        let ev = eigvecs3(&m)?;
        let cond: Vec<f64> = ev.iter().map(|e| 4.0 * e[0] * e[2] - e[1] * e[1]).collect();
        let i = if cond[0] < cond[1] { if cond[1] < cond[2] { 2 } else { 1 } } else if cond[0] < cond[2] { 2 } else { 0 };
        let e = ev[i];
        let mut norm = (e[0] * e[0] + e[1] * e[1] + e[2] * e[2]).sqrt();
        let sg = |v: f64| if v < 0.0 { -1.0 } else { 1.0 };
        if sg(e[0]) * sg(e[1]) * sg(e[2]) <= 0.0 {
            norm = -norm;
        }
        let pv = [e[0] / norm, e[1] / norm, e[2] / norm];
        let q = [
            (tm[0][0] * pv[0] + tm[0][1] * pv[1] + tm[0][2] * pv[2]) / ts,
            (tm[1][0] * pv[0] + tm[1][1] * pv[1] + tm[1][2] * pv[2]) / ts,
            (tm[2][0] * pv[0] + tm[2][1] * pv[1] + tm[2][2] * pv[2]) / ts,
        ];
        let u1 = pv[2] * q[0] * q[0] - pv[1] * q[0] * q[1] + pv[0] * q[1] * q[1] + pv[1] * pv[1] * q[2];
        let u2 = pv[0] * pv[2] * q[2];
        let l1 = (pv[1] * pv[1] + (pv[0] - pv[2]) * (pv[0] - pv[2])).sqrt();
        let l2 = pv[0] + pv[2];
        let l3 = pv[1] * pv[1] - 4.0 * pv[0] * pv[2];
        let p1 = 2.0 * pv[2] * q[0] - pv[1] * q[1];
        let p2 = 2.0 * pv[0] * q[1] - pv[1] * q[0];
        let x0 = (p1 / l3 / scale) + c[0];
        let y0 = (p2 / l3 / scale) + c[1];
        let a = 2f64.sqrt() * ((u1 - 4.0 * u2) / ((l1 - l2) * l3)).sqrt() / scale;
        let b = 2f64.sqrt() * (-1.0 * ((u1 - 4.0 * u2) / ((l1 + l2) * l3))).sqrt() / scale;
        let pi = std::f64::consts::PI;
        let theta = if pv[1] == 0.0 { if pv[0] < pv[2] { 0.0 } else { pi / 2.0 } } else { pi / 2.0 + 0.5 * pv[1].atan2(pv[0] - pv[2]) };
        let (mut w, mut h) = ((2.0 * a) as f32, (2.0 * b) as f32);
        let angle;
        if w > h {
            std::mem::swap(&mut w, &mut h);
            angle = ((90.0 + theta * 180.0 / pi) % 180.0) as f32;
        } else {
            angle = ((theta * 180.0 / pi) % 180.0) as f32;
        }
        return Some((x0 as f32, y0 as f32, w, h, angle));
    }
    None
}
