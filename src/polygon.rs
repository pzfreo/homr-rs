//! homr's polygon overlap test (bounding_boxes.do_polygons_overlap) and the
//! OpenCV 4.11 routines it uses, ported step for step: intersectConvexConvex
//! (O'Rourke's algorithm, modules/imgproc/src/geometry.cpp), pointPolygonTest,
//! contourArea, and ellipse2Poly (drawing.cpp).

use crate::sintable::SIN_TABLE;

type P = [f32; 2];

fn area_sign(a: P, b: P, c: P) -> i32 {
    let area2 = (b[0] - a[0]) as f64 * (c[1] - a[1]) as f64 - (c[0] - a[0]) as f64 * (b[1] - a[1]) as f64;
    if area2 > 1e-5 {
        1
    } else if area2 < -1e-5 {
        -1
    } else {
        0
    }
}

fn between(a: P, b: P, c: P) -> bool {
    if a[0] != b[0] {
        (a[0] <= c[0] && c[0] <= b[0]) || (a[0] >= c[0] && c[0] >= b[0])
    } else {
        (a[1] <= c[1] && c[1] <= b[1]) || (a[1] >= c[1] && c[1] >= b[1])
    }
}

#[derive(PartialEq, Clone, Copy)]
enum Ls {
    None,
    Single,
    Overlap,
    Endpoint,
}

fn parallel_int(a: P, b: P, c: P, d: P, p: &mut P, q: &mut P) -> Ls {
    if area_sign(a, b, c) != 0 {
        Ls::None
    } else if between(a, b, c) && between(a, b, d) {
        (*p, *q) = (c, d);
        Ls::Overlap
    } else if between(c, d, a) && between(c, d, b) {
        (*p, *q) = (a, b);
        Ls::Overlap
    } else if between(a, b, c) && between(c, d, b) {
        (*p, *q) = (c, b);
        Ls::Overlap
    } else if between(a, b, c) && between(c, d, a) {
        (*p, *q) = (c, a);
        Ls::Overlap
    } else if between(a, b, d) && between(c, d, b) {
        (*p, *q) = (d, b);
        Ls::Overlap
    } else if between(a, b, d) && between(c, d, a) {
        (*p, *q) = (d, a);
        Ls::Overlap
    } else {
        Ls::None
    }
}

fn intersect_segments(a: P, b: P, c: P, d: P, p: &mut P, q: &mut P) -> Ls {
    let denom = (a[0] - b[0]) as f64 * (d[1] - c[1]) as f64 - (a[1] - b[1]) as f64 * (d[0] - c[0]) as f64;
    if denom == 0.0 {
        return parallel_int(a, b, c, d, p, q);
    }
    let num = (d[1] - a[1]) as f64 * (a[0] - c[0]) as f64 + (a[0] - d[0]) as f64 * (a[1] - c[1]) as f64;
    let s = num / denom;
    let num = (b[1] - a[1]) as f64 * (a[0] - c[0]) as f64 + (c[1] - a[1]) as f64 * (b[0] - a[0]) as f64;
    let t = num / denom;
    *p = [(a[0] as f64 + s * (b[0] - a[0]) as f64) as f32, (a[1] as f64 + s * (b[1] - a[1]) as f64) as f32];
    *q = *p;
    if s < 0.0 || s > 1.0 || t < 0.0 || t > 1.0 {
        Ls::None
    } else if s == 0.0 || s == 1.0 || t == 0.0 || t == 1.0 {
        Ls::Endpoint
    } else {
        Ls::Single
    }
}

#[derive(PartialEq, Clone, Copy)]
enum In {
    P,
    Q,
    Unknown,
}

fn push_new(result: &mut Vec<P>, p: P) {
    if p != *result.last().unwrap() {
        result.push(p);
    }
}

/// intersectConvexConvex_: returns (number of vertices, area); -1 if it didn't converge.
fn intersect_cc(pp: &[P], qq: &[P], result_size: usize) -> (i64, f32) {
    let (n, m) = (pp.len(), qq.len());
    let (mut a, mut b) = (0usize, 0usize);
    let mut inflag = In::Unknown;
    let (mut aa, mut ba) = (0usize, 0usize);
    let mut first = true;
    let mut result: Vec<P> = vec![[f32::MAX, f32::MAX]];
    loop {
        let a1 = (a + n - 1) % n;
        let b1 = (b + m - 1) % m;
        let av = [pp[a][0] - pp[a1][0], pp[a][1] - pp[a1][1]];
        let bv = [qq[b][0] - qq[b1][0], qq[b][1] - qq[b1][1]];
        let cross = area_sign([0.0, 0.0], av, bv);
        let a_hb = area_sign(qq[b1], qq[b], pp[a]);
        let b_ha = area_sign(pp[a1], pp[a], qq[b]);
        let (mut p, mut q) = ([0f32; 2], [0f32; 2]);
        let code = intersect_segments(pp[a1], pp[a], qq[b1], qq[b], &mut p, &mut q);
        if code == Ls::Single || code == Ls::Endpoint {
            if inflag == In::Unknown && first {
                aa = 0;
                ba = 0;
                first = false;
                result.push(p);
            }
            push_new(&mut result, p);
            inflag = if a_hb > 0 { In::P } else if b_ha > 0 { In::Q } else { inflag };
        }
        let dot = av[0] as f64 * bv[0] as f64 + av[1] as f64 * bv[1] as f64;
        if code == Ls::Overlap && dot < 0.0 {
            push_new(&mut result, p);
            push_new(&mut result, q);
            return (result.len() as i64, 0.0); // OpenCV returns here without computing the area
        }
        let advance_a = |a: &mut usize, aa: &mut usize, inside: bool, result: &mut Vec<P>| {
            if inside {
                push_new(result, pp[*a]);
            }
            *aa += 1;
            *a = (*a + 1) % n;
        };
        let advance_b = |b: &mut usize, ba: &mut usize, inside: bool, result: &mut Vec<P>| {
            if inside {
                push_new(result, qq[*b]);
            }
            *ba += 1;
            *b = (*b + 1) % m;
        };
        if cross == 0 && a_hb < 0 && b_ha < 0 {
            return (result.len() as i64, 0.0);
        } else if cross == 0 && a_hb == 0 && b_ha == 0 {
            if inflag == In::P {
                advance_b(&mut b, &mut ba, inflag == In::Q, &mut result);
            } else {
                advance_a(&mut a, &mut aa, inflag == In::P, &mut result);
            }
        } else if cross >= 0 {
            if b_ha > 0 {
                advance_a(&mut a, &mut aa, inflag == In::P, &mut result);
            } else {
                advance_b(&mut b, &mut ba, inflag == In::Q, &mut result);
            }
        } else if a_hb > 0 {
            advance_b(&mut b, &mut ba, inflag == In::Q, &mut result);
        } else {
            advance_a(&mut a, &mut aa, inflag == In::P, &mut result);
        }
        if !(((aa < n) || (ba < m)) && (aa < 2 * n) && (ba < 2 * m) && result.len() <= result_size) {
            break;
        }
    }
    let mut nr = result.len();
    if nr > result_size {
        return (-1, -1.0);
    }
    let mut area = 0f64;
    let mut prev = result[nr - 1];
    for i in 1..nr {
        result[i - 1] = result[i];
        area += prev[0] as f64 * result[i][1] as f64 - prev[1] as f64 * result[i][0] as f64;
        prev = result[i];
    }
    if nr > 1 && result[nr - 2] == result[0] {
        nr -= 1;
    }
    (nr as i64 - 1, (area * 0.5) as f32)
}

/// pointPolygonTest(contour, pt, measureDist=false) for float contours: 1 inside, 0 on, -1 outside.
fn point_polygon_test_f(contour: &[P], pt: P) -> i32 {
    let mut counter = 0;
    let mut v = contour[contour.len() - 1];
    for &c in contour {
        let v0 = v;
        v = c;
        if (v0[1] <= pt[1] && v[1] <= pt[1]) || (v0[1] > pt[1] && v[1] > pt[1]) || (v0[0] < pt[0] && v[0] < pt[0]) {
            if pt[1] == v[1] && (pt[0] == v[0] || (pt[1] == v0[1] && ((v0[0] <= pt[0] && pt[0] <= v[0]) || (v[0] <= pt[0] && pt[0] <= v0[0])))) {
                return 0;
            }
            continue;
        }
        let mut dist = (pt[1] - v0[1]) as f64 * (v[0] - v0[0]) as f64 - (pt[0] - v0[0]) as f64 * (v[1] - v0[1]) as f64;
        if dist == 0.0 {
            return 0;
        }
        if v[1] < v0[1] {
            dist = -dist;
        }
        counter += (dist > 0.0) as i32;
    }
    if counter % 2 == 0 {
        -1
    } else {
        1
    }
}

/// pointPolygonTest for integer contours and an integer point (OpenCV's integer branch).
pub fn point_polygon_test_i(contour: &[[i64; 2]], pt: [i64; 2]) -> i32 {
    let mut counter = 0;
    let mut v = contour[contour.len() - 1];
    for &c in contour {
        let v0 = v;
        v = c;
        if (v0[1] <= pt[1] && v[1] <= pt[1]) || (v0[1] > pt[1] && v[1] > pt[1]) || (v0[0] < pt[0] && v[0] < pt[0]) {
            if pt[1] == v[1] && (pt[0] == v[0] || (pt[1] == v0[1] && ((v0[0] <= pt[0] && pt[0] <= v[0]) || (v[0] <= pt[0] && pt[0] <= v0[0])))) {
                return 0;
            }
            continue;
        }
        let mut dist = (pt[1] - v0[1]) * (v[0] - v0[0]) - (pt[0] - v0[0]) * (v[1] - v0[1]);
        if dist == 0 {
            return 0;
        }
        if v[1] < v0[1] {
            dist = -dist;
        }
        counter += (dist > 0) as i32;
    }
    if counter % 2 == 0 {
        -1
    } else {
        1
    }
}

fn contour_area(c: &[P]) -> f64 {
    let mut a = 0f64;
    let mut prev = c[c.len() - 1];
    for &p in c {
        a += prev[0] as f64 * p[1] as f64 - prev[1] as f64 * p[0] as f64;
        prev = p;
    }
    (a * 0.5).abs()
}

/// cv2.intersectConvexConvex(p1, p2) (handleNested=True): the intersection area.
pub fn intersect_convex_convex(p1: &[[i64; 2]], p2: &[[i64; 2]]) -> f32 {
    let (n, m) = (p1.len(), p2.len());
    if n < 2 || m < 2 {
        return 0.0;
    }
    let prep = |p: &[[i64; 2]]| -> Vec<P> {
        let mut d: Vec<P> = p.iter().map(|c| [c[0] as f32, c[1] as f32]).collect();
        let len = d.len();
        let diff0 = [d[0][0] - d[len - 1][0], d[0][1] - d[len - 1][1]];
        for i in 1..len {
            let e = [d[i][0] - d[i - 1][0], d[i][1] - d[i - 1][1]];
            let s = diff0[0] as f64 * e[1] as f64 - diff0[1] as f64 * e[0] as f64;
            if s != 0.0 {
                if s < 0.0 {
                    d.reverse();
                }
                break;
            }
        }
        d
    };
    let (f1, f2) = (prep(p1), prep(p2));
    let (nr, mut area) = intersect_cc(&f1, &f2, n + m + 1);
    if nr < 0 {
        return -1.0;
    }
    if nr == 0 {
        if f2.iter().all(|&p| point_polygon_test_f(&f1, p) >= 0) {
            area = contour_area(&f2) as f32;
        } else if f1.iter().all(|&p| point_polygon_test_f(&f2, p) >= 0) {
            area = contour_area(&f1) as f32;
        } else {
            return 0.0;
        }
    }
    area.abs()
}

/// homr's do_polygons_overlap.
pub fn polygons_overlap(a: &[[i64; 2]], b: &[[i64; 2]]) -> bool {
    if intersect_convex_convex(a, b) > 0.0 {
        return true;
    }
    a.iter().any(|&p| point_polygon_test_i(b, p) >= 0) || b.iter().any(|&p| point_polygon_test_i(a, p) >= 0)
}

/// cv2.ellipse2Poly(center, axes, angle, 0, 360, 1) with integer arguments.
pub fn ellipse2poly(center: [i64; 2], axes: [i64; 2], angle: i64) -> Vec<[i64; 2]> {
    let mut angle = angle;
    while angle < 0 {
        angle += 360;
    }
    while angle > 360 {
        angle -= 360;
    }
    let alpha = SIN_TABLE[(450 - angle) as usize]; // cos
    let beta = SIN_TABLE[angle as usize]; // sin
    let mut out: Vec<[i64; 2]> = Vec::new();
    let mut prev = [i64::MIN, i64::MIN];
    let mut i = 0i64;
    while i < 360 + 1 {
        let a = i.min(360);
        let x = axes[0] as f64 * SIN_TABLE[(450 - a) as usize] as f64;
        let y = axes[1] as f64 * SIN_TABLE[a as usize] as f64;
        let px = center[0] as f64 + x * alpha as f64 - y * beta as f64;
        let py = center[1] as f64 + x * beta as f64 + y * alpha as f64;
        let pt = [px.round_ties_even() as i64, py.round_ties_even() as i64];
        if pt != prev {
            out.push(pt);
            prev = pt;
        }
        i += 1;
    }
    if out.len() == 1 {
        out = vec![center, center];
    }
    out
}
