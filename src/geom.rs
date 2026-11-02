//! Rotated boxes and the OpenCV geometry homr uses on them
//! (homr/bounding_boxes.py: AngledBoundingBox, RotatedBoundingBox).

use std::f64::consts::PI;

pub const NO_FRAGMENT: usize = usize::MAX;

#[derive(Clone, Debug)]
pub struct RBox {
    pub cx: f64,
    pub cy: f64,
    pub w: f64,
    pub h: f64,
    pub angle: f64,
    /// cv2.boxPoints(box).astype(np.int64), as homr keeps it for overlap tests.
    pub poly: [[i64; 2]; 4],
    /// Index of the staff-line fragment this is (identity, as homr's box equality), or NO_FRAGMENT.
    pub frag: usize,
}

/// OpenCV's RotatedRect::points, in float32 as OpenCV does it.
pub fn box_points_f32(cx: f64, cy: f64, w: f64, h: f64, angle: f64) -> [[f32; 2]; 4] {
    let (cx, cy, w, h) = (cx as f32, cy as f32, w as f32, h as f32);
    let a_rad = (angle as f32) as f64 * PI / 180.0;
    let b = (libm::cos(a_rad) as f32) * 0.5f32;
    let a = (libm::sin(a_rad) as f32) * 0.5f32;
    let p0 = [cx - a * h - b * w, cy + b * h - a * w];
    let p1 = [cx + a * h - b * w, cy - b * h - a * w];
    let p2 = [2.0 * cx - p0[0], 2.0 * cy - p0[1]];
    let p3 = [2.0 * cx - p1[0], 2.0 * cy - p1[1]];
    [p0, p1, p2, p3]
}

/// AngledBoundingBox's angle normalisation into [-45, 45].
fn normalise(w: f64, h: f64, angle: f64) -> (f64, f64, f64) {
    if angle > 135.0 {
        (w, h, angle - 180.0)
    } else if angle < -135.0 {
        (w, h, angle + 180.0)
    } else if angle > 45.0 {
        (h, w, angle - 90.0)
    } else if angle < -45.0 {
        (h, w, angle + 90.0)
    } else {
        (w, h, angle)
    }
}

impl RBox {
    /// RotatedBoundingBox(box): the polygon comes from the box as given, the
    /// rest from the normalised box.
    pub fn new(cx: f64, cy: f64, w: f64, h: f64, angle: f64, frag: usize) -> RBox {
        let pts = box_points_f32(cx, cy, w, h, angle);
        let poly = pts.map(|p| [p[0] as f64 as i64, p[1] as f64 as i64]);
        let (w, h, angle) = normalise(w, h, angle);
        RBox {
            cx,
            cy,
            w,
            h,
            angle,
            poly,
            frag,
        }
    }
    pub fn left_x(&self) -> f64 {
        self.cx - self.w / 2.0 // bottom_left / top_left x
    }
    pub fn right_x(&self) -> f64 {
        self.cx + self.w / 2.0 // bottom_right / top_right x
    }
    pub fn move_x(&self, dx: f64) -> RBox {
        RBox::new(self.cx + dx, self.cy, self.w, self.h, self.angle, self.frag)
    }
    pub fn taller(&self, t: f64) -> RBox {
        RBox::new(self.cx, self.cy, self.w, self.h + t, self.angle, self.frag)
    }
    /// get_center_extrapolated
    pub fn center_at(&self, x: f64) -> f64 {
        (x - self.cx) * libm::tan(self.angle / 180.0 * PI) + self.cy
    }
    fn can_touch(&self, o: &RBox) -> bool {
        let d = ((self.cx - o.cx).powi(2) + (self.cy - o.cy).powi(2)).sqrt();
        d.partial_cmp(&(self.w.max(self.h) + o.w.max(o.h))) != Some(std::cmp::Ordering::Greater)
    }
    /// is_intersecting: cv2.rotatedRectangleIntersection != INTERSECT_NONE
    pub fn is_intersecting(&self, o: &RBox) -> bool {
        self.can_touch(o) && rotated_rects_intersect(self, o)
    }
    /// is_overlapping: polygons (integer corners) intersect or touch.
    pub fn is_overlapping(&self, o: &RBox) -> bool {
        self.can_touch(o) && convex_touch_i64(&self.poly, &o.poly)
    }
    pub fn is_overlapping_extrapolated(&self, o: &RBox, unit: f64) -> bool {
        let (l, r) = if self.cx > o.cx { (o, self) } else { (self, o) };
        let center_x = (l.cx + r.cx) * 0.5;
        let tol = unit / 3.0;
        let max_gap = 5.0 * unit;
        if center_x - l.cx - (l.w / 2.0).floor() > max_gap
            || r.cx - center_x - (r.w / 2.0).floor() > max_gap
        {
            return false;
        }
        let la = libm::tan(l.angle * PI / 180.0);
        let ra = libm::tan(r.angle * PI / 180.0);
        let ly = (center_x - l.cx) * la + l.cy;
        let ry = (center_x - r.cx) * ra + r.cy;
        (ly - ry).abs() <= tol
    }
}

/// OpenCV's rotatedRectangleIntersection (imgproc/src/intersection.cpp),
/// float32 step for step, reduced to "is there any intersection point".
pub fn rotated_rects_intersect(r1: &RBox, r2: &RBox) -> bool {
    let (w1, h1, w2, h2) = (r1.w as f32, r1.h as f32, r2.w as f32, r2.h as f32);
    if w1 <= 0.0 || h1 <= 0.0 || w2 <= 0.0 || h2 <= 0.0 {
        return false; // Size2f::empty()
    }
    // The public wrapper first moves both centres by their average, in float32.
    let (c1x, c1y, c2x, c2y) = (r1.cx as f32, r1.cy as f32, r2.cx as f32, r2.cy as f32);
    let (ax, ay) = ((c1x + c2x) / 2.0f32, (c1y + c2y) / 2.0f32);
    let p1 = box_points_f32((c1x - ax) as f64, (c1y - ay) as f64, r1.w, r1.h, r1.angle);
    let p2 = box_points_f32((c2x - ax) as f64, (c2y - ay) as f64, r2.w, r2.h, r2.angle);
    let area = |r: &RBox| (r.w as f32) * (r.h as f32);
    let eps = 1e-6f32 * area(r1).max(area(r2));
    if (0..4).all(|i| !((p1[i][0] - p2[i][0]).abs() > eps || (p1[i][1] - p2[i][1]).abs() > eps)) {
        return true; // same rectangle
    }
    let vec = |p: &[[f32; 2]; 4]| -> [[f32; 2]; 4] {
        let mut v = [[0f32; 2]; 4];
        for i in 0..4 {
            v[i] = [p[(i + 1) % 4][0] - p[i][0], p[(i + 1) % 4][1] - p[i][1]];
        }
        v
    };
    let (v1, v2) = (vec(&p1), vec(&p2));
    for i in 0..4 {
        for j in 0..4 {
            let x21 = p2[j][0] - p1[i][0];
            let y21 = p2[j][1] - p1[i][1];
            let (vx1, vy1, vx2, vy2) = (v1[i][0], v1[i][1], v2[j][0], v2[j][1]);
            let det = vx2 * vy1 - vx1 * vy2;
            if (det as f64).abs() < 1e-12 {
                continue;
            }
            let inv = 1.0f32 / det;
            let t1 = (vx2 * y21 - vy2 * x21) * inv;
            let t2 = (vx1 * y21 - vy1 * x21) * inv;
            if !t1.is_finite() || !t2.is_finite() {
                continue;
            }
            if (0.0..=1.0).contains(&t1) && (0.0..=1.0).contains(&t2) {
                return true;
            }
        }
    }
    // _isOnPositiveSide: vy*(lx - px) >= vx*(ly - py)
    let inside = |p: &[[f32; 2]; 4], q: &[[f32; 2]; 4], vq: &[[f32; 2]; 4]| {
        p.iter().any(|&[x, y]| {
            let pos = (0..4)
                .filter(|&j| vq[j][1] * (q[j][0] - x) >= vq[j][0] * (q[j][1] - y))
                .count();
            pos == 4 || pos == 0
        })
    };
    inside(&p1, &p2, &v2) || inside(&p2, &p1, &v1)
}

/// Separating-axis test for two closed convex quadrilaterals (touching counts).
#[allow(dead_code)]
fn convex_touch_f64(a: &[[f64; 2]; 4], b: &[[f64; 2]; 4]) -> bool {
    for poly in [a, b] {
        for i in 0..4 {
            let (p, q) = (poly[i], poly[(i + 1) % 4]);
            let n = [q[1] - p[1], p[0] - q[0]];
            if n[0] == 0.0 && n[1] == 0.0 {
                continue;
            }
            let proj = |s: &[[f64; 2]; 4]| {
                let v: Vec<f64> = s.iter().map(|c| c[0] * n[0] + c[1] * n[1]).collect();
                (
                    v.iter().cloned().fold(f64::INFINITY, f64::min),
                    v.iter().cloned().fold(f64::NEG_INFINITY, f64::max),
                )
            };
            let (amin, amax) = proj(a);
            let (bmin, bmax) = proj(b);
            if amax < bmin || bmax < amin {
                return false;
            }
        }
    }
    true
}

fn convex_touch_i64(a: &[[i64; 2]; 4], b: &[[i64; 2]; 4]) -> bool {
    for poly in [a, b] {
        for i in 0..4 {
            let (p, q) = (poly[i], poly[(i + 1) % 4]);
            let n = [(q[1] - p[1]) as i128, (p[0] - q[0]) as i128];
            if n[0] == 0 && n[1] == 0 {
                continue;
            }
            let proj = |s: &[[i64; 2]; 4]| {
                let v: Vec<i128> = s
                    .iter()
                    .map(|c| c[0] as i128 * n[0] + c[1] as i128 * n[1])
                    .collect();
                (*v.iter().min().unwrap(), *v.iter().max().unwrap())
            };
            let (amin, amax) = proj(a);
            let (bmin, bmax) = proj(b);
            if amax < bmin || bmax < amin {
                return false;
            }
        }
    }
    true
}

/// cv2.minAreaRect, ported from OpenCV (modules/imgproc/src/rotcalipers.cpp
/// and convhull.cpp) step for step, so the float32 results match exactly.
/// The default follows OpenCV 4.11 (as in Pyodide, which Partsong runs);
/// the `opencv5` feature follows OpenCV 5.0, which reports the box with a
/// different width/height/angle convention.
pub fn min_area_rect(points: &[[i32; 2]]) -> RBox {
    let (cx, cy, w, h, angle) = min_area_rect_raw(points);
    RBox::new(
        cx as f64,
        cy as f64,
        w as f64,
        h as f64,
        angle as f64,
        NO_FRAGMENT,
    )
}

/// cv2.minAreaRect as OpenCV returns it: (cx, cy, w, h, angle) in float32.
pub fn min_area_rect_raw(points: &[[i32; 2]]) -> (f32, f32, f32, f32, f32) {
    let hull: Vec<[f32; 2]> = convex_hull_cv(points)
        .iter()
        .map(|&k| [points[k][0] as f32, points[k][1] as f32])
        .collect();
    if cfg!(feature = "opencv5") {
        box_v5(&hull)
    } else {
        box_v4(&hull)
    }
}

/// OpenCV 4.x: width along out[1], angle = atan2 of out[1] (radians kept as
/// float32, then converted to degrees).
fn box_v4(hull: &[[f32; 2]]) -> (f32, f32, f32, f32, f32) {
    let n = hull.len();
    let (mut cx, mut cy, mut w, mut h, mut angle) = (0f32, 0f32, 0f32, 0f32, 0f32);
    if n > 2 {
        let out = rotating_calipers(hull);
        cx = out[0][0] + (out[1][0] + out[2][0]) * 0.5f32;
        cy = out[0][1] + (out[1][1] + out[2][1]) * 0.5f32;
        w = ((out[1][0] as f64) * out[1][0] as f64 + (out[1][1] as f64) * out[1][1] as f64).sqrt()
            as f32;
        h = ((out[2][0] as f64) * out[2][0] as f64 + (out[2][1] as f64) * out[2][1] as f64).sqrt()
            as f32;
        angle = libm::atan2(out[1][1] as f64, out[1][0] as f64) as f32;
    } else if n == 2 {
        cx = (hull[0][0] + hull[1][0]) * 0.5f32;
        cy = (hull[0][1] + hull[1][1]) * 0.5f32;
        let dx = (hull[1][0] - hull[0][0]) as f64;
        let dy = (hull[1][1] - hull[0][1]) as f64;
        w = (dx * dx + dy * dy).sqrt() as f32;
        h = 0.0;
        angle = libm::atan2(dy, dx) as f32;
    } else if n == 1 {
        cx = hull[0][0];
        cy = hull[0][1];
    }
    // box.angle (float, radians) * 180 is a float multiplication; then / CV_PI in double
    (cx, cy, w, h, ((angle * 180.0f32) as f64 / PI) as f32)
}

fn box_v5(hull: &[[f32; 2]]) -> (f32, f32, f32, f32, f32) {
    let n = hull.len();
    let (cx, cy, w, h): (f32, f32, f32, f32);
    let mut angle_d = -PI / 2.0;
    if n > 2 {
        let out = rotating_calipers(hull);
        cx = out[0][0] + (out[1][0] + out[2][0]) * 0.5f32;
        cy = out[0][1] + (out[1][1] + out[2][1]) * 0.5f32;
        let mut ww = ((out[2][0] as f64) * out[2][0] as f64 + (out[2][1] as f64) * out[2][1] as f64)
            .sqrt() as f32;
        let mut hh = ((out[1][0] as f64) * out[1][0] as f64 + (out[1][1] as f64) * out[1][1] as f64)
            .sqrt() as f32;
        if out[1][0] == 0.0 && out[1][1] > 0.0 {
            std::mem::swap(&mut ww, &mut hh);
        } else {
            angle_d = -libm::atan2(out[1][0] as f64, out[1][1] as f64);
        }
        w = ww;
        h = hh;
    } else if n == 2 {
        cx = (hull[0][0] + hull[1][0]) * 0.5f32;
        cy = (hull[0][1] + hull[1][1]) * 0.5f32;
        let dx = (hull[0][0] - hull[1][0]) as f64;
        let dy = (hull[0][1] - hull[1][1]) as f64;
        let (mut ww, mut hh) = (0f32, (dx * dx + dy * dy).sqrt() as f32);
        if dx == 0.0 {
            std::mem::swap(&mut ww, &mut hh);
        } else if dy < 0.0 {
            angle_d = libm::atan2(dy, dx);
            std::mem::swap(&mut ww, &mut hh);
        } else if dy > 0.0 {
            angle_d = -libm::atan2(dx, dy);
        }
        w = ww;
        h = hh;
    } else {
        let c = if n == 1 { hull[0] } else { [0.0, 0.0] };
        cx = c[0];
        cy = c[1];
        w = 0.0;
        h = 0.0;
    }
    (cx, cy, w, h, (angle_d * 180.0 / PI) as f32)
}

fn rotating_calipers(points: &[[f32; 2]]) -> [[f32; 2]; 3] {
    let n = points.len();
    let mut minarea = f32::MAX;
    let mut vect = vec![[0f32; 2]; n];
    let mut inv_len = vec![0f32; n];
    let (mut left, mut bottom, mut right, mut top) = (0usize, 0usize, 0usize, 0usize);
    let mut pt0 = points[0];
    let (mut left_x, mut right_x, mut top_y, mut bottom_y) = (pt0[0], pt0[0], pt0[1], pt0[1]);
    for i in 0..n {
        if pt0[0] < left_x {
            left_x = pt0[0];
            left = i;
        }
        if pt0[0] > right_x {
            right_x = pt0[0];
            right = i;
        }
        if pt0[1] > top_y {
            top_y = pt0[1];
            top = i;
        }
        if pt0[1] < bottom_y {
            bottom_y = pt0[1];
            bottom = i;
        }
        let pt = points[if i + 1 < n { i + 1 } else { 0 }];
        let dx = (pt[0] - pt0[0]) as f64;
        let dy = (pt[1] - pt0[1]) as f64;
        vect[i] = [dx as f32, dy as f32];
        inv_len[i] = (1.0 / (dx * dx + dy * dy).sqrt()) as f32;
        pt0 = pt;
    }
    let (mut base_a, mut base_b); // orientation 1 (counter-clockwise hull) only sets the first base
    let mut seq = [bottom, right, top, left];
    let mut buf = (0usize, 0f32, 0f32, 0f32, 0f32, 0usize); // left, a, width, b, height, bottom
    let right_of = |v1: [f32; 2], v2: [f32; 2]| v1[1] * v2[0] + (-v1[0]) * v2[1] < 0.0;
    for _ in 0..n {
        let rot = [
            vect[seq[0]],
            [vect[seq[1]][1], -vect[seq[1]][0]],
            [-vect[seq[2]][0], -vect[seq[2]][1]],
            [-vect[seq[3]][1], vect[seq[3]][0]],
        ];
        let mut main = 0;
        for i in 1..4 {
            if right_of(rot[i], rot[main]) {
                main = i;
            }
        }
        let p = seq[main];
        let lead_x = vect[p][0] * inv_len[p];
        let lead_y = vect[p][1] * inv_len[p];
        (base_a, base_b) = match main {
            0 => (lead_x, lead_y),
            1 => (lead_y, -lead_x),
            2 => (-lead_x, -lead_y),
            _ => (-lead_y, lead_x),
        };
        seq[main] = if seq[main] + 1 == n { 0 } else { seq[main] + 1 };
        let dx = points[seq[1]][0] - points[seq[3]][0];
        let dy = points[seq[1]][1] - points[seq[3]][1];
        let width = dx * base_a + dy * base_b;
        let dx = points[seq[2]][0] - points[seq[0]][0];
        let dy = points[seq[2]][1] - points[seq[0]][1];
        let height = -dx * base_b + dy * base_a;
        let area = width * height;
        if area <= minarea {
            minarea = area;
            buf = (seq[3], base_a, width, base_b, height, seq[0]);
        }
    }
    let (li, a1, width, b1, height, bi) = buf;
    let (a2, b2) = (-b1, a1);
    let c1 = a1 * points[li][0] + points[li][1] * b1;
    let c2 = a2 * points[bi][0] + points[bi][1] * b2;
    let idet = 1.0f32 / (a1 * b2 - a2 * b1);
    let px = (c1 * b2 - c2 * b1) * idet;
    let py = (a1 * c2 - a2 * c1) * idet;
    [
        [px, py],
        [a1 * width, b1 * width],
        [a2 * height, b2 * height],
    ]
}

fn sign(v: i64) -> i32 {
    (v > 0) as i32 - (v < 0) as i32
}

/// OpenCV's Sklansky_<int, int64>; `array` holds indices into `pts` (sorted order).
fn sklansky(
    pts: &[[i32; 2]],
    array: &[usize],
    start: i64,
    end: i64,
    nsign: i32,
    sign2: i32,
) -> Vec<i64> {
    let p = |k: i64| pts[array[k as usize]];
    let incr = if end > start { 1 } else { -1 };
    let (mut pprev, mut pcur, mut pnext) = (start, start + incr, start + 2 * incr);
    if start == end || p(start) == p(end) {
        return vec![start];
    }
    let mut stack: Vec<i64> = vec![0; array.len() + 4];
    stack[0] = pprev;
    stack[1] = pcur;
    stack[2] = pnext;
    let mut size: usize = 3;
    let end = end + incr;
    while pnext != end {
        let cury = p(pcur)[1] as i64;
        let nexty = p(pnext)[1] as i64;
        let by = nexty - cury;
        if sign(by) != nsign {
            let a = [(p(pcur)[0] - p(pprev)[0]) as i64, cury - p(pprev)[1] as i64];
            let b = [(p(pnext)[0] - p(pcur)[0]) as i64, by];
            let convexity = a[1] * b[0] - a[0] * b[1];
            if sign(convexity) == sign2 && (a[0] != 0 || a[1] != 0) {
                pprev = pcur;
                pcur = pnext;
                pnext += incr;
                stack[size] = pnext;
                size += 1;
            } else if pprev == start {
                pcur = pnext;
                stack[1] = pcur;
                pnext += incr;
                stack[2] = pnext;
            } else {
                stack[size - 2] = pnext;
                pcur = pprev;
                pprev = stack[size - 4];
                size -= 1;
            }
        } else {
            pnext += incr;
            stack[size - 1] = pnext;
        }
    }
    stack.truncate(size - 1);
    stack
}

/// OpenCV's convexHull(points, clockwise=false, returnPoints=true) for int
/// points: the indices of the hull points in OpenCV's order.
fn convex_hull_cv(pts: &[[i32; 2]]) -> Vec<usize> {
    let total = pts.len();
    if total == 0 {
        return vec![];
    }
    let mut ptr: Vec<usize> = (0..total).collect();
    ptr.sort_by(|&a, &b| (pts[a][0], pts[a][1], a).cmp(&(pts[b][0], pts[b][1], b)));
    let (mut miny, mut maxy) = (0usize, 0usize);
    for i in 1..total {
        let y = pts[ptr[i]][1];
        if pts[ptr[miny]][1] > y {
            miny = i;
        }
        if pts[ptr[maxy]][1] < y {
            maxy = i;
        }
    }
    let mut hull: Vec<i64> = Vec::new();
    let last = total as i64 - 1;
    if pts[ptr[0]] == pts[ptr[total - 1]] {
        hull.push(0);
    } else {
        let tl = sklansky(pts, &ptr, 0, maxy as i64, -1, 1);
        let tr = sklansky(pts, &ptr, last, maxy as i64, -1, -1);
        let (tl, tr) = (tr, tl); // counter-clockwise: swap
        hull.extend(tl.iter().take(tl.len().saturating_sub(1)));
        for i in (1..tr.len()).rev() {
            hull.push(tr[i]);
        }
        let stop_idx: i64 = if tr.len() > 2 {
            tr[1]
        } else if tl.len() > 2 {
            tl[tl.len() - 2]
        } else {
            -1
        };
        let mut bl = sklansky(pts, &ptr, 0, miny as i64, 1, -1);
        let mut br = sklansky(pts, &ptr, last, miny as i64, 1, 1);
        if stop_idx >= 0 {
            let check_idx: i64 = if bl.len() > 2 {
                bl[1]
            } else if bl.len() + br.len() > 2 {
                br[2 - bl.len()]
            } else {
                -1
            };
            if check_idx == stop_idx
                || (check_idx >= 0 && pts[ptr[check_idx as usize]] == pts[ptr[stop_idx as usize]])
            {
                bl.truncate(2);
                br.truncate(2);
            }
        }
        hull.extend(bl.iter().take(bl.len().saturating_sub(1)));
        for i in (1..br.len()).rev() {
            hull.push(br[i]);
        }
    }
    let mut hb: Vec<usize> = hull.iter().map(|&k| ptr[k as usize]).collect();
    let nout = hb.len();
    if nout >= 3 {
        let (mut min_idx, mut max_idx, mut lt) = (0usize, 0usize, 0usize);
        for i in 1..nout {
            let idx = hb[i];
            lt += (hb[i - 1] < idx) as usize;
            if lt > 1 && lt + 2 <= i {
                break;
            }
            if idx < hb[min_idx] {
                min_idx = i;
            }
            if idx > hb[max_idx] {
                max_idx = i;
            }
        }
        let mmdist = (max_idx as i64 - min_idx as i64).unsigned_abs() as usize;
        if (mmdist == 1 || mmdist == nout - 1) && (lt <= 1 || lt + 2 >= nout) {
            let ascending = (max_idx + 1) % nout == min_idx;
            let i0 = if ascending { min_idx } else { max_idx };
            if i0 > 0 {
                let mut tmp = vec![0usize; nout];
                let mut j = i0;
                let mut i = 0;
                while i < nout {
                    let curr = hb[j];
                    tmp[i] = curr;
                    let next_j = if j + 1 < nout { j + 1 } else { 0 };
                    if i < nout - 1 && (ascending != (curr < hb[next_j])) {
                        break;
                    }
                    j = next_j;
                    i += 1;
                }
                if i == nout {
                    hb = tmp;
                }
            }
        }
    }
    hb
}

#[allow(dead_code)]
fn convex_hull(pts: &[(i64, i64)]) -> Vec<(i64, i64)> {
    if pts.len() < 3 {
        return pts.to_vec();
    }
    let cross = |o: (i64, i64), a: (i64, i64), b: (i64, i64)| {
        (a.0 - o.0) * (b.1 - o.1) - (a.1 - o.1) * (b.0 - o.0)
    };
    let mut lower: Vec<(i64, i64)> = Vec::new();
    for &p in pts {
        while lower.len() >= 2 && cross(lower[lower.len() - 2], lower[lower.len() - 1], p) <= 0 {
            lower.pop();
        }
        lower.push(p);
    }
    let mut upper: Vec<(i64, i64)> = Vec::new();
    for &p in pts.iter().rev() {
        while upper.len() >= 2 && cross(upper[upper.len() - 2], upper[upper.len() - 1], p) <= 0 {
            upper.pop();
        }
        upper.push(p);
    }
    lower.pop();
    upper.pop();
    lower.extend(upper);
    lower
}
