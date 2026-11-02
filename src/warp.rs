//! OpenCV 4.11 image warping used to prepare staff images: resize (uint8,
//! INTER_LINEAR), getAffineTransform (6x6 LU solve), warpAffine (uint8,
//! INTER_LINEAR, BORDER_CONSTANT) and Subdiv2D (incremental Delaunay).
//! Ported from modules/imgproc/src/{resize,imgwarp,subdivision2d}.cpp and
//! modules/core/src/matrix_decomp.cpp.

/// A single-channel uint8 image, row major.
#[derive(Clone, Debug)]
pub struct Gray {
    pub w: usize,
    pub h: usize,
    pub data: Vec<u8>,
}

impl Gray {
    pub fn new(w: usize, h: usize, fill: u8) -> Gray {
        Gray {
            w,
            h,
            data: vec![fill; w * h],
        }
    }
    pub fn at(&self, x: usize, y: usize) -> u8 {
        self.data[y * self.w + x]
    }
    /// numpy slicing image[y0:y1, x0:x1] (bounds clamped as numpy does).
    pub fn slice(&self, x0: i64, y0: i64, x1: i64, y1: i64) -> Gray {
        let cl = |v: i64, n: usize| v.clamp(0, n as i64) as usize;
        let (x0, x1, y0, y1) = (
            cl(x0, self.w),
            cl(x1, self.w),
            cl(y0, self.h),
            cl(y1, self.h),
        );
        let (w, h) = (x1.saturating_sub(x0), y1.saturating_sub(y0));
        let mut out = Gray::new(w, h, 0);
        for y in 0..h {
            out.data[y * w..(y + 1) * w]
                .copy_from_slice(&self.data[(y0 + y) * self.w + x0..(y0 + y) * self.w + x0 + w]);
        }
        out
    }
}

fn cv_floor(v: f64) -> i64 {
    v.floor() as i64
}

/// cv2.resize(img, (dw, dh)) for uint8 single channel with INTER_LINEAR.
pub fn resize(src: &Gray, dw: usize, dh: usize) -> Gray {
    resize_region(src, dw, dh, 0, 0, dw, dh)
}

/// The part [x0, x1) x [y0, y1) of cv2.resize(img, (dw, dh)), computed alone
/// (each output pixel only depends on the source pixels around it).
pub fn resize_region(
    src: &Gray,
    dw: usize,
    dh: usize,
    x0: usize,
    y0: usize,
    x1: usize,
    y1: usize,
) -> Gray {
    let (x1, y1) = (x1.min(dw), y1.min(dh));
    let (ow, oh) = (x1.saturating_sub(x0), y1.saturating_sub(y0));
    if dw == src.w && dh == src.h {
        return src.slice(x0 as i64, y0 as i64, x1 as i64, y1 as i64);
    }
    let inv_scale_x = dw as f64 / src.w as f64;
    let inv_scale_y = dh as f64 / src.h as f64;
    let scale_x = 1.0 / inv_scale_x;
    let scale_y = 1.0 / inv_scale_y;
    let iscale_x = scale_x.round_ties_even() as i64;
    let iscale_y = scale_y.round_ties_even() as i64;
    let is_area_fast = (scale_x - iscale_x as f64).abs() < f64::EPSILON
        && (scale_y - iscale_y as f64).abs() < f64::EPSILON;
    if is_area_fast && iscale_x == 2 && iscale_y == 2 {
        return resize_area_fast_2x(src, dw, dh).slice(x0 as i64, y0 as i64, x1 as i64, y1 as i64);
    }
    const BITS: i32 = 11;
    const SCALE: f32 = (1 << BITS) as f32;
    let (sw, sh) = (src.w as i64, src.h as i64);
    // xmax: the first output column whose right neighbour is past the image
    let mut xmax = dw;
    let mut xofs = vec![0i64; ow];
    let mut ialpha = vec![0i32; ow * 2];
    for dx in 0..dw {
        let fx0 = ((dx as f64 + 0.5) * scale_x - 0.5) as f32;
        let sx0 = cv_floor(fx0 as f64);
        if sx0 + 1 >= sw {
            xmax = xmax.min(dx);
        }
        if dx < x0 || dx >= x1 {
            if dx >= x1 && xmax <= dx {
                break;
            }
            continue;
        }
        let mut fx = fx0;
        let mut sx = sx0;
        fx -= sx as f32;
        if sx < 0 {
            fx = 0.0;
            sx = 0;
        }
        if sx + 1 >= sw && sx >= sw - 1 {
            fx = 0.0;
            sx = sw - 1;
        }
        let i = dx - x0;
        xofs[i] = sx;
        ialpha[i * 2] = ((1.0f32 - fx) * SCALE).round_ties_even() as i32;
        ialpha[i * 2 + 1] = (fx * SCALE).round_ties_even() as i32;
    }
    let mut yofs = vec![0i64; oh];
    let mut ibeta = vec![0i32; oh * 2];
    for dy in y0..y1 {
        let mut fy = ((dy as f64 + 0.5) * scale_y - 0.5) as f32;
        let sy = cv_floor(fy as f64);
        fy -= sy as f32;
        let i = dy - y0;
        yofs[i] = sy;
        ibeta[i * 2] = ((1.0f32 - fy) * SCALE).round_ties_even() as i32;
        ibeta[i * 2 + 1] = (fy * SCALE).round_ties_even() as i32;
    }
    let hresize = |sy: i64| -> Vec<i32> {
        let row = &src.data[sy as usize * src.w..(sy as usize + 1) * src.w];
        let mut out = vec![0i32; ow];
        for i in 0..ow {
            let sx = xofs[i] as usize;
            out[i] = if x0 + i < xmax {
                row[sx] as i32 * ialpha[i * 2] + row[sx + 1] as i32 * ialpha[i * 2 + 1]
            } else {
                row[sx] as i32 * (1 << BITS)
            };
        }
        out
    };
    let clip = |y: i64| {
        if y >= 0 {
            if y < sh {
                y
            } else {
                sh - 1
            }
        } else {
            0
        }
    };
    let mut dst = Gray::new(ow, oh, 0);
    let mut cache: Vec<(i64, Vec<i32>)> = Vec::new();
    for i in 0..oh {
        let sy0 = yofs[i];
        for k in 0..2 {
            let sy = clip(sy0 + k);
            if !cache.iter().any(|c| c.0 == sy) {
                cache.push((sy, hresize(sy)));
                if cache.len() > 4 {
                    cache.remove(0);
                }
            }
        }
        let r0 = &cache.iter().find(|c| c.0 == clip(sy0)).unwrap().1;
        let r1 = &cache.iter().find(|c| c.0 == clip(sy0 + 1)).unwrap().1;
        let (b0, b1) = (ibeta[i * 2], ibeta[i * 2 + 1]);
        for x in 0..ow {
            let v = (((b0 * (r0[x] >> 4)) >> 16) + ((b1 * (r1[x] >> 4)) >> 16) + 2) >> 2;
            dst.data[i * ow + x] = v.clamp(0, 255) as u8;
        }
    }
    dst
}

/// resizeAreaFast_ for an exact 2x downscale (scalar path: rounds half to even).
fn resize_area_fast_2x(src: &Gray, dw: usize, dh: usize) -> Gray {
    let mut dst = Gray::new(dw, dh, 0);
    for dy in 0..dh {
        let sy0 = dy * 2;
        for dx in 0..dw {
            let sx0 = dx * 2;
            if sy0 + 2 <= src.h && sx0 + 2 <= src.w {
                let s = src.at(sx0, sy0) as i32
                    + src.at(sx0 + 1, sy0) as i32
                    + src.at(sx0, sy0 + 1) as i32
                    + src.at(sx0 + 1, sy0 + 1) as i32;
                dst.data[dy * dw + dx] = ((s as f32 * 0.25f32) as f64)
                    .round_ties_even()
                    .clamp(0.0, 255.0) as u8;
            } else {
                let (mut sum, mut count) = (0i32, 0i32);
                for sy in sy0..(sy0 + 2).min(src.h) {
                    for sx in sx0..(sx0 + 2).min(src.w) {
                        sum += src.at(sx, sy) as i32;
                        count += 1;
                    }
                }
                dst.data[dy * dw + dx] = if count > 0 {
                    ((sum as f32 / count as f32) as f64)
                        .round_ties_even()
                        .clamp(0.0, 255.0) as u8
                } else {
                    0
                };
            }
        }
    }
    dst
}

/// LUImpl (OpenCV): solves A x = b in place (b becomes x); false if singular.
fn lu_solve(a: &mut [f64], m: usize, b: &mut [f64]) -> bool {
    let eps = f64::EPSILON * 100.0;
    for i in 0..m {
        let mut k = i;
        for j in i + 1..m {
            if a[j * m + i].abs() > a[k * m + i].abs() {
                k = j;
            }
        }
        if a[k * m + i].abs() < eps {
            return false;
        }
        if k != i {
            for j in i..m {
                a.swap(i * m + j, k * m + j);
            }
            b.swap(i, k);
        }
        let d = -1.0 / a[i * m + i];
        for j in i + 1..m {
            let alpha = a[j * m + i] * d;
            for k in i + 1..m {
                a[j * m + k] += alpha * a[i * m + k];
            }
            b[j] += alpha * b[i];
        }
    }
    for i in (0..m).rev() {
        let mut s = b[i];
        for k in i + 1..m {
            s -= a[i * m + k] * b[k];
        }
        b[i] = s / a[i * m + i];
    }
    true
}

/// cv2.getAffineTransform(src, dst) for float32 points: 2x3 matrix, row major
/// (all zeros if the system is singular, as cv::solve leaves it).
pub fn get_affine_transform(src: &[[f32; 2]; 3], dst: &[[f32; 2]; 3]) -> [f64; 6] {
    let mut a = [0f64; 36];
    let mut b = [0f64; 6];
    for i in 0..3 {
        let (j, k) = (i * 12, i * 12 + 6);
        a[j] = src[i][0] as f64;
        a[k + 3] = src[i][0] as f64;
        a[j + 1] = src[i][1] as f64;
        a[k + 4] = src[i][1] as f64;
        a[j + 2] = 1.0;
        a[k + 5] = 1.0;
        b[i * 2] = dst[i][0] as f64;
        b[i * 2 + 1] = dst[i][1] as f64;
    }
    if lu_solve(&mut a, 6, &mut b) {
        b
    } else {
        [0.0; 6]
    }
}

/// cv2.warpAffine(src, M, (dw, dh), flags=INTER_LINEAR, borderMode=BORDER_CONSTANT, borderValue=fill).
pub fn warp_affine(src: &Gray, m0: &[f64; 6], dw: usize, dh: usize, fill: u8) -> Gray {
    let mut m = *m0;
    let d = m[0] * m[4] - m[1] * m[3];
    let d = if d != 0.0 { 1.0 / d } else { 0.0 };
    let (a11, a22) = (m[4] * d, m[0] * d);
    m[0] = a11;
    m[1] *= -d;
    m[3] *= -d;
    m[4] = a22;
    let b1 = -m[0] * m[2] - m[1] * m[5];
    let b2 = -m[3] * m[2] - m[4] * m[5];
    m[2] = b1;
    m[5] = b2;
    const AB_BITS: i32 = 10;
    const AB_SCALE: f64 = 1024.0;
    const INTER_BITS: i32 = 5;
    const TAB: i32 = 32;
    let sat =
        |v: f64| -> i64 { v.round_ties_even().clamp(i32::MIN as f64, i32::MAX as f64) as i64 };
    let adelta: Vec<i64> = (0..dw).map(|x| sat(m[0] * x as f64 * AB_SCALE)).collect();
    let bdelta: Vec<i64> = (0..dw).map(|x| sat(m[3] * x as f64 * AB_SCALE)).collect();
    let round_delta = (AB_SCALE as i64) / TAB as i64 / 2;
    let (sw, sh) = (src.w as i64, src.h as i64);
    let mut dst = Gray::new(dw, dh, 0);
    let s16 = |v: i64| v.clamp(i16::MIN as i64, i16::MAX as i64);
    for y in 0..dh {
        let x0 = sat((m[1] * y as f64 + m[2]) * AB_SCALE) + round_delta;
        let y0 = sat((m[4] * y as f64 + m[5]) * AB_SCALE) + round_delta;
        for x in 0..dw {
            let xx = (x0 + adelta[x]) >> (AB_BITS - INTER_BITS);
            let yy = (y0 + bdelta[x]) >> (AB_BITS - INTER_BITS);
            let sx = s16(xx >> INTER_BITS);
            let sy = s16(yy >> INTER_BITS);
            let fx = xx & (TAB as i64 - 1);
            let fy = yy & (TAB as i64 - 1);
            let w = [
                (TAB as i64 - fx) * (TAB as i64 - fy),
                fx * (TAB as i64 - fy),
                (TAB as i64 - fx) * fy,
                fx * fy,
            ]
            .map(|v| v * 32);
            let px = |xx: i64, yy: i64| -> i64 {
                if xx >= 0 && yy >= 0 && xx < sw && yy < sh {
                    src.at(xx as usize, yy as usize) as i64
                } else {
                    fill as i64
                }
            };
            let v = if (sx as u64) < (sw - 1).max(0) as u64
                && (sy as u64) < (sh - 1).max(0) as u64
                && sx >= 0
                && sy >= 0
            {
                let sxu = sx as usize;
                let syu = sy as usize;
                ((src.at(sxu, syu) as i64 * w[0]
                    + src.at(sxu + 1, syu) as i64 * w[1]
                    + src.at(sxu, syu + 1) as i64 * w[2]
                    + src.at(sxu + 1, syu + 1) as i64 * w[3]
                    + (1 << 14))
                    >> 15)
                    .clamp(0, 255)
            } else if sx >= sw || sx + 1 < 0 || sy >= sh || sy + 1 < 0 {
                fill as i64
            } else {
                ((px(sx, sy) * w[0]
                    + px(sx + 1, sy) * w[1]
                    + px(sx, sy + 1) * w[2]
                    + px(sx + 1, sy + 1) * w[3]
                    + (1 << 14))
                    >> 15)
                    .clamp(0, 255)
            };
            dst.data[y * dw + x] = v as u8;
        }
    }
    dst
}

/// cv2.boundingRect for float32 points: (x, y, w, h).
pub fn bounding_rect_f(pts: &[[f32; 2]]) -> (i64, i64, i64, i64) {
    let xmin = pts.iter().map(|p| p[0]).fold(f32::INFINITY, f32::min);
    let xmax = pts.iter().map(|p| p[0]).fold(f32::NEG_INFINITY, f32::max);
    let ymin = pts.iter().map(|p| p[1]).fold(f32::INFINITY, f32::min);
    let ymax = pts.iter().map(|p| p[1]).fold(f32::NEG_INFINITY, f32::max);
    let (x0, x1, y0, y1) = (
        cv_floor(xmin as f64),
        cv_floor(xmax as f64),
        cv_floor(ymin as f64),
        cv_floor(ymax as f64),
    );
    (x0, y0, x1 - x0 + 1, y1 - y0 + 1)
}

// ---------- Subdiv2D ----------

const NEXT_AROUND_LEFT: i32 = 0x13;
const PREV_AROUND_ORG: i32 = 0x11;
const PREV_AROUND_DST: i32 = 0x33;

#[derive(Clone, Copy)]
struct QuadEdge {
    next: [i32; 4],
    pt: [i32; 4],
}

#[derive(Clone, Copy)]
#[allow(dead_code)] // kind: OpenCV's virtual/free flag, kept for fidelity
struct Vertex {
    first_edge: i32,
    kind: i32,
    pt: [f32; 2],
}

pub struct Subdiv2D {
    qedges: Vec<QuadEdge>,
    vtx: Vec<Vertex>,
    free_qedge: i32,
    free_point: i32,
    recent_edge: i32,
    top_left: [f32; 2],
    bottom_right: [f32; 2],
}

fn triangle_area(a: [f32; 2], b: [f32; 2], c: [f32; 2]) -> f64 {
    (b[0] as f64 - a[0] as f64) * (c[1] as f64 - a[1] as f64)
        - (b[1] as f64 - a[1] as f64) * (c[0] as f64 - a[0] as f64)
}

fn in_circle(pt: [f32; 2], a: [f32; 2], b: [f32; 2], c: [f32; 2]) -> i32 {
    let eps = f32::EPSILON as f64 * 0.125;
    let sq = |p: [f32; 2]| p[0] as f64 * p[0] as f64 + p[1] as f64 * p[1] as f64;
    let mut val = sq(a) * triangle_area(b, c, pt);
    val -= sq(b) * triangle_area(a, c, pt);
    val += sq(c) * triangle_area(a, b, pt);
    val -= sq(pt) * triangle_area(a, b, c);
    if val > eps {
        1
    } else if val < -eps {
        -1
    } else {
        0
    }
}

impl Subdiv2D {
    /// Subdiv2D(rect) with rect = (x, y, w, h).
    pub fn new(rect: (i64, i64, i64, i64)) -> Subdiv2D {
        let big = 3.0f32 * (rect.2.max(rect.3) as f32);
        let (rx, ry) = (rect.0 as f32, rect.1 as f32);
        let mut s = Subdiv2D {
            qedges: vec![QuadEdge {
                next: [0; 4],
                pt: [0; 4],
            }],
            vtx: vec![Vertex {
                first_edge: 0,
                kind: -1,
                pt: [0.0, 0.0],
            }],
            free_qedge: 0,
            free_point: 0,
            recent_edge: 0,
            top_left: [rx, ry],
            bottom_right: [rx + rect.2 as f32, ry + rect.3 as f32],
        };
        let pa = s.new_point([rx + big, ry], false, 0);
        let pb = s.new_point([rx, ry + big], false, 0);
        let pc = s.new_point([rx - big, ry - big], false, 0);
        let ab = s.new_edge();
        let bc = s.new_edge();
        let ca = s.new_edge();
        s.set_edge_points(ab, pa, pb);
        s.set_edge_points(bc, pb, pc);
        s.set_edge_points(ca, pc, pa);
        s.splice(ab, ca ^ 2);
        s.splice(bc, ab ^ 2);
        s.splice(ca, bc ^ 2);
        s.recent_edge = ab;
        s
    }
    fn rotate(edge: i32, r: i32) -> i32 {
        (edge & !3) + ((edge + r) & 3)
    }
    fn next_edge(&self, edge: i32) -> i32 {
        self.qedges[(edge >> 2) as usize].next[(edge & 3) as usize]
    }
    fn get_edge(&self, edge: i32, t: i32) -> i32 {
        let e = self.qedges[(edge >> 2) as usize].next[((edge + t) & 3) as usize];
        (e & !3) + ((e + (t >> 4)) & 3)
    }
    fn org(&self, edge: i32) -> i32 {
        self.qedges[(edge >> 2) as usize].pt[(edge & 3) as usize]
    }
    fn dst(&self, edge: i32) -> i32 {
        self.qedges[(edge >> 2) as usize].pt[((edge + 2) & 3) as usize]
    }
    fn pt(&self, v: i32) -> [f32; 2] {
        self.vtx[v as usize].pt
    }
    fn splice(&mut self, a: i32, b: i32) {
        let a_next = self.qedges[(a >> 2) as usize].next[(a & 3) as usize];
        let b_next = self.qedges[(b >> 2) as usize].next[(b & 3) as usize];
        let a_rot = Self::rotate(a_next, 1);
        let b_rot = Self::rotate(b_next, 1);
        self.qedges[(a >> 2) as usize].next[(a & 3) as usize] = b_next;
        self.qedges[(b >> 2) as usize].next[(b & 3) as usize] = a_next;
        let a_rot_next = self.qedges[(a_rot >> 2) as usize].next[(a_rot & 3) as usize];
        let b_rot_next = self.qedges[(b_rot >> 2) as usize].next[(b_rot & 3) as usize];
        self.qedges[(a_rot >> 2) as usize].next[(a_rot & 3) as usize] = b_rot_next;
        self.qedges[(b_rot >> 2) as usize].next[(b_rot & 3) as usize] = a_rot_next;
    }
    fn set_edge_points(&mut self, edge: i32, org: i32, dst: i32) {
        self.qedges[(edge >> 2) as usize].pt[(edge & 3) as usize] = org;
        self.qedges[(edge >> 2) as usize].pt[((edge + 2) & 3) as usize] = dst;
        self.vtx[org as usize].first_edge = edge;
        self.vtx[dst as usize].first_edge = edge ^ 2;
    }
    fn connect_edges(&mut self, a: i32, b: i32) -> i32 {
        let edge = self.new_edge();
        let al = self.get_edge(a, NEXT_AROUND_LEFT);
        self.splice(edge, al);
        self.splice(edge ^ 2, b);
        let (o, d) = (self.dst(a), self.org(b));
        self.set_edge_points(edge, o, d);
        edge
    }
    fn swap_edges(&mut self, edge: i32) {
        let sedge = edge ^ 2;
        let a = self.get_edge(edge, PREV_AROUND_ORG);
        let b = self.get_edge(sedge, PREV_AROUND_ORG);
        self.splice(edge, a);
        self.splice(sedge, b);
        let (o, d) = (self.dst(a), self.dst(b));
        self.set_edge_points(edge, o, d);
        let al = self.get_edge(a, NEXT_AROUND_LEFT);
        self.splice(edge, al);
        let bl = self.get_edge(b, NEXT_AROUND_LEFT);
        self.splice(sedge, bl);
    }
    fn is_right_of(&self, pt: [f32; 2], edge: i32) -> i32 {
        let (o, d) = (self.pt(self.org(edge)), self.pt(self.dst(edge)));
        let a = triangle_area(pt, d, o);
        (a > 0.0) as i32 - (a < 0.0) as i32
    }
    fn new_edge(&mut self) -> i32 {
        if self.free_qedge <= 0 {
            self.qedges.push(QuadEdge {
                next: [0; 4],
                pt: [0; 4],
            });
            self.free_qedge = self.qedges.len() as i32 - 1;
        }
        let edge = self.free_qedge * 4;
        self.free_qedge = self.qedges[(edge >> 2) as usize].next[1];
        self.qedges[(edge >> 2) as usize] = QuadEdge {
            next: [edge, edge + 3, edge + 2, edge + 1],
            pt: [0; 4],
        };
        edge
    }
    fn delete_edge(&mut self, edge: i32) {
        let p = self.get_edge(edge, PREV_AROUND_ORG);
        self.splice(edge, p);
        let sedge = edge ^ 2;
        let p = self.get_edge(sedge, PREV_AROUND_ORG);
        self.splice(sedge, p);
        let e = (edge >> 2) as usize;
        self.qedges[e].next[0] = 0;
        self.qedges[e].next[1] = self.free_qedge;
        self.free_qedge = e as i32;
    }
    fn new_point(&mut self, pt: [f32; 2], is_virtual: bool, first_edge: i32) -> i32 {
        if self.free_point == 0 {
            self.vtx.push(Vertex {
                first_edge: 0,
                kind: -1,
                pt: [0.0, 0.0],
            });
            self.free_point = self.vtx.len() as i32 - 1;
        }
        let v = self.free_point;
        self.free_point = self.vtx[v as usize].first_edge;
        self.vtx[v as usize] = Vertex {
            first_edge,
            kind: is_virtual as i32,
            pt,
        };
        v
    }
    /// locate: (location, edge, vertex); location 0 inside, 1 vertex, 2 on edge, -2 error.
    fn locate(&mut self, pt: [f32; 2]) -> Result<(i32, i32, i32), String> {
        const INSIDE: i32 = 0;
        const VERTEX: i32 = 1;
        const ON_EDGE: i32 = 2;
        const ERROR: i32 = -2;
        let max_edges = (self.qedges.len() * 4) as i32;
        if self.qedges.len() < 4 {
            return Err("Subdivision is empty".into());
        }
        if pt[0] < self.top_left[0]
            || pt[1] < self.top_left[1]
            || pt[0] >= self.bottom_right[0]
            || pt[1] >= self.bottom_right[1]
        {
            return Err("point outside the subdivision (OpenCV raises StsOutOfRange)".into());
        }
        let mut edge = self.recent_edge;
        let mut location = ERROR;
        let mut right_of_curr = self.is_right_of(pt, edge);
        if right_of_curr > 0 {
            edge ^= 2;
            right_of_curr = -right_of_curr;
        }
        for _ in 0..max_edges {
            let onext = self.next_edge(edge);
            let dprev = self.get_edge(edge, PREV_AROUND_DST);
            let right_of_onext = self.is_right_of(pt, onext);
            let right_of_dprev = self.is_right_of(pt, dprev);
            if right_of_dprev > 0 {
                if right_of_onext > 0 || (right_of_onext == 0 && right_of_curr == 0) {
                    location = INSIDE;
                    break;
                } else {
                    right_of_curr = right_of_onext;
                    edge = onext;
                }
            } else if right_of_onext > 0 {
                if right_of_dprev == 0 && right_of_curr == 0 {
                    location = INSIDE;
                    break;
                } else {
                    right_of_curr = right_of_dprev;
                    edge = dprev;
                }
            } else if right_of_curr == 0 && self.is_right_of(self.pt(self.dst(onext)), edge) >= 0 {
                edge ^= 2;
            } else {
                right_of_curr = right_of_onext;
                edge = onext;
            }
        }
        self.recent_edge = edge;
        let mut vertex = 0;
        if location == INSIDE {
            let (o, d) = (self.pt(self.org(edge)), self.pt(self.dst(edge)));
            let t1 = (pt[0] - o[0]).abs() as f64 + (pt[1] - o[1]).abs() as f64;
            let t2 = (pt[0] - d[0]).abs() as f64 + (pt[1] - d[1]).abs() as f64;
            let t3 = (o[0] - d[0]).abs() as f64 + (o[1] - d[1]).abs() as f64;
            let eps = f32::EPSILON as f64;
            if t1 < eps {
                location = VERTEX;
                vertex = self.org(edge);
                edge = 0;
            } else if t2 < eps {
                location = VERTEX;
                vertex = self.dst(edge);
                edge = 0;
            } else if (t1 < t3 || t2 < t3) && triangle_area(pt, o, d).abs() < eps {
                location = ON_EDGE;
                vertex = 0;
            }
        }
        if location == ERROR {
            edge = 0;
            vertex = 0;
        }
        Ok((location, edge, vertex))
    }
    pub fn insert(&mut self, pt: [f32; 2]) -> Result<i32, String> {
        let (location, mut curr_edge, curr_point) = self.locate(pt)?;
        match location {
            1 => return Ok(curr_point),
            2 => {
                let deleted = curr_edge;
                curr_edge = self.get_edge(curr_edge, PREV_AROUND_ORG);
                self.recent_edge = curr_edge;
                self.delete_edge(deleted);
            }
            0 => {}
            _ => return Err("Subdiv2D::locate failed".into()),
        }
        let curr_point = self.new_point(pt, false, 0);
        let mut base_edge = self.new_edge();
        let first_point = self.org(curr_edge);
        self.set_edge_points(base_edge, first_point, curr_point);
        self.splice(base_edge, curr_edge);
        loop {
            base_edge = self.connect_edges(curr_edge, base_edge ^ 2);
            curr_edge = self.get_edge(base_edge, PREV_AROUND_ORG);
            if self.dst(curr_edge) == first_point {
                break;
            }
        }
        curr_edge = self.get_edge(base_edge, PREV_AROUND_ORG);
        let max_edges = (self.qedges.len() * 4) as i32;
        for _ in 0..max_edges {
            let temp_edge = self.get_edge(curr_edge, PREV_AROUND_ORG);
            let temp_dst = self.dst(temp_edge);
            let curr_org = self.org(curr_edge);
            let curr_dst = self.dst(curr_edge);
            if self.is_right_of(self.pt(temp_dst), curr_edge) > 0
                && in_circle(
                    self.pt(curr_org),
                    self.pt(temp_dst),
                    self.pt(curr_dst),
                    self.pt(curr_point),
                ) < 0
            {
                self.swap_edges(curr_edge);
                curr_edge = self.get_edge(curr_edge, PREV_AROUND_ORG);
            } else if curr_org == first_point {
                break;
            } else {
                let n = self.next_edge(curr_edge);
                curr_edge = self.get_edge(n, 0x20); // PREV_AROUND_LEFT
            }
        }
        Ok(curr_point)
    }
    /// getTriangleList: (a, b, c) corner points of each triangle inside the rect.
    pub fn triangle_list(&self) -> Vec<[[f32; 2]; 3]> {
        let total = self.qedges.len() * 4;
        let mut mask = vec![false; total];
        let (x, y) = (self.top_left[0], self.top_left[1]);
        let (w, h) = (self.bottom_right[0] - x, self.bottom_right[1] - y);
        let contains = |p: [f32; 2]| x <= p[0] && p[0] < x + w && y <= p[1] && p[1] < y + h;
        let mut out = Vec::new();
        let mut i = 4;
        while i < total {
            if mask[i] {
                i += 2;
                continue;
            }
            let ea = i as i32;
            let a = self.pt(self.org(ea));
            if !contains(a) {
                i += 2;
                continue;
            }
            let eb = self.get_edge(ea, NEXT_AROUND_LEFT);
            let b = self.pt(self.org(eb));
            if !contains(b) {
                i += 2;
                continue;
            }
            let ec = self.get_edge(eb, NEXT_AROUND_LEFT);
            let c = self.pt(self.org(ec));
            if !contains(c) {
                i += 2;
                continue;
            }
            mask[ea as usize] = true;
            mask[eb as usize] = true;
            mask[ec as usize] = true;
            out.push([a, b, c]);
            i += 2;
        }
        out
    }
}
