//! homr's prepare_staff_image (staff_parsing.py) and staff dewarping
//! (staff_dewarping.py): from the page image and one staff to the canvas the
//! transformer model reads.

use crate::braces::{get_at, point_unit};
use crate::contours::find_contours;
use crate::npmath::{np_mean, py_int, py_round};
use crate::raster::fill_convex_poly;
use crate::staff::{Staff, StaffPoint};
use crate::warp::{
    bounding_rect_f, get_affine_transform, resize, resize_region, warp_affine, Gray, Subdiv2D,
};

pub const CANVAS_W: usize = 1280;
pub const CANVAS_H: usize = 256;

/// StaffRegions: the (min_y, max_y) of every staff on the page.
pub struct Regions {
    pub centers: Vec<(f64, f64)>,
}

impl Regions {
    fn above(&self, y: f64) -> f64 {
        self.centers
            .iter()
            .filter(|c| c.0 < y)
            .map(|c| c.1)
            .fold(None, |m: Option<f64>, v| Some(m.map_or(v, |m| m.max(v))))
            .unwrap_or(0.0)
    }
    fn below(&self, y: f64) -> f64 {
        self.centers
            .iter()
            .filter(|c| c.1 > y)
            .map(|c| c.0)
            .fold(None, |m: Option<f64>, v| Some(m.map_or(v, |m| m.min(v))))
            .unwrap_or(1e12)
    }
}

fn median(mut v: Vec<f64>) -> f64 {
    v.sort_by(|a, b| a.partial_cmp(b).unwrap());
    let n = v.len();
    if n % 2 == 1 {
        v[n / 2]
    } else {
        (v[n / 2 - 1] + v[n / 2]) / 2.0
    }
}

fn unit_size(s: &Staff) -> f64 {
    median(s.grid.iter().map(point_unit).collect())
}

/// get_tr_omr_canvas_size((height, width)) -> [width, height]
fn canvas_size(height: i64, width: i64) -> (i64, i64) {
    let ratio = CANVAS_H as f64 / CANVAS_W as f64;
    if height as f64 / width as f64 > ratio {
        (
            py_int(width as f64 / height as f64 * CANVAS_H as f64),
            CANVAS_H as i64,
        )
    } else {
        (
            CANVAS_W as i64,
            py_int(height as f64 / width as f64 * CANVAS_W as f64),
        )
    }
}

/// crop_image_and_return_new_top's rectangle (x0, y0, x1, y1) on a w x h image
fn crop_rect(w: usize, h: usize, x1: f64, y1: f64, x2: f64, y2: f64) -> (i64, i64, i64, i64) {
    let lx = |x: f64| 0.max((w as i64 - 1).min(py_round(x)));
    let ly = |y: f64| 0.max((h as i64 - 1).min(py_round(y)));
    (
        lx(x1.min(x2)),
        ly(y1.min(y2)),
        lx(x1.max(x2)),
        ly(y1.max(y2)),
    )
}

/// crop_image_and_return_new_top
fn crop(img: &Gray, x1: f64, y1: f64, x2: f64, y2: f64) -> (Gray, (i64, i64)) {
    let lx = |x: f64| 0.max((img.w as i64 - 1).min(py_round(x)));
    let ly = |y: f64| 0.max((img.h as i64 - 1).min(py_round(y)));
    let (xa, xb, ya, yb) = (
        lx(x1.min(x2)),
        lx(x1.max(x2)),
        ly(y1.min(y2)),
        ly(y1.max(y2)),
    );
    (img.slice(xa, ya, xb, yb), (xa, ya))
}

/// Staff.transform_coordinates with (x - dx) * s, (y - dy) * s.
fn transform_staff(s: &Staff, dx: f64, dy: f64, scale: f64) -> Result<Staff, String> {
    let grid = s
        .grid
        .iter()
        .map(|p| {
            let xs: Vec<f64> = p.y.iter().map(|_| (p.x - dx) * scale).collect();
            StaffPoint {
                x: np_mean(&xs),
                y: p.y.iter().map(|y| (y - dy) * scale).collect(),
                angle: p.angle,
            }
        })
        .collect();
    Staff::from_grid(grid)
}

fn is_point_on_image(p: (i64, i64), img: &Gray) -> bool {
    let m = 10;
    !(p.0 < m || p.0 > img.w as i64 - m || p.1 < m || p.1 > img.h as i64 - m)
}

type Lines = Vec<Vec<(i64, i64)>>;

fn span_and_optimal_points(staff: &Staff, img: &Gray) -> (Lines, Lines) {
    let (mut span, mut optimal) = (Vec::new(), Vec::new());
    let step = py_int(img.h as f64 / 6.0);
    if step == 0 {
        return (span, optimal);
    }
    let mut first: Option<f64> = None;
    let mut y = 2i64;
    while y < img.h as i64 - 2 {
        let mut line = Vec::new();
        let mut x = 2i64;
        while x < img.w as i64 {
            if let Some(p) = get_at(staff, x as f64) {
                let off = p.y[2];
                let delta = match first {
                    Some(f) if f != 0.0 => py_int(off - f),
                    _ => {
                        first = Some(off);
                        0
                    }
                };
                let pt = (x, y + delta);
                if is_point_on_image(pt, img) {
                    line.push(pt);
                }
            }
            x += 80;
        }
        if line.len() > 2 {
            let avg = line.iter().map(|p| p.1).sum::<i64>() as f64 / line.len() as f64;
            optimal.push(line.iter().map(|p| (p.0, py_int(avg))).collect());
            span.push(line);
        }
        y += step;
    }
    (span, optimal)
}

pub struct Tri {
    src: [[f32; 2]; 3],
    dst: [[f32; 2]; 3],
    /// estimate(): getAffineTransform(src, dst), None for a degenerate triangle
    mat: Option<[f64; 6]>,
}

fn degenerate(t: &[[f32; 2]; 3]) -> bool {
    let v1 = [t[1][0] - t[0][0], t[1][1] - t[0][1]];
    let v2 = [t[2][0] - t[0][0], t[2][1] - t[0][1]];
    let area = (v1[0] * v2[1] - v1[1] * v2[0]).abs() / 2.0f32;
    area < 1e-6f32
}

/// PiecewiseAffineTransform.estimate: the triangles homr warps (None if dewarping failed).
fn dewarp_triangles(img: &Gray, staff: &Staff) -> Option<Vec<Tri>> {
    let (mut src, mut dst) = span_and_optimal_points(staff, img);
    let (w, h) = (img.w as i64, img.h as i64);
    for lines in [&mut src, &mut dst] {
        for line in lines.iter_mut() {
            let (y0, y1) = (line[0].1, line[line.len() - 1].1);
            line.insert(0, (0, y0));
            line.push((w, y1));
        }
        lines.insert(0, vec![(0, 0), (w, 0)]);
        lines.push(vec![(0, h), (w, h)]);
    }
    let sp: Vec<[f32; 2]> = src
        .iter()
        .flatten()
        .map(|p| [p.0 as f32, p.1 as f32])
        .collect();
    let dp: Vec<[f32; 2]> = dst
        .iter()
        .flatten()
        .map(|p| [p.0 as f32, p.1 as f32])
        .collect();
    if sp.len() < 3 {
        return None;
    }
    let r = bounding_rect_f(&sp);
    let mut sub = Subdiv2D::new((r.0 - 10, r.1 - 10, r.2 + 20, r.3 + 20));
    for p in &sp {
        if sub.insert(*p).is_err() {
            return None;
        }
    }
    let find = |q: [f32; 2]| -> Option<usize> {
        let mut best = 0usize;
        let mut best_d = f32::INFINITY;
        for (i, p) in sp.iter().enumerate() {
            let (dx, dy) = (p[0] - q[0], p[1] - q[1]);
            let d = (dx * dx + dy * dy).sqrt();
            if d < best_d {
                best_d = d;
                best = i;
            }
        }
        (best_d < 1e-3).then_some(best)
    };
    let mut tris = Vec::new();
    for t in sub.triangle_list() {
        if let (Some(a), Some(b), Some(c)) = (find(t[0]), find(t[1]), find(t[2])) {
            let (src, dst) = ([sp[a], sp[b], sp[c]], [dp[a], dp[b], dp[c]]);
            let mat = if degenerate(&src) || degenerate(&dst) {
                None
            } else {
                Some(get_affine_transform(&src, &dst))
            };
            tris.push(Tri { src, dst, mat });
        }
    }
    Some(tris)
}

/// PiecewiseAffineTransform.warp_image(image, fill_color=1, order=1)
fn warp_image(img: &Gray, tris: &[Tri]) -> Gray {
    let mut out = Gray::new(img.w, img.h, 1);
    for t in tris {
        if degenerate(&t.src) || degenerate(&t.dst) {
            continue; // estimate() stored None for this triangle
        }
        let sr = bounding_rect_f(&t.src);
        let dr = bounding_rect_f(&t.dst);
        if sr.2 <= 0 || sr.3 <= 0 || dr.2 <= 0 || dr.3 <= 0 {
            continue;
        }
        let shift = |tri: &[[f32; 2]; 3], r: (i64, i64, i64, i64)| {
            tri.map(|p| [p[0] - r.0 as f32, p[1] - r.1 as f32])
        };
        let (sc, dc) = (shift(&t.src, sr), shift(&t.dst, dr));
        if degenerate(&sc) || degenerate(&dc) {
            continue;
        }
        let m = get_affine_transform(&sc, &dc);
        let src_cropped = img.slice(sr.0, sr.1, sr.0 + sr.2, sr.1 + sr.3);
        if src_cropped.w == 0 || src_cropped.h == 0 {
            continue;
        }
        let (dw, dh) = (dr.2 as usize, dr.3 as usize);
        let warped = warp_affine(&src_cropped, &m, dw, dh, 1);
        let mut mask = vec![0u8; dw * dh];
        let poly: Vec<[i64; 2]> = dc.iter().map(|p| [p[0] as i64, p[1] as i64]).collect();
        fill_convex_poly(&mut mask, dh, dw, &poly, 255);
        let (y1, y2, x1, x2) = (dr.1, dr.1 + dr.3, dr.0, dr.0 + dr.2);
        let (yc1, yc2, xc1, xc2) = (
            y1.max(0),
            y2.min(img.h as i64),
            x1.max(0),
            x2.min(img.w as i64),
        );
        if yc2 <= yc1 || xc2 <= xc1 {
            continue;
        }
        for y in yc1..yc2 {
            for x in xc1..xc2 {
                let (my, mx) = ((y - y1) as usize, (x - x1) as usize);
                if mask[my * dw + mx] > 0 {
                    out.data[y as usize * img.w + x as usize] = warped.data[my * dw + mx];
                }
            }
        }
    }
    out
}

/// remove_black_contours_at_edges_of_image
fn remove_black_contours(gray: &mut Gray, unit: f64) {
    let thresh: Vec<u8> = gray
        .data
        .iter()
        .map(|&v| if v > 97 { 0 } else { 255 })
        .collect();
    let limit = 2.0 * unit;
    for c in find_contours(&thresh, gray.h, gray.w) {
        let x = c.iter().map(|p| p[0]).min().unwrap() as i64;
        let y = c.iter().map(|p| p[1]).min().unwrap() as i64;
        let w = c.iter().map(|p| p[0]).max().unwrap() as i64 - x + 1;
        let h = c.iter().map(|p| p[1]).max().unwrap() as i64 - y + 1;
        if (w as f64) < limit || (h as f64) < limit {
            continue;
        }
        if !(x == 0 || y == 0 || x + w == gray.w as i64 || y + h == gray.h as i64) {
            continue;
        }
        let mut sum = 0u64;
        for yy in y..y + h {
            for xx in x..x + w {
                sum += thresh[yy as usize * gray.w + xx as usize] as u64;
            }
        }
        if (sum as f64 / (w * h) as f64) < 127.0 {
            continue;
        }
        for yy in y..y + h {
            for xx in x..x + w {
                gray.data[yy as usize * gray.w + xx as usize] = 255;
            }
        }
    }
}

/// center_image_on_canvas
fn center_on_canvas(img: &Gray, size: (i64, i64)) -> Gray {
    let resized = resize(img, size.0 as usize, size.1 as usize);
    let mut out = Gray::new(CANVAS_W, CANVAS_H, 255);
    let y_off = (CANVAS_H as i64 - resized.h as i64).div_euclid(2) as usize;
    for y in 0..resized.h {
        out.data[(y_off + y) * CANVAS_W..(y_off + y) * CANVAS_W + resized.w]
            .copy_from_slice(&resized.data[y * resized.w..(y + 1) * resized.w]);
    }
    out
}

/// What prepare_staff_image did, to map points on the canvas back to the page
/// (its `to_page` chain).
pub struct StaffMapping {
    canvas_y_offset: i64,
    before_canvas: (usize, usize),  // (h, w)
    canvas_content: (usize, usize), // (h, w)
    crop2: (i64, i64),
    tris: Option<Vec<Tri>>,
    crop1: (i64, i64),
    page: (usize, usize),
    resized: (usize, usize),
}

/// undo_resize(shape_before, shape_after) applied to a point
fn undo_resize(p: (f64, f64), before: (usize, usize), after: (usize, usize)) -> (f64, f64) {
    let sx = after.1 as f64 / before.1 as f64;
    let sy = after.0 as f64 / before.0 as f64;
    (p.0 / sx, p.1 / sy)
}

fn point_in_triangle(p: (f64, f64), t: &[[f32; 2]; 3]) -> bool {
    let cross = |a: [f32; 2], b: [f32; 2]| -> f64 {
        ((b[0] - a[0]) * (p.1 as f32 - a[1]) - (b[1] - a[1]) * (p.0 as f32 - a[0])) as f64
    };
    let (d1, d2, d3) = (cross(t[0], t[1]), cross(t[1], t[2]), cross(t[2], t[0]));
    let neg = d1 < 0.0 || d2 < 0.0 || d3 < 0.0;
    let pos = d1 > 0.0 || d2 > 0.0 || d3 > 0.0;
    !(neg && pos)
}

/// cv2.invertAffineTransform for a double matrix
fn invert_affine(m: &[f64; 6]) -> [f64; 6] {
    let d = m[0] * m[4] - m[1] * m[3];
    let d = if d != 0.0 { 1.0 / d } else { 0.0 };
    let (a11, a22, a12, a21) = (m[4] * d, m[0] * d, -m[1] * d, -m[3] * d);
    let b1 = -a11 * m[2] - a12 * m[5];
    let b2 = -a21 * m[2] - a22 * m[5];
    [a11, a12, b1, a21, a22, b2]
}

impl StaffMapping {
    /// PiecewiseAffineTransform.inverse_transform_point
    fn undewarp(&self, p: (f64, f64)) -> (f64, f64) {
        let Some(tris) = &self.tris else { return p };
        for t in tris {
            let Some(m) = &t.mat else { continue };
            if !point_in_triangle(p, &t.dst) {
                continue;
            }
            let inv = invert_affine(m);
            let v = [p.0 as f32 as f64, p.1 as f32 as f64, 1.0f32 as f64];
            return (
                inv[0] * v[0] + inv[1] * v[1] + inv[2] * v[2],
                inv[3] * v[0] + inv[4] * v[1] + inv[5] * v[2],
            );
        }
        p
    }
    /// staff_to_page: a point on the canvas to the page image
    pub fn to_page(&self, p: (f64, f64)) -> (f64, f64) {
        let p = (p.0 + 0.0, p.1 + (-self.canvas_y_offset) as f64);
        let p = undo_resize(p, self.before_canvas, self.canvas_content);
        let p = (p.0 + self.crop2.0 as f64, p.1 + self.crop2.1 as f64);
        let p = self.undewarp(p);
        let p = (p.0 + self.crop1.0 as f64, p.1 + self.crop1.1 as f64);
        undo_resize(p, self.page, self.resized)
    }
}

pub struct PreparedStaff {
    pub canvas: Gray,
    /// the staff in the coordinates of the cropped, scaled staff image
    pub staff: Staff,
    pub mapping: StaffMapping,
}

/// prepare_staff_image: the uint8 canvas the transformer reads for this staff.
pub fn prepare_staff_image(staff: &Staff, image: &Gray, regions: &Regions) -> Result<Gray, String> {
    Ok(prepare_staff(staff, image, regions)?.canvas)
}

/// prepare_staff_image with the transformed staff and the mapping back to the page.
pub fn prepare_staff(
    staff: &Staff,
    image: &Gray,
    regions: &Regions,
) -> Result<PreparedStaff, String> {
    let u = unit_size(staff);
    let region = [
        py_int(staff.min_x - 2.0 * u),
        py_int((staff.min_y - 4.0 * u).max(regions.above(staff.min_y))),
        py_int(staff.max_x + 2.0 * u),
        py_int((staff.max_y + 4.0 * u).min(regions.below(staff.max_y))),
    ];
    let dims = canvas_size(region[3] - region[1], region[2] - region[0]);
    let scale = dims.1 as f64 / (region[3] - region[1]) as f64;
    let (rw, rh) = (
        py_int(image.w as f64 * scale) as usize,
        py_int(image.h as f64 * scale) as usize,
    );
    let rf: Vec<f64> = region
        .iter()
        .map(|&v| (v as f64 * scale).round_ties_even())
        .collect();
    // crop(resize(page)): only the cropped part of the resized page is computed
    let (xa, ya, xb, yb) = crop_rect(
        rw,
        rh,
        rf[0] - 10.0,
        rf[1] - 50.0,
        rf[2] + 10.0,
        rf[3] + 50.0,
    );
    let cropped = resize_region(
        image,
        rw,
        rh,
        xa as usize,
        ya as usize,
        xb as usize,
        yb as usize,
    );
    let tl = (xa, ya);
    let step2 = [
        rf[0] - tl.0 as f64,
        rf[1] - tl.1 as f64,
        rf[2] - tl.0 as f64,
        rf[3] - tl.1 as f64,
    ];
    let staff2 = transform_staff(staff, tl.0 as f64 / scale, tl.1 as f64 / scale, scale)?;
    let tris = dewarp_triangles(&cropped, &staff2);
    let dewarped = match &tris {
        Some(t) => warp_image(&cropped, t),
        None => cropped,
    };
    let (mut cropped2, tl2) = crop(&dewarped, step2[0], step2[1], step2[2], step2[3]);
    remove_black_contours(&mut cropped2, unit_size(&staff2));
    let before_canvas = (cropped2.h, cropped2.w);
    let canvas = center_on_canvas(&cropped2, dims);
    let content = (dims.1 as usize, dims.0 as usize);
    let mapping = StaffMapping {
        canvas_y_offset: (CANVAS_H as i64 - content.0 as i64).div_euclid(2),
        before_canvas,
        canvas_content: content,
        crop2: tl2,
        tris,
        crop1: tl,
        page: (image.h, image.w),
        resized: (rh, rw),
    };
    Ok(PreparedStaff {
        canvas,
        staff: staff2,
        mapping,
    })
}

/// The model's input from the canvas: (pixel / 255 - 0.7931) / 0.1738, float32.
pub fn to_model_input(canvas: &Gray) -> Vec<f32> {
    canvas
        .data
        .iter()
        .map(|&v| ((v as f64 / 255.0 - 0.7931) / 0.1738) as f32)
        .collect()
}
