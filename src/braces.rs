//! Braces, brackets and grand staffs: from staffs to homr's MultiStaffs
//! (homr/brace_dot_detection.py, model.py: MultiStaff, Staff.merge).

use crate::npmath::{np_mean, py_round};
use crate::raster::{ellipse_kernel, fill_poly, morph, subtract};
use crate::staff::{Staff, StaffPoint};
use crate::symbols::{rotated_boxes, Shape};

fn median(mut v: Vec<f64>) -> f64 {
    v.sort_by(|a, b| a.partial_cmp(b).unwrap());
    let n = v.len();
    if n % 2 == 1 {
        v[n / 2]
    } else {
        (v[n / 2 - 1] + v[n / 2]) / 2.0
    }
}

pub fn point_unit(p: &StaffPoint) -> f64 {
    let d: Vec<f64> = p.y.windows(2).map(|w| w[1] - w[0]).collect();
    np_mean(&d)
}

pub fn unit_size(s: &Staff) -> f64 {
    median(s.grid.iter().map(point_unit).collect())
}

/// Staff.get_at (homr's: the first grid point closest in x, within 50).
pub fn get_at(s: &Staff, x: f64) -> Option<&StaffPoint> {
    let mut best: Option<&StaffPoint> = None;
    for p in &s.grid {
        if best.is_none_or(|b| (p.x - x).abs() < (b.x - x).abs()) {
            best = Some(p);
        }
    }
    best.filter(|p| (p.x - x).abs() <= 50.0)
}

fn y_distance_to(s: &Staff, pt: (f64, f64)) -> f64 {
    match get_at(s, pt.0) {
        None => 1e10,
        Some(p) => {
            p.y.iter()
                .map(|y| (y - pt.1).abs())
                .fold(f64::INFINITY, f64::min)
        }
    }
}

/// prepare_brace_dot_image
pub fn brace_dot_image(symbols: &[u8], staff: &[u8], h: usize, w: usize) -> Vec<u8> {
    let d = subtract(symbols, staff);
    let out = morph(&d, h, w, &ellipse_kernel(1, 5), false);
    morph(&out, h, w, &ellipse_kernel(5, 35), true)
}

/// _trim_symbol_to_core_span
fn trim(s: &Shape) -> Shape {
    let c = &s.contour;
    let (x, y) = (
        c.iter().map(|p| p[0]).min().unwrap() as i64,
        c.iter().map(|p| p[1]).min().unwrap() as i64,
    );
    let w = (c.iter().map(|p| p[0]).max().unwrap() as i64 - x + 1) as usize;
    let h = (c.iter().map(|p| p[1]).max().unwrap() as i64 - y + 1) as usize;
    let mut mask = vec![0u8; w * h];
    let pts: Vec<[i64; 2]> = c
        .iter()
        .map(|p| [p[0] as i64 - x, p[1] as i64 - y])
        .collect();
    fill_poly(&mut mask, h, w, &pts, 255);
    let widths: Vec<usize> = (0..h)
        .map(|r| mask[r * w..(r + 1) * w].iter().filter(|&&v| v > 0).count())
        .collect();
    let max_width = *widths.iter().max().unwrap();
    if max_width == 0 {
        return s.clone();
    }
    let core: Vec<usize> = (0..h)
        .filter(|&r| widths[r] as f64 >= max_width as f64 * 0.5)
        .collect();
    if core.is_empty() {
        return s.clone();
    }
    let (cmin, cmax) = (y + core[0] as i64, y + *core.last().unwrap() as i64 + 1);
    if cmax - cmin <= 0 {
        return s.clone();
    }
    Shape::rect(
        [
            s.cx,
            (cmin + cmax) as f64 / 2.0,
            s.w,
            (cmax - cmin) as f64,
            s.angle,
        ],
        s.contour.clone(),
    )
}

fn filter_tall(brace_dot: Vec<Shape>, staffs: &[Staff], units: &[f64]) -> Vec<Shape> {
    let rough = units[0];
    brace_dot
        .into_iter()
        .filter(|s| s.h > 2.0 * rough && s.w < 3.0 * rough && s.w >= 5.0)
        .filter(|s| {
            let mut best = 0;
            let mut best_d = f64::INFINITY;
            for (k, st) in staffs.iter().enumerate() {
                let d = y_distance_to(st, (s.cx, s.cy));
                if d < best_d {
                    best_d = d;
                    best = k;
                }
            }
            s.h > 4.0 * units[best]
        })
        .collect()
}

/// make_box_thicker for a rotated box (t <= 0 keeps it as is).
fn thicker(s: &Shape, t: i64) -> Shape {
    if t <= 0 {
        return s.clone();
    }
    Shape::rect(
        [s.cx, s.cy, s.w + t as f64, s.h + t as f64, s.angle],
        vec![],
    )
}

/// StaffPoint.to_bounding_box: a zero-width box from the top to the bottom line.
fn point_box(p: &StaffPoint) -> Shape {
    let (x, y0, y1) = (p.x.trunc(), p.y[0].trunc(), p.y[p.y.len() - 1].trunc());
    let mut b = Shape::rect([x, (y0 + y1) / 2.0, 0.0, y1 - y0, 0.0], vec![]);
    // BoundingBox: no angle normalisation, size as given
    b.w = 0.0;
    b.h = y1 - y0;
    b
}

fn connections_at_lines(s1: &Staff, s2: &Staff, u1: f64, brace_dot: &[Shape]) -> Vec<usize> {
    let t = py_round(u1 * 2.0);
    (0..brace_dot.len())
        .filter(|&k| {
            let sym = &brace_dot[k];
            let th = thicker(sym, t);
            match (get_at(s1, sym.cx), get_at(s2, sym.cx)) {
                (Some(p1), Some(p2)) => {
                    th.is_overlapping(&point_box(p1)) && th.is_overlapping(&point_box(p2))
                }
                _ => false,
            }
        })
        .collect()
}

/// A MultiStaff: indices of its staffs (into the merged list) and connections.
#[derive(Clone, Debug)]
pub struct MultiStaff {
    pub staffs: Vec<Staff>,
    pub grand: Vec<bool>,
}

/// Staff.merge into a grand staff: the grid points both staffs have (by rounded x).
pub fn merge_staffs(a: &Staff, b: &Staff) -> Result<Staff, String> {
    let key = |p: &StaffPoint| py_round(p.x);
    let mut ga: std::collections::BTreeMap<i64, &StaffPoint> = std::collections::BTreeMap::new();
    for p in &a.grid {
        ga.insert(key(p), p);
    }
    let mut gb: std::collections::BTreeMap<i64, &StaffPoint> = std::collections::BTreeMap::new();
    for p in &b.grid {
        gb.insert(key(p), p);
    }
    let grid: Vec<StaffPoint> = ga
        .iter()
        .filter_map(|(k, pa)| {
            gb.get(k).map(|pb| {
                let mut y = pa.y.clone();
                y.extend_from_slice(&pb.y);
                y.sort_by(|u, v| u.partial_cmp(v).unwrap());
                StaffPoint {
                    x: pa.x,
                    y,
                    angle: (pa.angle + pb.angle) / 2.0,
                }
            })
        })
        .collect();
    let mut s = Staff::from_grid(grid)?;
    s.halves = Some(Box::new((a.clone(), b.clone())));
    Ok(s)
}

/// find_braces_brackets_and_grand_staff_lines: staffs (top to bottom) into MultiStaffs.
pub fn multi_staffs(
    staffs: &[Staff],
    symbols: &[u8],
    staff_mask: &[u8],
    h: usize,
    w: usize,
) -> Result<Vec<MultiStaff>, String> {
    let img = brace_dot_image(symbols, staff_mask, h, w);
    let brace_dot = rotated_boxes(&img, h, w, true, None, Some((100.0, -1.0)));
    let brace_dot: Vec<Shape> = brace_dot.iter().map(trim).collect();
    let units: Vec<f64> = staffs.iter().map(unit_size).collect();
    let brace_dot = filter_tall(brace_dot, staffs, &units);
    // each entry: staff indices (sorted by min_y) and connection keys (indices into brace_dot)
    let mut result: Vec<(Vec<usize>, Vec<usize>)> = Vec::new();
    let n = staffs.len();
    for i in 0..n {
        let mut neighbors = Vec::new();
        if i > 0 {
            neighbors.push(i - 1);
        }
        if i + 1 < n {
            neighbors.push(i + 1);
        }
        let mut any = false;
        for nb in neighbors {
            let conns = connections_at_lines(&staffs[i], &staffs[nb], units[i], &brace_dot);
            if !conns.is_empty() {
                let mut pair = vec![i, nb];
                pair.sort_by(|&a, &b| staffs[a].min_y.partial_cmp(&staffs[b].min_y).unwrap());
                result.push((pair, conns));
                any = true;
            }
        }
        if !any {
            result.push((vec![i], vec![]));
        }
    }
    // _merge_multi_staff_if_they_share_a_staff
    let mut merged: Vec<(Vec<usize>, Vec<usize>)> = Vec::new();
    for ms in result {
        match merged
            .iter()
            .position(|ex| ex.0.iter().any(|s| ms.0.contains(s)))
        {
            Some(k) => {
                let ex = merged.remove(k);
                let mut st = ex.0.clone();
                for s in &ms.0 {
                    if !st.contains(s) {
                        st.push(*s);
                    }
                }
                st.sort_by(|&a, &b| staffs[a].min_y.partial_cmp(&staffs[b].min_y).unwrap());
                let mut co = ex.1.clone();
                for c in &ms.1 {
                    if !co.iter().any(|x| same_box(&brace_dot[*x], &brace_dot[*c])) {
                        co.push(*c);
                    }
                }
                merged.push((st, co));
            }
            None => merged.push(ms),
        }
    }
    merged
        .into_iter()
        .map(|(idx, _)| create_grandstaffs(&idx, staffs, &units, &brace_dot))
        .collect()
}

fn same_box(a: &Shape, b: &Shape) -> bool {
    (a.cx, a.cy, a.w, a.h, a.angle) == (b.cx, b.cy, b.w, b.h, b.angle)
}

fn score(
    sym: &Shape,
    upper: &Staff,
    lower: &Staff,
    uu: f64,
    ul: f64,
    above: Option<&Staff>,
    below: Option<&Staff>,
) -> f64 {
    let unit = (uu + ul) / 2.0;
    let x_thr = 5.0 * unit;
    let y_thr = 0.5 * sym.h;
    let smin = sym.cy - sym.h / 2.0;
    let smax = sym.cy + sym.h / 2.0;
    if let Some(a) = above {
        if smin < (a.max_y + upper.min_y) / 2.0 {
            return 0.0;
        }
    }
    if let Some(b) = below {
        if smax > (lower.max_y + b.min_y) / 2.0 {
            return 0.0;
        }
    }
    let x_distance = (upper.min_x - sym.cx)
        .abs()
        .min((lower.min_x - sym.cx).abs());
    let y_overlap = smax.min(lower.max_y) - smin.max(upper.min_y);
    if !(x_distance < x_thr && y_overlap > y_thr && y_overlap > x_distance) {
        return 0.0;
    }
    let union = smax.max(lower.max_y) - smin.min(upper.min_y);
    let iou = if union > 0.0 { y_overlap / union } else { 0.0 };
    iou - x_distance / x_thr
}

fn create_grandstaffs(
    idx: &[usize],
    staffs: &[Staff],
    units: &[f64],
    brace_dot: &[Shape],
) -> Result<MultiStaff, String> {
    let plain = || MultiStaff {
        staffs: idx.iter().map(|&i| staffs[i].clone()).collect(),
        grand: vec![false; idx.len()],
    };
    if idx.len() < 2 {
        return Ok(plain());
    }
    if brace_dot.is_empty() {
        return Err("max() of an empty sequence (homr raises ValueError)".into());
    }
    let mut pairs: Vec<(usize, f64)> = Vec::new();
    for i in 0..idx.len() - 1 {
        let above = if i > 0 {
            Some(&staffs[idx[i - 1]])
        } else {
            None
        };
        let below = if i + 2 < idx.len() {
            Some(&staffs[idx[i + 2]])
        } else {
            None
        };
        let best = brace_dot
            .iter()
            .map(|s| {
                score(
                    s,
                    &staffs[idx[i]],
                    &staffs[idx[i + 1]],
                    units[idx[i]],
                    units[idx[i + 1]],
                    above,
                    below,
                )
            })
            .fold(f64::NEG_INFINITY, f64::max);
        if best > 0.0 {
            pairs.push((i, best));
        }
    }
    let mut order = pairs.clone();
    order.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap()); // sorted(reverse=True) is stable
    let mut chosen: Vec<usize> = Vec::new();
    let mut used: Vec<usize> = Vec::new();
    for (i, _) in order {
        if used.contains(&i) || used.contains(&(i + 1)) {
            continue;
        }
        chosen.push(i);
        used.push(i);
        used.push(i + 1);
    }
    if chosen.is_empty() {
        return Ok(plain());
    }
    let mut out = MultiStaff {
        staffs: vec![],
        grand: vec![],
    };
    let mut i = 0;
    while i < idx.len() {
        if chosen.iter().any(|&c| c == i || c + 1 == i) {
            out.staffs
                .push(merge_staffs(&staffs[idx[i]], &staffs[idx[i + 1]])?);
            out.grand.push(true);
            i += 2;
            continue;
        }
        out.staffs.push(staffs[idx[i]].clone());
        out.grand.push(false);
        i += 1;
    }
    // MultiStaff(...) sorts its staffs by min_y
    let mut z: Vec<(Staff, bool)> = out.staffs.into_iter().zip(out.grand).collect();
    z.sort_by(|a, b| a.0.min_y.partial_cmp(&b.0.min_y).unwrap());
    Ok(MultiStaff {
        staffs: z.iter().map(|p| p.0.clone()).collect(),
        grand: z.iter().map(|p| p.1).collect(),
    })
}
