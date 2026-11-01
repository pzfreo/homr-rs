//! homr/staff_detection.py: detect_staff and everything it calls, with
//! Partsong's speed patches (omr/runner.py: _find_staff_anchors_fast).

use crate::geom::{min_area_rect, RBox, NO_FRAGMENT};
use crate::npmath::*;

pub const LINES: usize = 5;
const X_TOLERANCE: f64 = 10.0; // staff_line_segment_x_tolerance

pub struct Fragments<'a> {
    pub boxes: &'a [RBox],
    pub contours: &'a [Vec<[i32; 2]>],
}

#[derive(Clone, Debug)]
pub struct StaffPoint {
    pub x: f64,
    pub y: Vec<f64>,
    pub angle: f64,
}

#[derive(Clone, Debug)]
pub struct Staff {
    pub grid: Vec<StaffPoint>,
    pub min_x: f64,
    pub max_x: f64,
    pub min_y: f64,
    pub max_y: f64,
}

impl Staff {
    pub fn from_grid(grid: Vec<StaffPoint>) -> Result<Staff, String> {
        Staff::new(grid)
    }
    fn new(grid: Vec<StaffPoint>) -> Result<Staff, String> {
        if grid.is_empty() {
            return Err("empty staff grid (homr raises IndexError)".into());
        }
        let min_y = grid.iter().map(|p| p.y.iter().cloned().fold(f64::INFINITY, f64::min)).fold(f64::INFINITY, f64::min);
        let max_y = grid.iter().map(|p| p.y.iter().cloned().fold(f64::NEG_INFINITY, f64::max)).fold(f64::NEG_INFINITY, f64::max);
        Ok(Staff { min_x: grid[0].x, max_x: grid[grid.len() - 1].x, min_y, max_y, grid })
    }
}

/// StaffLineSegment: fragments joined into one staff line, left to right.
#[derive(Clone, Debug)]
struct Segment {
    frags: Vec<usize>,
    min_x: f64,
    max_x: f64,
    min_y: f64,
    max_y: f64,
}

impl Segment {
    fn new(mut frags: Vec<usize>, f: &Fragments) -> Segment {
        let b = f.boxes;
        let min_x = frags.iter().map(|&i| b[i].cx - b[i].w / 2.0).fold(f64::INFINITY, f64::min);
        let max_x = frags.iter().map(|&i| b[i].cx + b[i].w / 2.0).fold(f64::NEG_INFINITY, f64::max);
        let min_y = frags.iter().map(|&i| b[i].cy - b[i].h / 2.0).fold(f64::INFINITY, f64::min);
        let max_y = frags.iter().map(|&i| b[i].cy + b[i].h / 2.0).fold(f64::NEG_INFINITY, f64::max);
        frags.sort_by(|&p, &q| b[p].cx.partial_cmp(&b[q].cx).unwrap());
        Segment { frags, min_x, max_x, min_y, max_y }
    }
    fn merge(&self, other: &Segment, f: &Fragments) -> Segment {
        let mut lines = self.frags.clone();
        for &g in &other.frags {
            if !lines.contains(&g) {
                lines.push(g);
            }
        }
        Segment::new(lines, f)
    }
    fn get_at(&self, x: f64, f: &Fragments) -> Option<usize> {
        self.frags.iter().copied().find(|&i| {
            let b = &f.boxes[i];
            x >= b.cx - b.w / 2.0 - X_TOLERANCE && x <= b.cx + b.w / 2.0 + X_TOLERANCE
        })
    }
    fn contains_all(&self, other: &Segment) -> bool {
        other.frags.iter().all(|g| self.frags.contains(g))
    }
}

#[derive(Clone, Debug)]
struct Anchor {
    lines: Vec<Segment>,
    symbol: RBox,
    avg_unit: f64,
    zone: (i64, i64),
}

impl Anchor {
    fn new(lines: Vec<Segment>, symbol: RBox, f: &Fragments) -> Anchor {
        let mut ys: Vec<f64> = lines.iter().map(|l| f.boxes[l.frags[0]].center_at(symbol.cx)).collect();
        ys.sort_by(|a, b| a.partial_cmp(b).unwrap());
        let deltas: Vec<f64> = ys.windows(2).map(|w| (w[1] - w[0]).abs()).collect();
        let avg_unit = if deltas.is_empty() { 0.0 } else { np_mean(&deltas) };
        let max_y = lines.iter().map(|l| l.max_y).fold(f64::NEG_INFINITY, f64::max);
        let min_y = lines.iter().map(|l| l.min_y).fold(f64::INFINITY, f64::min);
        let zone = (py_int(min_y - 5.0 * avg_unit), py_int(max_y + 5.0 * avg_unit));
        Anchor { lines, symbol, avg_unit, zone }
    }
}

struct RawStaff {
    lines: Vec<Segment>,
    anchors: Vec<usize>,
    rect: RBox,
}

impl RawStaff {
    fn new(lines: Vec<Segment>, anchors: Vec<usize>, f: &Fragments) -> RawStaff {
        let mut pts: Vec<[i32; 2]> = Vec::new();
        for l in &lines {
            for &g in &l.frags {
                pts.extend_from_slice(&f.contours[g]);
            }
        }
        RawStaff { rect: min_area_rect(&pts), lines, anchors }
    }
    fn min_x(&self) -> f64 {
        self.rect.cx - self.rect.w / 2.0
    }
    fn max_x(&self) -> f64 {
        self.rect.cx + self.rect.w / 2.0
    }
    fn merge(&self, other: &RawStaff, f: &Fragments) -> RawStaff {
        let lines = self.lines.iter().enumerate().map(|(i, l)| other.lines[i].merge(l, f)).collect();
        let mut anchors = self.anchors.clone();
        anchors.extend_from_slice(&other.anchors);
        RawStaff::new(lines, anchors, f)
    }
}

fn connect_staff_lines(lines: &[usize], unit: f64, f: &Fragments) -> Vec<Segment> {
    let b = f.boxes;
    let mut order: Vec<usize> = lines.to_vec();
    order.sort_by(|&p, &q| b[q].left_x().partial_cmp(&b[p].left_x()).unwrap()); // right to left, stable
    let mut result: Vec<Vec<usize>> = Vec::new();
    let mut active: Vec<usize> = Vec::new();
    let mut last_cleanup = 0.0;
    let max_gap = 5.0 * unit;
    while let Some(cur) = order.pop() {
        let x = b[cur].left_x();
        if x - last_cleanup > max_gap {
            active.retain(|&r| x - b[*result[r].last().unwrap()].right_x() < max_gap);
            last_cleanup = x;
        }
        if b[cur].w < unit / 5.0 {
            continue;
        }
        let mut connected = false;
        for &r in &active {
            let last = *result[r].last().unwrap();
            if b[last].is_overlapping_extrapolated(&b[cur], unit) {
                result[r].push(cur);
                connected = true;
            }
        }
        if !connected {
            result.push(vec![cur]);
            active.push(result.len() - 1);
        }
    }
    result.sort_by(|p, q| b[p[0]].cy.partial_cmp(&b[q[0]].cy).unwrap());
    result.into_iter().map(|l| Segment::new(l, f)).collect()
}

fn line_y_at(line: &Segment, x: f64, f: &Fragments) -> f64 {
    let g = line.get_at(x, f).unwrap_or(line.frags[0]);
    f.boxes[g].center_at(x)
}

fn are_lines_crossing(lines: &[Segment], f: &Fragments) -> bool {
    for i in 0..lines.len() {
        for j in i + 1..lines.len() {
            let start = lines[i].min_x.max(lines[j].min_x);
            let stop = lines[i].max_x.min(lines[j].max_x);
            if start >= stop {
                continue;
            }
            let ds = line_y_at(&lines[i], start, f) - line_y_at(&lines[j], start, f);
            let dt = line_y_at(&lines[i], stop, f) - line_y_at(&lines[j], stop, f);
            if ds == 0.0 || dt == 0.0 || (ds > 0.0) != (dt > 0.0) {
                return true;
            }
        }
    }
    false
}

fn are_lines_parallel(lines: &[Segment], unit: f64, f: &Fragments) -> bool {
    let all: Vec<usize> = lines.iter().flat_map(|l| l.frags.iter().copied()).collect();
    if all.is_empty() {
        return false;
    }
    let angles: Vec<f64> = all.iter().map(|&g| f.boxes[g].angle).collect();
    let avg = np_mean(&angles);
    !all.iter().any(|&g| (f.boxes[g].angle - avg).abs() > 10.0 && f.boxes[g].w > 2.0 * unit)
}

fn begins_or_ends_on_one_staff_line(sym: &RBox, lines: &[Segment], unit: f64, f: &Fragments) -> bool {
    lines.iter().any(|l| match l.get_at(sym.cx, f) {
        Some(g) => (f.boxes[g].center_at(sym.cx) - sym.cy).abs() < unit,
        None => false,
    })
}

fn find_staff_anchors_for(lines: &[usize], center: &RBox, are_clefs: bool, f: &Fragments, out: &mut Vec<Anchor>) {
    let shifts: &[f64] = if are_clefs { &[-10.0, 0.0, 10.0, 30.0, 60.0, 80.0] } else { &[-10.0, -5.0, 0.0, 5.0, 10.0] };
    for &dx in shifts {
        let sym = if dx == 0.0 { center.clone() } else { center.move_x(dx) };
        let unit = py_round(sym.h / 4.0) as f64;
        let thick = sym.taller(unit);
        let overlapping: Vec<usize> = lines.iter().copied().filter(|&g| f.boxes[g].is_intersecting(&thick)).collect();
        let mut connected = connect_staff_lines(&overlapping, unit, f);
        if connected.len() > LINES {
            connected.retain(|l| (l.max_x - l.min_x) > 2.0 * unit);
        }
        if connected.len() != LINES {
            continue;
        }
        if !are_lines_parallel(&connected, unit, f) || are_lines_crossing(&connected, f) {
            continue;
        }
        if !are_clefs && !begins_or_ends_on_one_staff_line(&sym, &connected, unit, f) {
            continue;
        }
        out.push(Anchor::new(connected, sym, f));
    }
}

/// find_staff_anchors with Partsong's prefilter: only fragments near enough
/// to touch any moved/taller copy of the symbol are tested exactly.
fn find_staff_anchors(lines: &[usize], symbols: &[RBox], are_clefs: bool, f: &Fragments) -> Vec<Anchor> {
    let mut out = Vec::new();
    let shift = if are_clefs { 80.0 } else { 10.0 };
    for s in symbols {
        let unit = py_round(s.h / 4.0) as f64;
        let near: Vec<usize> = lines
            .iter()
            .copied()
            .filter(|&g| {
                let b = &f.boxes[g];
                let d = ((b.cx - s.cx).powi(2) + (b.cy - s.cy).powi(2)).sqrt();
                d <= b.w.max(b.h) + s.w.max(s.h) + unit.abs() + shift + 1.0
            })
            .collect();
        find_staff_anchors_for(&near, s, are_clefs, f, &mut out);
    }
    out
}

pub fn find_peaks(x: &[f64], height: f64, distance: f64, prominence: f64) -> Vec<i64> {
    let n = x.len();
    if n < 3 {
        return vec![];
    }
    let mut peaks: Vec<usize> = Vec::new();
    let mut i = 1;
    while i < n - 1 {
        if x[i] > x[i - 1] || x[i] == x[i - 1] {
            let mut j = i;
            while j < n - 1 && x[j] == x[j + 1] {
                j += 1;
            }
            if j < n - 1 && x[j] > x[j + 1] {
                peaks.push((i + j) / 2);
            }
            i = j + 1;
        } else {
            i += 1;
        }
    }
    peaks.retain(|&p| x[p] >= height);
    peaks.retain(|&p| {
        let mut left_min = x[p];
        for k in (0..p).rev() {
            if x[k] > x[p] {
                break;
            }
            left_min = left_min.min(x[k]);
        }
        let mut right_min = x[p];
        for k in p + 1..n {
            if x[k] > x[p] {
                break;
            }
            right_min = right_min.min(x[k]);
        }
        x[p] - left_min.max(right_min) >= prominence
    });
    if peaks.len() > 1 {
        // np.argsort(x[peaks])[::-1]: highest first (ties: later index first, as a stable sort reversed)
        let mut order: Vec<usize> = peaks.clone();
        order.sort_by(|&a, &b| x[a].partial_cmp(&x[b]).unwrap());
        order.reverse();
        let mut keep: Vec<usize> = Vec::new();
        for p in order {
            if keep.iter().all(|&k| (k as f64 - p as f64).abs() >= distance) {
                keep.push(p);
            }
        }
        keep.sort();
        peaks = keep;
    }
    peaks.into_iter().map(|p| p as i64).collect()
}

fn find_horizontal_lines(image: &[u8], h: usize, w: usize, zone: (i64, i64), unit: f64) -> Result<Vec<Vec<i64>>, String> {
    let mut count = vec![0.0f64; h + 2];
    for y in 0..h {
        let row = &image[y * w..(y + 1) * w];
        count[y + 1] = row[zone.0 as usize..zone.1 as usize].iter().filter(|&&v| v > 0).count() as f64;
    }
    let mean = np_mean(&count);
    let std = np_std(&count);
    let norm: Vec<f64> = count.iter().map(|&c| (c - mean) / std).collect();
    let centers: Vec<i64> = find_peaks(&norm, 0.0, 0.7 * unit, 1.0).into_iter().map(|c| c - 1).collect();
    if centers.is_empty() {
        return Err("no line peaks (homr raises IndexError)".into());
    }
    // filter_line_peaks: only its grouping is used
    let mut gaps: Vec<f64> = centers.windows(2).map(|p| (p[1] - p[0]) as f64).collect();
    gaps.sort_by(|a, b| a.partial_cmp(b).unwrap());
    let take = 5.max(py_round(centers.len() as f64 * 0.2)) as usize;
    let smallest = &gaps[..take.min(gaps.len())];
    let approx = if smallest.is_empty() { f64::NAN } else { np_mean(smallest) };
    let max_gap = approx * 1.5;
    let mut ext: Vec<f64> = vec![centers[0] as f64 - max_gap - 1.0];
    ext.extend(centers.iter().map(|&c| c as f64));
    let mut groups: Vec<i64> = Vec::new();
    let mut group = -1;
    for i in 1..ext.len() {
        if ext[i] - ext[i - 1] > max_gap {
            group += 1;
        }
        groups.push(group);
    }
    let mut order: Vec<i64> = Vec::new();
    let mut grouped: std::collections::HashMap<i64, Vec<i64>> = std::collections::HashMap::new();
    for (i, &c) in centers.iter().enumerate() {
        if !grouped.contains_key(&groups[i]) {
            order.push(groups[i]);
        }
        grouped.entry(groups[i]).or_default().push(c);
    }
    Ok(order
        .into_iter()
        .filter_map(|g| {
            let mut v = grouped.remove(&g).unwrap();
            (v.len() == LINES).then(|| {
                v.sort();
                v
            })
        })
        .collect())
}

fn predict_other_anchors_from_clefs(clefs: &[Anchor], image: &[u8], h: usize, w: usize) -> Result<Vec<RBox>, String> {
    if clefs.is_empty() {
        return Ok(vec![]);
    }
    let avg_unit = np_mean(&clefs.iter().map(|a| a.avg_unit).collect::<Vec<_>>());
    // init_zone
    let mut ranges: Vec<(i64, i64)> = clefs
        .iter()
        .map(|c| (py_int(c.symbol.left_x()).max(0), py_int(c.symbol.right_x() + 10.0).min(w as i64)))
        .collect();
    ranges.sort_by_key(|r| r.0);
    let mut zones: Vec<(i64, i64)> = Vec::new();
    for r in ranges {
        match zones.last_mut() {
            Some(last) if r.0 < last.1 => last.1 = r.1,
            _ => zones.push(r),
        }
    }
    let mut result = Vec::new();
    for zone in zones {
        let zone = (zone.0, zone.1.max(zone.0));
        for group in find_horizontal_lines(image, h, w, zone, avg_unit)? {
            let (min_y, max_y) = (group[0], group[LINES - 1]);
            let center_y = (min_y + max_y) as f64 / 2.0;
            let center_x = zone.0 as f64 + (zone.1 - zone.0) as f64 / 2.0;
            result.push(RBox::new(
                py_int(center_x) as f64,
                py_int(center_y) as f64,
                (zone.1 - zone.0) as f64,
                (max_y - min_y) as f64,
                0.0,
                NO_FRAGMENT,
            ));
        }
    }
    Ok(result.into_iter().filter(|r| !clefs.iter().any(|c| r.is_overlapping(&c.symbol))).collect())
}

fn find_raw_staffs(anchors: &[Anchor], f: &Fragments) -> Vec<RawStaff> {
    let mut staffs: Vec<RawStaff> = Vec::new();
    let all: Vec<usize> = (0..f.boxes.len()).collect();
    for (ai, anchor) in anchors.iter().enumerate() {
        let existing = staffs
            .iter()
            .position(|s| anchor.lines.iter().enumerate().any(|(i, al)| s.lines[i].contains_all(al)));
        let frags: Vec<usize> = all
            .iter()
            .copied()
            .filter(|&g| f.boxes[g].cy >= anchor.zone.0 as f64 && f.boxes[g].cy <= anchor.zone.1 as f64)
            .collect();
        let connected = connect_staff_lines(&frags, anchor.avg_unit, f);
        let lines: Vec<Segment> = anchor
            .lines
            .iter()
            .map(|al| {
                let m: Vec<&Segment> = connected.iter().filter(|l| l.contains_all(al)).collect();
                if m.len() == 1 { m[0].clone() } else { al.clone() }
            })
            .collect();
        let new = RawStaff::new(lines, vec![ai], f);
        match existing {
            Some(k) => {
                let old = staffs.remove(k);
                staffs.push(old.merge(&new, f));
            }
            None => staffs.push(new),
        }
    }
    staffs
}

fn remove_duplicate_staffs(staffs: Vec<RawStaff>) -> Vec<RawStaff> {
    let mut result: Vec<RawStaff> = Vec::new();
    for s in staffs {
        let overlapping: Vec<usize> = (0..result.len()).filter(|&k| s.rect.is_overlapping(&result[k].rect)).collect();
        if overlapping.is_empty() {
            result.push(s);
        } else if overlapping.len() == 1 && result[overlapping[0]].anchors.len() < s.anchors.len() {
            result.remove(overlapping[0]);
            result.push(s);
        }
    }
    result
}

fn resample_segment(anchor: &Anchor, staff: &RawStaff, xs: impl Iterator<Item = i64>, f: &Fragments) -> Vec<StaffPoint> {
    let x0 = anchor.symbol.cx;
    let firsts: Vec<&RBox> = anchor.lines.iter().map(|l| &f.boxes[l.frags[0]]).collect();
    let mut prev = StaffPoint {
        x: x0,
        y: firsts.iter().map(|b| b.center_at(x0)).collect(),
        angle: np_mean(&firsts.iter().map(|b| b.angle).collect::<Vec<_>>()),
    };
    let half = 0.5 * anchor.avg_unit;
    let mut out = Vec::new();
    for xi in xs {
        let x = xi as f64;
        let lines: Vec<Option<usize>> = staff.lines.iter().map(|l| l.get_at(x, f)).collect();
        let mut c: Vec<Option<f64>> = lines.iter().map(|g| g.map(|g| f.boxes[g].center_at(x))).collect();
        let values: Vec<f64> = c.iter().flatten().copied().collect();
        if values.is_empty() {
            continue;
        }
        let deltas: Vec<f64> = values.windows(2).map(|p| p[1] - p[0]).collect();
        for (i, d) in deltas.iter().enumerate() {
            if *d < half {
                c[i] = None; // homr indexes by position among the found values (kept as is)
                c[i + 1] = None;
            }
        }
        for (i, &py) in prev.y.iter().enumerate() {
            if let Some(v) = c[i] {
                if (v - py).abs() > half {
                    c[i] = None;
                }
            }
        }
        let n = c.len();
        let mut prev_center: i64 = -1;
        for i in (0..n).chain((0..n).rev()) {
            if c[i].is_some() {
                prev_center = i as i64;
            } else if prev_center >= 0 {
                if let Some(v) = c[prev_center as usize] {
                    c[i] = Some(v + anchor.avg_unit * (i as i64 - prev_center) as f64);
                }
            }
        }
        if c.iter().any(|v| v.is_none()) {
            continue;
        }
        let angles: Vec<f64> = lines.iter().flatten().map(|&g| f.boxes[g].angle).collect();
        prev = StaffPoint { x, y: c.into_iter().flatten().collect(), angle: np_mean(&angles) };
        out.push(prev.clone());
    }
    out
}

fn resample_staff(staff: &RawStaff, anchors: &[Anchor], f: &Fragments) -> Result<Staff, String> {
    let mut l2r: Vec<&Anchor> = staff.anchors.iter().map(|&a| &anchors[a]).collect();
    l2r.sort_by(|a, b| a.symbol.cx.partial_cmp(&b.symbol.cx).unwrap());
    let rtd = |v: f64| py_round(v / 10.0) * 10;
    let start = py_floordiv(staff.min_x(), 10.0) * 10.0;
    let stop = (py_floordiv(staff.max_x(), 10.0) + 1.0) * 10.0;
    let mut grid = Vec::new();
    let mut x = start;
    for (i, a) in l2r.iter().enumerate() {
        let to_left = py_range(rtd(x), rtd(a.symbol.cx), 10);
        let (rs, re) = if i + 1 < l2r.len() {
            (py_int(a.symbol.cx), py_int((a.symbol.cx + l2r[i + 1].symbol.cx) / 2.0))
        } else {
            (rtd(a.symbol.cx), rtd(stop))
        };
        x = re as f64;
        let mut left = resample_segment(a, staff, to_left.into_iter().rev(), f);
        left.reverse();
        grid.extend(left);
        grid.extend(resample_segment(a, staff, py_range(rs, re, 10).into_iter(), f));
    }
    Staff::new(grid)
}

fn filter_unusual_anchors(anchors: Vec<Anchor>) -> Vec<Anchor> {
    if anchors.is_empty() {
        return anchors;
    }
    let units: Vec<f64> = anchors.iter().map(|a| a.avg_unit).collect();
    let (mean, std) = (np_mean(&units), np_std(&units));
    anchors.into_iter().filter(|a| !((a.avg_unit - mean).abs() > 3.0 * std)).collect()
}

fn filter_edge_of_vision(staffs: Vec<Staff>, h: usize, w: usize) -> Vec<Staff> {
    let widths: Vec<f64> = staffs.iter().map(|s| s.max_x - s.min_x).collect();
    let usual = np_mean(&widths);
    let (h, w) = (h as f64, w as f64);
    staffs
        .into_iter()
        .filter(|s| {
            if s.max_y >= h || s.min_y < 0.0 {
                return false;
            }
            let shorter = (s.max_x - s.min_x) < usual / 2.0;
            let edge = s.min_x < 0.01 * w || s.max_x > 0.99 * w;
            !(edge && shorter)
        })
        .collect()
}

/// Intermediate results, for comparing with homr step by step.
#[derive(Default, Debug)]
pub struct Mid {
    pub others: Vec<[f64; 5]>,
    pub anchors: Vec<([f64; 5], f64, i64, i64)>,
    pub raw: Vec<(f64, f64, f64, f64, usize, [f64; 5])>,
}

/// homr's detect_staff. `image` is the segmentation's staff mask (h x w).
pub fn detect_staff(image: &[u8], h: usize, w: usize, f: &Fragments, clefs: &[RBox], bars: &[RBox]) -> Result<Vec<Staff>, String> {
    detect_staff_mid(image, h, w, f, clefs, bars, &mut None)
}

fn b5(b: &RBox) -> [f64; 5] {
    [b.cx, b.cy, b.w, b.h, b.angle]
}

pub fn detect_staff_mid(image: &[u8], h: usize, w: usize, f: &Fragments, clefs: &[RBox], bars: &[RBox], mid: &mut Option<Mid>) -> Result<Vec<Staff>, String> {
    let all: Vec<usize> = (0..f.boxes.len()).collect();
    let mut anchors = find_staff_anchors(&all, clefs, true, f);
    let others = predict_other_anchors_from_clefs(&anchors, image, h, w)?;
    if let Some(m) = mid.as_mut() {
        m.others = others.iter().map(b5).collect();
    }
    anchors.extend(find_staff_anchors(&all, &others, true, f));
    anchors.extend(find_staff_anchors(&all, bars, false, f));
    let anchors = filter_unusual_anchors(anchors);
    if let Some(m) = mid.as_mut() {
        m.anchors = anchors.iter().map(|a| (b5(&a.symbol), a.avg_unit, a.zone.0, a.zone.1)).collect();
    }
    let raw = remove_duplicate_staffs(find_raw_staffs(&anchors, f));
    if let Some(m) = mid.as_mut() {
        m.raw = raw.iter().map(|s| (s.min_x(), s.max_x(), s.rect.cy - s.rect.h / 2.0, s.rect.cy + s.rect.h / 2.0, s.anchors.len(), b5(&s.rect))).collect();
    }
    let staffs = raw.iter().map(|s| resample_staff(s, &anchors, f)).collect::<Result<Vec<_>, _>>()?;
    let mut staffs = filter_edge_of_vision(staffs, h, w);
    staffs.sort_by(|a, b| a.min_y.partial_cmp(&b.min_y).unwrap());
    Ok(staffs)
}

/// For debugging: the fragments each moved copy of `sym` touches, and how they connect.
pub fn debug_symbol(sym: &RBox, f: &Fragments) -> Vec<(f64, Vec<usize>, Vec<Vec<usize>>)> {
    let all: Vec<usize> = (0..f.boxes.len()).collect();
    [-10.0, -5.0, 0.0, 5.0, 10.0]
        .iter()
        .map(|&dx| {
            let m = if dx == 0.0 { sym.clone() } else { sym.move_x(dx) };
            let unit = py_round(m.h / 4.0) as f64;
            let thick = m.taller(unit);
            let ov: Vec<usize> = all.iter().copied().filter(|&g| f.boxes[g].is_intersecting(&thick)).collect();
            let con = connect_staff_lines(&ov, unit, f).into_iter().map(|s| s.frags).collect();
            (dx, ov, con)
        })
        .collect()
}
