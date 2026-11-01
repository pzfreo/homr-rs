//! From segmentation masks to the symbols staff detection needs
//! (homr/main.py: predict_symbols and the start of detect_staffs_in_image;
//! bounding_boxes.py, bar_line_detection.py, note_detection.py).

use crate::contours::find_contours;
use crate::ellipse::fit_ellipse;
use crate::geom::{box_points_f32, min_area_rect_raw, RBox};
use crate::polygon::{ellipse2poly, polygons_overlap};

/// homr's RotatedBoundingBox / BoundingEllipse: the box as OpenCV gave it
/// (`raw`), normalised (angle in [-45, 45]), its polygon and its contour.
#[derive(Clone, Debug)]
pub struct Shape {
    pub raw: [f64; 5],
    pub cx: f64,
    pub cy: f64,
    pub w: f64,
    pub h: f64,
    pub angle: f64,
    pub poly: Vec<[i64; 2]>,
    pub contour: Vec<[i32; 2]>,
    pub ellipse: bool,
}

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

impl Shape {
    pub fn rect(raw: [f64; 5], contour: Vec<[i32; 2]>) -> Shape {
        let poly = box_points_f32(raw[0], raw[1], raw[2], raw[3], raw[4]).iter().map(|p| [p[0] as f64 as i64, p[1] as f64 as i64]).collect();
        let (w, h, angle) = normalise(raw[2], raw[3], raw[4]);
        Shape { raw, cx: raw[0], cy: raw[1], w, h, angle, poly, contour, ellipse: false }
    }
    pub fn ellipse(raw: [f64; 5], contour: Vec<[i32; 2]>) -> Shape {
        let t = |v: f64| v.trunc() as i64;
        let poly = ellipse2poly([t(raw[0]), t(raw[1])], [t(raw[2] / 2.0), t(raw[3] / 2.0)], t(raw[4]));
        let (w, h, angle) = normalise(raw[2], raw[3], raw[4]);
        Shape { raw, cx: raw[0], cy: raw[1], w, h, angle, poly, contour, ellipse: true }
    }
    fn can_touch(&self, o: &Shape) -> bool {
        let d = ((self.cx - o.cx).powi(2) + (self.cy - o.cy).powi(2)).sqrt();
        !(d > self.w.max(self.h) + o.w.max(o.h))
    }
    pub fn is_overlapping(&self, o: &Shape) -> bool {
        self.can_touch(o) && polygons_overlap(&self.poly, &o.poly)
    }
    /// BoundingEllipse.make_box_thicker (from the normalised box).
    fn thicker_ellipse(&self, t: f64) -> Shape {
        Shape::ellipse([self.cx, self.cy, self.w + t, self.h + t, self.angle], self.contour.clone())
    }
    /// As the staff detection's RBox (polygon from the box as OpenCV gave it).
    pub fn to_rbox(&self, frag: usize) -> RBox {
        RBox::new(self.raw[0], self.raw[1], self.raw[2], self.raw[3], self.raw[4], frag)
    }
}

fn valid(raw: &[f64; 5]) -> bool {
    !raw[2].is_nan() && !raw[3].is_nan() && raw[2] > 0.0 && raw[3] > 0.0
}

fn rect_of(points: &[[i32; 2]]) -> [f64; 5] {
    let (cx, cy, w, h, a) = min_area_rect_raw(points);
    [cx as f64, cy as f64, w as f64, h as f64, a as f64]
}

/// _merge_groups_optimized: connected groups of overlapping boxes, in order of
/// each group's first box, boxes in index order.
fn merge_groups(boxes: &[Shape]) -> Vec<Vec<usize>> {
    let n = boxes.len();
    let mut parent: Vec<usize> = (0..n).collect();
    fn find(p: &mut Vec<usize>, x: usize) -> usize {
        let mut r = x;
        while p[r] != r {
            r = p[r];
        }
        let mut x = x;
        while p[x] != r {
            let nx = p[x];
            p[x] = r;
            x = nx;
        }
        r
    }
    // Sweep by x so only boxes that can touch are tested.
    let reach: Vec<f64> = boxes.iter().map(|b| b.w.max(b.h)).collect();
    let max_reach = reach.iter().cloned().fold(0.0, f64::max);
    let mut order: Vec<usize> = (0..n).collect();
    order.sort_by(|&a, &b| boxes[a].cx.partial_cmp(&boxes[b].cx).unwrap());
    for (k, &i) in order.iter().enumerate() {
        for &j in &order[k + 1..] {
            if boxes[j].cx - boxes[i].cx > reach[i] + max_reach {
                break;
            }
            if find(&mut parent, i) != find(&mut parent, j) && boxes[i].is_overlapping(&boxes[j]) {
                let (ri, rj) = (find(&mut parent, i), find(&mut parent, j));
                parent[ri.max(rj)] = ri.min(rj);
            }
        }
    }
    let mut groups: Vec<Vec<usize>> = Vec::new();
    let mut slot: Vec<Option<usize>> = vec![None; n];
    for i in 0..n {
        let r = find(&mut parent, i);
        match slot[r] {
            Some(g) => groups[g].push(i),
            None => {
                slot[r] = Some(groups.len());
                groups.push(vec![i]);
            }
        }
    }
    groups
}

fn whole_group(boxes: &[Shape], groups: Vec<Vec<usize>>, ellipse: bool) -> Vec<Shape> {
    groups
        .into_iter()
        .map(|g| {
            let contour: Vec<[i32; 2]> = g.iter().flat_map(|&i| boxes[i].contour.iter().copied()).collect();
            let raw = rect_of(&contour);
            if ellipse { Shape::ellipse(raw, contour) } else { Shape::rect(raw, contour) }
        })
        .collect()
}

/// create_rotated_bounding_boxes
pub fn rotated_boxes(mask: &[u8], h: usize, w: usize, skip_merging: bool, min_size: Option<(f64, f64)>, max_size: Option<(f64, f64)>) -> Vec<Shape> {
    let mut boxes = Vec::new();
    for c in find_contours(mask, h, w) {
        let raw = rect_of(&c);
        if !valid(&raw) {
            continue;
        }
        let b = Shape::rect(raw, c);
        if let Some((a, bb)) = min_size {
            if b.w < a || b.h < bb {
                continue;
            }
        }
        if let Some((a, bb)) = max_size {
            if (a > 0.0 && b.w > a) || (bb > 0.0 && b.h > bb) {
                continue;
            }
        }
        boxes.push(b);
    }
    if skip_merging {
        return boxes;
    }
    let groups = merge_groups(&boxes);
    whole_group(&boxes, groups, false)
}

/// create_bounding_ellipses
pub fn bounding_ellipses(mask: &[u8], h: usize, w: usize, min_size: (f64, f64)) -> Vec<Shape> {
    let mut boxes = Vec::new();
    for c in find_contours(mask, h, w) {
        if c.len() < 5 {
            continue;
        }
        let (cx, cy, ew, eh, a) = fit_ellipse(&c);
        let raw = [cx as f64, cy as f64, ew as f64, eh as f64, a as f64];
        if !valid(&raw) {
            continue;
        }
        let b = Shape::ellipse(raw, c);
        if b.w < min_size.0 || b.h < min_size.1 {
            continue;
        }
        boxes.push(b);
    }
    let groups = merge_groups(&boxes);
    whole_group(&boxes, groups, true)
}

/// break_wide_fragments
pub fn break_wide_fragments(fragments: Vec<Shape>, limit: f64) -> Vec<Shape> {
    let mut result = Vec::new();
    for f in fragments {
        let mut rem = f;
        while rem.w > limit {
            let min_x = rem.contour.iter().map(|p| p[0]).min().unwrap() as f64;
            let mut left: Vec<[i32; 2]> = rem.contour.iter().copied().filter(|p| (p[0] as f64) < min_x + limit).collect();
            let mut right: Vec<[i32; 2]> = rem.contour.iter().copied().filter(|p| (p[0] as f64) >= min_x + limit).collect();
            left.sort_by_key(|p| p[0]);
            right.sort_by_key(|p| p[0]);
            if left.is_empty() || right.is_empty() {
                break;
            }
            left.push(right[0]);
            right.push(*left.last().unwrap());
            result.push(Shape::rect(rect_of(&left), left));
            rem = Shape::rect(rect_of(&right), right);
        }
        result.push(rem);
    }
    result
}

/// prepare_bar_line_image: dilate with a 5 (rows) x 3 (columns) kernel of ones.
pub fn dilate_5x3(mask: &[u8], h: usize, w: usize) -> Vec<u8> {
    let mut out = vec![0u8; h * w];
    for y in 0..h {
        for x in 0..w {
            let mut m = 0u8;
            for yy in y.saturating_sub(2)..(y + 3).min(h) {
                for xx in x.saturating_sub(1)..(x + 2).min(w) {
                    m = m.max(mask[yy * w + xx]);
                }
            }
            out[y * w + x] = m;
        }
    }
    out
}

pub struct NoteheadWithStem {
    pub notehead: Shape,
    pub stem: Option<usize>,
    pub stem_up: Option<bool>,
}

/// combine_noteheads_with_stems
pub fn combine_noteheads_with_stems(mut noteheads: Vec<Shape>, stems: &[Shape]) -> Vec<NoteheadWithStem> {
    noteheads.sort_by(|a, b| a.cy.partial_cmp(&b.cy).unwrap());
    noteheads
        .into_iter()
        .map(|n| {
            let thick = n.thicker_ellipse(15.0);
            match stems.iter().position(|s| s.is_overlapping(&thick)) {
                Some(k) => {
                    let up = stems[k].cy < n.cy;
                    NoteheadWithStem { notehead: n, stem: Some(k), stem_up: Some(up) }
                }
                None => NoteheadWithStem { notehead: n, stem: None, stem_up: None },
            }
        })
        .collect()
}

pub struct Masks<'a> {
    pub h: usize,
    pub w: usize,
    pub notehead: &'a [u8],
    pub staff: &'a [u8],
    pub clefs_keys: &'a [u8],
    pub stem_rests: &'a [u8],
}

pub struct PageSymbols {
    pub fragments: Vec<Shape>,
    pub clefs_keys: Vec<Shape>,
    pub stems_rest: Vec<Shape>,
    pub noteheads: Vec<NoteheadWithStem>,
    pub bar_lines: Vec<Shape>,
}

fn median(mut v: Vec<f64>) -> f64 {
    v.sort_by(|a, b| a.partial_cmp(b).unwrap());
    let n = v.len();
    if n % 2 == 1 { v[n / 2] } else { (v[n / 2 - 1] + v[n / 2]) / 2.0 }
}

/// predict_symbols, break_wide_fragments, combine_noteheads_with_stems and
/// the bar line filtering from detect_staffs_in_image.
pub fn page_symbols(m: &Masks) -> Result<PageSymbols, String> {
    let (h, w) = (m.h, m.w);
    let noteheads = bounding_ellipses(m.notehead, h, w, (4.0, 4.0));
    let fragments = rotated_boxes(m.staff, h, w, true, Some((5.0, 1.0)), Some((10000.0, 100.0)));
    let clefs_keys = rotated_boxes(m.clefs_keys, h, w, false, Some((20.0, 40.0)), Some((1000.0, 1000.0)));
    let stems_rest = rotated_boxes(m.stem_rests, h, w, false, None, None);
    let bar_img = dilate_5x3(m.stem_rests, h, w);
    let bar_candidates = rotated_boxes(&bar_img, h, w, true, Some((1.0, 5.0)), None);
    let fragments = break_wide_fragments(fragments, 100.0);
    let noteheads = combine_noteheads_with_stems(noteheads, &stems_rest);
    if noteheads.is_empty() {
        return Err("No noteheads found".into());
    }
    let unit = median(noteheads.iter().map(|n| n.notehead.h).collect());
    let stems: Vec<&Shape> = noteheads.iter().filter_map(|n| n.stem.map(|k| &stems_rest[k])).collect();
    let bar_lines = bar_candidates
        .into_iter()
        .filter(|l| !noteheads.iter().any(|n| l.is_overlapping(&n.notehead)) && !stems.iter().any(|s| l.is_overlapping(s)))
        .filter(|l| !(l.h < 3.0 * unit) && !(l.w > 2.0 * unit))
        .collect();
    Ok(PageSymbols { fragments, clefs_keys, stems_rest, noteheads, bar_lines })
}
