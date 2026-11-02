//! Raster operations homr uses, ported from OpenCV 4.11: elliptical
//! structuring elements, erode/dilate (binary masks, default borders),
//! saturating subtract, and polygon filling (fillPoly / drawContours FILLED,
//! modules/imgproc/src/drawing.cpp) with its Bresenham edges.

/// getStructuringElement(MORPH_ELLIPSE, (width, height)): rows of 0/1.
pub fn ellipse_kernel(width: usize, height: usize) -> Vec<Vec<u8>> {
    if width == 1 && height == 1 {
        return vec![vec![1]];
    }
    let r = (height / 2) as i64;
    let c = (width / 2) as i64;
    let inv_r2 = if r != 0 { 1.0 / ((r * r) as f64) } else { 0.0 };
    (0..height as i64)
        .map(|i| {
            let mut row = vec![0u8; width];
            let dy = i - r;
            if dy.abs() <= r {
                let dx = (c as f64 * (((r * r - dy * dy) as f64) * inv_r2).sqrt()).round_ties_even()
                    as i64;
                let j1 = (c - dx).max(0) as usize;
                let j2 = (c + dx + 1).min(width as i64) as usize;
                for v in &mut row[j1..j2] {
                    *v = 1;
                }
            }
            row
        })
        .collect()
}

/// out[i] = min/max of src[i+a ..= i+b] (clipped to the array; empty -> identity).
fn range_extreme(src: &[u8], a: i64, b: i64, dilate: bool, out: &mut [u8]) {
    let n = src.len() as i64;
    let better = |x: u8, y: u8| if dilate { x >= y } else { x <= y };
    let mut dq: std::collections::VecDeque<usize> = std::collections::VecDeque::new();
    let mut next = 0i64; // next index to push
    for i in 0..n {
        let (lo, hi) = ((i + a).max(0), (i + b).min(n - 1));
        while next <= hi {
            let v = src[next as usize];
            while let Some(&k) = dq.back() {
                if better(v, src[k]) {
                    dq.pop_back();
                } else {
                    break;
                }
            }
            dq.push_back(next as usize);
            next += 1;
        }
        while let Some(&k) = dq.front() {
            if (k as i64) < lo {
                dq.pop_front();
            } else {
                break;
            }
        }
        out[i as usize] = match dq.front() {
            Some(&k) if lo <= hi => src[k],
            _ => {
                if dilate {
                    0
                } else {
                    255
                }
            }
        };
    }
}

/// Erode (min) or dilate (max) an image with a kernel anchored at its centre;
/// pixels outside the image are ignored (OpenCV's default border). The kernel
/// is split into runs of rows with the same horizontal span, each done as a
/// horizontal then a vertical sliding window.
pub fn morph(img: &[u8], h: usize, w: usize, kernel: &[Vec<u8>], dilate: bool) -> Vec<u8> {
    let kh = kernel.len() as i64;
    let kw = kernel[0].len() as i64;
    let (ay, ax) = (kh / 2, kw / 2);
    // segments: (dy0, dy1, j1, j2) with columns j1..j2 set in rows dy0..=dy1
    let mut segs: Vec<(i64, i64, i64, i64)> = Vec::new();
    for i in 0..kh {
        let row = &kernel[i as usize];
        let j1 = row.iter().position(|&v| v != 0);
        let Some(j1) = j1 else { continue };
        let j2 = row.iter().rposition(|&v| v != 0).unwrap() + 1;
        assert!(
            row[j1..j2].iter().all(|&v| v != 0),
            "kernel rows must be contiguous"
        );
        let (j1, j2) = (j1 as i64, j2 as i64);
        match segs.last_mut() {
            Some(s) if s.1 == i - ay - 1 && s.2 == j1 - ax && s.3 == j2 - 1 - ax => s.1 = i - ay,
            _ => segs.push((i - ay, i - ay, j1 - ax, j2 - 1 - ax)),
        }
    }
    if let Some(v) = binary_value(img) {
        return morph_binary(img, h, w, &segs, dilate, v);
    }
    let identity = if dilate { 0u8 } else { 255u8 };
    let mut out = vec![identity; h * w];
    let mut hor = vec![0u8; h * w];
    let mut col = vec![0u8; h];
    let mut colout = vec![0u8; h];
    let mut done_span: Option<(i64, i64)> = None;
    for &(dy0, dy1, a, b) in &segs {
        if done_span != Some((a, b)) {
            for y in 0..h {
                range_extreme(
                    &img[y * w..(y + 1) * w],
                    a,
                    b,
                    dilate,
                    &mut hor[y * w..(y + 1) * w],
                );
            }
            done_span = Some((a, b));
        }
        for x in 0..w {
            for y in 0..h {
                col[y] = hor[y * w + x];
            }
            range_extreme(&col, dy0, dy1, dilate, &mut colout);
            for y in 0..h {
                let o = &mut out[y * w + x];
                *o = if dilate {
                    (*o).max(colout[y])
                } else {
                    (*o).min(colout[y])
                };
            }
        }
    }
    out
}

/// Some(v) if every pixel is 0 or v (v = 0 if all zero).
fn binary_value(img: &[u8]) -> Option<u8> {
    let mut v = 0u8;
    for &p in img {
        if p != 0 {
            if v == 0 {
                v = p;
            } else if p != v {
                return None;
            }
        }
    }
    Some(v)
}

/// morph for images of 0s and one other value v, on bit-packed rows (64
/// pixels a word): a window dilates to v if it holds any v, erodes to v if
/// all its in-image pixels are v. Same result as the byte-wise version.
fn morph_binary(
    img: &[u8],
    h: usize,
    w: usize,
    segs: &[(i64, i64, i64, i64)],
    dilate: bool,
    v: u8,
) -> Vec<u8> {
    let words = w.div_ceil(64);
    // 8 bytes -> 8 bits (bit i set if byte i is nonzero)
    let pack8 = |b: &[u8]| -> u64 {
        let x = u64::from_le_bytes(b.try_into().unwrap());
        let hi =
            ((x & 0x7f7f7f7f7f7f7f7f).wrapping_add(0x7f7f7f7f7f7f7f7f) | x) & 0x8080808080808080;
        (hi >> 7).wrapping_mul(0x0102040810204080) >> 56
    };
    let pack = |row: &[u8]| -> Vec<u64> {
        let mut out = vec![0u64; words];
        let full8 = row.len() / 8;
        for i in 0..full8 {
            out[i / 8] |= pack8(&row[i * 8..i * 8 + 8]) << ((i % 8) * 8);
        }
        for x in full8 * 8..row.len() {
            if row[x] != 0 {
                out[x / 64] |= 1 << (x % 64);
            }
        }
        out
    };
    let rows: Vec<Vec<u64>> = (0..h).map(|y| pack(&img[y * w..(y + 1) * w])).collect();
    // bits beyond the image width: for erode they count as set (outside is ignored)
    let tail_mask = if w.is_multiple_of(64) {
        !0u64
    } else {
        (1u64 << (w % 64)) - 1
    };
    let full = |r: &mut Vec<u64>| {
        if !dilate {
            let last = r.len() - 1;
            r[last] |= !tail_mask;
        }
    };
    // acc[i] op= (r shifted by k bits), with fill outside the row
    let shift_into = |acc: &mut [u64], r: &[u64], k: i64, fill: bool| {
        let fillw = if fill { !0u64 } else { 0 };
        let n = words as i64;
        let (q, rem) = (k.div_euclid(64), k.rem_euclid(64) as u32);
        let get = |i: i64| -> u64 {
            if i < 0 || i >= n {
                fillw
            } else {
                r[i as usize]
            }
        };
        for (i, x) in acc.iter_mut().enumerate() {
            let i = i as i64;
            let lo = get(i + q);
            let s = if rem == 0 {
                lo
            } else {
                (lo >> rem) | (get(i + q + 1) << (64 - rem))
            };
            *x = if dilate { *x | s } else { *x & s };
        }
    };
    let mut out: Vec<Vec<u64>> = vec![vec![if dilate { 0 } else { !0u64 }; words]; h];
    type ShiftCache = Vec<((i64, i64), Vec<Vec<u64>>)>;
    let mut cache: ShiftCache = Vec::new();
    for &(dy0, dy1, a, b) in segs {
        if !cache.iter().any(|c| c.0 == (a, b)) {
            let mut r = vec![0u64; words];
            let hor: Vec<Vec<u64>> = rows
                .iter()
                .map(|row| {
                    r.copy_from_slice(row);
                    full(&mut r);
                    let mut acc = vec![if dilate { 0 } else { !0u64 }; words];
                    for k in a..=b {
                        shift_into(&mut acc, &r, k, !dilate);
                    }
                    acc
                })
                .collect();
            cache.push(((a, b), hor));
        }
        let hor = &cache.iter().find(|c| c.0 == (a, b)).unwrap().1;
        for y in 0..h as i64 {
            let (lo, hi) = ((y + dy0).max(0), (y + dy1).min(h as i64 - 1));
            let o = &mut out[y as usize];
            if lo > hi {
                continue; // window entirely outside: identity
            }
            let mut acc = vec![if dilate { 0 } else { !0u64 }; words];
            for yy in lo..=hi {
                for (x, s) in acc.iter_mut().zip(&hor[yy as usize]) {
                    *x = if dilate { *x | s } else { *x & s };
                }
            }
            for (x, s) in o.iter_mut().zip(&acc) {
                *x = if dilate { *x | s } else { *x & s };
            }
        }
    }
    // 8 bits -> 8 bytes of v or 0
    let table: Vec<u64> = (0..256u64)
        .map(|b| {
            (0..8).fold(0u64, |t, i| {
                if (b >> i) & 1 == 1 {
                    t | (v as u64) << (8 * i)
                } else {
                    t
                }
            })
        })
        .collect();
    let spread = |b: u64| table[b as usize];
    let mut res = vec![0u8; h * w];
    for y in 0..h {
        let row = &mut res[y * w..(y + 1) * w];
        let full8 = w / 8;
        for i in 0..full8 {
            let byte = (out[y][i / 8] >> ((i % 8) * 8)) & 0xff;
            if byte != 0 {
                row[i * 8..i * 8 + 8].copy_from_slice(&spread(byte).to_le_bytes());
            }
        }
        for x in full8 * 8..w {
            if (out[y][x / 64] >> (x % 64)) & 1 == 1 {
                row[x] = v;
            }
        }
    }
    res
}

/// cv2.subtract for uint8 (saturating).
pub fn subtract(a: &[u8], b: &[u8]) -> Vec<u8> {
    a.iter()
        .zip(b)
        .map(|(&x, &y)| x.saturating_sub(y))
        .collect()
}

const XY_SHIFT: i64 = 16;
const XY_ONE: i64 = 1 << XY_SHIFT;

/// clipLine(Size, pt1, pt2): clips the segment to the image; false if it's outside.
pub fn clip_line(w: i64, h: i64, p1: &mut (i64, i64), p2: &mut (i64, i64)) -> bool {
    let (right, bottom) = (w - 1, h - 1);
    if w <= 0 || h <= 0 {
        return false;
    }
    let code = |x: i64, y: i64| {
        (x < 0) as i32 + (x > right) as i32 * 2 + (y < 0) as i32 * 4 + (y > bottom) as i32 * 8
    };
    let (mut x1, mut y1, mut x2, mut y2) = (p1.0, p1.1, p2.0, p2.1);
    let mut c1 = code(x1, y1);
    let mut c2 = code(x2, y2);
    if (c1 & c2) == 0 && (c1 | c2) != 0 {
        if c1 & 12 != 0 {
            let a = if c1 < 8 { 0 } else { bottom };
            x1 += ((a - y1) as f64 * (x2 - x1) as f64 / (y2 - y1) as f64) as i64;
            y1 = a;
            c1 = (x1 < 0) as i32 + (x1 > right) as i32 * 2;
        }
        if c2 & 12 != 0 {
            let a = if c2 < 8 { 0 } else { bottom };
            x2 += ((a - y2) as f64 * (x2 - x1) as f64 / (y2 - y1) as f64) as i64;
            y2 = a;
            c2 = (x2 < 0) as i32 + (x2 > right) as i32 * 2;
        }
        if (c1 & c2) == 0 && (c1 | c2) != 0 {
            if c1 != 0 {
                let a = if c1 == 1 { 0 } else { right };
                y1 += ((a - x1) as f64 * (y2 - y1) as f64 / (x2 - x1) as f64) as i64;
                x1 = a;
                c1 = 0;
            }
            if c2 != 0 {
                let a = if c2 == 1 { 0 } else { right };
                y2 += ((a - x2) as f64 * (y2 - y1) as f64 / (x2 - x1) as f64) as i64;
                x2 = a;
                c2 = 0;
            }
        }
    }
    *p1 = (x1, y1);
    *p2 = (x2, y2);
    (c1 | c2) == 0
}

/// Line(img, pt1, pt2, color, 8): LineIterator (clipped to the image),
/// connectivity 8, left to right.
fn line(img: &mut [u8], h: usize, w: usize, p1: (i64, i64), p2: (i64, i64), color: u8) {
    let (mut pt1, mut pt2) = (p1, p2);
    let outside = |p: (i64, i64)| p.0 < 0 || p.0 >= w as i64 || p.1 < 0 || p.1 >= h as i64;
    if (outside(pt1) || outside(pt2)) && !clip_line(w as i64, h as i64, &mut pt1, &mut pt2) {
        return;
    }
    let (mut delta_x, mut delta_y) = (1i64, 1i64);
    let mut dx = pt2.0 - pt1.0;
    let mut dy = pt2.1 - pt1.1;
    if dx < 0 {
        dx = -dx;
        dy = -dy;
        pt1 = pt2;
    }
    if dy < 0 {
        dy = -dy;
        delta_y = -1;
    }
    let vert = dy > dx;
    if vert {
        std::mem::swap(&mut dx, &mut dy);
        std::mem::swap(&mut delta_x, &mut delta_y);
    }
    let mut err = dx - (dy + dy);
    let plus_delta = dx + dx;
    let minus_delta = -(dy + dy);
    let (mut minus_shift, mut plus_shift, mut minus_step, mut plus_step) =
        (delta_x, 0i64, 0i64, delta_y);
    if vert {
        std::mem::swap(&mut plus_step, &mut plus_shift);
        std::mem::swap(&mut minus_step, &mut minus_shift);
    }
    let count = dx + 1;
    let mut p = pt1;
    for _ in 0..count {
        img[p.1 as usize * w + p.0 as usize] = color;
        let mask = if err < 0 { -1i64 } else { 0 };
        err += minus_delta + (plus_delta & mask);
        p.0 += minus_shift + (plus_shift & mask);
        p.1 += minus_step + (plus_step & mask);
    }
}

/// fillConvexPoly(img, pts, color) with LINE_8 and shift 0.
pub fn fill_convex_poly(img: &mut [u8], h: usize, w: usize, v: &[[i64; 2]], color: u8) {
    let npts = v.len();
    if npts == 0 {
        return;
    }
    let delta1 = XY_ONE >> 1;
    let delta2 = XY_ONE >> 1;
    let mut p0 = (v[npts - 1][0] << XY_SHIFT, v[npts - 1][1] << XY_SHIFT);
    let (mut xmin, mut xmax, mut ymin, mut ymax) = (v[0][0], v[0][0], v[0][1], v[0][1]);
    let mut imin = 0usize;
    for (i, q) in v.iter().enumerate() {
        if q[1] < ymin {
            ymin = q[1];
            imin = i;
        }
        ymax = ymax.max(q[1]);
        xmax = xmax.max(q[0]);
        xmin = xmin.min(q[0]);
        let p = (q[0] << XY_SHIFT, q[1] << XY_SHIFT);
        line(
            img,
            h,
            w,
            (p0.0 >> XY_SHIFT, p0.1 >> XY_SHIFT),
            (p.0 >> XY_SHIFT, p.1 >> XY_SHIFT),
            color,
        );
        p0 = p;
    }
    if npts < 3 || xmax < 0 || ymax < 0 || xmin >= w as i64 || ymin >= h as i64 {
        return;
    }
    let ymax = ymax.min(h as i64 - 1);
    #[derive(Clone, Copy)]
    struct E {
        idx: usize,
        di: usize,
        x: i64,
        dx: i64,
        ye: i64,
    }
    let mut edge = [
        E {
            idx: imin,
            di: 1,
            x: -XY_ONE,
            dx: 0,
            ye: ymin,
        },
        E {
            idx: imin,
            di: npts - 1,
            x: -XY_ONE,
            dx: 0,
            ye: ymin,
        },
    ];
    let mut edges = npts as i64;
    let mut y = ymin;
    loop {
        for e in edge.iter_mut() {
            if y >= e.ye {
                let mut idx0 = e.idx;
                let mut idx = idx0 + e.di;
                if idx >= npts {
                    idx -= npts;
                }
                loop {
                    let go = edges > 0;
                    edges -= 1;
                    if !go {
                        break;
                    }
                    let ty = v[idx][1];
                    if ty > y {
                        let xs = v[idx0][0] << XY_SHIFT;
                        let xe = v[idx][0] << XY_SHIFT;
                        e.ye = ty;
                        e.dx = ((xe - xs) * 2 + (ty - y)) / (2 * (ty - y));
                        e.x = xs;
                        e.idx = idx;
                        break;
                    }
                    idx0 = idx;
                    idx += e.di;
                    if idx >= npts {
                        idx -= npts;
                    }
                }
            }
        }
        if edges < 0 {
            break;
        }
        if y >= 0 {
            let (left, right) = if edge[0].x > edge[1].x {
                (1, 0)
            } else {
                (0, 1)
            };
            let mut xx1 = (edge[left].x + delta1) >> XY_SHIFT;
            let mut xx2 = (edge[right].x + delta2) >> XY_SHIFT;
            if xx2 >= 0 && xx1 < w as i64 {
                xx1 = xx1.max(0);
                xx2 = xx2.min(w as i64 - 1);
                for x in xx1..=xx2 {
                    img[y as usize * w + x as usize] = color;
                }
            }
        }
        edge[0].x += edge[0].dx;
        edge[1].x += edge[1].dx;
        y += 1;
        if y > ymax {
            break;
        }
    }
}

#[derive(Clone, Copy)]
struct Edge {
    y0: i64,
    y1: i64,
    x: i64,
    dx: i64,
    next: usize, // index into edges; usize::MAX = none
}

/// fillPoly(img, [pts], color) with LINE_8, shift 0, no offset.
pub fn fill_poly(img: &mut [u8], h: usize, w: usize, pts: &[[i64; 2]], color: u8) {
    let count = pts.len();
    if count == 0 {
        return;
    }
    let mut edges: Vec<Edge> = Vec::new();
    // CollectPolyEdges
    let mut pt0 = (pts[count - 1][0] << XY_SHIFT, pts[count - 1][1]);
    for v in pts {
        let pt1 = (v[0] << XY_SHIFT, v[1]);
        let t0 = ((pt0.0 + (XY_ONE >> 1)) >> XY_SHIFT, pt0.1);
        let t1 = ((pt1.0 + (XY_ONE >> 1)) >> XY_SHIFT, pt1.1);
        line(img, h, w, t0, t1, color);
        let (pt0c, pt1c) = ((t0.0 << XY_SHIFT, pt0.1), (t1.0 << XY_SHIFT, pt1.1));
        if pt0.1 != pt1.1 {
            let dx = (pt1c.0 - pt0c.0) / (pt1c.1 - pt0c.1);
            let e = if pt0.1 < pt1.1 {
                Edge {
                    y0: pt0.1,
                    y1: pt1.1,
                    x: pt0c.0 + (pt0.1 - pt0c.1) * dx,
                    dx,
                    next: usize::MAX,
                }
            } else {
                Edge {
                    y0: pt1.1,
                    y1: pt0.1,
                    x: pt1c.0 + (pt1.1 - pt1c.1) * dx,
                    dx,
                    next: usize::MAX,
                }
            };
            edges.push(e);
        }
        pt0 = pt1;
    }
    fill_edges(img, h, w, edges, color);
}

fn fill_edges(img: &mut [u8], h: usize, w: usize, mut edges: Vec<Edge>, color: u8) {
    let total = edges.len();
    if total < 2 {
        return;
    }
    let (mut y_max, mut y_min) = (i64::MIN, i64::MAX);
    let (mut x_max, mut x_min) = (-1i64, i64::MAX);
    for e in &edges {
        let x1 = e.x + (e.y1 - e.y0) * e.dx;
        y_min = y_min.min(e.y0);
        y_max = y_max.max(e.y1);
        x_min = x_min.min(e.x).min(x1);
        x_max = x_max.max(e.x).max(x1);
    }
    if y_max < 0 || y_min >= h as i64 || x_max < 0 || x_min >= (w as i64) << XY_SHIFT {
        return;
    }
    // CmpEdges: by y0, then x, then dx (std::sort; ties of all three keep no particular order,
    // but such edges are identical for filling)
    edges.sort_by_key(|a| (a.y0, a.x, a.dx));
    edges.push(Edge {
        y0: i64::MAX,
        y1: 0,
        x: 0,
        dx: 0,
        next: usize::MAX,
    });
    const TMP: usize = usize::MAX - 1; // the list head ("tmp" in OpenCV)
    let mut tmp_next: usize = usize::MAX;
    let next_of = |edges: &Vec<Edge>, tmp_next: usize, i: usize| {
        if i == TMP {
            tmp_next
        } else {
            edges[i].next
        }
    };
    let delta = XY_ONE - 1;
    let mut i = 0usize;
    let y_max = y_max.min(h as i64);
    let mut y = edges[0].y0;
    while y < y_max {
        let mut draw = false;
        let clipline = y < 0;
        let mut prelast = TMP;
        let mut last = tmp_next;
        loop {
            let e = i;
            if !(last != usize::MAX || edges[e].y0 == y) {
                break;
            }
            if last != usize::MAX && edges[last].y1 == y {
                let ln = edges[last].next;
                if prelast == TMP {
                    tmp_next = ln;
                } else {
                    edges[prelast].next = ln;
                }
                last = ln;
                continue;
            }
            let keep_prelast = prelast;
            if last != usize::MAX && (edges[e].y0 > y || edges[last].x < edges[e].x) {
                prelast = last;
                last = edges[last].next;
            } else if i < total {
                if prelast == TMP {
                    tmp_next = e;
                } else {
                    edges[prelast].next = e;
                }
                edges[e].next = last;
                prelast = e;
                i += 1;
            } else {
                break;
            }
            if draw {
                if !clipline {
                    let (kx, px) = (edges[keep_prelast].x, edges[prelast].x);
                    let (mut x1, mut x2) = if kx > px {
                        ((px + delta) >> XY_SHIFT, kx >> XY_SHIFT)
                    } else {
                        ((kx + delta) >> XY_SHIFT, px >> XY_SHIFT)
                    };
                    if x1 < w as i64 && x2 >= 0 {
                        x1 = x1.max(0);
                        x2 = x2.min(w as i64 - 1);
                        for x in x1..=x2 {
                            img[y as usize * w + x as usize] = color;
                        }
                    }
                }
                let kdx = edges[keep_prelast].dx;
                edges[keep_prelast].x += kdx;
                let pdx = edges[prelast].dx;
                edges[prelast].x += pdx;
            }
            draw = !draw;
        }
        // bubble sort the active list by x
        let mut keep_prelast = usize::MAX;
        loop {
            let mut prelast = TMP;
            let mut last = tmp_next;
            let mut last_exchange = usize::MAX;
            while last != keep_prelast && edges[last].next != usize::MAX {
                let te = edges[last].next;
                if edges[last].x > edges[te].x {
                    if prelast == TMP {
                        tmp_next = te;
                    } else {
                        edges[prelast].next = te;
                    }
                    edges[last].next = edges[te].next;
                    edges[te].next = last;
                    prelast = te;
                    last_exchange = prelast;
                } else {
                    prelast = last;
                    last = te;
                }
            }
            if last_exchange == usize::MAX {
                break;
            }
            keep_prelast = last_exchange;
            if keep_prelast == next_of(&edges, tmp_next, TMP) || keep_prelast == TMP {
                break;
            }
        }
        y += 1;
    }
}
