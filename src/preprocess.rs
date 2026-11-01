//! homr's page preprocessing and segmentation post-processing (main.py:
//! load_and_preprocess_predictions; autocrop.py, resize.py, color_adjust.py,
//! segmentation/inference_segnet.py, noise_filtering.py), with the OpenCV 4.11
//! and Pillow routines they use: cvtColor BGR2GRAY, calcHist, threshold,
//! morphology, findContours(RETR_EXTERNAL), Pillow's bicubic resize, CLAHE,
//! filter2D.

use crate::contours::find_external_contours;
use crate::raster::{ellipse_kernel, morph};
use crate::warp::Gray;

/// A 3-channel uint8 image (BGR, as cv2.imread gives it), row major.
#[derive(Clone, Debug)]
pub struct Bgr {
    pub w: usize,
    pub h: usize,
    pub data: Vec<u8>,
}

/// cvtColor(BGR2GRAY): fixed point, (B*3735 + G*19235 + R*9798 + 2^14) >> 15.
pub fn bgr_to_gray(img: &Bgr) -> Gray {
    let data = img.data.chunks_exact(3).map(|p| ((p[0] as u32 * 3735 + p[1] as u32 * 19235 + p[2] as u32 * 9798 + (1 << 14)) >> 15) as u8).collect();
    Gray { w: img.w, h: img.h, data }
}

/// autocrop_with_offset: the page cut out of a photo, and its top-left corner.
pub fn autocrop(img: &Bgr) -> (Bgr, (usize, usize)) {
    let gray = bgr_to_gray(img);
    let mut hist = [0u64; 256];
    for p in img.data.chunks_exact(3) {
        hist[p[0] as usize] += 1;
    }
    let dominant = (0..256).fold(0, |b, i| if hist[i] > hist[b] { i } else { b }) as i64;
    let t = dominant - 30;
    let thresh: Vec<u8> = gray.data.iter().map(|&v| if (v as i64) > t { 255 } else { 0 }).collect();
    let ones = |n: usize| vec![vec![1u8; n]; n];
    let closed = morph(&morph(&thresh, img.h, img.w, &ones(7), true), img.h, img.w, &ones(7), false);
    let eroded = morph(&closed, img.h, img.w, &ones(9), false);
    let mut best: Option<Vec<[i32; 2]>> = None;
    let mut best_area = 0.0f64;
    for c in find_external_contours(&eroded, img.h, img.w) {
        let mut a = 0f64;
        let mut prev = c[c.len() - 1];
        for &p in &c {
            a += prev[0] as f32 as f64 * p[1] as f32 as f64 - prev[1] as f32 as f64 * p[0] as f32 as f64;
            prev = p;
        }
        let area = (a * 0.5).abs();
        if area > best_area {
            best_area = area;
            best = Some(c);
        }
    }
    let Some(c) = best else { return (img.clone(), (0, 0)) };
    let x = c.iter().map(|p| p[0]).min().unwrap() as usize;
    let y = c.iter().map(|p| p[1]).min().unwrap() as usize;
    let w = c.iter().map(|p| p[0]).max().unwrap() as usize - x + 1;
    let h = c.iter().map(|p| p[1]).max().unwrap() as usize - y + 1;
    if (x as f64) < img.w as f64 * 0.25 || (y as f64) < img.h as f64 * 0.25 {
        return (img.clone(), (0, 0));
    }
    let (w, h) = (w.min(img.w - x), h.min(img.h - y));
    let mut data = Vec::with_capacity(w * h * 3);
    for yy in y..y + h {
        data.extend_from_slice(&img.data[(yy * img.w + x) * 3..(yy * img.w + x + w) * 3]);
    }
    (Bgr { w, h, data }, (x, y))
}

fn bicubic(x: f64) -> f64 {
    let a = -0.5;
    let x = x.abs();
    if x < 1.0 {
        ((a + 2.0) * x - (a + 3.0)) * x * x + 1.0
    } else if x < 2.0 {
        (((x - 5.0) * x + 8.0) * x - 4.0) * a
    } else {
        0.0
    }
}

/// Pillow's precompute_coeffs + normalize_coeffs_8bpc for the bicubic filter.
fn coeffs(in_size: usize, out_size: usize) -> (Vec<(usize, usize)>, Vec<Vec<i32>>) {
    let scale = in_size as f32 as f64 / out_size as f64;
    let filterscale = scale.max(1.0);
    let support = 2.0 * filterscale;
    let mut bounds = Vec::with_capacity(out_size);
    let mut kk = Vec::with_capacity(out_size);
    for xx in 0..out_size {
        let center = (xx as f64 + 0.5) * scale;
        let ss = 1.0 / filterscale;
        let xmin = ((center - support + 0.5) as i64).max(0);
        let xmax = ((center + support + 0.5) as i64).min(in_size as i64) - xmin;
        let mut k: Vec<f64> = (0..xmax).map(|x| bicubic((x as f64 + xmin as f64 - center + 0.5) * ss)).collect();
        let ww: f64 = k.iter().fold(0.0, |s, v| s + v);
        if ww != 0.0 {
            for v in &mut k {
                *v /= ww;
            }
        }
        let ki: Vec<i32> = k.iter().map(|&v| if v < 0.0 { (-0.5 + v * (1 << 22) as f64) as i32 } else { (0.5 + v * (1 << 22) as f64) as i32 }).collect();
        bounds.push((xmin as usize, xmax as usize));
        kk.push(ki);
    }
    (bounds, kk)
}

fn clip8(v: i32) -> u8 {
    (v >> 22).clamp(0, 255) as u8
}

/// PIL Image.resize((w, h)) with its default (bicubic) filter, 3 channels.
pub fn pil_resize(img: &Bgr, tw: usize, th: usize) -> Bgr {
    let need_h = tw != img.w;
    let need_v = th != img.h;
    let (hb, hk) = coeffs(img.w, tw);
    let (vb, vk) = coeffs(img.h, th);
    let mut cur = img.clone();
    if need_h {
        let first = vb[0].0;
        let last = vb[th - 1].0 + vb[th - 1].1;
        let rows = last - first;
        let mut out = vec![0u8; tw * rows * 3];
        for yy in 0..rows {
            let src = &img.data[(yy + first) * img.w * 3..(yy + first + 1) * img.w * 3];
            for xx in 0..tw {
                let (xmin, xmax) = hb[xx];
                let k = &hk[xx];
                for c in 0..3 {
                    let mut ss = 1i32 << 21;
                    for x in 0..xmax {
                        ss = ss.wrapping_add(src[(x + xmin) * 3 + c] as i32 * k[x]);
                    }
                    out[(yy * tw + xx) * 3 + c] = clip8(ss);
                }
            }
        }
        cur = Bgr { w: tw, h: rows, data: out };
        let vb2: Vec<(usize, usize)> = vb.iter().map(|&(a, b)| (a - first, b)).collect();
        return vertical(&cur, th, if need_v { Some((&vb2, &vk)) } else { None });
    }
    vertical(&cur, th, if need_v { Some((&vb, &vk)) } else { None })
}

fn vertical(img: &Bgr, th: usize, k: Option<(&Vec<(usize, usize)>, &Vec<Vec<i32>>)>) -> Bgr {
    let Some((vb, vk)) = k else { return img.clone() };
    let w = img.w;
    let mut out = vec![0u8; w * th * 3];
    for yy in 0..th {
        let (ymin, ymax) = vb[yy];
        let kk = &vk[yy];
        for xx in 0..w {
            for c in 0..3 {
                let mut ss = 1i32 << 21;
                for y in 0..ymax {
                    ss = ss.wrapping_add(img.data[((y + ymin) * w + xx) * 3 + c] as i32 * kk[y]);
                }
                out[(yy * w + xx) * 3 + c] = clip8(ss);
            }
        }
    }
    Bgr { w, h: th, data: out }
}

/// resize_image: 1920 wide, aspect kept (height rounded half to even).
pub fn resize_image(img: &Bgr) -> Bgr {
    let (tw, th) = if img.w == 1920 { (img.w, img.h) } else { (1920, (img.h as f64 * (1920.0 / img.w as f64)).round_ties_even() as usize) };
    if tw == img.w && th == img.h {
        return img.clone();
    }
    pil_resize(img, tw, th)
}

/// copyMakeBorder BORDER_REFLECT_101 index
fn reflect101(p: i64, n: i64) -> i64 {
    if n == 1 {
        return 0;
    }
    let mut p = p;
    loop {
        if p < 0 {
            p = -p;
        } else if p >= n {
            p = 2 * n - 2 - p;
        } else {
            return p;
        }
    }
}

/// createCLAHE(clipLimit=1.0, tileGridSize=(8, 8)).apply(gray)
pub fn clahe(src: &Gray) -> Gray {
    let (tiles_x, tiles_y) = (8usize, 8usize);
    let (ew, eh) = if src.w % tiles_x == 0 && src.h % tiles_y == 0 {
        (src.w, src.h)
    } else {
        (src.w + tiles_x - src.w % tiles_x, src.h + tiles_y - src.h % tiles_y)
    };
    // the image extended (bottom and right) by reflection, for the lookup tables
    let ext = |x: usize, y: usize| src.at(reflect101(x as i64, src.w as i64) as usize, reflect101(y as i64, src.h as i64) as usize);
    let (tw, th) = (ew / tiles_x, eh / tiles_y);
    let total = (tw * th) as i64;
    let lut_scale = 255.0f32 / total as f32;
    let clip_limit = ((1.0 * total as f64 / 256.0) as i64).max(1);
    let mut lut = vec![[0u8; 256]; tiles_x * tiles_y];
    for k in 0..tiles_x * tiles_y {
        let (ty, tx) = (k / tiles_x, k % tiles_x);
        let mut hist = [0i64; 256];
        for y in ty * th..(ty + 1) * th {
            for x in tx * tw..(tx + 1) * tw {
                hist[ext(x, y) as usize] += 1;
            }
        }
        let mut clipped = 0i64;
        for h in hist.iter_mut() {
            if *h > clip_limit {
                clipped += *h - clip_limit;
                *h = clip_limit;
            }
        }
        let batch = clipped / 256;
        let mut residual = clipped - batch * 256;
        for h in hist.iter_mut() {
            *h += batch;
        }
        if residual != 0 {
            let step = (256 / residual).max(1) as usize;
            let mut i = 0;
            while i < 256 && residual > 0 {
                hist[i] += 1;
                i += step;
                residual -= 1;
            }
        }
        let mut sum = 0i64;
        for i in 0..256 {
            sum += hist[i];
            lut[k][i] = ((sum as f32 * lut_scale) as f64).round_ties_even().clamp(0.0, 255.0) as u8;
        }
    }
    let inv_tw = 1.0f32 / tw as f32;
    let inv_th = 1.0f32 / th as f32;
    let xs: Vec<(usize, usize, f32, f32)> = (0..src.w)
        .map(|x| {
            let txf = x as f32 * inv_tw - 0.5f32;
            let tx1 = (txf as f64).floor() as i64;
            let xa = txf - tx1 as f32;
            ((tx1.max(0)) as usize, ((tx1 + 1).min(tiles_x as i64 - 1)) as usize, xa, 1.0f32 - xa)
        })
        .collect();
    let mut out = Gray::new(src.w, src.h, 0);
    for y in 0..src.h {
        let tyf = y as f32 * inv_th - 0.5f32;
        let ty1 = (tyf as f64).floor() as i64;
        let ya = tyf - ty1 as f32;
        let ya1 = 1.0f32 - ya;
        let (t1, t2) = (ty1.max(0) as usize, (ty1 + 1).min(tiles_y as i64 - 1) as usize);
        for x in 0..src.w {
            let v = src.at(x, y) as usize;
            let (i1, i2, xa, xa1) = xs[x];
            let (p1, p2) = (&lut[t1 * tiles_x..], &lut[t2 * tiles_x..]);
            let res = (p1[i1][v] as f32 * xa1 + p1[i2][v] as f32 * xa) * ya1 + (p2[i1][v] as f32 * xa1 + p2[i2][v] as f32 * xa) * ya;
            out.data[y * src.w + x] = (res as f64).round_ties_even().clamp(0.0, 255.0) as u8;
        }
    }
    out
}

pub struct Preprocessed {
    /// the page (cropped, 1920 wide), BGR: homr's "original"
    pub image: Bgr,
    /// its contrast-adjusted greyscale (CLAHE): what segnet and the staff images use
    pub gray: Gray,
    pub crop: (usize, usize),
}

/// The first half of load_and_preprocess_predictions.
pub fn preprocess(img: &Bgr) -> Preprocessed {
    let (cropped, crop) = autocrop(img);
    let image = resize_image(&cropped);
    let gray = clahe(&bgr_to_gray(&image));
    Preprocessed { image, gray, crop }
}

pub const WIN: usize = 320;

/// The tiles segnet reads (top-left corners), in homr's order (step 320).
pub fn tile_origins(h: usize, w: usize) -> Vec<(i64, i64)> {
    let mut out = Vec::new();
    let mut yl = 0;
    while yl < h.max(WIN) {
        let y = (yl as i64).min(h as i64 - WIN as i64);
        let mut xl = 0;
        while xl < w.max(WIN) {
            let x = (xl as i64).min(w as i64 - WIN as i64);
            out.push((y, x));
            xl += WIN;
        }
        yl += WIN;
    }
    out
}

/// One tile as segnet's input: 3 x 320 x 320 float32, white padded.
pub fn tile_input(gray: &Gray, y: i64, x: i64) -> Vec<f32> {
    let mut out = vec![255f32; 3 * WIN * WIN];
    let (y0, x0) = (y.max(0) as usize, x.max(0) as usize);
    let (y1, x1) = (((y + WIN as i64) as usize).min(gray.h), ((x + WIN as i64) as usize).min(gray.w));
    for c in 0..3 {
        for yy in y0..y1 {
            for xx in x0..x1 {
                out[c * WIN * WIN + (yy - y0) * WIN + (xx - x0)] = gray.at(xx, yy) as f32;
            }
        }
    }
    out
}

/// merge_patches: the per-tile class maps (argmax of segnet's output) into
/// one map for the page, averaging (and truncating) where tiles overlap.
pub fn merge_tiles(classes: &[Vec<u8>], h: usize, w: usize) -> Vec<u8> {
    let mut acc = vec![0f32; h * w];
    let mut weight = vec![0f32; h * w];
    let mut idx = 0;
    let mut iy = 0;
    while iy < h {
        let y = (iy as i64).min(h as i64 - WIN as i64);
        let (y0, y1) = (y.max(0) as usize, ((y + WIN as i64) as usize).min(h));
        let mut ix = 0;
        while ix < w {
            let x = (ix as i64).min(w as i64 - WIN as i64);
            let (x0, x1) = (x.max(0) as usize, ((x + WIN as i64) as usize).min(w));
            let p = &classes[idx];
            for yy in y0..y1 {
                for xx in x0..x1 {
                    acc[yy * w + xx] += p[(yy - y0) * WIN + (xx - x0)] as f32;
                    weight[yy * w + xx] += 1.0;
                }
            }
            idx += 1;
            ix += WIN;
        }
        iy += WIN;
    }
    acc.iter().zip(&weight).map(|(&a, &wt)| (a / if wt == 0.0 { 1.0 } else { wt }) as i64 as u8).collect()
}

pub struct SegMasks {
    pub h: usize,
    pub w: usize,
    pub staff: Vec<u8>,
    pub symbols: Vec<u8>,
    pub stems_rest: Vec<u8>,
    pub notehead: Vec<u8>,
    pub clefs_keys: Vec<u8>,
}

/// The class map split into homr's masks (1 stems/rests, 2 noteheads, 3 clefs/keys, 4 staff, 5 symbols).
pub fn masks(merged: &[u8], h: usize, w: usize) -> SegMasks {
    let m = |c: u8| merged.iter().map(|&v| (v == c) as u8).collect::<Vec<u8>>();
    SegMasks { h, w, stems_rest: m(1), notehead: m(2), clefs_keys: m(3), staff: m(4), symbols: m(5) }
}

/// estimate_noise: sum |Laplacian-like filter2D| / area (BORDER_REFLECT_101).
fn estimate_noise(tile: &[u8], h: usize, w: usize) -> f64 {
    const K: [[i64; 3]; 3] = [[1, -2, 1], [-2, 4, -2], [1, -2, 1]];
    let mut sum = 0f64;
    for y in 0..h as i64 {
        for x in 0..w as i64 {
            let mut v = 0i64;
            for dy in -1..=1i64 {
                for dx in -1..=1i64 {
                    let yy = reflect101(y + dy, h as i64) as usize;
                    let xx = reflect101(x + dx, w as i64) as usize;
                    v += K[(dy + 1) as usize][(dx + 1) as usize] * tile[yy * w + xx] as i64;
                }
            }
            sum += (v.abs()) as f64;
        }
    }
    sum / (h * w) as f64
}

/// filter_predictions: a mask of the cells to keep, or None (keep everything).
pub fn noise_mask(staff: &[u8], h: usize, w: usize) -> Option<Vec<u8>> {
    let gray: Vec<u8> = staff.iter().map(|&v| v.wrapping_mul(255)).collect();
    let (m, n) = (h / 20, w / 20);
    let (gh, gw) = (h.div_ceil(m), w.div_ceil(n));
    let mut grid = vec![0u8; gh * gw];
    for (i, y1) in (0..h).step_by(m).enumerate() {
        for (j, x1) in (0..w).step_by(n).enumerate() {
            let (y2, x2) = ((y1 + m).min(h), (x1 + n).min(w));
            let mut tile = Vec::with_capacity((y2 - y1) * (x2 - x1));
            for y in y1..y2 {
                tile.extend_from_slice(&gray[y * w + x1..y * w + x2]);
            }
            let sigma = estimate_noise(&tile, y2 - y1, x2 - x1);
            grid[i * gw + j] = (sigma.trunc() as i64).rem_euclid(256) as u8; // numpy's cast to uint8
        }
    }
    let mut mask = vec![0u8; h * w];
    let (mut filtered, mut total) = (0usize, 0usize);
    for (i, y1) in (0..h).step_by(m).enumerate() {
        for (j, x1) in (0..w).step_by(n).enumerate() {
            let noise = grid[i * gw + j];
            let mut neighbors = Vec::new();
            if i > 0 {
                neighbors.push(grid[(i - 1) * gw + j]);
            }
            if j > 0 {
                neighbors.push(grid[i * gw + j - 1]);
            }
            if i < gh - 1 {
                neighbors.push(grid[(i + 1) * gw + j]);
            }
            if j < gw - 1 {
                neighbors.push(grid[i * gw + j + 1]);
            }
            if noise > 50 && neighbors.iter().any(|&v| v > 50) {
                filtered += 1;
            } else {
                for y in y1..(y1 + m).min(h) {
                    for x in x1..(x1 + n).min(w) {
                        mask[y * w + x] = 255;
                    }
                }
            }
            total += 1;
        }
    }
    if filtered as f64 / total as f64 > 0.5 || filtered == 0 {
        None
    } else {
        Some(mask)
    }
}

/// The second half of load_and_preprocess_predictions: noise filter (also
/// applied to the preprocessed image) and make_lines_stronger on the staff.
pub fn postprocess(mut m: SegMasks, gray: &mut Gray) -> SegMasks {
    if let Some(mask) = noise_mask(&m.staff, m.h, m.w) {
        for v in [&mut m.staff, &mut m.symbols, &mut m.stems_rest, &mut m.notehead, &mut m.clefs_keys] {
            for (p, &k) in v.iter_mut().zip(&mask) {
                if k == 0 {
                    *p = 0;
                }
            }
        }
        for (p, &k) in gray.data.iter_mut().zip(&mask) {
            if k == 0 {
                *p = 0;
            }
        }
    }
    let strong = morph(&m.staff, m.h, m.w, &ellipse_kernel(1, 2), true);
    m.staff = strong.iter().map(|&v| (v as f64 > 0.1) as u8).collect();
    m
}
