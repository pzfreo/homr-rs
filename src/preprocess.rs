//! homr's page preprocessing and segmentation post-processing (main.py:
//! load_and_preprocess_predictions; autocrop.py, resize.py, color_adjust.py,
//! segmentation/inference_segnet.py, noise_filtering.py), with the OpenCV 4.11
//! and Pillow routines they use: cvtColor BGR2GRAY, calcHist, threshold,
//! morphology, findContours(RETR_EXTERNAL), Pillow's bicubic resize, CLAHE,
//! filter2D.

use crate::contours::find_external_contours;
use crate::geom::convex_hull_points;
use crate::raster::{ellipse_kernel, fill_poly, morph};
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
    let data = img
        .data
        .as_chunks::<3>()
        .0
        .iter()
        .map(|p| {
            ((p[0] as u32 * 3735 + p[1] as u32 * 19235 + p[2] as u32 * 9798 + (1 << 14)) >> 15)
                as u8
        })
        .collect();
    Gray {
        w: img.w,
        h: img.h,
        data,
    }
}

/// Light and shadows change the brightness of paper but hardly its color. Mixed
/// daylight and lamp light change it at most by this much.
const MAX_CHROMATICITY_DIFFERENCE: f32 = 0.07;

/// _chromaticity: color without brightness, every channel divided by the sum of the
/// channels (float32, the sum left to right as numpy adds three values).
fn chromaticity(p: [u8; 3]) -> [f32; 3] {
    let v = [p[0] as f32 + 1.0, p[1] as f32 + 1.0, p[2] as f32 + 1.0];
    let sum = (v[0] + v[1]) + v[2];
    [v[0] / sum, v[1] / sum, v[2] / sum]
}

/// np.median of float32 values: the middle one, or for an even count the float32
/// mean of the two middle ones.
fn median_f32(v: &mut [f32]) -> f32 {
    let n = v.len();
    let (lower, mid, _) = v.select_nth_unstable_by(n / 2, |a, b| a.total_cmp(b));
    let upper = *mid;
    if n % 2 == 1 {
        upper
    } else {
        let below = lower.iter().copied().fold(f32::NEG_INFINITY, f32::max);
        (below + upper) / 2.0
    }
}

/// np.median(_chromaticity(pixels), axis=0).
fn median_chromaticity(pixels: &[[u8; 3]]) -> [f32; 3] {
    let mut out = [0f32; 3];
    for (c, o) in out.iter_mut().enumerate() {
        let mut v: Vec<f32> = pixels.iter().map(|p| chromaticity(*p)[c]).collect();
        *o = median_f32(&mut v);
    }
    out
}

/// np.abs(a - b).sum() <= max_chromaticity_difference, in float32 as numpy does it.
fn chromaticity_close(a: [f32; 3], b: [f32; 3]) -> bool {
    let d = [
        (a[0] - b[0]).abs(),
        (a[1] - b[1]).abs(),
        (a[2] - b[2]).abs(),
    ];
    (d[0] + d[1]) + d[2] <= MAX_CHROMATICITY_DIFFERENCE
}

/// _is_paper: the median color of the pixels is the paper's (no pixels count as paper).
fn is_paper(img: &Bgr, rows: (usize, usize), cols: (usize, usize), paper: [f32; 3]) -> bool {
    let mut pixels: Vec<[u8; 3]> = Vec::new();
    for y in rows.0..rows.1 {
        for x in cols.0..cols.1 {
            let i = (y * img.w + x) * 3;
            pixels.push([img.data[i], img.data[i + 1], img.data[i + 2]]);
        }
    }
    pixels.is_empty() || chromaticity_close(median_chromaticity(&pixels), paper)
}

/// _keep_sides_with_paper: extends the box to the image border on every side where
/// the strip beside it is paper, e.g. in a shadow. Only the strips beside the box,
/// the corners can contain e.g. a desk at another side. (x, y, w, h)
fn keep_sides_with_paper(
    img: &Bgr,
    (x, y, w, h): (usize, usize, usize, usize),
    paper: [f32; 3],
) -> (usize, usize, usize, usize) {
    let keep_left = is_paper(img, (y, y + h), (0, x), paper);
    let keep_right = is_paper(img, (y, y + h), (x + w, img.w), paper);
    let keep_top = is_paper(img, (0, y), (x, x + w), paper);
    let keep_bottom = is_paper(img, (y + h, img.h), (x, x + w), paper);
    let left = if keep_left { 0 } else { x };
    let top = if keep_top { 0 } else { y };
    let right = if keep_right { img.w } else { x + w };
    let bottom = if keep_bottom { img.h } else { y + h };
    (left, top, right - left, bottom - top)
}

/// cv2.blur(img, (7, 7)): the box sum of 7x7 (reflected at the border) divided by 49
/// and rounded. 49 is odd, so there are no ties and float32, float64 and
/// (sum + 24) / 49 all round alike.
fn blur7(img: &Bgr) -> Bgr {
    let (w, h) = (img.w, img.h);
    let mut rows = vec![0u32; w * h * 3];
    for y in 0..h {
        for c in 0..3 {
            for x in 0..w {
                let mut sum = 0u32;
                for k in -3i64..=3 {
                    sum += img.data[(y * w + reflect101(x as i64 + k, w as i64) as usize) * 3 + c]
                        as u32;
                }
                rows[(y * w + x) * 3 + c] = sum;
            }
        }
    }
    let mut data = vec![0u8; w * h * 3];
    for y in 0..h {
        for i in 0..w * 3 {
            let mut sum = 0u32;
            for k in -3i64..=3 {
                sum += rows[reflect101(y as i64 + k, h as i64) as usize * w * 3 + i];
            }
            data[y * w * 3 + i] = ((sum + 24) / 49) as u8;
        }
    }
    Bgr { w, h, data }
}

/// _remove_surroundings: removes what is beside the sheet, e.g. the corners of a desk
/// beside a rotated sheet, by filling it with the color of the paper. The sheet
/// itself is unchanged. `sheet` is the paper's contour in the crop's coordinates.
fn remove_surroundings(
    crop: &Bgr,
    sheet: &[[i32; 2]],
    paper: &[[u8; 3]],
    paper_chroma: [f32; 3],
) -> Bgr {
    let (w, h) = (crop.w, crop.h);
    let mut beside_sheet = vec![1u8; w * h];
    let hull: Vec<[i64; 2]> = convex_hull_points(sheet)
        .iter()
        .map(|p| [p[0] as i64, p[1] as i64])
        .collect();
    fill_poly(&mut beside_sheet, h, w, &hull, 0);
    // np.median(paper, axis=0) of uint8: float64, the mean of the middle two if even
    let mut paper_color = [0f64; 3];
    for (c, o) in paper_color.iter_mut().enumerate() {
        let mut v: Vec<u8> = paper.iter().map(|p| p[c]).collect();
        let n = v.len();
        let (lower, mid, _) = v.select_nth_unstable(n / 2);
        *o = if n % 2 == 1 {
            *mid as f64
        } else {
            (*lower.iter().max().unwrap() as f64 + *mid as f64) / 2.0
        };
    }
    // Single pixels are too noisy to compare their color
    let smooth = blur7(crop);
    let bright_over = 0.25 * (((paper_color[0] + paper_color[1]) + paper_color[2]) / 3.0);
    let mut background = vec![0u8; w * h];
    for i in 0..w * h {
        let p = [
            smooth.data[i * 3],
            smooth.data[i * 3 + 1],
            smooth.data[i * 3 + 2],
        ];
        // The color of dark pixels is noise, e.g. ink in a shadow must stay
        let bright = (p[0] as u32 + p[1] as u32 + p[2] as u32) as f64 / 3.0 > bright_over;
        let c = chromaticity(p);
        let d = [
            (c[0] - paper_chroma[0]).abs(),
            (c[1] - paper_chroma[1]).abs(),
            (c[2] - paper_chroma[2]).abs(),
        ];
        let difference = (d[0] + d[1]) + d[2];
        background[i] =
            (beside_sheet[i] != 0 && bright && difference > MAX_CHROMATICITY_DIFFERENCE) as u8;
    }
    // Also the pixels at the edge of the paper, which mix paper and background
    let kernel = vec![vec![1u8; 5]; 5];
    let background = morph(&background, h, w, &kernel, true);
    let fill = [
        paper_color[0] as u8,
        paper_color[1] as u8,
        paper_color[2] as u8,
    ];
    let mut result = crop.clone();
    for i in 0..w * h {
        if background[i] & beside_sheet[i] > 0 {
            result.data[i * 3..i * 3 + 3].copy_from_slice(&fill);
        }
    }
    result
}

/// autocrop_with_offset: the page cut out of a photo, and its top-left corner. Areas
/// beside the paper which have another color than the paper, e.g. a desk, are painted
/// in the paper color.
pub fn autocrop(img: &Bgr) -> (Bgr, (usize, usize)) {
    let gray = bgr_to_gray(img);
    let mut hist = [0u64; 256];
    for p in img.data.as_chunks::<3>().0 {
        hist[p[0] as usize] += 1;
    }
    let dominant = (0..256).fold(0, |b, i| if hist[i] > hist[b] { i } else { b }) as i64;
    let t = dominant - 30;
    let thresh: Vec<u8> = gray
        .data
        .iter()
        .map(|&v| if (v as i64) > t { 255 } else { 0 })
        .collect();
    let ones = |n: usize| vec![vec![1u8; n]; n];
    let closed = morph(
        &morph(&thresh, img.h, img.w, &ones(7), true),
        img.h,
        img.w,
        &ones(7),
        false,
    );
    let eroded = morph(&closed, img.h, img.w, &ones(9), false);
    let mut best: Option<Vec<[i32; 2]>> = None;
    let mut best_area = 0.0f64;
    for c in find_external_contours(&eroded, img.h, img.w) {
        let mut a = 0f64;
        let mut prev = c[c.len() - 1];
        for &p in &c {
            a += prev[0] as f32 as f64 * p[1] as f32 as f64
                - prev[1] as f32 as f64 * p[0] as f32 as f64;
            prev = p;
        }
        let area = (a * 0.5).abs();
        if area > best_area {
            best_area = area;
            best = Some(c);
        }
    }
    let Some(contour) = best else {
        return (img.clone(), (0, 0));
    };
    let x = contour.iter().map(|p| p[0]).min().unwrap() as usize;
    let y = contour.iter().map(|p| p[1]).min().unwrap() as usize;
    let w = contour.iter().map(|p| p[0]).max().unwrap() as usize - x + 1;
    let h = contour.iter().map(|p| p[1]).max().unwrap() as usize - y + 1;
    // If we can't find a large contour, then we assume that the picture doesn't have page borders
    if (w as f64) < img.w as f64 * 0.25 || (h as f64) < img.h as f64 * 0.25 {
        return (img.clone(), (0, 0));
    }
    // A sample of the paper is enough to know its color
    let mut paper: Vec<[u8; 3]> = Vec::new();
    let mut seen = 0usize;
    for yy in y..y + h {
        for xx in x..x + w {
            if thresh[yy * img.w + xx] > 0 {
                if seen.is_multiple_of(10) {
                    let i = (yy * img.w + xx) * 3;
                    paper.push([img.data[i], img.data[i + 1], img.data[i + 2]]);
                }
                seen += 1;
            }
        }
    }
    if paper.is_empty() {
        return (img.clone(), (0, 0));
    }
    let paper_chroma = median_chromaticity(&paper);
    // The contour was eroded, keep a margin so that staffs at the edge of the paper stay
    let margin = 5.max(img.w.max(img.h) / 100);
    let (x0, y0) = (x.saturating_sub(margin), y.saturating_sub(margin));
    let (x1, y1) = (img.w.min(x + w + margin), img.h.min(y + h + margin));
    let (x, y, w, h) = keep_sides_with_paper(img, (x0, y0, x1 - x0, y1 - y0), paper_chroma);
    if (x, y, w, h) == (0, 0, img.w, img.h) {
        return (img.clone(), (0, 0));
    }
    let mut data = Vec::with_capacity(w * h * 3);
    for yy in y..y + h {
        data.extend_from_slice(&img.data[(yy * img.w + x) * 3..(yy * img.w + x + w) * 3]);
    }
    let crop = Bgr { w, h, data };
    let sheet: Vec<[i32; 2]> = contour
        .iter()
        .map(|p| [p[0] - x as i32, p[1] - y as i32])
        .collect();
    (
        remove_surroundings(&crop, &sheet, &paper, paper_chroma),
        (x, y),
    )
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
        let mut k: Vec<f64> = (0..xmax)
            .map(|x| bicubic((x as f64 + xmin as f64 - center + 0.5) * ss))
            .collect();
        let ww: f64 = k.iter().fold(0.0, |s, v| s + v);
        if ww != 0.0 {
            for v in &mut k {
                *v /= ww;
            }
        }
        let ki: Vec<i32> = k
            .iter()
            .map(|&v| {
                if v < 0.0 {
                    (-0.5 + v * (1 << 22) as f64) as i32
                } else {
                    (0.5 + v * (1 << 22) as f64) as i32
                }
            })
            .collect();
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
    if need_h {
        let first = vb[0].0;
        let last = vb[th - 1].0 + vb[th - 1].1;
        let rows = last - first;
        let mut out = vec![0u8; tw * rows * 3];
        for yy in 0..rows {
            let src = &img.data[(yy + first) * img.w * 3..(yy + first + 1) * img.w * 3];
            let dst = &mut out[yy * tw * 3..(yy + 1) * tw * 3];
            for xx in 0..tw {
                let (xmin, xmax) = hb[xx];
                let k = &hk[xx][..xmax];
                let px = &src[xmin * 3..(xmin + xmax) * 3];
                let (mut s0, mut s1, mut s2) = (1i32 << 21, 1i32 << 21, 1i32 << 21);
                for (p, &kv) in px.as_chunks::<3>().0.iter().zip(k) {
                    s0 = s0.wrapping_add(p[0] as i32 * kv);
                    s1 = s1.wrapping_add(p[1] as i32 * kv);
                    s2 = s2.wrapping_add(p[2] as i32 * kv);
                }
                dst[xx * 3] = clip8(s0);
                dst[xx * 3 + 1] = clip8(s1);
                dst[xx * 3 + 2] = clip8(s2);
            }
        }
        let cur = Bgr {
            w: tw,
            h: rows,
            data: out,
        };
        let vb2: Vec<(usize, usize)> = vb.iter().map(|&(a, b)| (a - first, b)).collect();
        return vertical(&cur, th, if need_v { Some((&vb2, &vk)) } else { None });
    }
    vertical(img, th, if need_v { Some((&vb, &vk)) } else { None })
}

type ResizeKernel<'a> = (&'a Vec<(usize, usize)>, &'a Vec<Vec<i32>>);
fn vertical(img: &Bgr, th: usize, k: Option<ResizeKernel<'_>>) -> Bgr {
    let Some((vb, vk)) = k else {
        return img.clone();
    };
    let row = img.w * 3;
    let mut out = vec![0u8; row * th];
    let mut acc = vec![0i32; row];
    for yy in 0..th {
        let (ymin, ymax) = vb[yy];
        acc.iter_mut().for_each(|a| *a = 1 << 21);
        for (y, &kv) in vk[yy][..ymax].iter().enumerate() {
            let src = &img.data[(y + ymin) * row..(y + ymin + 1) * row];
            for (a, &p) in acc.iter_mut().zip(src) {
                *a = a.wrapping_add(p as i32 * kv);
            }
        }
        for (o, &a) in out[yy * row..(yy + 1) * row].iter_mut().zip(&acc) {
            *o = clip8(a);
        }
    }
    Bgr {
        w: img.w,
        h: th,
        data: out,
    }
}

/// resize_image: 1920 wide, aspect kept (height rounded half to even).
pub fn resize_image(img: &Bgr) -> Bgr {
    let (tw, th) = if img.w == 1920 {
        (img.w, img.h)
    } else {
        (
            1920,
            (img.h as f64 * (1920.0 / img.w as f64)).round_ties_even() as usize,
        )
    };
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
    let (ew, eh) = if src.w.is_multiple_of(tiles_x) && src.h.is_multiple_of(tiles_y) {
        (src.w, src.h)
    } else {
        (
            src.w + tiles_x - src.w % tiles_x,
            src.h + tiles_y - src.h % tiles_y,
        )
    };
    // the image extended (bottom and right) by reflection, for the lookup tables
    let ext = |x: usize, y: usize| {
        src.at(
            reflect101(x as i64, src.w as i64) as usize,
            reflect101(y as i64, src.h as i64) as usize,
        )
    };
    let (tw, th) = (ew / tiles_x, eh / tiles_y);
    let total = (tw * th) as i64;
    let lut_scale = 255.0f32 / total as f32;
    let clip_limit = ((1.0 * total as f64 / 256.0) as i64).max(1);
    let mut lut = vec![[0u8; 256]; tiles_x * tiles_y];
    for (k, table) in lut.iter_mut().enumerate() {
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
            table[i] = ((sum as f32 * lut_scale) as f64)
                .round_ties_even()
                .clamp(0.0, 255.0) as u8;
        }
    }
    let inv_tw = 1.0f32 / tw as f32;
    let inv_th = 1.0f32 / th as f32;
    let xs: Vec<(usize, usize, f32, f32)> = (0..src.w)
        .map(|x| {
            let txf = x as f32 * inv_tw - 0.5f32;
            let tx1 = (txf as f64).floor() as i64;
            let xa = txf - tx1 as f32;
            (
                (tx1.max(0)) as usize,
                ((tx1 + 1).min(tiles_x as i64 - 1)) as usize,
                xa,
                1.0f32 - xa,
            )
        })
        .collect();
    let mut out = Gray::new(src.w, src.h, 0);
    for y in 0..src.h {
        let tyf = y as f32 * inv_th - 0.5f32;
        let ty1 = (tyf as f64).floor() as i64;
        let ya = tyf - ty1 as f32;
        let ya1 = 1.0f32 - ya;
        let (t1, t2) = (
            ty1.max(0) as usize,
            (ty1 + 1).min(tiles_y as i64 - 1) as usize,
        );
        for (x, &(i1, i2, xa, xa1)) in xs.iter().enumerate() {
            let v = src.at(x, y) as usize;
            let (p1, p2) = (&lut[t1 * tiles_x..], &lut[t2 * tiles_x..]);
            let res = (p1[i1][v] as f32 * xa1 + p1[i2][v] as f32 * xa) * ya1
                + (p2[i1][v] as f32 * xa1 + p2[i2][v] as f32 * xa) * ya;
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
    let (y1, x1) = (
        ((y + WIN as i64) as usize).min(gray.h),
        ((x + WIN as i64) as usize).min(gray.w),
    );
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
    acc.iter()
        .zip(&weight)
        .map(|(&a, &wt)| (a / if wt == 0.0 { 1.0 } else { wt }) as i64 as u8)
        .collect()
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
    SegMasks {
        h,
        w,
        stems_rest: m(1),
        notehead: m(2),
        clefs_keys: m(3),
        staff: m(4),
        symbols: m(5),
    }
}

/// estimate_noise: sum |Laplacian-like filter2D| / area (BORDER_REFLECT_101).
fn estimate_noise(tile: &[u8], h: usize, w: usize) -> f64 {
    let xs: Vec<[usize; 3]> = (0..w as i64)
        .map(|x| {
            [
                reflect101(x - 1, w as i64) as usize,
                x as usize,
                reflect101(x + 1, w as i64) as usize,
            ]
        })
        .collect();
    let mut sum = 0u64;
    for y in 0..h as i64 {
        let r = [
            reflect101(y - 1, h as i64) as usize,
            y as usize,
            reflect101(y + 1, h as i64) as usize,
        ];
        let (a, b, c) = (
            &tile[r[0] * w..r[0] * w + w],
            &tile[r[1] * w..r[1] * w + w],
            &tile[r[2] * w..r[2] * w + w],
        );
        for x in &xs {
            // [[1, -2, 1], [-2, 4, -2], [1, -2, 1]]
            let v = a[x[0]] as i32 - 2 * a[x[1]] as i32 + a[x[2]] as i32 - 2 * b[x[0]] as i32
                + 4 * b[x[1]] as i32
                - 2 * b[x[2]] as i32
                + c[x[0]] as i32
                - 2 * c[x[1]] as i32
                + c[x[2]] as i32;
            sum += v.unsigned_abs() as u64;
        }
    }
    sum as f64 / (h * w) as f64
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
        for v in [
            &mut m.staff,
            &mut m.symbols,
            &mut m.stems_rest,
            &mut m.notehead,
            &mut m.clefs_keys,
        ] {
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
