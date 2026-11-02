//! CPU simulation of the WGSL data movement/lane ownership, not GPU execution.
//! Constants and indices mirror tools/encoder_gpu_kernels.mjs. Ordered f32
//! arithmetic uses hardware fma/div/sqrt here; WGSL primitives have own probes.
use homr_rs::encoder::gpu::{self, ADD, CONV, DEPTHWISE, MATMUL, NORM, POSITION, SCALE};

pub fn simulate(q: &[u32; 16], a: &[f32], r: &[f32], weights: &[f32]) -> Vec<f32> {
    let [kind, rows, k, n, wo, bo, eps, h, w, c, kernel, stride, pad, ow, _, _] =
        q.map(|v| v as usize);
    let mut out = vec![0.0; rows * n];
    match kind as u32 {
        MATMUL | CONV => {
            let conv = kind as u32 == CONV;
            let stem = conv && c == 1;
            let blocked = conv && !stem;
            let block = if blocked { 128 } else { k };
            let cols = n.div_ceil(32);
            for group in 0..rows.div_ceil(16) * cols {
                let row_base = group / cols * 16;
                let col_base = group % cols * 32;
                let mut result = [[[0.0f32; 4]; 2]; 64];
                for start in (0..k).step_by(block) {
                    let mut sum = [[[0.0f32; 4]; 2]; 64];
                    if stem {
                        for (lane, lane_sum) in sum.iter_mut().enumerate() {
                            for v in lane_sum {
                                for (j, x) in v.iter_mut().enumerate() {
                                    let col = col_base + lane % 8 * 4 + j;
                                    if col < n {
                                        *x = weights[bo + col];
                                    }
                                }
                            }
                        }
                    }
                    for base in (start..(start + block).min(k)).step_by(16) {
                        let mut tile_a = [[0.0f32; 4]; 64];
                        let mut tile_b = [[0.0f32; 4]; 128];
                        for (lane, av) in tile_a.iter_mut().enumerate() {
                            let ar = row_base + lane / 4;
                            let ak = base + lane % 4 * 4;
                            if ar < rows {
                                for (j, value) in av.iter_mut().enumerate() {
                                    let kk = ak + j;
                                    if kk < k {
                                        *value = if conv {
                                            let channel = kk / (kernel * kernel);
                                            let ky = kk / kernel % kernel;
                                            let kx = kk % kernel;
                                            let iy =
                                                (ar / ow * stride + ky) as isize - pad as isize;
                                            let ix =
                                                (ar % ow * stride + kx) as isize - pad as isize;
                                            if iy >= 0
                                                && ix >= 0
                                                && iy < h as isize
                                                && ix < w as isize
                                            {
                                                a[(iy as usize * w + ix as usize) * c + channel]
                                            } else {
                                                0.0
                                            }
                                        } else {
                                            a[ar * k + kk]
                                        };
                                    }
                                }
                            }
                            for i in (lane..128).step_by(64) {
                                let bk = base + i / 8;
                                let bc = col_base + i % 8 * 4;
                                if bk < k && bc < n {
                                    tile_b[i].copy_from_slice(
                                        &weights[wo + bk * n + bc..wo + bk * n + bc + 4],
                                    );
                                }
                            }
                        }
                        // All lanes have completed their loads (WG barrier).
                        for (lane, lane_sum) in sum.iter_mut().enumerate() {
                            for kk in 0..16.min(k - base) {
                                let b = tile_b[kk * 8 + lane % 8];
                                for (half, values) in lane_sum.iter_mut().enumerate() {
                                    let x = tile_a[(lane / 8 + half * 8) * 4 + kk / 4][kk % 4];
                                    for j in 0..4 {
                                        values[j] = x.mul_add(b[j], values[j]);
                                    }
                                }
                            }
                        }
                        // All lanes finish before the next tile overwrites it.
                    }
                    for (lane, lane_sum) in sum.iter().enumerate() {
                        for half in 0..2 {
                            for j in 0..4 {
                                if blocked {
                                    result[lane][half][j] += lane_sum[half][j];
                                } else {
                                    result[lane][half][j] = lane_sum[half][j];
                                }
                            }
                        }
                    }
                }
                for (lane, lane_result) in result.iter().enumerate() {
                    for (half, values) in lane_result.iter().enumerate() {
                        let row = row_base + lane / 8 + half * 8;
                        for (j, &value) in values.iter().enumerate() {
                            let col = col_base + lane % 8 * 4 + j;
                            if row < rows && col < n {
                                out[row * n + col] = if stem {
                                    value
                                } else {
                                    value + weights[bo + col]
                                };
                            }
                        }
                    }
                }
            }
        }
        DEPTHWISE => {
            let channels = c.div_ceil(16);
            let tiles_w = w.div_ceil(4);
            let extent = 4 + kernel - 1;
            for group in 0..h.div_ceil(4) * tiles_w * channels {
                let spatial = group / channels;
                let ch = group % channels * 16;
                let y0 = spatial / tiles_w * 4;
                let x0 = spatial % tiles_w * 4;
                let mut tile_a = [[0.0f32; 4]; 400];
                let mut tile_b = [[0.0f32; 4]; 196];
                for lane in 0..64 {
                    for i in (lane..extent * extent * 4).step_by(64) {
                        let pixel = i / 4;
                        let cc = ch + i % 4 * 4;
                        let y = (y0 + pixel / extent) as isize - pad as isize;
                        let x = (x0 + pixel % extent) as isize - pad as isize;
                        if y >= 0 && x >= 0 && y < h as isize && x < w as isize && cc < c {
                            let offset = (y as usize * w + x as usize) * c + cc;
                            tile_a[i].copy_from_slice(&a[offset..offset + 4]);
                        }
                    }
                    for i in (lane..kernel * kernel * 4).step_by(64) {
                        let cc = ch + i % 4 * 4;
                        if cc < c {
                            let offset = wo + i / 4 * c + cc;
                            tile_b[i].copy_from_slice(&weights[offset..offset + 4]);
                        }
                    }
                }
                for lane in 0..64 {
                    let pixel = lane / 4;
                    let cv = lane % 4;
                    let y = y0 + pixel / 4;
                    let x = x0 + pixel % 4;
                    let cc = ch + cv * 4;
                    if y >= h || x >= w || cc >= c {
                        continue;
                    }
                    let mut sum = [0.0f32; 4];
                    for ky in 0..kernel {
                        for kx in 0..kernel {
                            let v = tile_a[((pixel / 4 + ky) * extent + pixel % 4 + kx) * 4 + cv];
                            let b = tile_b[(ky * kernel + kx) * 4 + cv];
                            for j in 0..4 {
                                sum[j] = v[j].mul_add(b[j], sum[j]);
                            }
                        }
                    }
                    for j in 0..4 {
                        out[(y * w + x) * c + cc + j] = sum[j] + weights[bo + cc + j];
                    }
                }
            }
        }
        NORM => {
            for group in 0..rows.div_ceil(64) {
                for lane in 0..64 {
                    let row = group * 64 + lane;
                    if row >= rows {
                        continue;
                    }
                    let (mut mean, mut m2) = (0.0f32, 0.0f32);
                    for j in 0..n {
                        let value = a[row * n + j];
                        let delta = value - mean;
                        mean += delta / (j + 1) as f32;
                        m2 = delta.mul_add(value - mean, m2);
                    }
                    let std = (m2 / n as f32 + f32::from_bits(eps as u32)).sqrt();
                    for j in 0..n {
                        out[row * n + j] = ((a[row * n + j] - mean) / std)
                            .mul_add(weights[wo + j], weights[bo + j]);
                    }
                }
            }
        }
        SCALE | ADD | POSITION => {
            for group in 0..(rows * n).div_ceil(256) {
                for lane in 0..64 {
                    let start = (group * 64 + lane) * 4;
                    for i in start..(start + 4).min(rows * n) {
                        out[i] = match kind as u32 {
                            SCALE => a[i] * weights[wo + i % n],
                            ADD => a[i] + r[i],
                            _ => a[i] + weights[wo + i],
                        };
                    }
                }
            }
        }
        // Activation arithmetic/scheduling is unchanged and has its own
        // primitive and SIMD oracle checks; do not duplicate its polynomial.
        _ => return gpu::reference(q, a, r, weights),
    }
    out
}
