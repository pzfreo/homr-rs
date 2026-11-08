//! homr's segmentation network (`segnet_*.onnx`): a ResNet-18 U-Net which labels every pixel of a
//! 320 x 320 tile as one of six classes (background, staff, symbols, stems and rests, noteheads,
//! clefs and keys). The graph is read from the ONNX file and checked, not built in: this
//! executor runs the operations that graph uses (normalising by channel, convolutions, ReLU,
//! a max-pool, add, concat and 2x nearest upsampling) on float32 tensors with a fixed input
//! of 3 x 320 x 320, and says what it can't run.
//!
//! The results are those of any float32 implementation of the graph: sums in another order
//! than onnxruntime's differ in the last bits, so a pixel where two classes are about equally
//! likely can come out the other way. `tools/dump_segnet.py` and `check_segnet` measure that.

use crate::onnx::{Graph, Tensor};
use crate::preprocess::{tile_input, tile_origins, WIN};
use crate::warp::Gray;
use std::collections::HashMap;

pub const CLASSES: usize = 6;
/// Output columns computed together, and output channels (the convolution's register tile).
const LANES: usize = 16;
const BLOCK: usize = 4;

/// a * b + c, fused where the hardware does it in one instruction.
#[cfg(not(any(
    target_arch = "aarch64",
    all(target_arch = "wasm32", target_feature = "simd128")
)))]
#[inline(always)]
fn fma(a: f32, b: f32, c: f32) -> f32 {
    #[cfg(any(target_arch = "aarch64", target_feature = "fma"))]
    {
        a.mul_add(b, c)
    }
    #[cfg(not(any(target_arch = "aarch64", target_feature = "fma")))]
    {
        a * b + c
    }
}

/// A convolution with its weights laid out for the kernel: [output block of 4][input channel]
/// [ky][kx][4 output channels], the last block padded with zeros.
struct Conv {
    ic: usize,
    oc: usize,
    k: usize,
    stride: usize,
    pad: usize,
    ih: usize,
    iw: usize,
    oh: usize,
    ow: usize,
    weights: Vec<f32>,
    bias: Vec<f32>,
    relu: bool,
}

impl Conv {
    /// `weights` is [oc][ic][k][k] as in ONNX.
    #[allow(clippy::too_many_arguments)]
    fn new(
        weights: &[f32],
        bias: &[f32],
        ic: usize,
        oc: usize,
        k: usize,
        stride: usize,
        pad: usize,
        ih: usize,
        iw: usize,
    ) -> Conv {
        let blocks = oc.div_ceil(BLOCK);
        let mut w = vec![0f32; blocks * ic * k * k * BLOCK];
        for o in 0..oc {
            for i in 0..ic {
                for t in 0..k * k {
                    w[(((o / BLOCK) * ic + i) * k * k + t) * BLOCK + o % BLOCK] =
                        weights[(o * ic + i) * k * k + t];
                }
            }
        }
        let mut b = bias.to_vec();
        b.resize(blocks * BLOCK, 0.0);
        Conv {
            ic,
            oc,
            k,
            stride,
            pad,
            ih,
            iw,
            oh: (ih + 2 * pad - k) / stride + 1,
            ow: (iw + 2 * pad - k) / stride + 1,
            weights: w,
            bias: b,
            relu: false,
        }
    }

    fn run(&self, input: &[f32], out: &mut [f32]) {
        let (s, k, p) = (self.stride, self.k, self.pad);
        let ow_round = self.ow.div_ceil(LANES) * LANES;
        // The zero-padded input, split by column phase so that a strided convolution reads
        // contiguous columns too: plane q holds the padded columns q, q + s, q + 2s...
        let wq = ow_round + (k - 1) / s + 1;
        let hp = self.ih + 2 * p;
        let plane_len = hp * wq;
        let mut planes = vec![0f32; s * self.ic * plane_len];
        for c in 0..self.ic {
            for y in 0..self.ih {
                let src = &input[(c * self.ih + y) * self.iw..][..self.iw];
                for (x, &v) in src.iter().enumerate() {
                    let u = x + p;
                    planes[((u % s) * self.ic + c) * plane_len + (y + p) * wq + u / s] = v;
                }
            }
        }
        for ob in 0..self.oc.div_ceil(BLOCK) {
            let wblock = &self.weights[ob * self.ic * k * k * BLOCK..][..self.ic * k * k * BLOCK];
            for y in 0..self.oh {
                for x0 in (0..ow_round).step_by(LANES) {
                    let acc = accumulate(self, &planes, wblock, (y, x0, wq, plane_len));
                    for (o, a) in acc.iter().enumerate() {
                        let oc = ob * BLOCK + o;
                        if oc >= self.oc {
                            break;
                        }
                        for (j, &v) in a.iter().enumerate() {
                            let x = x0 + j;
                            if x < self.ow {
                                let v = v + self.bias[oc];
                                out[(oc * self.oh + y) * self.ow + x] =
                                    if self.relu { v.max(0.0) } else { v };
                            }
                        }
                    }
                }
            }
        }
    }
}

/// The sums of one tile of the output (4 channels by 16 columns) over all input channels and
/// kernel positions. `(y, x0, wq, plane_len)`: the output row and first column, and the size of
/// the padded input planes. Every version adds the same products in the same order.
#[cfg(target_arch = "aarch64")]
fn accumulate(
    c: &Conv,
    planes: &[f32],
    wblock: &[f32],
    at: (usize, usize, usize, usize),
) -> [[f32; LANES]; BLOCK] {
    use std::arch::aarch64::*;
    let (y, x0, wq, plane_len) = at;
    let (s, k) = (c.stride, c.k);
    let mut out = [[0f32; LANES]; BLOCK];
    // SAFETY: every read is inside `planes` (the padded planes are wide enough for 16 columns
    // from x0 at each kernel position, see `Conv::run`) and `wblock` (4 weights a position).
    unsafe {
        let zero = vdupq_n_f32(0.0);
        let mut acc = [[zero; 4]; BLOCK];
        for ch in 0..c.ic {
            for ky in 0..k {
                let row = (y * s + ky) * wq;
                for kx in 0..k {
                    let base = ((kx % s) * c.ic + ch) * plane_len + row + kx / s + x0;
                    debug_assert!(base + LANES <= planes.len());
                    let p = planes.as_ptr().add(base);
                    let src = [
                        vld1q_f32(p),
                        vld1q_f32(p.add(4)),
                        vld1q_f32(p.add(8)),
                        vld1q_f32(p.add(12)),
                    ];
                    let w = vld1q_f32(wblock.as_ptr().add(((ch * k + ky) * k + kx) * BLOCK));
                    macro_rules! row {
                        ($o:literal) => {
                            for v in 0..4 {
                                acc[$o][v] = vfmaq_laneq_f32::<$o>(acc[$o][v], src[v], w);
                            }
                        };
                    }
                    row!(0);
                    row!(1);
                    row!(2);
                    row!(3);
                }
            }
        }
        for (row, regs) in out.iter_mut().zip(&acc) {
            for (v, reg) in regs.iter().enumerate() {
                vst1q_f32(row.as_mut_ptr().add(4 * v), *reg);
            }
        }
    }
    out
}

#[cfg(all(target_arch = "wasm32", target_feature = "simd128"))]
fn accumulate(
    c: &Conv,
    planes: &[f32],
    wblock: &[f32],
    at: (usize, usize, usize, usize),
) -> [[f32; LANES]; BLOCK] {
    use std::arch::wasm32::*;
    let (y, x0, wq, plane_len) = at;
    let (s, k) = (c.stride, c.k);
    let mut out = [[0f32; LANES]; BLOCK];
    // SAFETY: as for the aarch64 version.
    unsafe {
        let zero = f32x4_splat(0.0);
        let mut acc = [[zero; 4]; BLOCK];
        for ch in 0..c.ic {
            for ky in 0..k {
                let row = (y * s + ky) * wq;
                for kx in 0..k {
                    let base = ((kx % s) * c.ic + ch) * plane_len + row + kx / s + x0;
                    debug_assert!(base + LANES <= planes.len());
                    let p = planes.as_ptr().add(base) as *const v128;
                    let src = [
                        v128_load(p),
                        v128_load(p.add(1)),
                        v128_load(p.add(2)),
                        v128_load(p.add(3)),
                    ];
                    let w = wblock.as_ptr().add(((ch * k + ky) * k + kx) * BLOCK);
                    for o in 0..BLOCK {
                        let wo = f32x4_splat(*w.add(o));
                        for v in 0..4 {
                            acc[o][v] = f32x4_add(acc[o][v], f32x4_mul(src[v], wo));
                        }
                    }
                }
            }
        }
        for o in 0..BLOCK {
            for v in 0..4 {
                v128_store(out[o].as_mut_ptr().add(4 * v) as *mut v128, acc[o][v]);
            }
        }
    }
    out
}

#[cfg(not(any(
    target_arch = "aarch64",
    all(target_arch = "wasm32", target_feature = "simd128")
)))]
fn accumulate(
    c: &Conv,
    planes: &[f32],
    wblock: &[f32],
    at: (usize, usize, usize, usize),
) -> [[f32; LANES]; BLOCK] {
    let (y, x0, wq, plane_len) = at;
    let (s, k) = (c.stride, c.k);
    let mut acc = [[0f32; LANES]; BLOCK];
    for ch in 0..c.ic {
        for ky in 0..k {
            let row = (y * s + ky) * wq;
            for kx in 0..k {
                let base = ((kx % s) * c.ic + ch) * plane_len + row + kx / s + x0;
                let src = &planes[base..base + LANES];
                let w = &wblock[((ch * k + ky) * k + kx) * BLOCK..][..BLOCK];
                for o in 0..BLOCK {
                    for j in 0..LANES {
                        acc[o][j] = fma(w[o], src[j], acc[o][j]);
                    }
                }
            }
        }
    }
    acc
}

enum Op {
    /// (x - mean) / std for each channel, as the graph's Sub and Div
    Sub {
        input: usize,
        output: usize,
        per_channel: [f32; 3],
    },
    Div {
        input: usize,
        output: usize,
        per_channel: [f32; 3],
    },
    Conv {
        input: usize,
        output: usize,
        conv: Box<Conv>,
    },
    Relu {
        input: usize,
        output: usize,
    },
    MaxPool {
        input: usize,
        output: usize,
        k: usize,
        stride: usize,
        pad: usize,
    },
    Add {
        a: usize,
        b: usize,
        output: usize,
    },
    Concat {
        inputs: Vec<usize>,
        output: usize,
    },
    /// Resize, nearest, asymmetric, floor
    Upsample {
        input: usize,
        output: usize,
    },
}

/// A tensor of the graph: channels, height, width (the batch is one).
#[derive(Clone, Copy, PartialEq, Debug)]
struct Shape {
    c: usize,
    h: usize,
    w: usize,
}

impl Shape {
    fn len(self) -> usize {
        self.c * self.h * self.w
    }
}

pub struct Segnet {
    ops: Vec<Op>,
    shapes: Vec<Shape>,
    input: usize,
    output: usize,
    /// the index of the last op which reads each tensor, to free it after
    last_use: Vec<usize>,
}

fn int64s(t: &Tensor) -> Result<Vec<i64>, String> {
    if t.kind != 7 {
        return Err("expected int64 initializer".into());
    }
    Ok(t.data
        .as_chunks::<8>()
        .0
        .iter()
        .map(|b| i64::from_le_bytes(*b))
        .collect())
}

fn per_channel(t: &Tensor) -> Result<[f32; 3], String> {
    let v = t.floats()?;
    v.try_into()
        .map_err(|_| "expected one value per channel".to_string())
}

/// A new tensor of the graph, known by its name.
fn add(shapes: &mut Vec<Shape>, ids: &mut HashMap<String, usize>, name: &str, s: Shape) -> usize {
    shapes.push(s);
    ids.insert(name.to_string(), shapes.len() - 1);
    shapes.len() - 1
}

impl Segnet {
    pub fn from_onnx(bytes: &[u8]) -> Result<Segnet, String> {
        let g = Graph::read(bytes)?;
        // how often each tensor is read, to fuse a ReLU into the convolution it follows
        let mut uses: HashMap<&str, usize> = HashMap::new();
        for n in &g.nodes {
            for i in &n.inputs {
                *uses.entry(i.as_str()).or_default() += 1;
            }
        }
        let mut ids: HashMap<String, usize> = HashMap::new();
        let mut shapes: Vec<Shape> = Vec::new();
        let mut consts: HashMap<String, Vec<i64>> = HashMap::new();
        let mut ops: Vec<Op> = Vec::new();
        let input = add(
            &mut shapes,
            &mut ids,
            "input",
            Shape {
                c: 3,
                h: WIN,
                w: WIN,
            },
        );
        let get = |name: &str, ids: &HashMap<String, usize>| -> Result<usize, String> {
            ids.get(name)
                .copied()
                .ok_or_else(|| format!("tensor {name} isn't computed before it is read"))
        };
        for n in &g.nodes {
            let out = n
                .outputs
                .first()
                .ok_or("a node without an output")?
                .as_str();
            match n.op.as_str() {
                "Sub" | "Div" => {
                    let x = get(&n.inputs[0], &ids)?;
                    let v = per_channel(g.tensor(&n.inputs[1])?)?;
                    let o = {
                        let same = shapes[x];
                        add(&mut shapes, &mut ids, out, same)
                    };
                    ops.push(if n.op == "Sub" {
                        Op::Sub {
                            input: x,
                            output: o,
                            per_channel: v,
                        }
                    } else {
                        Op::Div {
                            input: x,
                            output: o,
                            per_channel: v,
                        }
                    });
                }
                "Conv" => {
                    let x = get(&n.inputs[0], &ids)?;
                    let (w, b) = (g.tensor(&n.inputs[1])?, g.tensor(&n.inputs[2])?);
                    let ints = |k: &str| n.ints.get(k).cloned().unwrap_or_default();
                    let (strides, pads, dilations) =
                        (ints("strides"), ints("pads"), ints("dilations"));
                    if w.dims.len() != 4 || w.dims[2] != w.dims[3] {
                        return Err("a convolution which isn't square".into());
                    }
                    if n.ints.get("group").is_some_and(|g| g != &[1])
                        || dilations.iter().any(|&d| d != 1)
                        || strides.len() != 2
                        || strides[0] != strides[1]
                        || pads.len() != 4
                        || pads.iter().any(|&p| p != pads[0])
                        || n.strings.get("auto_pad").is_some_and(|a| a != "NOTSET")
                    {
                        return Err(
                            "a convolution with groups, dilations, uneven strides or padding"
                                .into(),
                        );
                    }
                    let (oc, ic, k) = (w.dims[0], w.dims[1], w.dims[2]);
                    let s = shapes[x];
                    if ic != s.c {
                        return Err(format!("a convolution of {ic} channels reads {}", s.c));
                    }
                    let conv = Conv::new(
                        &w.floats()?,
                        &b.floats()?,
                        ic,
                        oc,
                        k,
                        strides[0] as usize,
                        pads[0] as usize,
                        s.h,
                        s.w,
                    );
                    let o = add(
                        &mut shapes,
                        &mut ids,
                        out,
                        Shape {
                            c: oc,
                            h: conv.oh,
                            w: conv.ow,
                        },
                    );
                    ops.push(Op::Conv {
                        input: x,
                        output: o,
                        conv: Box::new(conv),
                    });
                }
                "Relu" => {
                    let x = get(&n.inputs[0], &ids)?;
                    // a convolution only this ReLU reads has it applied as it stores its result
                    if let Some(Op::Conv { output, conv, .. }) = ops.last_mut() {
                        if *output == x && uses.get(n.inputs[0].as_str()) == Some(&1) {
                            conv.relu = true;
                            ids.insert(out.to_string(), x);
                            continue;
                        }
                    }
                    let o = {
                        let same = shapes[x];
                        add(&mut shapes, &mut ids, out, same)
                    };
                    ops.push(Op::Relu {
                        input: x,
                        output: o,
                    });
                }
                "MaxPool" => {
                    let x = get(&n.inputs[0], &ids)?;
                    let ints = |k: &str| n.ints.get(k).cloned().unwrap_or_default();
                    let (kernel, strides, pads) =
                        (ints("kernel_shape"), ints("strides"), ints("pads"));
                    if kernel.len() != 2
                        || kernel[0] != kernel[1]
                        || strides.len() != 2
                        || strides[0] != strides[1]
                        || pads.len() != 4
                        || pads.iter().any(|&p| p != pads[0])
                        || n.ints.get("ceil_mode").is_some_and(|c| c != &[0])
                        || ints("dilations").iter().any(|&d| d != 1)
                    {
                        return Err("a max-pool other than square, uniform, floor mode".into());
                    }
                    let (k, st, p, s) = (
                        kernel[0] as usize,
                        strides[0] as usize,
                        pads[0] as usize,
                        shapes[x],
                    );
                    let o = add(
                        &mut shapes,
                        &mut ids,
                        out,
                        Shape {
                            c: s.c,
                            h: (s.h + 2 * p - k) / st + 1,
                            w: (s.w + 2 * p - k) / st + 1,
                        },
                    );
                    ops.push(Op::MaxPool {
                        input: x,
                        output: o,
                        k,
                        stride: st,
                        pad: p,
                    });
                }
                "Add" => {
                    let (a, b) = (get(&n.inputs[0], &ids)?, get(&n.inputs[1], &ids)?);
                    if shapes[a] != shapes[b] {
                        return Err("an addition of different shapes".into());
                    }
                    let o = {
                        let same = shapes[a];
                        add(&mut shapes, &mut ids, out, same)
                    };
                    ops.push(Op::Add { a, b, output: o });
                }
                "Shape" => {
                    // only feeds a Resize: the dimensions it names are known here
                    let x = get(&n.inputs[0], &ids)?;
                    let s = shapes[x];
                    let dims = [1, s.c as i64, s.h as i64, s.w as i64];
                    let (start, end) = (
                        n.ints.get("start").map_or(0, |v| v[0]),
                        n.ints.get("end").map_or(4, |v| v[0]),
                    );
                    consts.insert(out.to_string(), dims[start as usize..end as usize].to_vec());
                }
                "Concat" => {
                    let axis = n.ints.get("axis").map(|a| a[0]);
                    if axis == Some(0) {
                        let mut all = Vec::new();
                        for i in &n.inputs {
                            all.extend(match consts.get(i) {
                                Some(c) => c.clone(),
                                None => int64s(g.tensor(i)?)?,
                            });
                        }
                        consts.insert(out.to_string(), all);
                    } else if axis == Some(1) {
                        let inputs: Vec<usize> = n
                            .inputs
                            .iter()
                            .map(|i| get(i, &ids))
                            .collect::<Result<_, _>>()?;
                        let s0 = shapes[inputs[0]];
                        if inputs
                            .iter()
                            .any(|&i| shapes[i].h != s0.h || shapes[i].w != s0.w)
                        {
                            return Err("a concatenation of different sizes".into());
                        }
                        let c = inputs.iter().map(|&i| shapes[i].c).sum();
                        let o = add(
                            &mut shapes,
                            &mut ids,
                            out,
                            Shape {
                                c,
                                h: s0.h,
                                w: s0.w,
                            },
                        );
                        ops.push(Op::Concat { inputs, output: o });
                    } else {
                        return Err("a concatenation along another axis".into());
                    }
                }
                "Resize" => {
                    let x = get(&n.inputs[0], &ids)?;
                    let mode = |k: &str| n.strings.get(k).map(String::as_str);
                    if mode("mode") != Some("nearest")
                        || mode("coordinate_transformation_mode") != Some("asymmetric")
                        || mode("nearest_mode") != Some("floor")
                    {
                        return Err("a resize other than nearest, asymmetric, floor".into());
                    }
                    let sizes = consts
                        .get(n.inputs.get(3).map_or("", |s| s.as_str()))
                        .ok_or("a resize without sizes")?;
                    let s = shapes[x];
                    if sizes.len() != 4 || sizes[2] != 2 * s.h as i64 || sizes[3] != 2 * s.w as i64
                    {
                        return Err("a resize which isn't a doubling".into());
                    }
                    let o = add(
                        &mut shapes,
                        &mut ids,
                        out,
                        Shape {
                            c: s.c,
                            h: 2 * s.h,
                            w: 2 * s.w,
                        },
                    );
                    ops.push(Op::Upsample {
                        input: x,
                        output: o,
                    });
                }
                other => return Err(format!("segmentation graph: unsupported operation {other}")),
            }
        }
        let output = get("output", &ids)?;
        if shapes[output]
            != (Shape {
                c: CLASSES,
                h: WIN,
                w: WIN,
            })
        {
            return Err(format!(
                "segmentation output is {:?}, not {CLASSES} x {WIN} x {WIN}",
                shapes[output]
            ));
        }
        let mut last_use = vec![0; shapes.len()];
        for (i, op) in ops.iter().enumerate() {
            let reads: Vec<usize> = match op {
                Op::Sub { input, .. }
                | Op::Div { input, .. }
                | Op::Conv { input, .. }
                | Op::Relu { input, .. }
                | Op::MaxPool { input, .. }
                | Op::Upsample { input, .. } => vec![*input],
                Op::Add { a, b, .. } => vec![*a, *b],
                Op::Concat { inputs, .. } => inputs.clone(),
            };
            for r in reads {
                last_use[r] = i;
            }
        }
        Ok(Segnet {
            ops,
            shapes,
            input,
            output,
            last_use,
        })
    }

    /// The network's output for one tile (`tile_input`: 3 x 320 x 320 float32): 6 x 320 x 320 logits.
    pub fn logits(&self, input: &[f32]) -> Result<Vec<f32>, String> {
        if input.len() != self.shapes[self.input].len() {
            return Err(format!(
                "segmentation input of {} values, not {}",
                input.len(),
                self.shapes[self.input].len()
            ));
        }
        let mut t: Vec<Vec<f32>> = vec![Vec::new(); self.shapes.len()];
        t[self.input] = input.to_vec();
        for (i, op) in self.ops.iter().enumerate() {
            let (output, result) = match op {
                Op::Sub {
                    input,
                    output,
                    per_channel,
                }
                | Op::Div {
                    input,
                    output,
                    per_channel,
                } => {
                    let plane = self.shapes[*input].h * self.shapes[*input].w;
                    let sub = matches!(op, Op::Sub { .. });
                    let mut r = t[*input].clone();
                    for (c, v) in r.chunks_mut(plane).enumerate() {
                        for x in v {
                            *x = if sub {
                                *x - per_channel[c]
                            } else {
                                *x / per_channel[c]
                            };
                        }
                    }
                    (*output, r)
                }
                Op::Conv {
                    input,
                    output,
                    conv,
                } => {
                    let mut r = vec![0f32; self.shapes[*output].len()];
                    conv.run(&t[*input], &mut r);
                    (*output, r)
                }
                Op::Relu { input, output } => {
                    (*output, t[*input].iter().map(|v| v.max(0.0)).collect())
                }
                Op::MaxPool {
                    input,
                    output,
                    k,
                    stride,
                    pad,
                } => {
                    let (s, o) = (self.shapes[*input], self.shapes[*output]);
                    let mut r = vec![f32::NEG_INFINITY; o.len()];
                    for c in 0..s.c {
                        for y in 0..o.h {
                            for x in 0..o.w {
                                let mut m = f32::NEG_INFINITY;
                                for ky in 0..*k {
                                    for kx in 0..*k {
                                        let (yy, xx) = (
                                            (y * stride + ky) as isize - *pad as isize,
                                            (x * stride + kx) as isize - *pad as isize,
                                        );
                                        if yy >= 0
                                            && xx >= 0
                                            && (yy as usize) < s.h
                                            && (xx as usize) < s.w
                                        {
                                            m = m.max(
                                                t[*input]
                                                    [(c * s.h + yy as usize) * s.w + xx as usize],
                                            );
                                        }
                                    }
                                }
                                r[(c * o.h + y) * o.w + x] = m;
                            }
                        }
                    }
                    (*output, r)
                }
                Op::Add { a, b, output } => (
                    *output,
                    t[*a].iter().zip(&t[*b]).map(|(x, y)| x + y).collect(),
                ),
                Op::Concat { inputs, output } => {
                    let mut r = Vec::with_capacity(self.shapes[*output].len());
                    for i in inputs {
                        r.extend_from_slice(&t[*i]);
                    }
                    (*output, r)
                }
                Op::Upsample { input, output } => {
                    let (s, o) = (self.shapes[*input], self.shapes[*output]);
                    let mut r = vec![0f32; o.len()];
                    for c in 0..s.c {
                        for y in 0..o.h {
                            for x in 0..o.w {
                                r[(c * o.h + y) * o.w + x] =
                                    t[*input][(c * s.h + y / 2) * s.w + x / 2];
                            }
                        }
                    }
                    (*output, r)
                }
            };
            t[output] = result;
            for (id, &last) in self.last_use.iter().enumerate() {
                if last == i && id != self.output && id != output {
                    t[id] = Vec::new();
                }
            }
        }
        Ok(std::mem::take(&mut t[self.output]))
    }

    /// The class of every pixel of a tile: the first of the largest logits, as argmax does.
    pub fn classes(&self, input: &[f32]) -> Result<Vec<u8>, String> {
        let logits = self.logits(input)?;
        let plane = WIN * WIN;
        Ok((0..plane)
            .map(|i| {
                let mut best = 0;
                for c in 1..CLASSES {
                    if logits[c * plane + i] > logits[best * plane + i] {
                        best = c;
                    }
                }
                best as u8
            })
            .collect())
    }

    /// The class map of every tile of a page (the contrast-adjusted page, 1920 wide), in
    /// the order `merge_tiles` takes them.
    pub fn segment(&self, page: &Gray) -> Result<Vec<Vec<u8>>, String> {
        tile_origins(page.h, page.w)
            .into_iter()
            .map(|(y, x)| self.classes(&tile_input(page, y, x)))
            .collect()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[allow(clippy::too_many_arguments)]
    fn naive(
        input: &[f32],
        w: &[f32],
        b: &[f32],
        ic: usize,
        oc: usize,
        k: usize,
        s: usize,
        p: usize,
        ih: usize,
        iw: usize,
    ) -> Vec<f32> {
        let oh = (ih + 2 * p - k) / s + 1;
        let ow = (iw + 2 * p - k) / s + 1;
        let mut out = vec![0f32; oc * oh * ow];
        for o in 0..oc {
            for y in 0..oh {
                for x in 0..ow {
                    let mut sum = 0f64;
                    for i in 0..ic {
                        for ky in 0..k {
                            for kx in 0..k {
                                let (yy, xx) = (
                                    (y * s + ky) as isize - p as isize,
                                    (x * s + kx) as isize - p as isize,
                                );
                                if yy >= 0 && xx >= 0 && (yy as usize) < ih && (xx as usize) < iw {
                                    sum += input[(i * ih + yy as usize) * iw + xx as usize] as f64
                                        * w[((o * ic + i) * k + ky) * k + kx] as f64;
                                }
                            }
                        }
                    }
                    out[(o * oh + y) * ow + x] = sum as f32 + b[o];
                }
            }
        }
        out
    }

    #[test]
    fn the_convolution_matches_a_plain_one() {
        // sizes and strides the network uses, and some it doesn't: widths which aren't a multiple of 16
        for &(ic, oc, k, s, p, ih, iw) in &[
            (3, 8, 7, 2, 3, 20, 33),
            (5, 6, 3, 1, 1, 9, 21),
            (4, 7, 3, 2, 1, 16, 16),
            (6, 5, 1, 2, 0, 11, 10),
            (2, 4, 3, 1, 1, 5, 40),
        ] {
            let mut seed = 12345u32;
            let mut rnd = || {
                seed = seed.wrapping_mul(1664525).wrapping_add(1013904223);
                (seed >> 8) as f32 / (1 << 24) as f32 - 0.5
            };
            let input: Vec<f32> = (0..ic * ih * iw).map(|_| rnd()).collect();
            let w: Vec<f32> = (0..oc * ic * k * k).map(|_| rnd()).collect();
            let b: Vec<f32> = (0..oc).map(|_| rnd()).collect();
            let conv = Conv::new(&w, &b, ic, oc, k, s, p, ih, iw);
            let mut out = vec![0f32; oc * conv.oh * conv.ow];
            conv.run(&input, &mut out);
            let want = naive(&input, &w, &b, ic, oc, k, s, p, ih, iw);
            let worst = out
                .iter()
                .zip(&want)
                .map(|(a, b)| (a - b).abs())
                .fold(0f32, f32::max);
            assert!(
                worst < 1e-4,
                "ic {ic} oc {oc} k {k} stride {s}: worst difference {worst}"
            );
        }
    }

    /// Prints how many instructions the convolution takes: cargo test --release speed -- --ignored --nocapture
    #[test]
    #[ignore]
    fn speed() {
        for &(ic, oc, k, s, p, h, w) in &[
            (64, 64, 3, 1, 1, 80, 80),
            (256, 256, 3, 1, 1, 20, 20),
            (32, 16, 3, 1, 1, 320, 320),
            (3, 64, 7, 2, 3, 320, 320),
        ] {
            let input = vec![0.5f32; ic * h * w];
            let weights = vec![0.01f32; oc * ic * k * k];
            let conv = Conv::new(&weights, &vec![0.0; oc], ic, oc, k, s, p, h, w);
            let mut out = vec![0f32; oc * conv.oh * conv.ow];
            // retired instructions (macOS), which another process can't change
            extern "C" {
                fn proc_pid_rusage(pid: i32, flavor: i32, buffer: *mut u64) -> i32;
                fn getpid() -> i32;
            }
            let instructions = || {
                let mut buf = [0u64; 96];
                assert_eq!(unsafe { proc_pid_rusage(getpid(), 4, buf.as_mut_ptr()) }, 0);
                buf[2 + 29] as f64
            };
            conv.run(&input, &mut out); // warm
            let start = instructions();
            conv.run(&input, &mut out);
            let n = instructions() - start;
            let macs = (conv.oh * conv.ow * oc * ic * k * k) as f64;
            println!(
                "{ic}->{oc} {k}x{k} stride {s} at {h}x{w}: {:.1} M instructions, {:.3} instructions a multiply-add",
                n / 1e6,
                n / macs
            );
        }
    }
}
