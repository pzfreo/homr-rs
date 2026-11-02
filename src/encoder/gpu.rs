//! Read-only GPU execution plan derived from an already validated Encoder.
//! No ONNX parser or alternate graph optimizer lives in the JavaScript host.
use super::{kernels, Conv, Encoder, Linear, Norm, CONTEXT_VALUES, IMAGE_VALUES};
use serde_json::{json, Value};

pub const ABI_VERSION: u32 = 1;
// Uniform words, shared with encoder_gpu_kernels.mjs. The last word is a
// runtime *zero* integer rounding barrier, not a floating-point shader constant.
pub const WORDS: usize = 16;
pub const MATMUL: u32 = 0;
pub const CONV: u32 = 1;
pub const DEPTHWISE: u32 = 2;
pub const NORM: u32 = 3;
pub const GELU: u32 = 4;
pub const SCALE: u32 = 5;
pub const ADD: u32 = 6;
pub const POSITION: u32 = 7;
#[cfg(test)]
mod certificate;

pub struct Op {
    pub name: String,
    pub params: [u32; WORDS],
    pub input: usize,
    pub residual: usize,
    pub output: usize,
}
impl Op {
    pub fn values(&self) -> usize {
        self.params[1] as usize * self.params[3] as usize
    }
}
pub struct Plan {
    pub weights: Vec<f32>,
    pub ops: Vec<Op>,
    pub buffers: [usize; 7],
    pub json: Vec<u8>,
}
impl Plan {
    pub fn new(encoder: &Encoder) -> Self {
        let mut p = Self {
            weights: vec![],
            ops: vec![],
            buffers: [IMAGE_VALUES, 0, 0, 0, 0, 0, CONTEXT_VALUES],
            json: vec![],
        };
        let (mut h, mut w) = (64, 320);
        p.conv("/stem/stem.0/Conv", &encoder.stem, 256, 1280, 0, 1, false);
        p.norm(
            "/stem/stem.1/LayerNormalization_output_0",
            &encoder.norm,
            h * w,
            1,
            2,
        );
        let mut x = 2;
        for (s, stage) in encoder.stages.iter().enumerate() {
            if let Some((norm, conv)) = &stage.down {
                p.norm(
                    &format!("/stages.{s}/downsample/downsample.0/LayerNormalization_output_0"),
                    norm,
                    h * w,
                    x,
                    3,
                );
                let out = 3 - x;
                p.conv(
                    &format!("/stages.{s}/downsample/downsample.1/Conv_output_0"),
                    conv,
                    h,
                    w,
                    3,
                    out,
                    false,
                );
                h /= 2;
                w /= 2;
                x = out;
            }
            for b in &stage.blocks {
                p.conv(
                    &format!("{}/conv_dw/Conv_output_0", b.name),
                    &b.dw,
                    h,
                    w,
                    x,
                    3,
                    true,
                );
                p.norm(
                    &format!("{}/norm/LayerNormalization_output_0", b.name),
                    &b.norm,
                    h * w,
                    3,
                    4,
                );
                p.linear(
                    &format!("{}/mlp/fc1/Add_output_0", b.name),
                    &b.fc1,
                    h * w,
                    4,
                    5,
                );
                p.simple(
                    &format!("{}/mlp/act/Mul_1_output_0", b.name),
                    GELU,
                    h * w,
                    b.fc1.n,
                    5,
                    0,
                    5,
                    0,
                );
                p.linear(
                    &format!("{}/mlp/fc2/Add_output_0", b.name),
                    &b.fc2,
                    h * w,
                    5,
                    3,
                );
                let gamma = p.append(&b.gamma);
                p.simple(
                    &format!("{}/Mul", b.name),
                    SCALE,
                    h * w,
                    b.fc2.n,
                    3,
                    0,
                    3,
                    gamma,
                );
                let out = 3 - x;
                p.simple(
                    &format!("{}/Add", b.name),
                    ADD,
                    h * w,
                    b.fc2.n,
                    3,
                    x,
                    out,
                    0,
                );
                x = out;
            }
        }
        p.linear("/proj/Add_output_0", &encoder.projection, h * w, x, 6);
        let pos = p.append(&encoder.position);
        p.simple("output", POSITION, 1280, 512, 6, 0, 6, pos);
        p.json=serde_json::to_vec(&json!({"abi":ABI_VERSION,"weights":p.weights.len(),
            "buffers":p.buffers,"input":0,"output":6,"max_channels":384,
            "ops":p.ops.iter().map(|op|json!({"name":op.name,"params":op.params,"input":op.input,"residual":op.residual,"output":op.output})).collect::<Vec<_>>(),
            "self_tests":self_tests()})).expect("GPU plan JSON");
        p
    }
    fn append(&mut self, x: &[f32]) -> u32 {
        let offset = u32::try_from(self.weights.len()).expect("GPU weights overflow");
        self.weights.extend_from_slice(x);
        offset
    }
    fn push(
        &mut self,
        name: &str,
        params: [u32; WORDS],
        input: usize,
        residual: usize,
        output: usize,
    ) {
        self.buffers[output] = self.buffers[output].max(params[1] as usize * params[3] as usize);
        self.ops.push(Op {
            name: name.into(),
            params,
            input,
            residual,
            output,
        });
    }
    fn norm(&mut self, name: &str, norm: &Norm, rows: usize, input: usize, output: usize) {
        let scale = self.append(&norm.scale);
        let bias = self.append(&norm.bias);
        let mut q = [0; WORDS];
        q[0] = NORM;
        q[1] = rows as u32;
        q[3] = norm.scale.len() as u32;
        q[4] = scale;
        q[5] = bias;
        q[6] = norm.epsilon.to_bits();
        self.push(name, q, input, 0, output);
    }
    fn linear(&mut self, name: &str, linear: &Linear, rows: usize, input: usize, output: usize) {
        let weights = self.append(&linear.weights);
        let bias = self.append(&linear.bias);
        let mut q = [0; WORDS];
        q[0] = MATMUL;
        q[1] = rows as u32;
        q[2] = linear.k as u32;
        q[3] = linear.n as u32;
        q[4] = weights;
        q[5] = bias;
        self.push(name, q, input, 0, output);
    }
    #[allow(clippy::too_many_arguments)]
    fn conv(
        &mut self,
        name: &str,
        conv: &Conv,
        h: usize,
        w: usize,
        input: usize,
        output: usize,
        dw: bool,
    ) {
        let k = conv.k * conv.k * if dw { 1 } else { conv.c };
        let mut packed = vec![0.0; conv.weights.len()];
        for j in 0..conv.n {
            for i in 0..k {
                packed[i * conv.n + j] = conv.weights[j * k + i];
            }
        }
        let weights = self.append(&packed);
        let bias = self.append(&conv.bias);
        let mut q = [0; WORDS];
        q[0] = if dw { DEPTHWISE } else { CONV };
        let oh = (h + 2 * conv.pad - conv.k) / conv.stride + 1;
        let ow = (w + 2 * conv.pad - conv.k) / conv.stride + 1;
        q[1] = (oh * ow) as u32;
        q[2] = k as u32;
        q[3] = conv.n as u32;
        q[4] = weights;
        q[5] = bias;
        q[7] = h as u32;
        q[8] = w as u32;
        q[9] = conv.c as u32;
        q[10] = conv.k as u32;
        q[11] = conv.stride as u32;
        q[12] = conv.pad as u32;
        q[13] = ow as u32;
        self.push(name, q, input, 0, output);
    }
    #[allow(clippy::too_many_arguments)]
    fn simple(
        &mut self,
        name: &str,
        kind: u32,
        rows: usize,
        c: usize,
        input: usize,
        residual: usize,
        output: usize,
        weights: u32,
    ) {
        let mut q = [0; WORDS];
        q[0] = kind;
        q[1] = rows as u32;
        q[3] = c as u32;
        q[4] = weights;
        self.push(name, q, input, residual, output);
    }
    pub fn reference(&self, op: &Op, input: &[f32], residual: &[f32]) -> Vec<f32> {
        reference(&op.params, input, residual, &self.weights)
    }
}

/// Independent CPU oracle for GPU probes. Uses the existing ordered CPU kernels.
pub fn reference(q: &[u32; WORDS], a: &[f32], residual: &[f32], weights: &[f32]) -> Vec<f32> {
    let [kind, m, k, n, wo, bo, eps, h, w, c, kernel, stride, pad, _, _, _] = q.map(|v| v as usize);
    let size = m * n;
    match kind as u32 {
        MATMUL => kernels::matmul(
            a,
            &weights[wo..wo + k * n],
            m,
            k,
            n,
            Some(&weights[bo..bo + n]),
        ),
        CONV | DEPTHWISE => {
            let mut unpacked = vec![0.0; k * n];
            for j in 0..n {
                for i in 0..k {
                    unpacked[j * k + i] = weights[wo + i * n + j];
                }
            }
            if kind as u32 == DEPTHWISE {
                kernels::depthwise(a, h, w, c, &unpacked, &weights[bo..bo + n], kernel, pad)
            } else {
                kernels::conv(
                    a,
                    h,
                    w,
                    c,
                    &unpacked,
                    &weights[bo..bo + n],
                    n,
                    kernel,
                    stride,
                    pad,
                )
            }
        }
        NORM => kernels::layer_norm(
            a,
            n,
            &weights[wo..wo + n],
            &weights[bo..bo + n],
            f32::from_bits(eps as u32),
        ),
        GELU => {
            let mut v = a.to_vec();
            kernels::gelu_in_place(&mut v);
            v
        }
        SCALE => (0..size).map(|i| a[i] * weights[wo + i % n]).collect(),
        ADD => (0..size).map(|i| a[i] + residual[i]).collect(),
        POSITION => (0..size).map(|i| a[i] + weights[wo + i]).collect(),
        _ => panic!("unknown GPU operation"),
    }
}

/// Encoder's inner Erf exponential (not the decoder softmax exponential).
#[allow(clippy::excessive_precision)]
pub fn erf_exp(x: f32) -> f32 {
    let x = x.max(-88.3762626647949);
    let biased = kernels::fma(x, std::f32::consts::LOG2_E, 12582912.0);
    let m = biased - 12582912.0;
    let r = kernels::fma(m, -6.93145752e-1, x);
    let r = kernels::fma(m, -1.42860677e-6, r);
    let scale = f32::from_bits(biased.to_bits().wrapping_shl(23).wrapping_add(0x3f800000));
    let mut p = 1.38319808e-3;
    for c in [
        8.37550033e-3,
        4.16689515e-2,
        1.66664466e-1,
        4.99999851e-1,
        1.0,
        1.0,
    ] {
        p = kernels::fma(p, r, c);
    }
    p * scale
}

/// Operand of the inner exponential, including SIMD's unselected large branch.
#[allow(clippy::excessive_precision)]
pub fn erf_exp_input(v:f32)->f32 {
    let a=v.abs().min(3.925);
    let mut p=1.72948930e-5;
    for b in [-3.83208680e-4,3.88393435e-3,-2.42545605e-2,1.06777847e-1,6.34846687e-1,1.28717512e-1] {p=kernels::fma(p,a,b);}
    -kernels::fma(p,a,a)
}

fn bits(x: &[f32]) -> Vec<u32> {
    x.iter().map(|x| x.to_bits()).collect()
}
fn self_tests() -> Vec<Value> {
    let mut tests = Vec::new();
    for (test_index, kind) in [
        MATMUL, CONV, DEPTHWISE, NORM, GELU, SCALE, ADD, POSITION, CONV,
    ]
    .into_iter()
    .enumerate()
    {
        let mut q = [0; WORDS];
        q[0] = kind;
        q[1] = 4;
        q[2] = 32;
        q[3] = 16;
        if kind == CONV {
            q[2] = 48 * 4;
            q[7] = 4;
            q[8] = 4;
            q[9] = 48;
            q[10] = 2;
            q[11] = 2;
            q[13] = 2;
        }
        if test_index == 8 {
            q[2] = 16;
            q[7] = 8;
            q[8] = 8;
            q[9] = 1;
            q[10] = 4;
            q[11] = 4;
            q[13] = 2;
        }
        if kind == DEPTHWISE {
            q[2] = 49;
            q[7] = 2;
            q[8] = 2;
            q[9] = 16;
            q[10] = 7;
            q[11] = 1;
            q[12] = 3;
            q[13] = 2;
        }
        q[6] = 1e-6f32.to_bits();
        let input_len = match kind {
            MATMUL => 4 * 32,
            CONV => {
                if test_index == 8 {
                    64
                } else {
                    4 * 4 * 48
                }
            }
            DEPTHWISE => 2 * 2 * 16,
            _ => 64,
        };
        let mut a: Vec<_> = (0..input_len)
            .map(|i| ((i * 17 % 71) as f32 - 35.0) * 0.03125)
            .collect();
        let mut weights: Vec<_> = (0..(q[2] * q[3]) as usize)
            .map(|i| ((i * 13 % 59) as f32 - 29.0) * 0.015625)
            .collect();
        if kind == NORM {
            weights = vec![0.75; 16];
        }
        if kind == SCALE {
            weights = vec![f32::from_bits(0x3f7ffffe); 16];
            a[0] = f32::from_bits(0x3f800001);
        }
        if kind == POSITION {
            weights = vec![-1.0; 64];
        }
        q[5] = weights.len() as u32;
        weights.extend((0..16).map(|i| i as f32 * 0.125));
        let residual: Vec<_> = (0..64).map(|i| i as f32 * 0.0625 - 2.0).collect();
        if test_index == 8 {
            a.fill(0.0);
            weights.fill(0.0);
            a[0] = f32::from_bits(0x3f800001);
            weights[0] = f32::from_bits(0x3f7ffffe);
            weights[q[5] as usize] = 16777218.0;
        }
        if kind == MATMUL {
            a.fill(0.0);
            weights.fill(0.0);
            a[0] = f32::from_bits(0x3f800001);
            weights[0] = f32::from_bits(0x3f7ffffe);
            weights[q[5] as usize] = 16777218.0;
            a[32] = -1.0;
            a[33] = f32::from_bits(0x3f800001);
            weights[1] = 1.0;
            weights[17] = f32::from_bits(0x3f7ffffe);
            a[64] = 16777216.0;
            a[65] = 1.0;
            a[66] = -16777216.0;
            weights[2] = 1.0;
            weights[18] = 1.0;
            weights[34] = 1.0;
        }
        let expected = reference(&q, &a, &residual, &weights);
        tests.push(json!({"params":q,"a":bits(&a),"r":bits(&residual),"weights":bits(&weights),"expected":bits(&expected)}));
    }
    tests
}

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn exponential_oracle_reconstructs_the_actual_cpu_erf_large_branch() {
        for i in 0..100_000 {
            let x=0.921875+(4.25-0.921875)*(i as f32+0.5)/100_000.0;
            assert_eq!((1.0-erf_exp(erf_exp_input(x))).to_bits(),kernels::erf(x).to_bits());
        }
    }
    #[test]
    fn load_time_probes_cover_every_kernel_with_finite_normal_data() {
        let probes = self_tests();
        assert_eq!(probes.len(), 9);
        for (kind, p) in probes.iter().enumerate() {
            assert_eq!(p["params"][0], if kind == 8 { CONV } else { kind as u32 });
            for name in ["a", "r", "weights", "expected"] {
                for b in p[name].as_array().unwrap() {
                    let v = f32::from_bits(b.as_u64().unwrap() as u32);
                    assert!(v.is_finite() && (v == 0.0 || v.is_normal()));
                }
            }
        }
    }
}
