//! Fixed-graph execution of homr's float32 ConvNeXt encoder.
pub mod kernels;
#[cfg(any(target_arch = "wasm32", target_arch = "aarch64"))]
mod simd;
#[cfg(target_arch = "wasm32")]
mod wasm;

use crate::onnx::Graph;
pub const IMAGE_VALUES: usize = 256 * 1280;
pub const CONTEXT_VALUES: usize = 1280 * 512;

struct Norm {
    scale: Vec<f32>,
    bias: Vec<f32>,
    epsilon: f32,
}
impl Norm {
    fn load(g: &Graph, name: &str, c: usize) -> Result<Self, String> {
        let n = g.node(name)?;
        let scale = g.tensor(&n.inputs[1])?.floats()?;
        let bias = g.tensor(&n.inputs[2])?.floats()?;
        if scale.len() != c || bias.len() != c {
            return Err("encoder norm shape".into());
        }
        Ok(Self {
            scale,
            bias,
            epsilon: *n.floats.get("epsilon").ok_or("encoder norm epsilon")?,
        })
    }
    fn run(&self, x: &[f32]) -> Vec<f32> {
        kernels::layer_norm(x, self.scale.len(), &self.scale, &self.bias, self.epsilon)
    }
}
struct Linear {
    weights: Vec<f32>,
    bias: Vec<f32>,
    k: usize,
    n: usize,
}
impl Linear {
    fn load(g: &Graph, prefix: &str, k: usize, n: usize) -> Result<Self, String> {
        let node = g.node(&format!("{prefix}/MatMul"))?;
        let t = g.tensor(&node.inputs[1])?;
        if t.dims != [k, n] {
            return Err("encoder matrix shape".into());
        }
        let add = g.node(&format!("{prefix}/Add"))?;
        let name = add
            .inputs
            .iter()
            .find(|s| g.tensors.contains_key(*s))
            .ok_or("matrix bias missing")?;
        let bias = g.tensor(name)?.floats()?;
        if bias.len() != n {
            return Err("encoder matrix bias".into());
        }
        Ok(Self {
            weights: t.floats()?,
            bias,
            k,
            n,
        })
    }
    fn run(&self, x: &[f32]) -> Vec<f32> {
        kernels::matmul(
            x,
            &self.weights,
            x.len() / self.k,
            self.k,
            self.n,
            Some(&self.bias),
        )
    }
}
struct Conv {
    weights: Vec<f32>,
    bias: Vec<f32>,
    c: usize,
    n: usize,
    k: usize,
    stride: usize,
    pad: usize,
}
impl Conv {
    #[allow(clippy::too_many_arguments)]
    fn load(
        g: &Graph,
        name: &str,
        c: usize,
        n: usize,
        k: usize,
        stride: usize,
        pad: usize,
        depthwise: bool,
    ) -> Result<Self, String> {
        let node = g.node(name)?;
        let t = g.tensor(&node.inputs[1])?;
        if t.dims != [n, if depthwise { 1 } else { c }, k, k]
            || node.ints.get("strides") != Some(&vec![stride as i64; 2])
            || node.ints.get("pads") != Some(&vec![pad as i64; 4])
        {
            return Err("encoder convolution shape/attributes".into());
        }
        let bias = g.tensor(&node.inputs[2])?.floats()?;
        if bias.len() != n {
            return Err("encoder conv bias".into());
        }
        Ok(Self {
            weights: t.floats()?,
            bias,
            c,
            n,
            k,
            stride,
            pad,
        })
    }
    fn run(&self, x: &[f32], h: usize, w: usize) -> Vec<f32> {
        kernels::conv(
            x,
            h,
            w,
            self.c,
            &self.weights,
            &self.bias,
            self.n,
            self.k,
            self.stride,
            self.pad,
        )
    }
    fn depthwise(&self, x: &[f32], h: usize, w: usize) -> Vec<f32> {
        kernels::depthwise(x, h, w, self.c, &self.weights, &self.bias, self.k, self.pad)
    }
}
struct Block {
    name: String,
    dw: Conv,
    norm: Norm,
    fc1: Linear,
    fc2: Linear,
    gamma: Vec<f32>,
}
struct Stage {
    down: Option<(Norm, Conv)>,
    blocks: Vec<Block>,
}
/// Loads upstream float32 ONNX weights at runtime; no weights are embedded.
pub struct Encoder {
    stem: Conv,
    norm: Norm,
    stages: Vec<Stage>,
    projection: Linear,
    position: Vec<f32>,
}
impl Encoder {
    pub fn from_onnx(bytes: &[u8]) -> Result<Self, String> {
        #[cfg(all(target_arch = "wasm32", target_feature = "relaxed-simd"))]
        if !crate::decoder::fused_simd_supported() {
            return Err("encoder requires fused relaxed SIMD on this artifact".into());
        }
        let g = Graph::read(bytes)?;
        let mut inventory = std::collections::HashMap::new();
        for n in &g.nodes {
            *inventory.entry(n.op.as_str()).or_insert(0) += 1;
        }
        for (op, count) in [
            ("Conv", 18),
            ("MatMul", 31),
            ("LayerNormalization", 18),
            ("Erf", 15),
            ("Transpose", 37),
            ("Add", 62),
            ("Div", 15),
            ("Mul", 45),
            ("Reshape", 1),
        ] {
            if inventory.remove(op) != Some(count) {
                return Err(format!("unsupported encoder graph: {op}"));
            }
        }
        if !inventory.is_empty() {
            return Err("unsupported encoder operators".into());
        }
        let stem = Conv::load(&g, "/stem/stem.0/Conv", 1, 96, 4, 4, 0, false)?;
        let norm = Norm::load(&g, "/stem/stem.1/LayerNormalization", 96)?;
        let mut stages = Vec::new();
        for (stage, (c, depth)) in [(96, 3), (192, 3), (384, 9)].into_iter().enumerate() {
            let down = if stage == 0 {
                None
            } else {
                Some((
                    Norm::load(
                        &g,
                        &format!("/stages.{stage}/downsample/downsample.0/LayerNormalization"),
                        c / 2,
                    )?,
                    Conv::load(
                        &g,
                        &format!("/stages.{stage}/downsample/downsample.1/Conv"),
                        c / 2,
                        c,
                        2,
                        2,
                        0,
                        false,
                    )?,
                ))
            };
            let mut blocks = Vec::new();
            for b in 0..depth {
                let name = format!("/stages.{stage}/blocks/blocks.{b}");
                let mul = g.node(&format!("{name}/Mul"))?;
                let gamma = g.tensor(&mul.inputs[1])?.floats()?;
                if gamma.len() != c {
                    return Err("encoder layer scale shape".into());
                }
                blocks.push(Block {
                    dw: Conv::load(&g, &format!("{name}/conv_dw/Conv"), c, c, 7, 1, 3, true)?,
                    norm: Norm::load(&g, &format!("{name}/norm/LayerNormalization"), c)?,
                    fc1: Linear::load(&g, &format!("{name}/mlp/fc1"), c, c * 4)?,
                    fc2: Linear::load(&g, &format!("{name}/mlp/fc2"), c * 4, c)?,
                    gamma,
                    name,
                });
            }
            stages.push(Stage { down, blocks });
        }
        let projection = Linear::load(&g, "/proj", 384, 512)?;
        let position = g.tensor("/Concat_output_0")?.floats()?;
        if position.len() != CONTEXT_VALUES {
            return Err("encoder positional shape".into());
        }
        Ok(Self {
            stem,
            norm,
            stages,
            projection,
            position,
        })
    }
    pub fn encode(&self, image: &[f32]) -> Result<Vec<f32>, String> {
        self.encode_observed(image, |name, data, h, w| {
            #[cfg(all(target_arch = "wasm32", feature = "encoder-trace"))]
            unsafe {
                trace(name.as_ptr(), name.len(), data.as_ptr(), data.len(), h, w);
            }
            let _ = (data, h, w);
            #[cfg(all(target_arch = "wasm32", feature = "encoder-profile"))]
            unsafe {
                mark(phase(name) as u32);
            }
            let _ = name;
        })
    }
    /// Diagnostics receive NHWC buffers without additional arithmetic or copies.
    pub fn encode_observed(
        &self,
        image: &[f32],
        mut observe: impl FnMut(&str, &[f32], usize, usize),
    ) -> Result<Vec<f32>, String> {
        if image.len() != IMAGE_VALUES || image.iter().any(|v| !v.is_finite()) {
            return Err("encoder expects finite [1,1,256,1280] image".into());
        }
        let (mut h, mut w) = (64, 320);
        let mut x = self.stem.run(image, 256, 1280);
        observe("/stem/stem.0/Conv", &x, h, w);
        x = self.norm.run(&x);
        observe("/stem/stem.1/LayerNormalization_output_0", &x, h, w);
        for (s, stage) in self.stages.iter().enumerate() {
            if let Some((norm, conv)) = &stage.down {
                x = norm.run(&x);
                observe(
                    &format!("/stages.{s}/downsample/downsample.0/LayerNormalization_output_0"),
                    &x,
                    h,
                    w,
                );
                x = conv.run(&x, h, w);
                h /= 2;
                w /= 2;
                observe(
                    &format!("/stages.{s}/downsample/downsample.1/Conv_output_0"),
                    &x,
                    h,
                    w,
                );
            }
            for b in &stage.blocks {
                let mut y = b.dw.depthwise(&x, h, w);
                observe(&format!("{}/conv_dw/Conv_output_0", b.name), &y, h, w);
                y = b.norm.run(&y);
                observe(
                    &format!("{}/norm/LayerNormalization_output_0", b.name),
                    &y,
                    h,
                    w,
                );
                y = b.fc1.run(&y);
                observe(&format!("{}/mlp/fc1/Add_output_0", b.name), &y, h, w);
                kernels::gelu_in_place(&mut y);
                observe(&format!("{}/mlp/act/Mul_1_output_0", b.name), &y, h, w);
                y = b.fc2.run(&y);
                observe(&format!("{}/mlp/fc2/Add_output_0", b.name), &y, h, w);
                for (i, v) in y.iter_mut().enumerate() {
                    *v *= b.gamma[i % b.gamma.len()];
                }
                observe(&format!("{}/Mul", b.name), &y, h, w);
                for (v, r) in y.iter_mut().zip(x) {
                    *v += r;
                }
                x = y;
                observe(&format!("{}/Add", b.name), &x, h, w);
            }
        }
        let mut result = self.projection.run(&x);
        observe("/proj/Add_output_0", &result, h, w);
        for (v, p) in result.iter_mut().zip(&self.position) {
            *v += p;
        }
        observe("output", &result, h, w);
        Ok(result)
    }
}

/// Stable phase IDs used only by the optional instruction benchmark hook.
pub fn phase(name: &str) -> usize {
    if name == "output" {
        10
    } else if name.contains("/stem/stem.0") {
        0
    } else if name.contains("downsample.1") {
        1
    } else if name.contains("conv_dw") {
        2
    } else if name.contains("LayerNormalization") {
        3
    } else if name.contains("fc1") {
        4
    } else if name.contains("fc2") {
        6
    } else if name.contains("/act/") {
        5
    } else if name.contains("/proj/") {
        9
    } else if name.ends_with("/Mul") {
        7
    } else {
        8
    }
}
pub const PHASE_NAMES: [&str; 11] = [
    "stem_conv",
    "downsample_conv",
    "depthwise",
    "norm",
    "fc1",
    "gelu",
    "fc2",
    "scale",
    "residual",
    "projection",
    "position",
];
#[cfg(all(target_arch = "wasm32", feature = "encoder-profile"))]
#[link(wasm_import_module = "encoder_profile")]
extern "C" {
    fn mark(phase: u32);
}

#[cfg(all(target_arch = "wasm32", feature = "encoder-trace"))]
#[link(wasm_import_module = "encoder_trace")]
extern "C" {
    fn trace(name: *const u8, name_len: usize, data: *const f32, len: usize, h: usize, w: usize);
}
