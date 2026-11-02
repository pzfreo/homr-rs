//! Native, stateful execution of homr's quantized transformer decoder.
//!
//! This is a model-specific executor, not a general ONNX runtime. Weights are
//! read at runtime from the upstream decoder ONNX bytes. A session owns its KV
//! caches; no cache crosses the JavaScript boundary between steps.
mod activation;
mod math;
mod onnx;
use activation::{gelu, sigmoid, softmax};
use math::{norm, Linear, Quantized};
use onnx::Graph;

pub const CONTEXT_POSITIONS: usize = 1280;
pub const DIM: usize = 512;
pub const MAX_STEPS: usize = 608;
const HEADS: usize = 8;
const HD: usize = 64;

struct Embedding {
    data: Vec<f32>,
    rows: usize,
}
impl Embedding {
    fn load(g: &Graph, name: &str) -> Result<Self, String> {
        let prefix = format!("model.{name}_emb.emb.weight");
        let t = g.tensor(&format!("{prefix}_quantized"))?;
        if t.kind != 2 || t.dims.len() != 2 || t.dims[1] != DIM {
            return Err("embedding shape/type".into());
        }
        let scale = g.tensor(&format!("{prefix}_scale"))?.scalar()?;
        let zero = g.tensor(&format!("{prefix}_zero_point"))?;
        if zero.kind != 2 || zero.data.len() != 1 {
            return Err("embedding zero point".into());
        }
        Ok(Self {
            rows: t.dims[0],
            data: t
                .data
                .iter()
                .map(|&v| (v as i32 - zero.data[0] as i32) as f32 * scale)
                .collect(),
        })
    }
    fn row(&self, i: usize) -> &[f32] {
        &self.data[i * DIM..(i + 1) * DIM]
    }
}
struct Block {
    gains: [Vec<f32>; 3],
    qkv: Linear,
    self_out: Linear,
    q: Linear,
    k: Linear,
    v: Linear,
    cross_out: Linear,
    ff_in: Linear,
    ff_out: Linear,
}
/// Immutable model weights. One model can serve multiple independent sessions.
pub struct Decoder {
    embeddings: [Embedding; 6],
    emb_gain: Vec<f32>,
    emb_bias: Vec<f32>,
    blocks: Vec<Block>,
    final_gain: Vec<f32>,
    heads: [Linear; 6],
}
/// One chosen symbol; token order follows ONNX outputs: rhythm, pitch, lift,
/// position, articulation, slur. EOS is included in `generate`'s steps.
#[derive(Debug, Clone)]
pub struct Step {
    pub tokens: [usize; 6],
    pub attention: [f32; 2],
    pub logits: [Vec<f32>; 6],
}
impl Step {
    pub fn symbol(&self, vocab: &crate::tokens::Vocab) -> Option<crate::tokens::Symbol> {
        vocab.symbol(
            self.tokens,
            Some(self.attention.iter().map(|&v| v as f64).collect()),
        )
    }
}
#[derive(Default)]
struct Cache {
    k: Vec<f32>,
    v: Vec<f32>,
}
impl Cache {
    // Position-major storage lets append preserve all previous positions in place.
    fn append(&mut self, k: &[f32], v: &[f32]) {
        self.k.extend_from_slice(k);
        self.v.extend_from_slice(v);
    }
    fn attend(&self, q: &[f32]) -> Vec<f32> {
        let mut out = vec![0.0; DIM];
        let len = self.k.len() / DIM;
        let mut scores = vec![0.0; len];
        for h in 0..HEADS {
            let start = h * HD;
            for (p, s) in scores.iter_mut().enumerate() {
                *s = dot_float(
                    &q[start..start + HD],
                    &self.k[p * DIM + start..p * DIM + start + HD],
                ) * 0.125;
            }
            softmax(&mut scores);
            for (p, &s) in scores.iter().enumerate() {
                for d in 0..HD {
                    out[start + d] = s.mul_add(self.v[p * DIM + start + d], out[start + d]);
                }
            }
        }
        out
    }
    fn map(&self, q: &[f32]) -> Vec<f32> {
        let len = self.k.len() / DIM;
        let mut avg = vec![0.0; HEADS * len];
        let mut scores = vec![0.0; len];
        // The centroid has a separate scaled Einsum/Softmax path in the graph.
        for h in 0..HEADS {
            for (p, s) in scores.iter_mut().enumerate() {
                *s = dot_float(
                    &q[h * HD..(h + 1) * HD],
                    &self.k[p * DIM + h * HD..p * DIM + (h + 1) * HD],
                ) * 0.125;
            }
            softmax(&mut scores);
            avg[h * len..(h + 1) * len].copy_from_slice(&scores);
        }
        avg
    }
}
fn dot_float(a: &[f32], b: &[f32]) -> f32 {
    a.iter().zip(b).fold(0.0, |s, (&a, &b)| a.mul_add(b, s))
}
struct BlockState {
    self_cache: Cache,
    cross_cache: Cache,
    extra_k: Vec<f32>,
    extra_v: Vec<f32>,
}
/// Per-staff state. Start a new session for each encoder output.
pub struct Session<'a> {
    model: &'a Decoder,
    blocks: Vec<BlockState>,
    position: usize,
}
fn gain(g: &Graph, name: &str) -> Result<Vec<f32>, String> {
    let n = g.node(name)?;
    let v = g
        .tensor(n.inputs.get(1).ok_or("gain input missing")?)?
        .floats()?;
    if v.len() != DIM {
        return Err("gain shape".into());
    }
    Ok(v)
}
impl Decoder {
    pub fn from_onnx(bytes: &[u8]) -> Result<Self, String> {
        let g = Graph::read(bytes)?;
        if g.nodes
            .iter()
            .filter(|n| n.op == "MultiHeadAttention")
            .count()
            != 16
            || g.nodes.iter().filter(|n| n.op == "MatMulInteger").count() != 70
        {
            return Err("unsupported decoder graph (expected homr 465 quantized decoder)".into());
        }
        let embeddings = [
            Embedding::load(&g, "rhythm")?,
            Embedding::load(&g, "pitch")?,
            Embedding::load(&g, "lift")?,
            Embedding::load(&g, "articulation")?,
            Embedding::load(&g, "slur")?,
            Embedding::load(&g, "pos")?,
        ];
        let mut blocks = Vec::new();
        for b in 0..8 {
            let s = 3 * b;
            let prefix = |i| format!("/model/attn_layers/layers.{i}");
            let mha = g.node(&format!("layers_{s}_1_MultiHeadAttention"))?;
            let split = g.producer(mha.inputs.first().ok_or("self query")?)?;
            let mul = g.producer(split.inputs.first().ok_or("qkv split")?)?;
            let cast = g.producer(mul.inputs.first().ok_or("qkv scale")?)?;
            let mm = g.producer(cast.inputs.first().ok_or("qkv cast")?)?;
            let qkv = Linear::load(&g, mm.inputs.get(1).ok_or("qkv weight")?, None)?;
            let linear =
                |i, tail: &str, bias| Linear::named(&g, &format!("{}.1/{tail}", prefix(i)), bias);
            let ff = s + 2;
            blocks.push(Block {
                gains: [
                    gain(&g, &format!("{}.0.0/Mul", prefix(s)))?,
                    gain(&g, &format!("{}.0.0/Mul", prefix(s + 1)))?,
                    gain(&g, &format!("{}.0.0/Mul", prefix(ff)))?,
                ],
                qkv,
                self_out: linear(s, "to_out/to_out.0/MatMul", None)?,
                q: linear(s + 1, "to_q/MatMul", None)?,
                k: linear(s + 1, "to_k/MatMul", None)?,
                v: linear(s + 1, "to_v/MatMul", None)?,
                cross_out: linear(s + 1, "to_out/to_out.0/MatMul", None)?,
                ff_in: linear(
                    ff,
                    "ff/ff.0/proj/MatMul",
                    Some(&format!("model.attn_layers.layers.{ff}.1.ff.0.proj.bias")),
                )?,
                ff_out: linear(
                    ff,
                    "ff/ff.2/MatMul",
                    Some(&format!("model.attn_layers.layers.{ff}.1.ff.2.bias")),
                )?,
            });
        }
        let head = |name: &str| {
            Linear::named(
                &g,
                &format!("/model/to_logits_{name}/MatMul"),
                Some(&format!("model.to_logits_{name}.bias")),
            )
        };
        let emb_gain = g.tensor("model.post_emb_norm.weight")?.floats()?;
        let emb_bias = g.tensor("model.post_emb_norm.bias")?.floats()?;
        if emb_gain.len() != DIM || emb_bias.len() != DIM {
            return Err("embedding normalization shape".into());
        }
        Ok(Self {
            embeddings,
            emb_gain,
            emb_bias,
            blocks,
            final_gain: gain(&g, "/model/attn_layers/final_norm/Mul")?,
            heads: [
                head("rhythm")?,
                head("pitch")?,
                head("lift")?,
                head("position")?,
                head("articulations")?,
                head("slurs")?,
            ],
        })
    }
    pub fn session(&self, context: &[f32]) -> Result<Session<'_>, String> {
        if context.len() != CONTEXT_POSITIONS * DIM || context.iter().any(|x| !x.is_finite()) {
            return Err("expected finite [1280,512] encoder output".into());
        }
        let full = Quantized::new(context);
        let first = Quantized::new(&context[..DIM]);
        let blocks = self
            .blocks
            .iter()
            .map(|b| BlockState {
                self_cache: Cache {
                    k: Vec::with_capacity(MAX_STEPS * DIM),
                    v: Vec::with_capacity(MAX_STEPS * DIM),
                },
                cross_cache: {
                    let mut c = Cache {
                        k: Vec::with_capacity((CONTEXT_POSITIONS + MAX_STEPS) * DIM),
                        v: Vec::with_capacity((CONTEXT_POSITIONS + MAX_STEPS) * DIM),
                    };
                    c.append(&b.k.quantized(&full), &b.v.quantized(&full));
                    c
                },
                extra_k: b.k.quantized(&first),
                extra_v: b.v.quantized(&first),
            })
            .collect();
        Ok(Session {
            model: self,
            blocks,
            position: 0,
        })
    }
    pub fn generate(&self, context: &[f32]) -> Result<Vec<Step>, String> {
        let mut session = self.session(context)?;
        let mut tokens = [1, 0, 0, 0, 0];
        let mut out = Vec::new();
        for _ in 0..MAX_STEPS {
            let step = session.step(tokens)?;
            let t = step.tokens;
            out.push(step);
            if t[0] == 2 {
                break;
            }
            tokens = [t[0], t[1], t[2], t[4], t[5]];
        }
        Ok(out)
    }
}
fn gated(x: Vec<f32>) -> Vec<f32> {
    let half = x.len() / 2;
    (0..half).map(|i| x[i] * sigmoid(x[half + i])).collect()
}
fn residual(x: &mut [f32], add: Vec<f32>) {
    for (v, a) in x.iter_mut().zip(add) {
        *v += a;
    }
}
impl Session<'_> {
    pub fn position(&self) -> usize {
        self.position
    }
    /// Inputs: previous rhythm, pitch, lift, articulation, slur; start [1,0,0,0,0].
    pub fn step(&mut self, tokens: [usize; 5]) -> Result<Step, String> {
        if self.position >= MAX_STEPS {
            return Err("decoder position exceeds 608".into());
        }
        for (&t, e) in tokens.iter().zip(&self.model.embeddings) {
            if t >= e.rows {
                return Err("token outside embedding vocabulary".into());
            }
        }
        let mut x = self.model.embeddings[0].row(tokens[0]).to_vec();
        for (i, &t) in tokens.iter().enumerate().skip(1) {
            for (x, &e) in x.iter_mut().zip(self.model.embeddings[i].row(t)) {
                *x += e;
            }
        }
        for (x, &p) in x
            .iter_mut()
            .zip(self.model.embeddings[5].row(self.position))
        {
            *x += p * 0.044194173;
        }
        x = norm(&x, &self.model.emb_gain, Some(&self.model.emb_bias));
        let mut maps = Vec::new();
        for (i, (b, state)) in self.model.blocks.iter().zip(&mut self.blocks).enumerate() {
            let input = norm(&x, &b.gains[0], None);
            let qkv = b.qkv.run(&input);
            state.self_cache.append(&qkv[DIM..2 * DIM], &qkv[2 * DIM..]);
            residual(
                &mut x,
                gated(b.self_out.run(&state.self_cache.attend(&qkv[..DIM]))),
            );
            let input = norm(&x, &b.gains[1], None);
            let q = b.q.run(&input);
            if self.position > 0 {
                state.cross_cache.append(&state.extra_k, &state.extra_v);
            }
            let a = state.cross_cache.attend(&q);
            if i >= 5 {
                maps.push(state.cross_cache.map(&q));
            }
            residual(&mut x, gated(b.cross_out.run(&a)));
            let input = norm(&x, &b.gains[2], None);
            let ff = b.ff_in.run(&input);
            let half = ff.len() / 2;
            let ff: Vec<f32> = (0..half)
                .map(|j| {
                    let v = ff[half + j];
                    ff[j] * gelu(v)
                })
                .collect();
            residual(&mut x, b.ff_out.run(&ff));
        }
        let x = norm(&x, &self.model.final_gain, None);
        let q = Quantized::new(&x);
        let logits = std::array::from_fn(|i| self.model.heads[i].quantized(&q));
        let tokens = std::array::from_fn(|i| {
            let mut best = 0;
            for j in 1..logits[i].len() {
                if logits[i][j] > logits[i][best] {
                    best = j;
                }
            }
            best
        });
        let (mut sum, mut sx, mut sy) = (0.0f32, 0.0f32, 0.0f32);
        for p in 0..CONTEXT_POSITIONS {
            let mut a = 0.0;
            let len = CONTEXT_POSITIONS + self.position;
            for h in 0..HEADS {
                let j = h * len + p;
                a += ((maps[0][j] + maps[1][j]) + maps[2][j]) * (1.0 / 3.0);
            }
            let a = (a * (1.0 / HEADS as f32)).max(1e-4);
            let a = libm::powf(a, 4.0);
            sum += a;
            sx += a * ((p % 80) as f32 + 0.5);
            sy += a * ((p / 80) as f32 + 0.5);
        }
        self.position += 1;
        Ok(Step {
            tokens,
            logits,
            attention: [sx / sum * 16.0, sy / sum * 16.0],
        })
    }
}
