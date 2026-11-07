//! Native, stateful execution of homr's quantized transformer decoder.
//!
//! This is a model-specific executor, not a general ONNX runtime. Weights are
//! read at runtime from the upstream decoder ONNX bytes. A session owns its KV
//! caches; no cache crosses the JavaScript boundary between steps.
pub(crate) mod activation;
mod math;
pub mod profile;
#[cfg(target_arch = "wasm32")]
pub(crate) mod wasm;
use crate::onnx::Graph;
use activation::{gelu, sigmoid, softmax};
use math::{norm, Linear, Quantized};

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
// The caches hold f32 on every target. The strict SIMD128 kernels widen each
// value to f64 as they read it: every f32 is exactly representable in f64, so
// the results are the same as caching f64, in half the memory.
type CacheFloat = f32;
struct Cache {
    // Channel-major K permits four independent scores in SIMD lanes. V stays
    // position-major, contiguous across output channels. Both start small and
    // grow as positions are appended (K's channel stride is its capacity), so
    // a short staff doesn't reserve room for the longest one.
    k: Vec<CacheFloat>,
    v: Vec<CacheFloat>,
    len: usize,
    capacity: usize,
    limit: usize,
}
// Positions reserved at first, and at least as many again each time it grows.
const CACHE_GROWTH: usize = 64;
impl Cache {
    #[cfg(test)]
    fn new(capacity: usize) -> Self {
        Self::growing(capacity, capacity)
    }
    /// Room for `start` positions at first, growing to at most `limit`.
    fn growing(start: usize, limit: usize) -> Self {
        let capacity = start.min(limit);
        let mut v = Vec::new();
        v.reserve_exact(DIM * capacity);
        Self {
            k: vec![0.0; DIM * capacity],
            v,
            len: 0,
            capacity,
            limit,
        }
    }
    fn grow(&mut self, needed: usize) {
        let capacity = (self.capacity + CACHE_GROWTH.max(self.capacity / 4))
            .max(needed)
            .min(self.limit);
        assert!(needed <= capacity);
        let mut k = vec![0.0; DIM * capacity];
        for d in 0..DIM {
            k[d * capacity..d * capacity + self.len]
                .copy_from_slice(&self.k[d * self.capacity..d * self.capacity + self.len]);
        }
        self.k = k;
        self.v.reserve_exact(DIM * capacity - self.v.len());
        self.capacity = capacity;
    }
    fn append(&mut self, k: &[f32], v: &[f32]) {
        let _g = profile::enter(profile::Phase::Copy);
        let rows = k.len() / DIM;
        assert_eq!(k.len(), v.len());
        if self.len + rows > self.capacity {
            self.grow(self.len + rows);
        }
        assert!(self.len + rows <= self.capacity);
        for d in 0..DIM {
            for r in 0..rows {
                self.k[d * self.capacity + self.len + r] = k[r * DIM + d];
            }
        }
        self.v.extend_from_slice(v);
        self.len += rows;
    }
    fn attend(&self, q: &[f32], mut map: Option<&mut [f32]>) -> Vec<f32> {
        let mut out = vec![0.0; DIM];
        let mut scores = vec![0.0; self.len];
        for h in 0..HEADS {
            {
                let _g = profile::enter(profile::Phase::Scores);
                self.scores(q, h, &mut scores);
            }
            softmax(&mut scores);
            if let Some(map) = map.as_deref_mut() {
                let _g = profile::enter(profile::Phase::Copy);
                map[h * self.len..(h + 1) * self.len].copy_from_slice(&scores);
            }
            {
                let _g = profile::enter(profile::Phase::WeightedV);
                // SAFETY: the cache contains len complete rows, q/out have DIM
                // elements, and h indexes a complete 64-channel head.
                #[cfg(any(
                    target_arch = "aarch64",
                    all(target_arch = "wasm32", target_feature = "simd128")
                ))]
                unsafe {
                    weighted_sum(&self.v, &scores, h, &mut out);
                }
                #[cfg(not(any(
                    target_arch = "aarch64",
                    all(target_arch = "wasm32", target_feature = "simd128")
                )))]
                for (p, &s) in scores.iter().enumerate() {
                    for d in 0..HD {
                        let j = h * HD + d;
                        out[j] = activation::fma(s, self.v[p * DIM + j], out[j]);
                    }
                }
            }
        }
        out
    }
    fn scores(&self, q: &[f32], h: usize, out: &mut [f32]) {
        let bulk = self.len / 4 * 4;
        // SAFETY: each transposed channel has capacity >= len elements and
        // the complete query/head dimensions are validated by the model loader.
        #[cfg(any(
            target_arch = "aarch64",
            all(target_arch = "wasm32", target_feature = "simd128")
        ))]
        unsafe {
            score_simd(q, &self.k, self.capacity, h, &mut out[..bulk]);
        }
        #[cfg(not(any(
            target_arch = "aarch64",
            all(target_arch = "wasm32", target_feature = "simd128")
        )))]
        let bulk = 0;
        for (p, s) in out.iter_mut().enumerate().skip(bulk) {
            let mut sum = 0.0;
            for d in 0..HD {
                sum = activation::fma(q[h * HD + d], self.k[(h * HD + d) * self.capacity + p], sum);
            }
            *s = sum * 0.125;
        }
    }
}
#[cfg(all(
    target_arch = "wasm32",
    target_feature = "simd128",
    not(target_feature = "relaxed-simd")
))]
#[inline(always)]
unsafe fn round2(v: core::arch::wasm32::v128) -> core::arch::wasm32::v128 {
    use core::arch::wasm32::*;
    f64x2_promote_low_f32x4(f32x4_demote_f64x2_zero(v))
}
/// Four f32 lanes as two f64x2 (lanes 0-1, lanes 2-3), exactly.
#[cfg(all(
    target_arch = "wasm32",
    target_feature = "simd128",
    not(target_feature = "relaxed-simd")
))]
#[inline(always)]
unsafe fn widen(
    v: core::arch::wasm32::v128,
) -> (core::arch::wasm32::v128, core::arch::wasm32::v128) {
    use core::arch::wasm32::*;
    (
        f64x2_promote_low_f32x4(v),
        f64x2_promote_low_f32x4(i32x4_shuffle::<2, 3, 2, 3>(v, v)),
    )
}
#[cfg(all(
    target_arch = "wasm32",
    target_feature = "simd128",
    not(target_feature = "relaxed-simd")
))]
unsafe fn score_simd(q: &[f32], keys: &[CacheFloat], stride: usize, h: usize, out: &mut [f32]) {
    use core::arch::wasm32::*;
    let queries: [v128; HD] =
        std::array::from_fn(|d| f64x2_splat(*q.get_unchecked(h * HD + d) as f64));
    for p in (0..out.len()).step_by(4) {
        let mut lo = f64x2_splat(0.0);
        let mut hi = f64x2_splat(0.0);
        for (d, &query) in queries.iter().enumerate() {
            let k = v128_load(keys.as_ptr().add((h * HD + d) * stride + p).cast());
            let (k_lo, k_hi) = widen(k);
            lo = round2(f64x2_add(f64x2_mul(k_lo, query), lo));
            hi = round2(f64x2_add(f64x2_mul(k_hi, query), hi));
        }
        let sum =
            i32x4_shuffle::<0, 1, 4, 5>(f32x4_demote_f64x2_zero(lo), f32x4_demote_f64x2_zero(hi));
        v128_store(
            out.as_mut_ptr().add(p).cast(),
            f32x4_mul(sum, f32x4_splat(0.125)),
        );
    }
}
#[cfg(target_arch = "aarch64")]
unsafe fn score_simd(q: &[f32], keys: &[f32], stride: usize, h: usize, out: &mut [f32]) {
    use core::arch::aarch64::*;
    for p in (0..out.len()).step_by(4) {
        let mut sum = vdupq_n_f32(0.0);
        for d in 0..HD {
            sum = vfmaq_n_f32(
                sum,
                vld1q_f32(keys.as_ptr().add((h * HD + d) * stride + p)),
                *q.get_unchecked(h * HD + d),
            );
        }
        vst1q_f32(out.as_mut_ptr().add(p), vmulq_n_f32(sum, 0.125));
    }
}
#[cfg(all(
    target_arch = "wasm32",
    target_feature = "simd128",
    not(target_feature = "relaxed-simd")
))]
unsafe fn weighted_sum(values: &[CacheFloat], scores: &[f32], h: usize, out: &mut [f32]) {
    use core::arch::wasm32::*;
    for d in (0..HD).step_by(8) {
        let mut a = f64x2_splat(0.0);
        let mut b = f64x2_splat(0.0);
        let mut c = f64x2_splat(0.0);
        let mut e = f64x2_splat(0.0);
        for (p, &s) in scores.iter().enumerate() {
            let scale = f64x2_splat(s as f64);
            let v = values.as_ptr().add(p * DIM + h * HD + d);
            let (v0, v1) = widen(v128_load(v.cast()));
            let (v2, v3) = widen(v128_load(v.add(4).cast()));
            a = round2(f64x2_add(f64x2_mul(v0, scale), a));
            b = round2(f64x2_add(f64x2_mul(v1, scale), b));
            c = round2(f64x2_add(f64x2_mul(v2, scale), c));
            e = round2(f64x2_add(f64x2_mul(v3, scale), e));
        }
        let lo =
            i32x4_shuffle::<0, 1, 4, 5>(f32x4_demote_f64x2_zero(a), f32x4_demote_f64x2_zero(b));
        let hi =
            i32x4_shuffle::<0, 1, 4, 5>(f32x4_demote_f64x2_zero(c), f32x4_demote_f64x2_zero(e));
        v128_store(out.as_mut_ptr().add(h * HD + d).cast(), lo);
        v128_store(out.as_mut_ptr().add(h * HD + d + 4).cast(), hi);
    }
}
#[cfg(target_arch = "aarch64")]
unsafe fn weighted_sum(values: &[f32], scores: &[f32], h: usize, out: &mut [f32]) {
    use core::arch::aarch64::*;
    for d in (0..HD).step_by(8) {
        let mut lo = vdupq_n_f32(0.0);
        let mut hi = vdupq_n_f32(0.0);
        for (p, &s) in scores.iter().enumerate() {
            let v = values.as_ptr().add(p * DIM + h * HD + d);
            lo = vfmaq_n_f32(lo, vld1q_f32(v), s);
            hi = vfmaq_n_f32(hi, vld1q_f32(v.add(4)), s);
        }
        vst1q_f32(out.as_mut_ptr().add(h * HD + d), lo);
        vst1q_f32(out.as_mut_ptr().add(h * HD + d + 4), hi);
    }
}

// This optional artifact requires a host whose relaxed multiply-add is fused.
// The strict SIMD128 artifact remains the fallback for other engines.
#[cfg(all(target_arch = "wasm32", target_feature = "relaxed-simd"))]
unsafe fn score_simd(q: &[f32], keys: &[CacheFloat], stride: usize, h: usize, out: &mut [f32]) {
    use core::arch::wasm32::*;
    for p in (0..out.len()).step_by(4) {
        let mut sum = f32x4_splat(0.0);
        for d in 0..HD {
            let v = v128_load(keys.as_ptr().add((h * HD + d) * stride + p).cast());
            sum = f32x4_relaxed_madd(v, f32x4_splat(*q.get_unchecked(h * HD + d)), sum);
        }
        v128_store(
            out.as_mut_ptr().add(p).cast(),
            f32x4_mul(sum, f32x4_splat(0.125)),
        );
    }
}
#[cfg(all(target_arch = "wasm32", target_feature = "relaxed-simd"))]
unsafe fn weighted_sum(values: &[CacheFloat], scores: &[f32], h: usize, out: &mut [f32]) {
    use core::arch::wasm32::*;
    for d in (0..HD).step_by(8) {
        let mut lo = f32x4_splat(0.0);
        let mut hi = f32x4_splat(0.0);
        for (p, &s) in scores.iter().enumerate() {
            let scale = f32x4_splat(s);
            let v = values.as_ptr().add(p * DIM + h * HD + d);
            lo = f32x4_relaxed_madd(v128_load(v.cast()), scale, lo);
            hi = f32x4_relaxed_madd(v128_load(v.add(4).cast()), scale, hi);
        }
        v128_store(out.as_mut_ptr().add(h * HD + d).cast(), lo);
        v128_store(out.as_mut_ptr().add(h * HD + d + 4).cast(), hi);
    }
}
#[cfg(target_arch = "wasm32")]
pub(crate) fn fused_simd_supported() -> bool {
    #[cfg(not(target_feature = "relaxed-simd"))]
    {
        false
    }
    #[cfg(target_feature = "relaxed-simd")]
    {
        use core::arch::wasm32::*;
        let a = std::hint::black_box([1.0000001f32; 4]);
        let b = std::hint::black_box([0.9999999f32; 4]);
        let c = std::hint::black_box([-1.0f32; 4]);
        let mut out = [0.0f32; 4];
        // SAFETY: these are four-element arrays in the current linear memory.
        unsafe {
            v128_store(
                out.as_mut_ptr().cast(),
                f32x4_relaxed_madd(
                    v128_load(a.as_ptr().cast()),
                    v128_load(b.as_ptr().cast()),
                    v128_load(c.as_ptr().cast()),
                ),
            );
        }
        out.iter().enumerate().all(|(i, &v)| {
            v.to_bits() == ((a[i] as f64 * b[i] as f64 + c[i] as f64) as f32).to_bits()
        })
    }
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
        #[cfg(all(target_arch = "wasm32", target_feature = "relaxed-simd"))]
        if !fused_simd_supported() {
            return Err(
                "relaxed SIMD host does not fuse multiply-add; use the SIMD128 fallback artifact"
                    .into(),
            );
        }

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
        for (e, rows) in embeddings.iter().zip([260, 72, 7, 62, 5, MAX_STEPS]) {
            if e.rows != rows {
                return Err("unsupported embedding vocabulary".into());
            }
        }
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
        let model = Self {
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
        };
        for b in &model.blocks {
            for (matrix, expected) in [
                (&b.qkv, (DIM, 3 * DIM)),
                (&b.self_out, (DIM, 2 * DIM)),
                (&b.q, (DIM, DIM)),
                (&b.k, (DIM, DIM)),
                (&b.v, (DIM, DIM)),
                (&b.cross_out, (DIM, 2 * DIM)),
                (&b.ff_in, (DIM, 8 * DIM)),
                (&b.ff_out, (4 * DIM, DIM)),
            ] {
                if matrix.shape() != expected {
                    return Err("unsupported decoder matrix dimensions".into());
                }
            }
        }
        for (head, n) in model.heads.iter().zip([260, 72, 7, 5, 62, 5]) {
            if head.shape() != (DIM, n) {
                return Err("unsupported output head dimensions".into());
            }
        }
        Ok(model)
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
                self_cache: Cache::growing(CACHE_GROWTH, MAX_STEPS),
                cross_cache: {
                    let mut c = Cache::growing(
                        CONTEXT_POSITIONS + CACHE_GROWTH,
                        CONTEXT_POSITIONS + MAX_STEPS,
                    );
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
    let _g = profile::enter(profile::Phase::Activation);
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
        let _step = profile::enter(profile::Phase::Other);
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
                gated(b.self_out.run(&state.self_cache.attend(&qkv[..DIM], None))),
            );
            let input = norm(&x, &b.gains[1], None);
            let q = b.q.run(&input);
            if self.position > 0 {
                state.cross_cache.append(&state.extra_k, &state.extra_v);
            }
            let mut map = if i >= 5 {
                Some(vec![0.0; HEADS * (CONTEXT_POSITIONS + self.position)])
            } else {
                None
            };
            let a = state.cross_cache.attend(&q, map.as_deref_mut());
            if let Some(map) = map {
                maps.push(map);
            }
            residual(&mut x, gated(b.cross_out.run(&a)));
            let input = norm(&x, &b.gains[2], None);
            let ff = b.ff_in.run(&input);
            let half = ff.len() / 2;
            let ff: Vec<f32> = {
                let _g = profile::enter(profile::Phase::Activation);
                (0..half)
                    .map(|j| {
                        let v = ff[half + j];
                        ff[j] * gelu(v)
                    })
                    .collect()
            };
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
        let _centroid = profile::enter(profile::Phase::Centroid);
        let mut sums = [[[0.0f32; 4]; 2]; 3];
        for p in 0..CONTEXT_POSITIONS {
            let mut a = 0.0;
            let len = CONTEXT_POSITIONS + self.position;
            for h in 0..HEADS {
                let j = h * len + p;
                a += ((maps[0][j] + maps[1][j]) + maps[2][j]) / 3.0;
            }
            let a = (a * (1.0 / HEADS as f32)).max(1e-4);
            let a = libm::powf(a, 4.0);
            sums[0][p % 8 / 4][p % 4] += a;
            sums[1][p % 8 / 4][p % 4] += a * ((p % 80) as f32 + 0.5);
            sums[2][p % 8 / 4][p % 4] += a * ((p / 80) as f32 + 0.5);
        }
        let [sum, sx, sy] = sums.map(|s| {
            let v: [f32; 4] = std::array::from_fn(|i| s[0][i] + s[1][i]);
            (v[0] + v[1]) + (v[2] + v[3])
        });
        self.position += 1;
        Ok(Step {
            tokens,
            logits,
            attention: [sx / sum * 16.0, sy / sum * 16.0],
        })
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn attention_tiles_and_tails_match_scalar_order() {
        for n in [1, 3, 4, 5, 17] {
            let keys: Vec<f32> = (0..n * DIM)
                .map(|i| ((i * 71 % 511) as f32 - 255.0) / 513.0)
                .collect();
            let values: Vec<f32> = (0..n * DIM)
                .map(|i| ((i * 43 % 509) as f32 - 254.0) / 517.0)
                .collect();
            let query: Vec<f32> = (0..DIM)
                .map(|i| ((i * 131 % 503) as f32 - 251.0) / 519.0)
                .collect();
            let mut cache = Cache::new(n + 3);
            cache.append(&keys, &values);
            let actual = cache.attend(&query, None);
            let mut expected = vec![0.0; DIM];
            for h in 0..HEADS {
                let mut scores: Vec<f32> = (0..n)
                    .map(|p| {
                        let mut s = 0.0;
                        for d in 0..HD {
                            s = activation::fma(query[h * HD + d], keys[p * DIM + h * HD + d], s);
                        }
                        s * 0.125
                    })
                    .collect();
                softmax(&mut scores);
                for (p, &s) in scores.iter().enumerate() {
                    for d in 0..HD {
                        let j = h * HD + d;
                        expected[j] = activation::fma(s, values[p * DIM + j], expected[j]);
                    }
                }
            }
            assert_eq!(
                actual.iter().map(|v| v.to_bits()).collect::<Vec<_>>(),
                expected.iter().map(|v| v.to_bits()).collect::<Vec<_>>()
            );
        }
    }
    #[test]
    fn a_growing_cache_attends_like_a_full_one() {
        let row = |i: usize, salt: usize| -> Vec<f32> {
            (0..DIM)
                .map(|j| (((i * 97 + j * salt) % 509) as f32 - 254.0) / 511.0)
                .collect()
        };
        let query: Vec<f32> = (0..DIM)
            .map(|i| ((i * 131 % 503) as f32 - 251.0) / 519.0)
            .collect();
        let mut full = Cache::new(300);
        let mut grown = Cache::growing(2, 300);
        // A block first (like the cross cache's context), then rows one at a time.
        let block_k: Vec<f32> = (0..5).flat_map(|i| row(i, 71)).collect();
        let block_v: Vec<f32> = (0..5).flat_map(|i| row(i, 43)).collect();
        full.append(&block_k, &block_v);
        grown.append(&block_k, &block_v);
        for i in 5..300 {
            full.append(&row(i, 71), &row(i, 43));
            grown.append(&row(i, 71), &row(i, 43));
            if i % 37 == 0 || i == 299 {
                let bits = |c: &Cache| {
                    c.attend(&query, None)
                        .iter()
                        .map(|v| v.to_bits())
                        .collect::<Vec<_>>()
                };
                assert_eq!(bits(&full), bits(&grown), "after {} rows", i + 1);
            }
        }
        assert_eq!(grown.capacity, 300);
    }
}
