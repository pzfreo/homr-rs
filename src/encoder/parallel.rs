//! Synchronous, allocation-free jobs for a host-managed WASM worker pool.
//!
//! Only the coordinator allocates or enters the graph executor. Workers read
//! immutable inputs and borrow only their disjoint output range. A job lives
//! on the coordinator stack until dispatch has joined every worker. No K
//! reduction is divided, and every GEMM chunk contains whole 4-row tiles.
#![allow(clippy::too_many_arguments)]

#[repr(C)]
#[derive(Clone, Copy)]
struct Payload {
    op: u32,
    a: *const f32,
    b: *const f32,
    bias: *const f32,
    out: *mut f32,
    dims: [usize; 6],
    epsilon: f32,
    weights_bounded: bool,
}

#[link(wasm_import_module = "encoder_threads")]
extern "C" {
    /// Must publish the job, execute ranks 0..N and join before returning.
    fn dispatch(job: *const u8, units: usize);
}

use super::chunks::Cursor;
#[cfg(feature = "encoder-chunk-bench")]
use std::sync::atomic::{AtomicUsize, Ordering};
struct Job {
    cursor: Cursor,
    payload: Payload,
    units: usize,
    chunk: usize,
    #[cfg(feature = "encoder-chunk-bench")]
    bench: diagnostic::State,
}

fn chunk_size(p: &Payload) -> usize {
    let base = match p.op {
        // Work budgets, not data-dependent shortcuts. A unit is a whole tile.
        0 => (1_000_000 / (4 * p.dims[1] * p.dims[2]).max(1)).clamp(1, 64),
        1 => 2048, // SIMD vectors: 8192 float32 values.
        2 => (250_000 / (p.dims[2] * p.dims[3] * p.dims[3])).max(1),
        3 => (4096 / p.dims[1]).max(1),
        _ => unreachable!(),
    };
    // The calibrated heterogeneous-rank sweep preferred finer fused chunks;
    // strict's much dearer arithmetic preferred the larger base budgets.
    #[cfg(target_feature = "relaxed-simd")]
    let base = (base / 4).max(1);
    base
}

fn submit(payload: Payload, units: usize) {
    let job = Job {
        cursor: Cursor::new(),
        chunk: chunk_size(&payload).min(units).max(1),
        payload,
        units,
        #[cfg(feature = "encoder-chunk-bench")]
        bench: diagnostic::State::new(),
    };
    // SAFETY: every caller validates sizes and retains its buffers. The host
    // contract joins all readers/writers before this stack job is released.
    unsafe { dispatch((&job as *const Job).cast(), units) }
}

pub(super) fn product(
    a: &[f32],
    b: &[f32],
    out: &mut [f32],
    m: usize,
    k: usize,
    n: usize,
    bias: Option<&[f32]>,
) {
    submit(
        Payload {
            op: 0,
            a: a.as_ptr(),
            b: b.as_ptr(),
            bias: bias.map_or(std::ptr::null(), |v| v.as_ptr()),
            out: out.as_mut_ptr(),
            dims: [m, k, n, 0, 0, 0],
            epsilon: 0.0,
            weights_bounded: super::simd::bounded_weights(b, k, bias),
        },
        m / 4,
    );
}

pub(super) fn activation(x: &mut [f32], gelu: bool) {
    submit(
        Payload {
            op: 1,
            a: std::ptr::null(),
            b: std::ptr::null(),
            bias: std::ptr::null(),
            out: x.as_mut_ptr(),
            dims: [x.len(), gelu as usize, 0, 0, 0, 0],
            epsilon: 0.0,
            weights_bounded: false,
        },
        x.len() / 4,
    );
}

pub(super) fn depthwise(
    a: &[f32],
    b: &[f32],
    bias: &[f32],
    out: &mut [f32],
    h: usize,
    w: usize,
    c: usize,
    kernel: usize,
    pad: usize,
) {
    submit(
        Payload {
            op: 2,
            a: a.as_ptr(),
            b: b.as_ptr(),
            bias: bias.as_ptr(),
            out: out.as_mut_ptr(),
            dims: [h, w, c, kernel, pad, 0],
            epsilon: 0.0,
            weights_bounded: false,
        },
        h * w,
    );
}

pub(super) fn norm(
    x: &[f32],
    c: usize,
    scale: &[f32],
    bias: &[f32],
    epsilon: f32,
    out: &mut [f32],
) {
    submit(
        Payload {
            op: 3,
            a: x.as_ptr(),
            b: scale.as_ptr(),
            bias: bias.as_ptr(),
            out: out.as_mut_ptr(),
            dims: [x.len() / c, c, 0, 0, 0, 0],
            epsilon,
            weights_bounded: false,
        },
        x.len() / c,
    );
}

/// Execute one rank by claiming disjoint chunks until the job is exhausted.
/// Workers must never call graph/allocator exports concurrently.
/// # Safety
/// `job` must be a live descriptor published by this module's dispatch import,
/// using identical modules and shared memory. Execute ranks exactly once and
/// join before dispatch returns. Inputs remain immutable, outputs are exclusively
/// lent to this job, and workers require private stacks/TLS. The relaxed cursor
/// only assigns ownership; the host's publication/join atomics provide ordering.
#[no_mangle]
pub unsafe extern "C" fn encoder_work(job: *const u8, rank: usize, threads: usize) -> u32 {
    if job.is_null() || threads == 0 || threads > 32 || rank >= threads {
        return 0;
    }
    let job = &*job.cast::<Job>();
    if job.payload.op > 3 {
        return 0;
    }
    // Copy immutable metadata once, independently of the contested cursor.
    let p = job.payload;
    #[cfg(feature = "encoder-chunk-bench")]
    if job.bench.is_static() {
        let start = job.units * rank / threads;
        let end = job.units * (rank + 1) / threads;
        if start < end {
            run_range(&p, start, end);
            job.bench.after(rank, end - start);
        }
        return 1;
    }
    #[cfg(feature = "encoder-chunk-bench")]
    let step = job.bench.chunk(job.chunk);
    #[cfg(not(feature = "encoder-chunk-bench"))]
    let step = job.chunk;
    while let Some(range) = job.cursor.claim(job.units, step) {
        let count = range.len();
        run_range(&p, range.start, range.end);
        #[cfg(feature = "encoder-chunk-bench")]
        job.bench.after(rank, count);
        let _ = count;
    }
    1
}

unsafe fn run_range(j: &Payload, start: usize, end: usize) {
    let [m, k, n, kernel, pad, _] = j.dims;
    match j.op {
        0 => {
            let rows = (end - start) * 4;
            let a = std::slice::from_raw_parts(j.a.add(start * 4 * k), rows * k);
            let b = std::slice::from_raw_parts(j.b, k * n);
            let out = std::slice::from_raw_parts_mut(j.out.add(start * 4 * n), rows * n);
            let bias = (!j.bias.is_null()).then(|| std::slice::from_raw_parts(j.bias, n));
            super::simd::product_prechecked(a, b, out, rows, k, n, bias, j.weights_bounded);
        }
        1 => {
            let out = std::slice::from_raw_parts_mut(j.out.add(start * 4), (end - start) * 4);
            super::simd::activation(out, k != 0);
        }
        2 => {
            let padded = std::slice::from_raw_parts(j.a, (m + pad * 2) * (k + pad * 2) * n);
            let packed = std::slice::from_raw_parts(j.b, kernel * kernel * n);
            let bias = std::slice::from_raw_parts(j.bias, n);
            let out = std::slice::from_raw_parts_mut(j.out.add(start * n), (end - start) * n);
            super::simd::depthwise_into(padded, k, n, packed, bias, kernel, pad, start, out);
        }
        3 => {
            let input = std::slice::from_raw_parts(j.a.add(start * k), (end - start) * k);
            let scale = std::slice::from_raw_parts(j.b, k);
            let bias = std::slice::from_raw_parts(j.bias, k);
            let out = std::slice::from_raw_parts_mut(j.out.add(start * k), (end - start) * k);
            super::kernels::norm_into(input, k, scale, bias, j.epsilon, out);
        }
        _ => unreachable!(),
    }
}

// Diagnostic-only scheduler selection and sleeping slow ranks. No polling,
// delays, static scheduling or statistics are compiled into production builds.
#[cfg(feature = "encoder-chunk-bench")]
mod diagnostic {
    use super::*;
    use std::sync::atomic::{AtomicI32, AtomicU64};

    #[repr(align(128))]
    struct Rank {
        units: AtomicUsize,
        chunks: AtomicUsize,
        requested_ns: AtomicU64,
    }
    pub(super) struct State {
        mode: AtomicUsize,
        chunk: AtomicUsize,
        ns_per_unit: AtomicU64,
        slow_mask: AtomicUsize,
        ranks: [Rank; 32],
    }
    static WAIT: AtomicI32 = AtomicI32::new(0);
    impl State {
        pub(super) fn new() -> Self {
            Self {
                mode: AtomicUsize::new(1),
                chunk: AtomicUsize::new(0),
                ns_per_unit: AtomicU64::new(0),
                slow_mask: AtomicUsize::new(0),
                ranks: std::array::from_fn(|_| Rank {
                    units: AtomicUsize::new(0),
                    chunks: AtomicUsize::new(0),
                    requested_ns: AtomicU64::new(0),
                }),
            }
        }
        pub(super) fn is_static(&self) -> bool {
            self.mode.load(Ordering::Relaxed) == 0
        }
        pub(super) fn chunk(&self, default: usize) -> usize {
            let n = self.chunk.load(Ordering::Relaxed);
            if n == 0 {
                default
            } else {
                n
            }
        }
        pub(super) fn after(&self, rank: usize, units: usize) {
            self.ranks[rank].units.fetch_add(units, Ordering::Relaxed);
            self.ranks[rank].chunks.fetch_add(1, Ordering::Relaxed);
            if self.slow_mask.load(Ordering::Relaxed) & (1 << rank) != 0 {
                let ns = (f64::from_bits(self.ns_per_unit.load(Ordering::Relaxed)) * units as f64)
                    .clamp(0.0, i64::MAX as f64) as i64;
                if ns > 0 {
                    self.ranks[rank]
                        .requested_ns
                        .fetch_add(ns as u64, Ordering::Relaxed);
                    // Sleep, never burn dummy instructions on this shared host.
                    unsafe {
                        std::arch::wasm32::memory_atomic_wait32(WAIT.as_ptr(), 0, ns);
                    }
                }
            }
        }
    }

    /// Set only before dispatch publishes this live job, then leave unchanged.
    /// ns_per_unit is an additional sleeping budget, calibrated outside trials.
    /// # Safety
    /// Pointer must be the current descriptor supplied by dispatch; no rank may
    /// be executing it yet. Mode 0 selects static, 1 dynamic. Chunk 0 uses default.
    #[no_mangle]
    pub unsafe extern "C" fn encoder_bench_prepare(
        job: *const u8,
        mode: usize,
        chunk: usize,
        ns_per_unit: f64,
        slow_mask: usize,
    ) {
        let b = &(*job.cast::<Job>()).bench;
        b.mode.store(mode, Ordering::Relaxed);
        b.chunk.store(chunk, Ordering::Relaxed);
        b.ns_per_unit
            .store(ns_per_unit.to_bits(), Ordering::Relaxed);
        b.slow_mask.store(slow_mask, Ordering::Relaxed);
    }
    /// # Safety
    /// A live current dispatch descriptor, before publishing or after joining.
    #[no_mangle]
    pub unsafe extern "C" fn encoder_bench_info(job: *const u8, field: usize) -> usize {
        let j = &*job.cast::<Job>();
        match field {
            0 => j.payload.op as usize,
            1 => j.units,
            2 => j.chunk,
            3..=8 => j.payload.dims[field - 3],
            _ => 0,
        }
    }
    /// # Safety
    /// A live descriptor after all ranks have joined, rank < actual threads.
    #[no_mangle]
    pub unsafe extern "C" fn encoder_bench_stat(job: *const u8, rank: usize, field: usize) -> f64 {
        let r = &(*job.cast::<Job>()).bench.ranks[rank];
        match field {
            0 => r.units.load(Ordering::Relaxed) as f64,
            1 => r.chunks.load(Ordering::Relaxed) as f64,
            _ => r.requested_ns.load(Ordering::Relaxed) as f64,
        }
    }
}
