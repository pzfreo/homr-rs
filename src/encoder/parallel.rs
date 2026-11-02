//! Synchronous, allocation-free jobs for a host-managed WASM worker pool.
//!
//! Only the coordinator allocates or enters the graph executor. Workers read
//! immutable inputs and borrow only their disjoint output range. A job lives
//! on the coordinator stack until dispatch has joined every worker. No K
//! reduction is divided, and every GEMM partition contains whole 4-row tiles.
#![allow(clippy::too_many_arguments)]

#[repr(C)]
#[derive(Clone, Copy)]
struct Job {
    op: u32,
    a: *const f32,
    b: *const f32,
    bias: *const f32,
    out: *mut f32,
    dims: [usize; 6],
    epsilon: f32,
}

#[link(wasm_import_module = "encoder_threads")]
extern "C" {
    /// Must publish the job, execute ranks 0..N and join before returning.
    fn dispatch(job: *const Job, units: usize);
}

fn submit(job: Job, units: usize) {
    // SAFETY: every caller validates sizes and retains its buffers. The host
    // contract joins all readers/writers before this stack job is released.
    unsafe { dispatch(&job, units) }
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
        Job {
            op: 0,
            a: a.as_ptr(),
            b: b.as_ptr(),
            bias: bias.map_or(std::ptr::null(), |v| v.as_ptr()),
            out: out.as_mut_ptr(),
            dims: [m, k, n, 0, 0, 0],
            epsilon: 0.0,
        },
        m / 4,
    );
}

pub(super) fn activation(x: &mut [f32], gelu: bool) {
    submit(
        Job {
            op: 1,
            a: std::ptr::null(),
            b: std::ptr::null(),
            bias: std::ptr::null(),
            out: x.as_mut_ptr(),
            dims: [x.len(), gelu as usize, 0, 0, 0, 0],
            epsilon: 0.0,
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
        Job {
            op: 2,
            a: a.as_ptr(),
            b: b.as_ptr(),
            bias: bias.as_ptr(),
            out: out.as_mut_ptr(),
            dims: [h, w, c, kernel, pad, 0],
            epsilon: 0.0,
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
        Job {
            op: 3,
            a: x.as_ptr(),
            b: scale.as_ptr(),
            bias: bias.as_ptr(),
            out: out.as_mut_ptr(),
            dims: [x.len() / c, c, 0, 0, 0, 0],
            epsilon,
        },
        x.len() / c,
    );
}

fn partition(units: usize, rank: usize, threads: usize) -> (usize, usize) {
    (units * rank / threads, units * (rank + 1) / threads)
}

/// Execute one rank of an encoder job. Workers must never call encoder_run,
/// decoder_alloc/free or other graph/allocator exports concurrently.
/// # Safety
/// `job` must be a live job published by this module's dispatch import, using
/// the same module and shared memory. Execute every rank exactly once, with
/// identical thread count, then join before returning from dispatch. Inputs
/// must remain immutable and workers need private stacks/TLS. No other host
/// call may mutate or release the graph/buffers while a job is outstanding.
#[no_mangle]
pub unsafe extern "C" fn encoder_work(job: *const u8, rank: usize, threads: usize) -> u32 {
    if job.is_null() || threads == 0 || threads > 32 || rank >= threads {
        return 0;
    }
    let j = job.cast::<Job>().read();
    let [m, k, n, kernel, pad, _] = j.dims;
    match j.op {
        0 => {
            let (start, end) = partition(m / 4, rank, threads);
            if start == end {
                return 1;
            }
            let rows = (end - start) * 4;
            let a = std::slice::from_raw_parts(j.a.add(start * 4 * k), rows * k);
            let b = std::slice::from_raw_parts(j.b, k * n);
            let out = std::slice::from_raw_parts_mut(j.out.add(start * 4 * n), rows * n);
            let bias = (!j.bias.is_null()).then(|| std::slice::from_raw_parts(j.bias, n));
            super::simd::product(a, b, out, rows, k, n, bias);
        }
        1 => {
            let (start, end) = partition(m / 4, rank, threads);
            let out = std::slice::from_raw_parts_mut(j.out.add(start * 4), (end - start) * 4);
            super::simd::activation(out, k != 0);
        }
        2 => {
            let (start, end) = partition(m * k, rank, threads);
            let padded = std::slice::from_raw_parts(j.a, (m + pad * 2) * (k + pad * 2) * n);
            let packed = std::slice::from_raw_parts(j.b, kernel * kernel * n);
            let bias = std::slice::from_raw_parts(j.bias, n);
            let out = std::slice::from_raw_parts_mut(j.out.add(start * n), (end - start) * n);
            super::simd::depthwise_into(padded, k, n, packed, bias, kernel, pad, start, out);
        }
        3 => {
            let (start, end) = partition(m, rank, threads);
            let input = std::slice::from_raw_parts(j.a.add(start * k), (end - start) * k);
            let scale = std::slice::from_raw_parts(j.b, k);
            let bias = std::slice::from_raw_parts(j.bias, k);
            let out = std::slice::from_raw_parts_mut(j.out.add(start * k), (end - start) * k);
            super::kernels::norm_into(input, k, scale, bias, j.epsilon, out);
        }
        _ => return 0,
    }
    1
}

#[cfg(test)]
mod tests {
    use super::partition;
    #[test]
    fn partitions_cover_uneven_tiles_without_overlap() {
        for units in [1, 2, 5, 16, 320, 5120] {
            for threads in [1, 2, 4, 6] {
                let mut last = 0;
                for rank in 0..threads {
                    let (start, end) = partition(units, rank, threads);
                    assert_eq!(start, last);
                    assert!(end >= start);
                    last = end;
                }
                assert_eq!(last, units);
            }
        }
    }
}
