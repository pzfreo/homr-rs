//! Optional instruction accounting. Never present in ordinary decoder builds.
//! Counters include all process threads; use one decoder thread for profiling.
#[derive(Clone, Copy)]
#[repr(usize)]
pub enum Phase {
    Other,
    Context,
    SelfQkv,
    SelfOut,
    CrossQ,
    CrossOut,
    FfIn,
    FfOut,
    Heads,
    Quantize,
    Scores,
    Softmax,
    WeightedV,
    Norm,
    Activation,
    Centroid,
    Alloc,
    Copy,
    Empty,
}
pub const N: usize = 19;
pub const NAMES: [&str; N] = [
    "other",
    "context_kv",
    "self_qkv",
    "self_out",
    "cross_q",
    "cross_out",
    "ff_in",
    "ff_out",
    "heads",
    "quantize",
    "scores",
    "softmax",
    "weighted_v",
    "norm",
    "sigmoid_gelu",
    "centroid",
    "alloc_free",
    "copies",
    "empty",
];
#[cfg(feature = "decoder-profile")]
mod enabled {
    use super::{Phase, N};
    use std::cell::RefCell;
    #[derive(Clone, Copy)]
    struct Frame {
        id: usize,
        start: u64,
        children: u64,
    }
    struct State {
        enabled: bool,
        totals: [u64; N],
        calls: [u64; N],
        frames: [Frame; 32],
        depth: usize,
    }
    thread_local! { static STATE: RefCell<State> = const {RefCell::new(State { enabled:false, totals:[0;N], calls:[0;N], frames:[Frame{id:0,start:0,children:0};32], depth:0 })}; }
    #[cfg(target_arch = "wasm32")]
    #[link(wasm_import_module = "decoder_profile")]
    extern "C" {
        fn instructions() -> f64;
    }
    fn count() -> u64 {
        #[cfg(target_arch = "wasm32")]
        // SAFETY: the profiling host supplies the synchronous instruction counter.
        unsafe {
            instructions() as u64
        }
        #[cfg(not(target_arch = "wasm32"))]
        {
            extern "C" {
                fn proc_pid_rusage(pid: i32, flavor: i32, buffer: *mut u64) -> i32;
                fn getpid() -> i32;
            }
            let mut b = [0u64; 96];
            // SAFETY: the macOS V4 structure fits in the supplied buffer.
            unsafe {
                assert_eq!(proc_pid_rusage(getpid(), 4, b.as_mut_ptr()), 0);
            }
            b[31]
        }
    }
    pub struct Guard(bool);
    pub fn enter(id: Phase) -> Guard {
        let active = STATE.with(|s| s.borrow().enabled);
        if active {
            let start = count();
            STATE.with(|s| {
                let mut s = s.borrow_mut();
                let i = s.depth;
                s.frames[i] = Frame {
                    id: id as usize,
                    start,
                    children: 0,
                };
                s.depth += 1;
            });
        }
        Guard(active)
    }
    impl Drop for Guard {
        fn drop(&mut self) {
            if !self.0 {
                return;
            }
            let end = count();
            STATE.with(|s| {
                let mut s = s.borrow_mut();
                s.depth -= 1;
                let f = s.frames[s.depth];
                let elapsed = end - f.start;
                s.totals[f.id] += elapsed - f.children;
                s.calls[f.id] += 1;
                if s.depth > 0 {
                    let i = s.depth - 1;
                    s.frames[i].children += elapsed;
                }
            });
        }
    }
    pub fn reset() {
        STATE.with(|s| {
            let mut s = s.borrow_mut();
            assert_eq!(s.depth, 0);
            s.totals.fill(0);
            s.calls.fill(0);
            s.enabled = true;
        });
    }
    pub fn stop() {
        STATE.with(|s| {
            let mut s = s.borrow_mut();
            assert_eq!(s.depth, 0);
            s.enabled = false;
        });
    }
    pub fn values() -> ([u64; N], [u64; N]) {
        STATE.with(|s| {
            let s = s.borrow();
            (s.totals, s.calls)
        })
    }
    pub fn calibrate() -> f64 {
        reset();
        for _ in 0..1000 {
            std::hint::black_box(enter(Phase::Empty));
        }
        stop();
        let (t, n) = values();
        t[Phase::Empty as usize] as f64 / n[Phase::Empty as usize] as f64
    }
    // All allocator bookkeeping is allocation-free, including TLS initialization.
    struct Allocator;
    #[global_allocator]
    static ALLOC: Allocator = Allocator;
    unsafe impl std::alloc::GlobalAlloc for Allocator {
        unsafe fn alloc(&self, l: std::alloc::Layout) -> *mut u8 {
            let _g = enter(Phase::Alloc);
            std::alloc::GlobalAlloc::alloc(&std::alloc::System, l)
        }
        unsafe fn alloc_zeroed(&self, l: std::alloc::Layout) -> *mut u8 {
            let _g = enter(Phase::Alloc);
            std::alloc::GlobalAlloc::alloc_zeroed(&std::alloc::System, l)
        }
        unsafe fn dealloc(&self, p: *mut u8, l: std::alloc::Layout) {
            let _g = enter(Phase::Alloc);
            std::alloc::GlobalAlloc::dealloc(&std::alloc::System, p, l);
        }
        unsafe fn realloc(&self, p: *mut u8, l: std::alloc::Layout, n: usize) -> *mut u8 {
            let _g = enter(Phase::Alloc);
            std::alloc::GlobalAlloc::realloc(&std::alloc::System, p, l, n)
        }
    }
}
#[cfg(feature = "decoder-profile")]
pub use enabled::*;
#[cfg(not(feature = "decoder-profile"))]
pub struct Guard;
#[cfg(not(feature = "decoder-profile"))]
#[inline(always)]
pub fn enter(_: Phase) -> Guard {
    Guard
}
