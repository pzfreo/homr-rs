//! A small C ABI for JavaScript hosts; no wasm-bindgen dependency. All pointers
//! refer to this module's linear memory. Decode returns 32 bytes per step:
//! six little-endian u32 token IDs followed by two float32 coordinates.
use super::{Decoder, DIM, MAX_STEPS};

/// Allocate a byte buffer owned by the host until `decoder_free`.
#[no_mangle]
pub extern "C" fn decoder_alloc(len: usize) -> *mut u8 {
    let layout = std::alloc::Layout::from_size_align(len.max(1), 16).unwrap();
    // SAFETY: layout is nonempty and valid; the corresponding free uses it verbatim.
    let ptr = unsafe { std::alloc::alloc_zeroed(layout) };
    if ptr.is_null() {
        std::alloc::handle_alloc_error(layout);
    }
    ptr
}
/// Release a buffer allocated by `decoder_alloc`.
/// # Safety
/// `ptr,len` must describe a live allocation from `decoder_alloc`, exactly once.
#[no_mangle]
pub unsafe extern "C" fn decoder_free(ptr: *mut u8, len: usize) {
    std::alloc::dealloc(
        ptr,
        std::alloc::Layout::from_size_align(len.max(1), 16).unwrap(),
    );
}
/// Load and prepack a model. Returns null on an unsupported/invalid model.
/// # Safety
/// `ptr,len` must describe readable bytes in linear memory for this call.
#[no_mangle]
pub unsafe extern "C" fn decoder_load(ptr: *const u8, len: usize) -> *mut Decoder {
    match Decoder::from_onnx(std::slice::from_raw_parts(ptr, len)) {
        Ok(model) => Box::into_raw(Box::new(model)),
        Err(_) => std::ptr::null_mut(),
    }
}
/// Release a model handle.
/// # Safety
/// `model` must be a live handle returned by `decoder_load`, exactly once.
#[no_mangle]
pub unsafe extern "C" fn decoder_drop(model: *mut Decoder) {
    drop(Box::from_raw(model));
}
/// Decode a staff; returns step count (including EOS), or zero on invalid input.
/// With `fixed != 0`, keep going after EOS (feed chord), for equal-work benchmarks.
/// # Safety
/// `model` must be live; `context` must contain 1280*512 aligned float32 values;
/// `output` must have room for `steps*32` bytes. Buffers must not alias the model.
#[no_mangle]
pub unsafe extern "C" fn decoder_run(
    model: *const Decoder,
    context: *const f32,
    output: *mut u8,
    steps: usize,
    fixed: u32,
) -> usize {
    if steps > MAX_STEPS {
        return 0;
    }
    let _all = super::profile::enter(super::profile::Phase::Other);
    let model = &*model;
    let Ok(mut session) = model.session(std::slice::from_raw_parts(context, 1280 * DIM)) else {
        return 0;
    };
    let mut last = [1, 0, 0, 0, 0];
    for i in 0..steps {
        let Ok(result) = session.step(last) else {
            return 0;
        };
        for (j, &t) in result.tokens.iter().enumerate() {
            std::ptr::copy_nonoverlapping(
                (t as u32).to_le_bytes().as_ptr(),
                output.add(i * 32 + j * 4),
                4,
            );
        }
        for (j, &a) in result.attention.iter().enumerate() {
            std::ptr::copy_nonoverlapping(
                a.to_le_bytes().as_ptr(),
                output.add(i * 32 + 24 + j * 4),
                4,
            );
        }
        let t = result.tokens;
        if t[0] == 2 && fixed == 0 {
            return i + 1;
        }
        last = [if t[0] == 2 { 3 } else { t[0] }, t[1], t[2], t[4], t[5]];
    }
    steps
}

#[cfg(feature = "decoder-profile")]
#[no_mangle]
pub extern "C" fn decoder_profile_reset() {
    super::profile::reset();
}
#[cfg(feature = "decoder-profile")]
#[no_mangle]
pub extern "C" fn decoder_profile_stop() {
    super::profile::stop();
}
#[cfg(feature = "decoder-profile")]
#[no_mangle]
pub extern "C" fn decoder_profile_value(id: usize, calls: u32) -> f64 {
    let (t, n) = super::profile::values();
    if calls == 0 {
        t[id] as f64
    } else {
        n[id] as f64
    }
}
#[cfg(feature = "decoder-profile")]
#[no_mangle]
pub extern "C" fn decoder_profile_calibrate() -> f64 {
    super::profile::calibrate()
}

/// Reports whether this optional relaxed-SIMD artifact observes fused arithmetic.
/// Hosts must still validate their engine; relaxed SIMD permits unfused results.
#[no_mangle]
pub extern "C" fn decoder_fused_simd_supported() -> u32 {
    super::fused_simd_supported() as u32
}
