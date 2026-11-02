//! Warmed native retired instructions and per-operation counts.
use homr_rs::encoder::{phase, Encoder, PHASE_NAMES};
use serde_json::json;
use std::{fs, path::Path};
fn floats(p: &Path) -> Vec<f32> {
    fs::read(p)
        .unwrap()
        .as_chunks::<4>()
        .0
        .iter()
        .map(|b| f32::from_le_bytes(*b))
        .collect()
}
fn count() -> u64 {
    unsafe {
        extern "C" {
            fn proc_pid_rusage(pid: i32, flavor: i32, buffer: *mut u64) -> i32;
            fn getpid() -> i32;
        }
        let mut b = [0u64; 96];
        assert_eq!(proc_pid_rusage(getpid(), 4, b.as_mut_ptr()), 0);
        b[31]
    }
}
fn main() -> Result<(), Box<dyn std::error::Error>> {
    let a: Vec<_> = std::env::args().collect();
    let model = Encoder::from_onnx(&fs::read(&a[1])?)?;
    let d = Path::new(&a[2]);
    let input = floats(&d.join("input.f32"));
    let reference = floats(&d.join("context.bin"));
    let mut rows = Vec::new();
    model.encode(&input)?;
    for _ in 0..3 {
        let start = count();
        let mut previous = start;
        let mut phases = [0u64; 11];
        let got = model.encode_observed(&input, |name, _, _, _| {
            let now = count();
            phases[phase(name)] += now - previous;
            previous = count();
        })?;
        let total = count() - start;
        assert!(got
            .iter()
            .zip(&reference)
            .all(|(a, b)| a.to_bits() == b.to_bits()));
        rows.push(json!({"instructions":total,"phases":phases}));
    }
    let report =
        json!({"counter":"proc_pid_rusage RUSAGE_INFO_V4","runs":rows,"phase_names":PHASE_NAMES});
    println!("{report}");
    fs::write(&a[3], serde_json::to_vec_pretty(&report)?)?;
    Ok(())
}
