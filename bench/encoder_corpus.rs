//! Full native encoder corpus, including all staffs on native failure pages.
use homr_rs::encoder::Encoder;
use serde_json::{json, Value};
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
fn instructions() -> u64 {
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
fn phase(name: &str) -> &'static str {
    if name == "output" {
        "position"
    } else if name.contains("/stem/stem.0") {
        "stem_conv"
    } else if name.contains("downsample.1") {
        "downsample_conv"
    } else if name.contains("conv_dw") {
        "depthwise"
    } else if name.contains("LayerNormalization") {
        "norm"
    } else if name.contains("fc1") {
        "fc1"
    } else if name.contains("fc2") {
        "fc2"
    } else if name.contains("/act/") {
        "gelu"
    } else if name.contains("/proj/") {
        "projection"
    } else if name.ends_with("/Mul") {
        "scale"
    } else {
        "residual"
    }
}
fn main() -> Result<(), Box<dyn std::error::Error>> {
    let a: Vec<_> = std::env::args().collect();
    let model = Encoder::from_onnx(&fs::read(&a[1])?)?;
    let root = Path::new(&a[2]);
    let out = Path::new(&a[3]);
    let corpus = fs::read_to_string(&a[4])?;
    let mut rows = Vec::new();
    let mut different = 0;
    for image in corpus.lines() {
        let page = Path::new(image).file_stem().unwrap().to_str().unwrap();
        let meta: Value = serde_json::from_slice(&fs::read(root.join(page).join("models.json"))?)?;
        let mut staffs = Vec::new();
        for staff in 0..meta["staffs"].as_u64().unwrap() {
            let d = root.join(page).join(staff.to_string());
            let input = floats(&d.join("input.f32"));
            let reference = floats(&d.join("context.bin"));
            let mut counts = std::collections::BTreeMap::<&str, u64>::new();
            let start = instructions();
            let mut previous = start;
            let got = model.encode_observed(&input, |name, _, _, _| {
                let now = instructions();
                *counts.entry(phase(name)).or_default() += now - previous;
                previous = instructions();
            })?;
            let count = instructions() - start;
            let bits = got
                .iter()
                .zip(&reference)
                .filter(|(a, b)| a.to_bits() != b.to_bits())
                .count();
            let max = got
                .iter()
                .zip(&reference)
                .map(|(a, b)| (*a as f64 - *b as f64).abs())
                .fold(0.0f64, f64::max);
            different += usize::from(bits != 0);
            let dst = out.join(page).join(staff.to_string());
            fs::create_dir_all(&dst)?;
            fs::write(
                dst.join("context.bin"),
                got.iter().flat_map(|v| v.to_le_bytes()).collect::<Vec<_>>(),
            )?;
            let row = json!({"staff":staff,"values":got.len(),"bit_differences":bits,"max_abs":max,"instructions":count,"phases":counts});
            println!("{page}/{staff}: {row}");
            staffs.push(row);
        }
        let row = json!({"page":page,"encoder":staffs});
        fs::write(
            out.join(page).join("metrics.json"),
            serde_json::to_vec(&row)?,
        )?;
        rows.push(row);
    }
    let report = json!({"pages":rows,"differing_staffs":different});
    fs::write(
        out.join("comparison.json"),
        serde_json::to_vec_pretty(&report)?,
    )?;
    if different != 0 {
        return Err("encoder corpus differs".into());
    }
    Ok(())
}
