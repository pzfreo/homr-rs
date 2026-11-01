//! Stage 2 (preprocessing + segmentation post-processing) `reps` times per page,
//! for instruction counts. WASI: cargo build --release --bin bench_pre --target wasm32-wasip1
//!   bench_pre <reps> <segnet dirs (tools/dump_segnet.py)...>
use homr_rs::preprocess::{masks, merge_tiles, postprocess, preprocess, Bgr};
use serde_json::Value;
use std::fs;

fn main() {
    let args: Vec<String> = std::env::args().collect();
    let reps: usize = args[1].parse().unwrap();
    let mut n = 0;
    for d in &args[2..] {
        let meta: Value = serde_json::from_str(&fs::read_to_string(format!("{d}/segnet.json")).unwrap()).unwrap();
        let sh: Vec<usize> = meta["image_shape"].as_array().unwrap().iter().map(|v| v.as_u64().unwrap() as usize).collect();
        let img = Bgr { w: sh[1], h: sh[0], data: fs::read(format!("{d}/image.bin")).unwrap() };
        let tiles = fs::read(format!("{d}/tiles.bin")).unwrap();
        let classes: Vec<Vec<u8>> = tiles.chunks_exact(320 * 320).map(|c| c.to_vec()).collect();
        for _ in 0..reps {
            let mut pre = preprocess(&img);
            let (h, w) = (pre.gray.h, pre.gray.w);
            let merged = merge_tiles(&classes, h, w);
            let m = postprocess(masks(&merged, h, w), &mut pre.gray);
            n += m.staff.len().min(1);
        }
    }
    println!("rust: {n} runs");
}
