//! Preprocessing and segmentation post-processing against homr in Pyodide
//! (tools/dump_segnet.py + tools/pyodide_pre.mjs): resized page, CLAHE image,
//! merged class map, and the final masks after noise filtering.
use homr_rs::preprocess::{masks, merge_tiles, postprocess, preprocess, tile_input, tile_origins, Bgr};
use serde_json::Value;
use std::fs;

fn report(label: &str, ours: &[u8], theirs: &[u8], problems: &mut Vec<String>) {
    if ours.len() != theirs.len() {
        problems.push(format!("{label}: {} vs {} bytes", ours.len(), theirs.len()));
    } else {
        let n = ours.iter().zip(theirs).filter(|(a, b)| a != b).count();
        if n > 0 {
            problems.push(format!("{label}: {n} of {} differ", ours.len()));
        }
    }
}

fn main() {
    let seg = std::env::args().nth(1).expect("segnet dir");
    let refs = std::env::args().nth(2).expect("reference dir");
    let mut dirs: Vec<_> = fs::read_dir(&refs).unwrap().map(|e| e.unwrap().path()).filter(|p| p.join("pre.json").exists()).collect();
    dirs.sort();
    let (mut same, mut diff) = (0, 0);
    for r in dirs {
        let name = r.file_name().unwrap().to_string_lossy().to_string();
        let d = std::path::Path::new(&seg).join(&name);
        let meta: Value = serde_json::from_str(&fs::read_to_string(d.join("segnet.json")).unwrap()).unwrap();
        let sh: Vec<usize> = meta["image_shape"].as_array().unwrap().iter().map(|v| v.as_u64().unwrap() as usize).collect();
        let img = Bgr { w: sh[1], h: sh[0], data: fs::read(d.join("image.bin")).unwrap() };
        let mut problems = Vec::new();
        let mut pre = preprocess(&img);
        report("resized", &pre.image.data, &fs::read(r.join("resized.bin")).unwrap(), &mut problems);
        report("clahe", &pre.gray.data, &fs::read(r.join("preprocessed.bin")).unwrap(), &mut problems);
        // (the native segnet input is not a reference: native OpenCV's CLAHE
        // uses fused multiply-add, so it differs from Pyodide's in a few pixels)
        let (h, w) = (pre.gray.h, pre.gray.w);
        let origins = tile_origins(h, w);
        let n = meta["tiles"].as_u64().unwrap() as usize;
        if origins.len() != n {
            problems.push(format!("{} tiles, homr {n}", origins.len()));
        }
        let _ = tile_input(&pre.gray, origins[0].0, origins[0].1);
        let tiles_raw = fs::read(d.join("tiles.bin")).unwrap();
        let classes: Vec<Vec<u8>> = tiles_raw.chunks_exact(320 * 320).map(|c| c.to_vec()).collect();
        let merged = merge_tiles(&classes, h, w);
        report("merged", &merged, &fs::read(r.join("merged.bin")).unwrap(), &mut problems);
        let m = postprocess(masks(&merged, h, w), &mut pre.gray);
        for (label, ours) in [("staff", &m.staff), ("symbols", &m.symbols), ("notehead", &m.notehead), ("clefs_keys", &m.clefs_keys), ("stems_rest", &m.stems_rest)] {
            report(label, ours, &fs::read(r.join(format!("final_{label}.bin"))).unwrap(), &mut problems);
        }
        report("filtered page", &pre.gray.data, &fs::read(r.join("final_preprocessed.bin")).unwrap(), &mut problems);
        if problems.is_empty() {
            same += 1;
        } else {
            diff += 1;
            println!("DIFF {name}: {}", problems.join("; "));
        }
    }
    println!("{same} pages same, {diff} different");
}
