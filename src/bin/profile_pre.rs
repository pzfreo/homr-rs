//! Instructions per part of stage 2 for one page (tools/dump_segnet.py dir).
use homr_rs::preprocess::*;
use serde_json::Value;
use std::fs;
extern "C" {
    fn proc_pid_rusage(pid: i32, flavor: i32, buffer: *mut u64) -> i32;
    fn getpid() -> i32;
}
fn instr() -> u64 {
    let mut buf = [0u64; 96];
    unsafe { assert_eq!(proc_pid_rusage(getpid(), 4, buf.as_mut_ptr()), 0) };
    buf[2 + 29]
}
fn main() {
    let d = std::env::args().nth(1).unwrap();
    let meta: Value = serde_json::from_str(&fs::read_to_string(format!("{d}/segnet.json")).unwrap()).unwrap();
    let sh: Vec<usize> = meta["image_shape"].as_array().unwrap().iter().map(|v| v.as_u64().unwrap() as usize).collect();
    let img = Bgr { w: sh[1], h: sh[0], data: fs::read(format!("{d}/image.bin")).unwrap() };
    let tiles = fs::read(format!("{d}/tiles.bin")).unwrap();
    let classes: Vec<Vec<u8>> = tiles.chunks_exact(320 * 320).map(|c| c.to_vec()).collect();
    let mut t = instr();
    let mut lap = |l: &str| {
        let n = instr();
        println!("{l:20} {:>8.1}M", (n - t) as f64 / 1e6);
        t = instr();
    };
    let (c, _) = autocrop(&img);
    lap("autocrop");
    let r = resize_image(&c);
    lap("pillow resize");
    let g = bgr_to_gray(&r);
    lap("gray");
    let mut cl = clahe(&g);
    lap("clahe");
    let merged = merge_tiles(&classes, cl.h, cl.w);
    lap("merge");
    let m = masks(&merged, cl.h, cl.w);
    lap("masks");
    let _ = noise_mask(&m.staff, m.h, m.w);
    lap("noise mask");
    let _ = postprocess(m, &mut cl);
    lap("postprocess (all)");
    println!("image {}x{} -> {}x{}", img.w, img.h, r.w, r.h);
}
