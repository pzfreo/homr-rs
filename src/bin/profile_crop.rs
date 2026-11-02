//! Instructions per part of autocrop for one page (tools/dump_segnet.py dir).
use homr_rs::contours::find_external_contours;
use homr_rs::preprocess::*;
use homr_rs::raster::morph;
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
    let meta: Value =
        serde_json::from_str(&fs::read_to_string(format!("{d}/segnet.json")).unwrap()).unwrap();
    let sh: Vec<usize> = meta["image_shape"]
        .as_array()
        .unwrap()
        .iter()
        .map(|v| v.as_u64().unwrap() as usize)
        .collect();
    let img = Bgr {
        w: sh[1],
        h: sh[0],
        data: fs::read(format!("{d}/image.bin")).unwrap(),
    };
    let mut t = instr();
    let mut lap = |l: &str| {
        let n = instr();
        println!("{l:20} {:>8.1}M", (n - t) as f64 / 1e6);
        t = instr();
    };
    let gray = bgr_to_gray(&img);
    lap("gray");
    let mut hist = [0u64; 256];
    for p in img.data.as_chunks::<3>().0 {
        hist[p[0] as usize] += 1;
    }
    let dominant = (0..256).fold(0, |b, i| if hist[i] > hist[b] { i } else { b }) as i64;
    let thresh: Vec<u8> = gray
        .data
        .iter()
        .map(|&v| if (v as i64) > dominant - 30 { 255 } else { 0 })
        .collect();
    lap("hist+thresh");
    let ones = |n: usize| vec![vec![1u8; n]; n];
    let a = morph(&thresh, img.h, img.w, &ones(7), true);
    lap("dilate7");
    let b = morph(&a, img.h, img.w, &ones(7), false);
    lap("erode7");
    let e = morph(&b, img.h, img.w, &ones(9), false);
    lap("erode9");
    let c = find_external_contours(&e, img.h, img.w);
    lap("contours");
    println!("{} contours", c.len());
}
