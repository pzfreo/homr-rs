//! The Rust segmentation against native onnxruntime's: for each page of the dumps from
//! tools/dump_segnet.py, the page homr gave segnet, cut into tiles, run here, and the class maps
//! compared with the ones onnxruntime gave. Prints how many pixels differ and how many instructions
//! a tile takes (TILES=<n> runs only the first n tiles of each page).
//!   check_segnet <segnet dumps> <segnet.onnx> [pages...]
use homr_rs::segnet::Segnet;
use homr_rs::warp::Gray;
use serde_json::Value;
use std::fs;
use std::path::Path;
#[cfg(target_os = "macos")]
extern "C" {
    fn proc_pid_rusage(pid: i32, flavor: i32, buffer: *mut u64) -> i32;
    fn getpid() -> i32;
}

/// Instructions this process has retired (macOS only, 0 elsewhere): a busy machine doesn't change them.
#[cfg(not(target_os = "macos"))]
fn instructions() -> u64 {
    0
}

#[cfg(target_os = "macos")]
fn instructions() -> u64 {
    let mut buf = [0u64; 96];
    assert_eq!(unsafe { proc_pid_rusage(getpid(), 4, buf.as_mut_ptr()) }, 0);
    buf[2 + 29]
}

fn main() {
    let args: Vec<String> = std::env::args().collect();
    let dir = Path::new(&args[1]);
    let net = Segnet::from_onnx(&fs::read(&args[2]).unwrap()).unwrap();
    let mut names: Vec<String> = if args.len() > 3 {
        args[3..].to_vec()
    } else {
        fs::read_dir(dir)
            .unwrap()
            .map(|e| e.unwrap().file_name().to_string_lossy().to_string())
            .collect()
    };
    names.sort();
    let (mut pixels, mut wrong, mut tiles, mut wrong_tiles, mut pages, mut worst_page) =
        (0usize, 0usize, 0usize, 0usize, 0usize, 0.0f64);
    let mut retired = 0u64;
    let mut run_tiles = 0usize;
    for name in names {
        let d = dir.join(&name);
        let Ok(meta) = fs::read_to_string(d.join("segnet.json")) else {
            continue;
        };
        let meta: Value = serde_json::from_str(&meta).unwrap();
        let shape: Vec<usize> = meta["merged_shape"]
            .as_array()
            .unwrap()
            .iter()
            .map(|v| v.as_u64().unwrap() as usize)
            .collect();
        let data = fs::read(d.join("segnet_input.bin")).unwrap();
        if data.len() != shape[0] * shape[1] * shape.get(2).copied().unwrap_or(1) {
            println!(
                "SKIP {name}: segnet input of {} bytes for {shape:?}",
                data.len()
            );
            continue;
        }
        let page = Gray {
            h: shape[0],
            w: shape[1],
            data: data[..shape[0] * shape[1]].to_vec(),
        };
        let start = instructions();
        // the tiles, spread over threads (THREADS, 8 by default)
        let limit = std::env::var("TILES")
            .ok()
            .and_then(|t| t.parse::<usize>().ok());
        let origins: Vec<(i64, i64)> = homr_rs::preprocess::tile_origins(page.h, page.w)
            .into_iter()
            .take(limit.unwrap_or(usize::MAX))
            .collect();
        let threads: usize = std::env::var("THREADS")
            .ok()
            .and_then(|t| t.parse().ok())
            .unwrap_or(8);
        let per = origins.len().div_ceil(threads).max(1);
        let ours: Vec<Vec<u8>> = std::thread::scope(|scope| {
            let handles: Vec<_> = origins
                .chunks(per)
                .map(|chunk| {
                    let (net, page) = (&net, &page);
                    scope.spawn(move || {
                        chunk
                            .iter()
                            .map(|&(y, x)| {
                                net.classes(&homr_rs::preprocess::tile_input(page, y, x))
                                    .unwrap()
                            })
                            .collect::<Vec<_>>()
                    })
                })
                .collect();
            handles
                .into_iter()
                .flat_map(|h| h.join().unwrap())
                .collect()
        });
        retired += instructions() - start;
        run_tiles += ours.len();
        let theirs = fs::read(d.join("tiles.bin")).unwrap();
        let n = meta["tiles"].as_u64().unwrap() as usize;
        if ours.len() != n && std::env::var("TILES").is_err() {
            println!("DIFF {name}: {} tiles, onnxruntime {n}", ours.len());
            continue;
        }
        let (mut w, mut wt) = (0usize, 0usize);
        for (t, mine) in ours.iter().enumerate() {
            let other = &theirs[t * 320 * 320..(t + 1) * 320 * 320];
            let d = mine.iter().zip(other).filter(|(a, b)| a != b).count();
            w += d;
            wt += (d > 0) as usize;
        }
        let total = n * 320 * 320;
        pixels += total;
        wrong += w;
        tiles += n;
        wrong_tiles += wt;
        pages += 1;
        worst_page = worst_page.max(w as f64 / total as f64);
        println!("{name}: {n} tiles, {w} of {total} pixels differ ({wt} tiles)");
    }
    println!(
        "{pages} pages, {tiles} tiles: {wrong} of {pixels} pixels differ ({:.6}%), {wrong_tiles} tiles with any difference, worst page {:.5}%; {:.0} M instructions a tile",
        100.0 * wrong as f64 / pixels as f64,
        100.0 * worst_page,
        retired as f64 / run_tiles.max(1) as f64 / 1e6
    );
}
