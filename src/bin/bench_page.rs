//! A page from masks to MusicXML (page analysis, staff images, replayed
//! transformer output, MusicXML), `reps` times, for instruction counts by
//! difference. Builds for WASI: cargo build --release --bin bench_page --target wasm32-wasip1
//!   REPLAY_DIR=<tools/pyodide_replay.mjs output> bench_page <reps> <page dir>:<decoder dir> ...
use homr_rs::page::{analyse_page, PageMasks};
use homr_rs::score::{page_xml, PageMapping};
use homr_rs::tokens::Symbol;
use homr_rs::warp::Gray;
use serde_json::Value;
use std::fs;

fn main() {
    let args: Vec<String> = std::env::args().collect();
    let reps: usize = args[1].parse().unwrap();
    let mut ok = 0;
    for pair in &args[2..] {
        let (pd, dd) = pair.split_once(':').unwrap();
        let page: Value = serde_json::from_str(&fs::read_to_string(format!("{pd}/page.json")).unwrap()).unwrap();
        let sh: Vec<usize> = page["shape"].as_array().unwrap().iter().map(|v| v.as_u64().unwrap() as usize).collect();
        let ps: Vec<usize> = page["preprocessed_shape"].as_array().unwrap().iter().map(|v| v.as_u64().unwrap() as usize).collect();
        let rd = |n: &str| fs::read(format!("{pd}/{n}.bin")).unwrap();
        let (nh, sy, st, ck, sr) = (rd("notehead"), rd("symbols"), rd("staff"), rd("clefs_keys"), rd("stem_rests"));
        let masks = PageMasks { h: sh[0], w: sh[1], notehead: &nh, symbols: &sy, staff: &st, clefs_keys: &ck, stem_rests: &sr };
        let image = Gray { w: ps[1], h: ps[0], data: rd("preprocessed") };
        let dec: Value = serde_json::from_str(&fs::read_to_string(format!("{dd}/decoder.json")).unwrap()).unwrap();
        let cs: Vec<usize> = dec["crop"]["cropped_shape"].as_array().unwrap().iter().map(|v| v.as_u64().unwrap() as usize).collect();
        let tl: Vec<f64> = dec["crop"]["top_left"].as_array().unwrap().iter().map(|v| v.as_f64().unwrap()).collect();
        let mapping = PageMapping { cropped: (cs[0], cs[1]), preprocessed: (ps[0], ps[1]), crop: (tl[0], tl[1]) };
        let recorded: Vec<Vec<Symbol>> = dec["calls"].as_array().unwrap().iter().map(|c| c.as_array().unwrap().iter().map(|s| {
            let s = s.as_array().unwrap();
            let t = |i: usize| s[i].as_str().unwrap().to_string();
            Symbol { rhythm: t(0), pitch: t(1), lift: t(2), articulation: t(3), slur: t(4), position: t(5), coordinates: s[6].as_array().map(|a| a.iter().map(|v| v.as_f64().unwrap()).collect()), image_coordinates: None, extra: vec![] }
        }).collect()).collect();
        let mut last = None;
        for _ in 0..reps {
            let mut calls = recorded.clone();
            calls.reverse();
            let mut predict = |_: &Gray| -> Result<Vec<Symbol>, String> { calls.pop().ok_or_else(|| "ran out".to_string()) };
            last = Some(analyse_page(&masks).and_then(|m| page_xml(&m, &image, &mapping, &mut predict)));
        }
        if let Some(Ok(xml)) = last {
            let name = std::path::Path::new(dd).file_name().unwrap().to_string_lossy().to_string();
            let replays = std::env::var("REPLAY_DIR").unwrap_or_default();
            if fs::read_to_string(format!("{replays}/{name}/replay.musicxml")).map(|e| e == xml).unwrap_or(false) {
                ok += 1;
            }
        }
    }
    println!("rust: {ok} of {} pages same as homr", args.len() - 2);
}
