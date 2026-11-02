//! Whole pages to MusicXML against homr in Pyodide, with the transformer's
//! output replayed (tools/dump_decoder.py + tools/pyodide_replay.mjs):
//! masks and preprocessed page -> page analysis -> staff images -> recorded
//! symbols -> homr's MusicXML, compared byte for byte.
use homr_rs::page::{analyse_page, PageMasks};
use homr_rs::score::{page_xml, PageMapping};
use homr_rs::tokens::Symbol;
use homr_rs::warp::Gray;
use serde_json::Value;
use std::fs;

fn main() {
    let pages = std::env::args().nth(1).expect("pages dir (tools/dump_page.py)");
    let decs = std::env::args().nth(2).expect("decoder dir (tools/dump_decoder.py)");
    let refs = std::env::args().nth(3).expect("replay dir (tools/pyodide_replay.mjs)");
    // pages homr replayed to MusicXML, and pages where homr itself fails
    // (decoder.json has "error"): the port must fail the same way
    let mut names: Vec<String> = fs::read_dir(&decs).unwrap().map(|e| e.unwrap().file_name().to_string_lossy().to_string()).collect();
    names.sort();
    let (mut same, mut diff, mut same_err) = (0, 0, 0);
    for name in names {
        let r = std::path::Path::new(&refs).join(&name);
        let dec_path = std::path::Path::new(&decs).join(&name).join("decoder.json");
        if !dec_path.exists() {
            continue;
        }
        let dec: Value = serde_json::from_str(&fs::read_to_string(&dec_path).unwrap()).unwrap();
        let homr_error = dec["error"].as_str().map(|s| s.to_string());
        if homr_error.is_none() && !r.join("replay.musicxml").exists() {
            continue;
        }
        let d = std::path::Path::new(&pages).join(&name);
        let page: Value = serde_json::from_str(&fs::read_to_string(d.join("page.json")).unwrap()).unwrap();
        let sh: Vec<usize> = page["shape"].as_array().unwrap().iter().map(|v| v.as_u64().unwrap() as usize).collect();
        let ps: Vec<usize> = page["preprocessed_shape"].as_array().unwrap().iter().map(|v| v.as_u64().unwrap() as usize).collect();
        let rd = |n: &str| fs::read(d.join(format!("{n}.bin"))).unwrap();
        let (nh, sy, st, ck, sr) = (rd("notehead"), rd("symbols"), rd("staff"), rd("clefs_keys"), rd("stem_rests"));
        let masks = PageMasks { h: sh[0], w: sh[1], notehead: &nh, symbols: &sy, staff: &st, clefs_keys: &ck, stem_rests: &sr };
        let image = Gray { w: ps[1], h: ps[0], data: rd("preprocessed") };
        let cs: Vec<usize> = dec["crop"]["cropped_shape"].as_array().unwrap().iter().map(|v| v.as_u64().unwrap() as usize).collect();
        let tl: Vec<f64> = dec["crop"]["top_left"].as_array().unwrap().iter().map(|v| v.as_f64().unwrap()).collect();
        let mapping = PageMapping { cropped: (cs[0], cs[1]), preprocessed: (ps[0], ps[1]), crop: (tl[0], tl[1]) };
        let mut calls: Vec<Vec<Symbol>> = dec["calls"]
            .as_array()
            .unwrap()
            .iter()
            .map(|c| {
                c.as_array()
                    .unwrap()
                    .iter()
                    .map(|s| {
                        let s = s.as_array().unwrap();
                        let st = |i: usize| s[i].as_str().unwrap().to_string();
                        Symbol {
                            rhythm: st(0),
                            pitch: st(1),
                            lift: st(2),
                            articulation: st(3),
                            slur: st(4),
                            position: st(5),
                            coordinates: s[6].as_array().map(|a| a.iter().map(|v| v.as_f64().unwrap()).collect()),
                            image_coordinates: None,
                            extra: vec![],
                        }
                    })
                    .collect()
            })
            .collect();
        calls.reverse();
        let mut predict = |_: &Gray| -> Result<Vec<Symbol>, String> { calls.pop().ok_or_else(|| "replay ran out".to_string()) };
        let result = analyse_page(&masks).and_then(|multi| page_xml(&multi, &image, &mapping, &mut predict));
        if let Some(e) = homr_error {
            match result {
                Err(r) if r == e && calls.is_empty() => same_err += 1,
                Err(r) => {
                    diff += 1;
                    println!("DIFF {name}: homr {e}, rust {r}, {} recorded staffs unused", calls.len());
                }
                Ok(_) => {
                    diff += 1;
                    println!("DIFF {name}: homr {e}, rust succeeded");
                }
            }
            continue;
        }
        let expected = fs::read_to_string(r.join("replay.musicxml")).unwrap();
        match result {
            Ok(xml) if xml == expected && calls.is_empty() => same += 1,
            Ok(xml) => {
                diff += 1;
                let at = xml.bytes().zip(expected.bytes()).position(|(a, b)| a != b).unwrap_or(xml.len().min(expected.len()));
                let lo = at.saturating_sub(150);
                println!("DIFF {name}: {} vs {} bytes, {} recorded staffs unused; first difference at {at}:\n  rust: {}\n  homr: {}", xml.len(), expected.len(), calls.len(),
                    &xml[lo..(at + 150).min(xml.len())], &expected[lo..(at + 150).min(expected.len())]);
            }
            Err(e) => {
                diff += 1;
                println!("DIFF {name}: {e}");
            }
        }
    }
    println!("{same} pages same, {same_err} failing in homr and the port alike, {diff} different");
}
