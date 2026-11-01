//! From the segmentation masks (tools/dump_page.py) to staff detection's
//! inputs, compared with homr's (tools/dump_staff.py): staff-line fragments
//! (box and contour), clefs/keys and bar lines.
use homr_rs::symbols::{page_symbols, Masks, Shape};
use serde_json::Value;
use std::fs;

fn b5(s: &Shape) -> [f64; 5] {
    [s.cx, s.cy, s.w, s.h, s.angle]
}
fn v5(v: &Value) -> [f64; 5] {
    let a: Vec<f64> = v.as_array().unwrap().iter().map(|x| x.as_f64().unwrap()).collect();
    [a[0], a[1], a[2], a[3], a[4]]
}
fn main() {
    let pages = std::env::args().nth(1).expect("pages dir");
    let cases = std::env::args().nth(2).expect("staff cases dir");
    let mut dirs: Vec<_> = fs::read_dir(&pages).unwrap().map(|e| e.unwrap().path()).filter(|p| p.join("page.json").exists()).collect();
    dirs.sort();
    let (mut same, mut diff) = (0, 0);
    for d in dirs {
        let name = d.file_name().unwrap().to_string_lossy().to_string();
        let case_path = std::path::Path::new(&cases).join(&name).join("case.json");
        if !case_path.exists() {
            continue;
        }
        let page: Value = serde_json::from_str(&fs::read_to_string(d.join("page.json")).unwrap()).unwrap();
        let case: Value = serde_json::from_str(&fs::read_to_string(case_path).unwrap()).unwrap();
        let shape: Vec<usize> = page["shape"].as_array().unwrap().iter().map(|v| v.as_u64().unwrap() as usize).collect();
        let rd = |n: &str| fs::read(d.join(format!("{n}.bin"))).unwrap();
        let (nh, st, ck, sr) = (rd("notehead"), rd("staff"), rd("clefs_keys"), rd("stem_rests"));
        let m = Masks { h: shape[0], w: shape[1], notehead: &nh, staff: &st, clefs_keys: &ck, stem_rests: &sr };
        let mut problems = Vec::new();
        match page_symbols(&m) {
            Err(e) => problems.push(e),
            Ok(s) => {
                let ef = case["fragments"].as_array().unwrap();
                if s.fragments.len() != ef.len() {
                    problems.push(format!("{} fragments, homr {}", s.fragments.len(), ef.len()));
                } else if let Some(k) = (0..ef.len()).find(|&k| {
                    let c: Vec<[i32; 2]> = ef[k]["contour"].as_array().unwrap().iter().map(|p| [p[0].as_i64().unwrap() as i32, p[1].as_i64().unwrap() as i32]).collect();
                    b5(&s.fragments[k]) != v5(&ef[k]["box"]) || s.fragments[k].contour != c
                }) {
                    problems.push(format!("fragment {k}: {:?} vs {:?}", b5(&s.fragments[k]), v5(&ef[k]["box"])));
                }
                for (label, ours, key) in [("clefs", &s.clefs_keys, "clefs"), ("bars", &s.bar_lines, "bars")] {
                    let e = case[key].as_array().unwrap();
                    if ours.len() != e.len() {
                        problems.push(format!("{} {label}, homr {}", ours.len(), e.len()));
                    } else if let Some(k) = (0..e.len()).find(|&k| b5(&ours[k]) != v5(&e[k])) {
                        problems.push(format!("{label} {k}: {:?} vs {:?}", b5(&ours[k]), v5(&e[k])));
                    }
                }
            }
        }
        if problems.is_empty() {
            same += 1;
        } else {
            diff += 1;
            println!("DIFF {name}: {}", problems.join("; "));
        }
    }
    println!("{same} pages same, {diff} different");
}
