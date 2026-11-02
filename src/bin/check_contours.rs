//! Checks find_contours against cv2.findContours on the masks dumped by
//! tools/dump_page.py (contours in the same order with the same points).
//! EXTERNAL=1 checks RETR_EXTERNAL instead of RETR_TREE.
use homr_rs::contours::{find_contours, find_external_contours};
use serde_json::Value;
use std::fs;

fn main() {
    let root = std::env::args().nth(1).expect("pages dir");
    let mut dirs: Vec<_> = fs::read_dir(&root)
        .unwrap()
        .map(|e| e.unwrap().path())
        .filter(|p| p.join("page.json").exists())
        .collect();
    dirs.sort();
    let (mut same, mut diff) = (0, 0);
    for d in dirs {
        let page: Value =
            serde_json::from_str(&fs::read_to_string(d.join("page.json")).unwrap()).unwrap();
        let shape: Vec<usize> = page["shape"]
            .as_array()
            .unwrap()
            .iter()
            .map(|v| v.as_u64().unwrap() as usize)
            .collect();
        for (name, exp) in page["contours"].as_object().unwrap() {
            let mask = fs::read(d.join(format!("{name}.bin"))).unwrap();
            let ours = if std::env::var("EXTERNAL").is_ok() {
                find_external_contours(&mask, shape[0], shape[1])
            } else {
                find_contours(&mask, shape[0], shape[1])
            };
            let exp: Vec<Vec<[i32; 2]>> = exp
                .as_array()
                .unwrap()
                .iter()
                .map(|c| {
                    c.as_array()
                        .unwrap()
                        .iter()
                        .map(|p| [p[0].as_i64().unwrap() as i32, p[1].as_i64().unwrap() as i32])
                        .collect()
                })
                .collect();
            if ours == exp {
                same += 1;
            } else {
                diff += 1;
                let first = ours.iter().zip(&exp).position(|(a, b)| a != b);
                println!(
                    "DIFF {} {name}: {} contours vs {}; first differing contour {:?}",
                    d.file_name().unwrap().to_string_lossy(),
                    ours.len(),
                    exp.len(),
                    first
                );
            }
        }
    }
    println!("{same} masks same, {diff} different");
}
