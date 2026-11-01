//! Compares the Rust intermediate results with homr's (mid.json from tools/dump_mid.py).
use homr_rs::geom::{RBox, NO_FRAGMENT};
use homr_rs::staff::{detect_staff_mid, Fragments, Mid};
use serde_json::Value;
use std::fs;

fn rbox(v: &Value, frag: usize) -> RBox {
    let a: Vec<f64> = v.as_array().unwrap().iter().map(|x| x.as_f64().unwrap()).collect();
    RBox::new(a[0], a[1], a[2], a[3], a[4], frag)
}
fn main() {
    let d = std::path::PathBuf::from(std::env::args().nth(1).unwrap());
    let case: Value = serde_json::from_str(&fs::read_to_string(d.join("case.json")).unwrap()).unwrap();
    let py: Value = serde_json::from_str(&fs::read_to_string(d.join("mid.json")).unwrap()).unwrap();
    let shape: Vec<usize> = case["shape"].as_array().unwrap().iter().map(|v| v.as_u64().unwrap() as usize).collect();
    let image = fs::read(d.join("staff.bin")).unwrap();
    let frags = case["fragments"].as_array().unwrap();
    let boxes: Vec<RBox> = frags.iter().enumerate().map(|(i, f)| rbox(&f["box"], i)).collect();
    let contours: Vec<Vec<[i32; 2]>> = frags.iter().map(|f| f["contour"].as_array().unwrap().iter().map(|p| [p[0].as_i64().unwrap() as i32, p[1].as_i64().unwrap() as i32]).collect()).collect();
    let clefs: Vec<RBox> = case["clefs"].as_array().unwrap().iter().map(|b| rbox(b, NO_FRAGMENT)).collect();
    let bars: Vec<RBox> = case["bars"].as_array().unwrap().iter().map(|b| rbox(b, NO_FRAGMENT)).collect();
    let f = Fragments { boxes: &boxes, contours: &contours };
    let mut mid = Some(Mid::default());
    let _ = detect_staff_mid(&image, shape[0], shape[1], &f, &clefs, &bars, &mut mid);
    let m = mid.unwrap();
    println!("others: rust {} python {}", m.others.len(), py["others"].as_array().unwrap().len());
    let pa = py["anchors"].as_array().unwrap();
    println!("anchors: rust {} python {}", m.anchors.len(), pa.len());
    for (i, (r, p)) in m.anchors.iter().zip(pa).enumerate() {
        let pb: Vec<f64> = p[0].as_array().unwrap().iter().map(|x| x.as_f64().unwrap()).collect();
        if r.0.iter().zip(&pb).any(|(a, b)| (a - b).abs() > 1e-9) || (r.1 - p[1].as_f64().unwrap()).abs() > 1e-9 {
            println!("  first anchor difference at {i}: rust {:?} {} | python {:?} {}", r.0, r.1, pb, p[1]);
            break;
        }
    }
    let pr = py["raw"].as_array().unwrap();
    println!("raw staffs: rust {} python {}", m.raw.len(), pr.len());
    for (r, p) in m.raw.iter().zip(pr) {
        let p: Vec<String> = p.as_array().unwrap().iter().map(|v| v.to_string()).collect();
        println!("  rust min_x {:.4} max_x {:.4} anchors {} | python {}", r.0, r.1, r.4, p[..5].join(" "));
    }
}
