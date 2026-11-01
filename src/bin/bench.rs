//! Runs staff detection `reps` times on each case (for instruction counts by
//! difference), and checks the result against homr's. Builds for WASI too:
//!   cargo build --release --bin bench --target wasm32-wasip1
use homr_rs::geom::{RBox, NO_FRAGMENT};
use homr_rs::staff::{detect_staff, Fragments};
use serde_json::Value;
use std::fs;

fn rbox(v: &Value, frag: usize) -> RBox {
    let a: Vec<f64> = v.as_array().unwrap().iter().map(|x| x.as_f64().unwrap()).collect();
    RBox::new(a[0], a[1], a[2], a[3], a[4], frag)
}
fn main() {
    let args: Vec<String> = std::env::args().collect();
    let reps: usize = args[1].parse().unwrap();
    let mut ok = 0;
    for d in &args[2..] {
        let case: Value = serde_json::from_str(&fs::read_to_string(format!("{d}/case.json")).unwrap()).unwrap();
        let shape: Vec<usize> = case["shape"].as_array().unwrap().iter().map(|v| v.as_u64().unwrap() as usize).collect();
        let image = fs::read(format!("{d}/staff.bin")).unwrap();
        let frags = case["fragments"].as_array().unwrap();
        let boxes: Vec<RBox> = frags.iter().enumerate().map(|(i, f)| rbox(&f["box"], i)).collect();
        let contours: Vec<Vec<[i32; 2]>> = frags.iter().map(|f| f["contour"].as_array().unwrap().iter().map(|p| [p[0].as_i64().unwrap() as i32, p[1].as_i64().unwrap() as i32]).collect()).collect();
        let clefs: Vec<RBox> = case["clefs"].as_array().unwrap().iter().map(|b| rbox(b, NO_FRAGMENT)).collect();
        let bars: Vec<RBox> = case["bars"].as_array().unwrap().iter().map(|b| rbox(b, NO_FRAGMENT)).collect();
        let f = Fragments { boxes: &boxes, contours: &contours };
        let mut last = None;
        for _ in 0..reps {
            last = Some(detect_staff(&image, shape[0], shape[1], &f, &clefs, &bars).unwrap());
        }
        let staffs = last.unwrap();
        let exp = case["expected"].as_array().unwrap();
        let same = staffs.len() == exp.len()
            && staffs.iter().zip(exp).all(|(s, e)| {
                let e = e.as_array().unwrap();
                s.grid.len() == e.len() && s.grid.iter().zip(e).all(|(p, q)| p.y.iter().zip(q[1].as_array().unwrap()).all(|(a, b)| (a - b.as_f64().unwrap()).abs() < 1e-9))
            });
        ok += same as usize;
    }
    println!("rust: {ok} of {} cases same as homr", args.len() - 2);
}
