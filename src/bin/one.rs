//! Debug: the fragments one symbol (and copies moved by -10..10) touches, and how they connect.
use homr_rs::geom::RBox;
use homr_rs::staff::{debug_symbol, Fragments};
use serde_json::Value;
fn main() {
    let a: Vec<String> = std::env::args().collect();
    let case: Value =
        serde_json::from_str(&std::fs::read_to_string(format!("{}/case.json", a[1])).unwrap())
            .unwrap();
    let frags = case["fragments"].as_array().unwrap();
    let boxes: Vec<RBox> = frags
        .iter()
        .enumerate()
        .map(|(i, f)| {
            let b: Vec<f64> = f["box"]
                .as_array()
                .unwrap()
                .iter()
                .map(|x| x.as_f64().unwrap())
                .collect();
            RBox::new(b[0], b[1], b[2], b[3], b[4], i)
        })
        .collect();
    let contours: Vec<Vec<[i32; 2]>> = frags.iter().map(|_| vec![]).collect();
    let s: Vec<f64> = a[2].split(',').map(|v| v.parse().unwrap()).collect();
    let sym = RBox::new(s[0], s[1], s[2], s[3], s[4], usize::MAX);
    for (dx, ov, con) in debug_symbol(
        &sym,
        &Fragments {
            boxes: &boxes,
            contours: &contours,
        },
    ) {
        println!("{dx} overlap {ov:?}\n   connected {con:?}");
    }
}
