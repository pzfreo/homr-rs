//! np_argsort against numpy's: the arrays and numpy's argsort of them from
//! tools/argsort_cases.py.   check_argsort <cases.json> <native.json>
use homr_rs::npmath::np_argsort;
use serde_json::Value;
use std::fs;

fn main() {
    let args: Vec<String> = std::env::args().collect();
    let cases: Value = serde_json::from_str(&fs::read_to_string(&args[1]).unwrap()).unwrap();
    let cases: Vec<Vec<f64>> = cases["values"]
        .as_array()
        .unwrap()
        .iter()
        .map(|c| {
            c.as_array()
                .unwrap()
                .iter()
                .map(|v| v.as_f64().unwrap_or(f64::NAN))
                .collect()
        })
        .collect();
    for (name, path) in args[2..].iter().map(|p| (p.clone(), p)) {
        let r: Value = serde_json::from_str(&fs::read_to_string(path).unwrap()).unwrap();
        let (mut same, mut diff) = (0, 0);
        for (c, want) in cases.iter().zip(r["order"].as_array().unwrap()) {
            let want: Vec<usize> = want
                .as_array()
                .unwrap()
                .iter()
                .map(|v| v.as_u64().unwrap() as usize)
                .collect();
            if np_argsort(c) == want {
                same += 1;
            } else {
                diff += 1;
                if std::env::var("SHOW").is_ok() && c.len() <= 40 {
                    println!(
                        "  n={} values {:?}\n   rust  {:?}\n   numpy {:?}",
                        c.len(),
                        c,
                        np_argsort(c),
                        want
                    );
                }
            }
        }
        println!(
            "{name} (numpy {}): identical {same}, different {diff}",
            r["numpy"]
        );
    }
}
