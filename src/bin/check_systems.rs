//! repair_systems against homr's: the cases from tools/synth_systems.py (staffs grouped into
//! systems, and homr's repaired grouping or its error). Every staff's grid and which staffs
//! are grand staffs must be identical; the exit status is 1 if a case differs.
//!   check_systems <cases dir>
use homr_rs::braces::MultiStaff;
use homr_rs::staff::{Staff, StaffPoint};
use homr_rs::systems::repair_systems;
use serde_json::Value;
use std::fs;

fn rows(v: &Value) -> Vec<MultiStaff> {
    v.as_array()
        .unwrap()
        .iter()
        .map(|row| {
            let staffs: Vec<(Staff, bool)> = row
                .as_array()
                .unwrap()
                .iter()
                .map(|s| {
                    let grid = s["grid"]
                        .as_array()
                        .unwrap()
                        .iter()
                        .map(|p| StaffPoint {
                            x: p[0].as_f64().unwrap(),
                            y: p[1]
                                .as_array()
                                .unwrap()
                                .iter()
                                .map(|y| y.as_f64().unwrap())
                                .collect(),
                            angle: p[2].as_f64().unwrap(),
                        })
                        .collect();
                    (
                        Staff::from_grid(grid).unwrap(),
                        s["grand"].as_bool().unwrap(),
                    )
                })
                .collect();
            MultiStaff {
                grand: staffs.iter().map(|s| s.1).collect(),
                staffs: staffs.into_iter().map(|s| s.0).collect(),
            }
        })
        .collect()
}

/// Every staff as (grand, [(x, [y], angle)]), row by row.
fn shape(r: &[MultiStaff]) -> Vec<Vec<(bool, Vec<(f64, Vec<f64>, f64)>)>> {
    r.iter()
        .map(|m| {
            m.staffs
                .iter()
                .zip(&m.grand)
                .map(|(s, &g)| {
                    (
                        g,
                        s.grid.iter().map(|p| (p.x, p.y.clone(), p.angle)).collect(),
                    )
                })
                .collect()
        })
        .collect()
}

fn main() {
    let dir = std::env::args().nth(1).expect("cases dir");
    let mut files: Vec<_> = fs::read_dir(&dir)
        .unwrap()
        .map(|e| e.unwrap().path())
        .filter(|p| p.extension().is_some_and(|e| e == "json"))
        .collect();
    files.sort();
    let (mut same, mut diff, mut changed, mut errors) = (0, 0, 0, 0);
    for f in files {
        let case: Value = serde_json::from_str(&fs::read_to_string(&f).unwrap()).unwrap();
        let input = rows(&case["input"]);
        let before = shape(&input);
        let got = std::panic::catch_unwind(|| repair_systems(input));
        let name = f.file_name().unwrap().to_string_lossy().to_string();
        if case.get("error").is_some() {
            errors += 1;
            match got {
                Ok(Err(_)) | Err(_) => same += 1,
                Ok(Ok(_)) => {
                    diff += 1;
                    println!(
                        "DIFF {name}: homr failed ({}), the Rust did not",
                        case["error"]
                    );
                }
            }
            continue;
        }
        let want = shape(&rows(&case["output"]));
        match got {
            Ok(Ok(r)) if shape(&r) == want => {
                same += 1;
                changed += (want != before) as usize;
            }
            Ok(Ok(r)) => {
                diff += 1;
                let g = shape(&r);
                println!(
                    "DIFF {name}: rust {:?} staffs per system, homr {:?}",
                    g.iter()
                        .map(|x| x.iter().map(|s| s.0 as u8).collect::<Vec<_>>())
                        .collect::<Vec<_>>(),
                    want.iter()
                        .map(|x| x.iter().map(|s| s.0 as u8).collect::<Vec<_>>())
                        .collect::<Vec<_>>()
                );
            }
            Ok(Err(e)) => {
                diff += 1;
                println!("DIFF {name}: the Rust failed ({e}), homr did not");
            }
            Err(_) => {
                diff += 1;
                println!("DIFF {name}: the Rust panicked");
            }
        }
    }
    println!(
        "{same} same, {diff} different; homr changed the grouping in {changed}, failed in {errors}"
    );
    if diff > 0 {
        std::process::exit(1);
    }
}
