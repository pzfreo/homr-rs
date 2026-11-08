//! parse_staffs against homr's: cases from `PARSE=1 tools/synth_systems.py` (systems of staffs,
//! with the symbols the stand-in transformer gave each staff, in homr's calling order,
//! and the voices homr assembled from them). The Rust gets the repaired systems; the staffs
//! it asks for, in which order, and the voices it assembles must be identical.
//!   check_parse <cases dir>
use homr_rs::braces::MultiStaff;
use homr_rs::score::parse_staffs_with;
use homr_rs::staff::{Staff, StaffPoint};
use homr_rs::tokens::Symbol;
use serde_json::Value;
use std::fs;

fn staffs(v: &Value) -> Vec<MultiStaff> {
    v.as_array()
        .unwrap()
        .iter()
        .map(|row| {
            let items: Vec<(Staff, bool)> = row
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
                grand: items.iter().map(|s| s.1).collect(),
                staffs: items.into_iter().map(|s| s.0).collect(),
            }
        })
        .collect()
}

fn symbol(v: &Value) -> Symbol {
    let s = |k: &str| v[k].as_str().unwrap().to_string();
    let mut sym = Symbol::new(&s("rhythm"));
    sym.pitch = s("pitch");
    sym.lift = s("lift");
    sym.articulation = s("articulation");
    sym.slur = s("slur");
    sym.position = s("position");
    sym
}

/// One variant of a case: the systems, the calls homr's stand-in transformer got, and the
/// voices (or the error) homr gave. Returns what differs.
fn check(
    rows: &[MultiStaff],
    calls: &[Value],
    voices: Option<&Value>,
    error: Option<&Value>,
) -> (Vec<String>, bool) {
    let mut asked: Vec<String> = Vec::new();
    let asked_count = std::cell::Cell::new(0usize);
    let got = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
        parse_staffs_with(rows.to_vec(), &mut |i, staff, grand, _| {
            asked_count.set(asked_count.get() + 1);
            let call = calls.get(i).ok_or("asked for more staffs than homr")?;
            let want = call["staff"].as_array().unwrap();
            let fingerprint = [staff.min_x, staff.max_x, staff.min_y, staff.max_y];
            if fingerprint
                .iter()
                .zip(want)
                .any(|(a, b)| *a != b.as_f64().unwrap())
                || grand != want[4].as_bool().unwrap()
            {
                asked.push(format!(
                    "call {i}: staff {fingerprint:?} grand {grand}, homr asked for {want:?}"
                ));
            }
            Ok(call["symbols"]
                .as_array()
                .unwrap()
                .iter()
                .map(symbol)
                .collect())
        })
    }));
    let mut problems = asked;
    let mut hidden = false;
    if let Some(error) = error {
        match got {
            Ok(Err(_)) => {}
            Ok(Ok(_)) => problems.push(format!("homr failed ({error}), the Rust did not")),
            Err(_) => problems.push(format!("homr failed ({error}), the Rust panicked")),
        }
    } else {
        match got {
            Ok(Ok(got)) => {
                if asked_count.get() != calls.len() {
                    problems.push(format!(
                        "asked for {} staffs, homr for {}",
                        asked_count.get(),
                        calls.len()
                    ));
                }
                let want: Vec<Vec<String>> = voices
                    .unwrap()
                    .as_array()
                    .unwrap()
                    .iter()
                    .map(|v| {
                        v.as_array()
                            .unwrap()
                            .iter()
                            .map(|s| symbol(s).key())
                            .collect()
                    })
                    .collect();
                let got_keys: Vec<Vec<String>> = got
                    .iter()
                    .map(|v| v.iter().map(|s| s.key()).collect())
                    .collect();
                if got_keys != want {
                    problems.push(format!(
                        "voices differ ({} voices vs homr's {})",
                        got_keys.len(),
                        want.len()
                    ));
                }
                hidden = got.iter().flatten().any(|s| s.rhythm == "measureRest");
            }
            Ok(Err(e)) => problems.push(format!("the Rust failed ({e}), homr did not")),
            Err(_) => problems.push("the Rust panicked".into()),
        }
    }
    (problems, hidden)
}

fn main() {
    let dir = std::env::args().nth(1).expect("cases dir");
    let mut files: Vec<_> = fs::read_dir(&dir)
        .unwrap()
        .map(|e| e.unwrap().path())
        .filter(|p| p.extension().is_some_and(|e| e == "json"))
        .collect();
    files.sort();
    let (mut same, mut diff, mut skipped, mut hidden, mut errors) = (0, 0, 0, 0, 0);
    for f in files {
        let case: Value = serde_json::from_str(&fs::read_to_string(&f).unwrap()).unwrap();
        let name = f.file_name().unwrap().to_string_lossy().to_string();
        if case.get("output").is_none() || case.get("calls").is_none() {
            skipped += 1; // homr failed before parse_staffs (system repair)
            continue;
        }
        // as repaired (the pipeline), and as detected (parse_staffs alone, whatever the layout)
        for (key, systems) in [("", "output"), ("raw_", "input")] {
            let rows = staffs(&case[systems]);
            let calls = case[format!("{key}calls")].as_array().unwrap();
            let (problems, had_rests) = check(
                &rows,
                calls,
                case.get(format!("{key}voices")),
                case.get(format!("{key}parse_error")),
            );
            errors += case.get(format!("{key}parse_error")).is_some() as usize;
            hidden += had_rests as usize;
            if problems.is_empty() {
                same += 1;
            } else {
                diff += 1;
                println!(
                    "DIFF {name} ({}): {}",
                    if key.is_empty() {
                        "repaired"
                    } else {
                        "as detected"
                    },
                    problems.join("; ")
                );
            }
        }
    }
    println!("{same} same, {diff} different; {hidden} with measure rests for hidden staffs, {errors} where homr fails, {skipped} skipped");
    if diff > 0 {
        std::process::exit(1);
    }
}
