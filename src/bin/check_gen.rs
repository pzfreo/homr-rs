//! The MusicXML generator alone against homr's: for each case dir (made by
//! tools/dump_gen_cases.py) the voices homr gave its generate_xml and the file it
//! wrote. The Rust generator gets the same voices; the output must be identical.
//!   check_gen <cases dir> [OUT_DIR]   (OUT_DIR: where to save the Rust output of differing cases)
use homr_rs::musicxml::generate_xml;
use homr_rs::tokens::{remove_duplicated_symbols, Symbol};
use homr_rs::triplets::restore_unmarked_triplets;
use serde_json::Value;
use std::fs;

fn symbol(v: &Value) -> Symbol {
    let s = |k: &str| v[k].as_str().unwrap().to_string();
    let mut sym = Symbol::new(&s("rhythm"));
    sym.pitch = s("pitch");
    sym.lift = s("lift");
    sym.articulation = s("articulation");
    sym.slur = s("slur");
    sym.position = s("position");
    if let Some(c) = v["image_coordinates"].as_array() {
        sym.image_coordinates = Some((c[0].as_f64().unwrap(), c[1].as_f64().unwrap()));
    }
    sym
}

/// The calls homr made to `name` (remove_duplicated_symbols, restore_unmarked_triplets):
/// the Rust function on the same input must give the same symbols.
fn check_calls(case: &Value, name: &str, f: fn(&[Symbol]) -> Vec<Symbol>) -> (usize, usize) {
    let (mut same, mut diff) = (0, 0);
    for call in case[name].as_array().into_iter().flatten() {
        let Some(out) = call["out"].as_array() else {
            continue;
        }; // homr itself failed
        let input: Vec<Symbol> = call["in"].as_array().unwrap().iter().map(symbol).collect();
        let want: Vec<String> = out.iter().map(|v| symbol(v).key()).collect();
        let Ok(got) = std::panic::catch_unwind(|| f(&input)) else {
            println!("  {name}: the Rust function panicked");
            diff += 1;
            continue;
        };
        let got: Vec<String> = got.iter().map(|s| s.key()).collect();
        if got == want {
            same += 1;
        } else {
            diff += 1;
            let at = got
                .iter()
                .zip(&want)
                .position(|(a, b)| a != b)
                .unwrap_or(got.len().min(want.len()));
            println!(
                "  {name}: first difference at symbol {at}: rust {:?} homr {:?}",
                got.get(at),
                want.get(at)
            );
        }
    }
    (same, diff)
}

fn main() {
    let root = std::env::args().nth(1).expect("cases dir");
    let out = std::env::args().nth(2);
    let mut dirs: Vec<_> = fs::read_dir(&root)
        .unwrap()
        .map(|e| e.unwrap().path())
        .filter(|p| p.join("voices.json").exists())
        .collect();
    dirs.sort();
    let (mut same, mut diff, mut failed) = (0, 0, 0);
    let mut calls = [(0, 0), (0, 0)];
    for d in dirs {
        let name = d.file_name().unwrap().to_string_lossy().to_string();
        let case: Value =
            serde_json::from_str(&fs::read_to_string(d.join("voices.json")).unwrap()).unwrap();
        let voices: Vec<Vec<Symbol>> = case["voices"]
            .as_array()
            .unwrap()
            .iter()
            .map(|v| v.as_array().unwrap().iter().map(symbol).collect())
            .collect();
        let title = case["title"].as_str().unwrap_or("");
        for (k, (name_, f)) in [
            (
                "remove_duplicated_symbols",
                remove_duplicated_symbols as fn(&[Symbol]) -> Vec<Symbol>,
            ),
            (
                "restore_unmarked_triplets",
                restore_unmarked_triplets as fn(&[Symbol]) -> Vec<Symbol>,
            ),
        ]
        .into_iter()
        .enumerate()
        {
            let (a, b) = check_calls(&case, name_, f);
            if b > 0 {
                println!("DIFF {name} ({name_})");
            }
            calls[k].0 += a;
            calls[k].1 += b;
        }
        let want = fs::read_to_string(d.join(format!("{name}.musicxml")));
        let got = std::panic::catch_unwind(|| generate_xml(&voices, title));
        let Ok(want) = want else {
            // homr failed on this case (error.txt): the port must fail too
            let homr = fs::read_to_string(d.join("error.txt")).unwrap_or_default();
            match got {
                Ok(Err(_)) | Err(_) => same += 1,
                Ok(Ok(_)) => {
                    diff += 1;
                    println!("DIFF {name}: homr failed ({homr}) but the Rust generator did not");
                }
            }
            continue;
        };
        let got = match got {
            Ok(r) => r,
            Err(_) => {
                failed += 1;
                println!("PANIC {name}: the Rust generator panicked");
                continue;
            }
        };
        match got {
            Ok(got) if got == want => same += 1,
            Ok(got) => {
                diff += 1;
                let at = got
                    .bytes()
                    .zip(want.bytes())
                    .position(|(a, b)| a != b)
                    .unwrap_or(got.len().min(want.len()));
                let from = at.saturating_sub(60);
                let show = |s: &str| {
                    s.get(from..(at + 100).min(s.len()))
                        .unwrap_or("")
                        .to_string()
                };
                println!("DIFF {name}: first difference at byte {at} (rust {} bytes, homr {})\n  rust: {}\n  homr: {}", got.len(), want.len(), show(&got), show(&want));
                if let Some(o) = &out {
                    fs::create_dir_all(o).unwrap();
                    fs::write(format!("{o}/{name}.rust.musicxml"), got).unwrap();
                }
            }
            Err(e) => {
                failed += 1;
                println!("FAIL {name}: {e}");
            }
        }
    }
    println!(
        "remove_duplicated_symbols: identical {}, different {}",
        calls[0].0, calls[0].1
    );
    println!(
        "restore_unmarked_triplets: identical {}, different {}",
        calls[1].0, calls[1].1
    );
    println!("generate_xml: identical {same}, different {diff}, failed {failed}");
}
