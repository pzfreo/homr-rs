//! Compare recorded homr decoder steps (tools/record_decoder.py), and write
//! decoder.json from the Rust tokens and coordinates for check_xml.
use homr_rs::decoder::Decoder;
use homr_rs::tokens::Vocab;
use serde_json::{json, Value};
use std::{fs, path::Path};
fn floats(p: &Path) -> Vec<f32> {
    fs::read(p)
        .unwrap()
        .as_chunks::<4>()
        .0
        .iter()
        .map(|b| f32::from_le_bytes(*b))
        .collect()
}
fn run() -> Result<(), String> {
    let args: Vec<_> = std::env::args().collect();
    if args.len() < 4 {
        return Err("check_decoder MODEL REFERENCES OUTPUT [PAGE ...]".into());
    }
    let model = Decoder::from_onnx(&fs::read(&args[1]).map_err(|e| e.to_string())?)?;
    let root = Path::new(&args[2]);
    let out = Path::new(&args[3]);
    let mut pages: Vec<_> = if args.len() > 4 {
        args[4..].to_vec()
    } else {
        fs::read_dir(root)
            .unwrap()
            .map(|e| e.unwrap().file_name().to_string_lossy().into_owned())
            .collect()
    };
    pages.sort();
    let (mut staffs, mut steps, mut differences, mut length_differences) = (0, 0, 0, 0);
    let (mut max_abs, mut total_abs, mut attention_count) = (0.0f64, 0.0f64, 0usize);
    let mut errors = Vec::new();
    let mut absolute_errors = Vec::new();
    let mut compared_steps = 0;
    let vocab = Vocab::default();
    for page in pages {
        let src = root.join(&page);
        if !src.join("decoder.json").exists() {
            return Err(format!("missing completed page {page}"));
        }
        let mut meta: Value =
            serde_json::from_slice(&fs::read(src.join("decoder.json")).unwrap()).unwrap();
        let nstaff = meta["calls"].as_array().unwrap().len();
        let mut calls = Vec::new();
        for staff in 0..nstaff {
            let d = src.join(staff.to_string());
            let refs: Value =
                serde_json::from_slice(&fs::read(d.join("steps.json")).unwrap()).unwrap();
            let refs = refs.as_array().unwrap();
            let ref_logits = floats(&d.join("logits.bin"));
            let context = floats(&d.join("context.bin"));
            let mut session = model.session(&context)?;
            let mut input = [1, 0, 0, 0, 0];
            let mut symbols = Vec::new();
            let mut ended = false;
            // Always run the real greedy trajectory, never teacher-force reference tokens.
            for index in 0..homr_rs::decoder::MAX_STEPS {
                let result = session.step(input)?;
                let t = result.tokens;
                if let Some(reference) = refs.get(index) {
                    compared_steps += 1;
                    let expected: [usize; 6] =
                        std::array::from_fn(|i| reference["tokens"][i].as_u64().unwrap() as usize);
                    if t != expected {
                        differences += 1;
                        let mut offset = index * 411;
                        for h in 0..6 {
                            if t[h] != expected[h] {
                                let margin = result.logits[h][t[h]] - result.logits[h][expected[h]];
                                let ref_margin =
                                    ref_logits[offset + expected[h]] - ref_logits[offset + t[h]];
                                let msg = format!("{page}/{staff}/{index} head {h}: rust {} reference {}, margins rust {margin:e} reference {ref_margin:e}",t[h],expected[h]);
                                eprintln!("{msg}");
                                errors.push(msg);
                            }
                            offset += result.logits[h].len();
                        }
                    }
                    for i in 0..2 {
                        let e = (result.attention[i] as f64
                            - reference["attention"][i].as_f64().unwrap())
                        .abs();
                        absolute_errors.push(e);
                        max_abs = max_abs.max(e);
                        total_abs += e;
                        attention_count += 1;
                    }
                }
                steps += 1;
                if t[0] == 2 {
                    ended = true;
                    if index + 1 != refs.len() {
                        length_differences += 1;
                        errors.push(format!(
                            "{page}/{staff}: length {} reference {}",
                            index + 1,
                            refs.len()
                        ));
                    }
                    break;
                }
                if let Some(s) = result.symbol(&vocab) {
                    symbols.push(json!([
                        s.rhythm,
                        s.pitch,
                        s.lift,
                        s.articulation,
                        s.slur,
                        s.position,
                        s.coordinates
                    ]));
                }
                input = [t[0], t[1], t[2], t[4], t[5]];
            }
            if !ended && refs.len() != homr_rs::decoder::MAX_STEPS {
                length_differences += 1;
            }
            calls.push(symbols);
            staffs += 1;
            println!("{page}/{staff}: {} steps", session.position());
        }
        meta["calls"] = json!(calls);
        fs::create_dir_all(out.join(&page)).unwrap();
        fs::write(
            out.join(&page).join("decoder.json"),
            serde_json::to_vec(&meta).unwrap(),
        )
        .unwrap();
    }
    absolute_errors.sort_by(f64::total_cmp);
    let percentile = |p: f64| {
        absolute_errors
            .get(((absolute_errors.len().saturating_sub(1)) as f64 * p) as usize)
            .copied()
            .unwrap_or(0.0)
    };
    let report = json!({"staffs":staffs,"steps":steps,"compared_steps":compared_steps,"attention_median_abs":percentile(0.5),"attention_p95_abs":percentile(0.95),"differing_steps":differences,"length_differences":length_differences,"attention_max_abs":max_abs,"attention_mean_abs":total_abs/attention_count as f64,"differences":errors});
    println!("{report}");
    fs::write(
        out.join("comparison.json"),
        serde_json::to_vec_pretty(&report).unwrap(),
    )
    .unwrap();
    if differences + length_differences > 0 {
        return Err("decoder differs".into());
    }
    Ok(())
}
fn main() {
    if let Err(e) = run() {
        eprintln!("{e}");
        std::process::exit(1);
    }
}
