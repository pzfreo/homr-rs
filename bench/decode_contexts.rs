//! Replay alternative encoder contexts through the exact native Rust decoder.
//! cargo run --release --bin decode_contexts -- MODEL NATIVE_REF CONTEXTS OUT [PAGE...]
//! Counts every overlapping step/head, plus missing/extra steps separately.
use homr_rs::{decoder::Decoder, tokens::Vocab};
use serde_json::{json, Value};
use std::{fs, path::Path};

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let args: Vec<_> = std::env::args().collect();
    if args.len() < 5 {
        return Err("decode_contexts MODEL NATIVE_REF CONTEXTS OUT [PAGE...]".into());
    }
    let model = Decoder::from_onnx(&fs::read(&args[1])?)?;
    let refs = Path::new(&args[2]);
    let contexts = Path::new(&args[3]);
    let out = Path::new(&args[4]);
    let mut pages: Vec<_> = if args.len() > 5 {
        args[5..].to_vec()
    } else {
        fs::read_dir(refs)?
            .filter_map(Result::ok)
            .filter(|e| e.path().join("decoder.json").is_file())
            .map(|e| e.file_name().to_string_lossy().into_owned())
            .collect()
    };
    pages.sort();
    let vocab = Vocab::default();
    let mut rows = Vec::new();
    for page in pages {
        let mut meta: Value =
            serde_json::from_slice(&fs::read(refs.join(&page).join("decoder.json"))?)?;
        let mut calls = Vec::new();
        let mut staff_rows = Vec::new();
        for staff in 0..meta["calls"].as_array().ok_or("missing calls")?.len() {
            let d = Path::new(&page).join(staff.to_string());
            let bytes = fs::read(contexts.join(&d).join("context.bin"))?;
            if bytes.len() != 1280 * 512 * 4 {
                return Err("unexpected context size".into());
            }
            let context: Vec<_> = bytes
                .as_chunks::<4>()
                .0
                .iter()
                .map(|b| f32::from_le_bytes(*b))
                .collect();
            let reference: Value =
                serde_json::from_slice(&fs::read(refs.join(&d).join("steps.json"))?)?;
            let reference = reference.as_array().ok_or("invalid steps")?;
            let mut session = model.session(&context)?;
            let mut input = [1, 0, 0, 0, 0];
            let (mut steps, mut tokens_different, mut steps_different) = (Vec::new(), 0, 0);
            let mut first = None;
            let mut symbols = Vec::new();
            for index in 0..homr_rs::decoder::MAX_STEPS {
                let result = session.step(input)?;
                let tokens = result.tokens;
                if let Some(r) = reference.get(index) {
                    let expected: [usize; 6] =
                        std::array::from_fn(|h| r["tokens"][h].as_u64().unwrap() as usize);
                    let differences = tokens
                        .iter()
                        .zip(expected)
                        .filter(|(a, b)| **a != *b)
                        .count();
                    tokens_different += differences;
                    if differences != 0 {
                        steps_different += 1;
                        if first.is_none() {
                            first = Some(
                                json!({"step": index, "native": expected, "alternative": tokens}),
                            );
                        }
                    }
                }
                steps.push(json!({"tokens": tokens, "attention": result.attention}));
                if tokens[0] == 2 {
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
                input = [tokens[0], tokens[1], tokens[2], tokens[4], tokens[5]];
            }
            let missing = reference.len().saturating_sub(steps.len());
            let extra = steps.len().saturating_sub(reference.len());
            let row = json!({"staff": staff, "native_steps": reference.len(), "steps": steps.len(),
                "overlapping_steps": steps.len().min(reference.len()), "differing_steps": steps_different,
                "differing_tokens": tokens_different, "missing_steps": missing, "extra_steps": extra,
                "different": tokens_different + missing + extra != 0, "first_difference": first});
            let dir = out.join(&d);
            fs::create_dir_all(&dir)?;
            fs::write(dir.join("steps.json"), serde_json::to_vec(&steps)?)?;
            println!("{page}/{staff}: {row}");
            staff_rows.push(row);
            calls.push(symbols);
        }
        meta["calls"] = json!(calls);
        fs::create_dir_all(out.join(&page))?;
        fs::write(
            out.join(&page).join("decoder.json"),
            serde_json::to_vec(&meta)?,
        )?;
        rows.push(json!({"page": page, "staffs": staff_rows}));
    }
    let staffs: Vec<_> = rows
        .iter()
        .flat_map(|p| p["staffs"].as_array().unwrap())
        .collect();
    let sum = |key: &str| staffs.iter().map(|s| s[key].as_u64().unwrap()).sum::<u64>();
    let report = json!({"summary": {"pages": rows.len(), "staffs": staffs.len(),
        "differing_staffs": staffs.iter().filter(|s| s["different"] == true).count(),
        "steps": sum("steps"), "native_steps": sum("native_steps"),
        "overlapping_steps": sum("overlapping_steps"), "differing_steps": sum("differing_steps"),
        "differing_tokens": sum("differing_tokens"), "extra_steps": sum("extra_steps"), "missing_steps": sum("missing_steps")},
        "pages": rows});
    fs::write(
        out.join("comparison.json"),
        serde_json::to_vec_pretty(&report)?,
    )?;
    println!("{}", report["summary"]);
    Ok(())
}
