//! Compare whole encoder and native optimized intermediate tensors.
use homr_rs::encoder::Encoder;
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
fn main() -> Result<(), Box<dyn std::error::Error>> {
    let a: Vec<_> = std::env::args().collect();
    let model = Encoder::from_onnx(&fs::read(&a[1])?)?;
    let input = floats(Path::new(&a[2]));
    let stages = Path::new(&a[3]);
    let rows: Value = serde_json::from_slice(&fs::read(stages.join("stages.json"))?)?;
    let mut results = Vec::new();
    let mut failed = false;
    let output = model.encode_observed(&input, |name, data, h, w| {
        let Some(row) = rows.as_array().unwrap().iter().find(|r| r["name"] == name) else {
            return;
        };
        let mut reference = floats(&stages.join(row["file"].as_str().unwrap()));
        let shape: Vec<_> = row["shape"]
            .as_array()
            .unwrap()
            .iter()
            .map(|v| v.as_u64().unwrap() as usize)
            .collect();
        if shape.len() == 4 && shape[2] == h && shape[3] == w {
            let c = shape[1];
            reference = (0..reference.len())
                .map(|i| reference[(i % c) * h * w + i / c])
                .collect();
        }
        assert_eq!(data.len(), reference.len());
        let different = data
            .iter()
            .zip(&reference)
            .filter(|(a, b)| a.to_bits() != b.to_bits())
            .count();
        let max = data
            .iter()
            .zip(&reference)
            .map(|(a, b)| (*a as f64 - *b as f64).abs())
            .fold(0.0f64, f64::max);
        let row =
            json!({"name":name,"values":data.len(),"bit_differences":different,"max_abs":max});
        println!("{row}");
        results.push(row);
        failed |= different != 0;
    })?;
    let bytes: Vec<_> = output.iter().flat_map(|v| v.to_le_bytes()).collect();
    fs::write(&a[4], bytes)?;
    fs::write(
        format!("{}.json", a[4]),
        serde_json::to_vec_pretty(&results)?,
    )?;
    if failed {
        return Err("encoder stage differs".into());
    }
    Ok(())
}
