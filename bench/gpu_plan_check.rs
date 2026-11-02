//! Verify packing/wiring of every GPU-plan operation against the CPU encoder.
//! gpu_plan_check MODEL INPUT.f32 NATIVE_CONTEXT [SCRATCH_RESULT.json] [--simulate]
use homr_rs::encoder::{gpu::Plan, Encoder};
use serde_json::json;
use std::fs;
mod gpu_schedule;
fn read(path: &str) -> Vec<f32> {
    fs::read(path)
        .unwrap()
        .as_chunks::<4>()
        .0
        .iter()
        .map(|v| f32::from_le_bytes(*v))
        .collect()
}
fn main() -> Result<(), Box<dyn std::error::Error>> {
    let args: Vec<_> = std::env::args().collect();
    if args.len() < 3 || (args.len() < 4 && args[2] != "--self-tests") {
        return Err(
            "gpu_plan_check MODEL INPUT.f32 NATIVE_CONTEXT [RESULT.json] [--simulate]; or MODEL --self-tests".into(),
        );
    }
    let encoder = Encoder::from_onnx(&fs::read(&args[1])?)?;
    let plan = Plan::new(&encoder);
    let only_tests = args[2] == "--self-tests";
    let simulate = only_tests || args.iter().any(|s| s == "--simulate");
    if simulate {
        let meta: serde_json::Value = serde_json::from_slice(&plan.json)?;
        for test in meta["self_tests"].as_array().unwrap() {
            let words = |name: &str| -> Vec<u32> {
                test[name]
                    .as_array()
                    .unwrap()
                    .iter()
                    .map(|v| v.as_u64().unwrap() as u32)
                    .collect()
            };
            let floats = |name: &str| {
                words(name)
                    .into_iter()
                    .map(f32::from_bits)
                    .collect::<Vec<_>>()
            };
            let q: [u32; 16] = words("params").try_into().unwrap();
            let result = gpu_schedule::simulate(&q, &floats("a"), &floats("r"), &floats("weights"));
            assert_eq!(
                result.iter().map(|v| v.to_bits()).collect::<Vec<_>>(),
                words("expected")
            );
        }
        if only_tests {
            println!(
                "{}",
                json!({"simulated_self_tests":meta["self_tests"].as_array().unwrap().len(),"bit_differences":0,"gpu_execution":false})
            );
            return Ok(());
        }
    }
    let input = read(&args[2]);
    let native = read(&args[3]);
    let mut buffers: Vec<Vec<f32>> = (0..7).map(|_| vec![]).collect();
    buffers[0] = input.clone();
    let mut index = 0;
    let mut rows = vec![];
    let output = encoder.encode_observed(&input, |name, values, _, _| {
        let op = &plan.ops[index];
        assert_eq!(op.name, name);
        let result = if simulate {
            gpu_schedule::simulate(
                &op.params,
                &buffers[op.input],
                &buffers[op.residual],
                &plan.weights,
            )
        } else {
            plan.reference(op, &buffers[op.input], &buffers[op.residual])
        };
        assert_eq!(result.len(), values.len());
        let differing = result
            .iter()
            .zip(values)
            .filter(|(a, b)| a.to_bits() != b.to_bits())
            .count();
        assert_eq!(differing, 0, "{name}");
        rows.push(json!({"name":name,"values":values.len(),"bit_differences":differing}));
        buffers[op.output] = result;
        index += 1;
    })?;
    assert_eq!(index, plan.ops.len());
    assert_eq!(output.len(), native.len());
    assert!(output
        .iter()
        .zip(&native)
        .all(|(a, b)| a.to_bits() == b.to_bits()));
    let report = json!({"simulated_tiled_schedule":simulate,"ops":index,"weights":plan.weights.len(),"buffers":plan.buffers,"context_values":output.len(),"context_bit_differences":0,"rows":rows});
    if let Some(file) = args.get(4).filter(|s| !s.starts_with("--")) {
        fs::write(file, serde_json::to_vec_pretty(&report)?)?;
    }
    println!(
        "{}",
        json!({"ops":index,"weights":plan.weights.len(),"buffers":plan.buffers,"context_bit_differences":0})
    );
    Ok(())
}
