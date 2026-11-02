//! Equal-work native decoder benchmark; pair decode with load and subtract
//! /usr/bin/time -l retired instructions. Never used by CI.
use homr_rs::decoder::Decoder;
use std::{fs, path::Path};
fn main() {
    let args: Vec<_> = std::env::args().collect();
    assert!(
        args.len() >= 5,
        "bench_decoder load|decode MODEL REFERENCES PAGE/STAFF ..."
    );
    let model = Decoder::from_onnx(&fs::read(&args[2]).unwrap()).unwrap();
    let data: Vec<_> = args[4..]
        .iter()
        .map(|name| {
            let d = Path::new(&args[3]).join(name);
            let context = fs::read(d.join("context.bin"))
                .unwrap()
                .as_chunks::<4>()
                .0
                .iter()
                .map(|b| f32::from_le_bytes(*b))
                .collect::<Vec<_>>();
            let refs: serde_json::Value =
                serde_json::from_slice(&fs::read(d.join("steps.json")).unwrap()).unwrap();
            (context, refs.as_array().unwrap().len())
        })
        .collect();
    if args[1] == "load" {
        std::hint::black_box(model);
        wait();
        return;
    }
    assert!(["decode", "profile"].contains(&args[1].as_str()));
    #[cfg(feature = "decoder-profile")]
    let overhead = homr_rs::decoder::profile::calibrate();
    #[cfg(feature = "decoder-profile")]
    homr_rs::decoder::profile::reset();
    let mut total = 0;
    for (context, count) in data {
        let _all = homr_rs::decoder::profile::enter(homr_rs::decoder::profile::Phase::Other);
        let mut session = model.session(&context).unwrap();
        let mut input = [1, 0, 0, 0, 0];
        for _ in 0..count {
            let step = session.step(input).unwrap();
            let t = step.tokens;
            input = [if t[0] == 2 { 3 } else { t[0] }, t[1], t[2], t[4], t[5]];
            std::hint::black_box(step);
        }
        total += count;
    }
    #[cfg(feature = "decoder-profile")]
    {
        homr_rs::decoder::profile::stop();
        let (totals, calls) = homr_rs::decoder::profile::values();
        let phases:Vec<_>=(0..homr_rs::decoder::profile::N).map(|i| serde_json::json!({"phase":homr_rs::decoder::profile::NAMES[i],"raw":totals[i],"calls":calls[i],"corrected":(totals[i] as f64-calls[i] as f64*overhead).max(0.0)})).collect();
        println!(
            "{}",
            serde_json::json!({"steps":total,"overhead":overhead,"phases":phases})
        );
    }
    #[cfg(not(feature = "decoder-profile"))]
    println!("{total} steps");
    wait();
}

fn wait() {
    if std::env::var_os("BENCH_WAIT").is_some() {
        use std::io::Write;
        println!("BENCH_READY");
        std::io::stdout().flush().unwrap();
        std::io::stdin().read_line(&mut String::new()).unwrap();
    }
}
