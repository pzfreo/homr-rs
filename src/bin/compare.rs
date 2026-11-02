//! Runs the Rust staff detection on cases dumped from homr (tools/dump_staff.py),
//! compares the staffs with homr's, and counts retired instructions.
use homr_rs::geom::{RBox, NO_FRAGMENT};
use homr_rs::staff::{detect_staff, Fragments, Staff};
use serde_json::Value;
use std::fs;

extern "C" {
    fn proc_pid_rusage(pid: i32, flavor: i32, buffer: *mut u64) -> i32;
    fn getpid() -> i32;
}
fn instructions() -> u64 {
    let mut buf = [0u64; 96]; // rusage_info_v4: 16-byte uuid, then u64 fields
    unsafe { assert_eq!(proc_pid_rusage(getpid(), 4, buf.as_mut_ptr()), 0) };
    buf[2 + 29] // ri_instructions
}

fn rbox(v: &Value, frag: usize) -> RBox {
    let a: Vec<f64> = v
        .as_array()
        .unwrap()
        .iter()
        .map(|x| x.as_f64().unwrap())
        .collect();
    RBox::new(a[0], a[1], a[2], a[3], a[4], frag)
}

fn compare(ours: &[Staff], expected: &Value) -> Result<f64, String> {
    let exp = expected.as_array().unwrap();
    if ours.len() != exp.len() {
        return Err(format!("{} staffs, homr {}", ours.len(), exp.len()));
    }
    let mut worst: f64 = 0.0;
    for (k, (s, e)) in ours.iter().zip(exp).enumerate() {
        let e = e.as_array().unwrap();
        if s.grid.len() != e.len() {
            return Err(format!(
                "staff {k}: {} points, homr {}",
                s.grid.len(),
                e.len()
            ));
        }
        for (p, q) in s.grid.iter().zip(e) {
            let q = q.as_array().unwrap();
            let qx = q[0].as_f64().unwrap();
            if p.x != qx {
                return Err(format!("staff {k}: x {} vs {}", p.x, qx));
            }
            for (y, qy) in p.y.iter().zip(q[1].as_array().unwrap()) {
                worst = worst.max((y - qy.as_f64().unwrap()).abs());
            }
            worst = worst.max((p.angle - q[2].as_f64().unwrap()).abs());
        }
    }
    Ok(worst)
}

fn main() {
    let root = std::env::args().nth(1).expect("cases dir");
    let mut dirs: Vec<_> = fs::read_dir(&root)
        .unwrap()
        .map(|e| e.unwrap().path())
        .collect();
    dirs.sort();
    let (mut same, mut diff, mut py_total, mut rs_total) = (0, 0, 0u64, 0u64);
    for d in dirs {
        let case: Value =
            serde_json::from_str(&fs::read_to_string(d.join("case.json")).unwrap()).unwrap();
        let shape: Vec<usize> = case["shape"]
            .as_array()
            .unwrap()
            .iter()
            .map(|v| v.as_u64().unwrap() as usize)
            .collect();
        let image = fs::read(d.join("staff.bin")).unwrap();
        let frags = case["fragments"].as_array().unwrap();
        let boxes: Vec<RBox> = frags
            .iter()
            .enumerate()
            .map(|(i, f)| rbox(&f["box"], i))
            .collect();
        let contours: Vec<Vec<[i32; 2]>> = frags
            .iter()
            .map(|f| {
                f["contour"]
                    .as_array()
                    .unwrap()
                    .iter()
                    .map(|p| [p[0].as_i64().unwrap() as i32, p[1].as_i64().unwrap() as i32])
                    .collect()
            })
            .collect();
        let clefs: Vec<RBox> = case["clefs"]
            .as_array()
            .unwrap()
            .iter()
            .map(|b| rbox(b, NO_FRAGMENT))
            .collect();
        let bars: Vec<RBox> = case["bars"]
            .as_array()
            .unwrap()
            .iter()
            .map(|b| rbox(b, NO_FRAGMENT))
            .collect();
        let f = Fragments {
            boxes: &boxes,
            contours: &contours,
        };
        let _ = detect_staff(&image, shape[0], shape[1], &f, &clefs, &bars); // warm, as the Python count
        let a = instructions();
        let result = detect_staff(&image, shape[0], shape[1], &f, &clefs, &bars);
        let rs = instructions() - a;
        let py = case["python_instructions"].as_u64().unwrap();
        let name = d.file_name().unwrap().to_string_lossy().to_string();
        let verdict = match result {
            Ok(staffs) => compare(&staffs, &case["expected"]),
            Err(e) => Err(e),
        };
        match verdict {
            Ok(worst) => {
                same += 1;
                py_total += py;
                rs_total += rs;
                println!(
                    "SAME {name:40} max diff {worst:.1e}  python {:>6.0}M  rust {:>5.1}M  x{:.0}",
                    py as f64 / 1e6,
                    rs as f64 / 1e6,
                    py as f64 / rs as f64
                );
            }
            Err(e) => {
                diff += 1;
                println!("DIFF {name:40} {e}");
            }
        }
    }
    println!("\n{same} same, {diff} different; on the same ones python {:.1}G instructions, rust {:.2}G (x{:.0})", py_total as f64 / 1e9, rs_total as f64 / 1e9, py_total as f64 / rs_total.max(1) as f64);
}
