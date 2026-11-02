//! Whole page analysis (masks -> MultiStaffs) against homr in Pyodide
//! (multi.json from tools/pyodide_ref.mjs). Also counts instructions.
use homr_rs::page::{analyse_page, PageMasks};
use serde_json::Value;
use std::fs;

extern "C" {
    fn proc_pid_rusage(pid: i32, flavor: i32, buffer: *mut u64) -> i32;
    fn getpid() -> i32;
}
fn instructions() -> u64 {
    let mut buf = [0u64; 96];
    unsafe { assert_eq!(proc_pid_rusage(getpid(), 4, buf.as_mut_ptr()), 0) };
    buf[2 + 29]
}

fn main() {
    let pages = std::env::args().nth(1).expect("pages dir");
    let refs = std::env::args().nth(2).expect("reference dir");
    let mut dirs: Vec<_> = fs::read_dir(&pages)
        .unwrap()
        .map(|e| e.unwrap().path())
        .filter(|p| p.join("page.json").exists())
        .collect();
    dirs.sort();
    let (mut same, mut diff, mut total) = (0, 0, 0u64);
    for d in dirs {
        let name = d.file_name().unwrap().to_string_lossy().to_string();
        let r = std::path::Path::new(&refs).join(&name);
        let page: Value =
            serde_json::from_str(&fs::read_to_string(d.join("page.json")).unwrap()).unwrap();
        let shape: Vec<usize> = page["shape"]
            .as_array()
            .unwrap()
            .iter()
            .map(|v| v.as_u64().unwrap() as usize)
            .collect();
        let rd = |n: &str| fs::read(d.join(format!("{n}.bin"))).unwrap();
        let (nh, sy, st, ck, sr) = (
            rd("notehead"),
            rd("symbols"),
            rd("staff"),
            rd("clefs_keys"),
            rd("stem_rests"),
        );
        let m = PageMasks {
            h: shape[0],
            w: shape[1],
            notehead: &nh,
            symbols: &sy,
            staff: &st,
            clefs_keys: &ck,
            stem_rests: &sr,
        };
        let a = instructions();
        let ours = analyse_page(&m);
        total += instructions() - a;
        let expected: Option<Value> = fs::read_to_string(r.join("multi.json"))
            .ok()
            .map(|t| serde_json::from_str(&t).unwrap());
        let verdict: Result<(), String> = match (&ours, &expected) {
            (Err(_), None) => Ok(()), // both fail
            (Err(e), Some(_)) => Err(format!("rust failed: {e}")),
            (Ok(_), None) => Err("homr failed, rust didn't".into()),
            (Ok(ms), Some(exp)) => {
                let exp = exp.as_array().unwrap();
                if ms.len() != exp.len() {
                    Err(format!("{} systems, homr {}", ms.len(), exp.len()))
                } else {
                    let mut err = Ok(());
                    'outer: for (k, (m, e)) in ms.iter().zip(exp).enumerate() {
                        let e = e.as_array().unwrap();
                        if m.staffs.len() != e.len() {
                            err = Err(format!(
                                "system {k}: {} staffs, homr {}",
                                m.staffs.len(),
                                e.len()
                            ));
                            break;
                        }
                        for (j, (s, es)) in m.staffs.iter().zip(e).enumerate() {
                            if m.grand[j] != es["grand"].as_bool().unwrap() {
                                err = Err(format!(
                                    "system {k} staff {j}: grand {} vs {}",
                                    m.grand[j], es["grand"]
                                ));
                                break 'outer;
                            }
                            let g = es["grid"].as_array().unwrap();
                            let same_grid = s.grid.len() == g.len()
                                && s.grid.iter().zip(g).all(|(p, q)| {
                                    let q = q.as_array().unwrap();
                                    p.x == q[0].as_f64().unwrap()
                                        && p.angle == q[2].as_f64().unwrap()
                                        && p.y
                                            .iter()
                                            .zip(q[1].as_array().unwrap())
                                            .all(|(a, b)| *a == b.as_f64().unwrap())
                                });
                            if !same_grid {
                                let first = s.grid.iter().zip(g).position(|(p, q)| {
                                    let q = q.as_array().unwrap();
                                    !(p.x == q[0].as_f64().unwrap()
                                        && p.angle == q[2].as_f64().unwrap()
                                        && p.y
                                            .iter()
                                            .zip(q[1].as_array().unwrap())
                                            .all(|(a, b)| *a == b.as_f64().unwrap()))
                                });
                                let detail = first
                                    .map(|i| {
                                        format!(
                                            "point {i}: rust {:?} homr {}",
                                            (s.grid[i].x, &s.grid[i].y, s.grid[i].angle),
                                            g[i]
                                        )
                                    })
                                    .unwrap_or_default();
                                err = Err(format!(
                                    "system {k} staff {j}: grid differs ({} vs {} points) {detail}",
                                    s.grid.len(),
                                    g.len()
                                ));
                                break 'outer;
                            }
                        }
                    }
                    err
                }
            }
        };
        match verdict {
            Ok(()) => same += 1,
            Err(e) => {
                diff += 1;
                println!("DIFF {name}: {e}");
            }
        }
    }
    println!(
        "{same} pages same, {diff} different; rust {:.2}G instructions in all",
        total as f64 / 1e9
    );
}
