//! Staff images (prepare_staff_image) against homr in Pyodide
//! (tools/pyodide_staffs.mjs): the canvas the model reads, pixel for pixel.
use homr_rs::staff::{Staff, StaffPoint};
use homr_rs::staff_image::{prepare_staff_image, Regions};
use homr_rs::warp::Gray;
use serde_json::Value;
use std::fs;

fn diff_count(a: &[u8], b: &[u8]) -> usize {
    a.iter().zip(b).filter(|(x, y)| x != y).count()
}

fn main() {
    let pages = std::env::args().nth(1).expect("pages dir");
    let refs = std::env::args().nth(2).expect("staff reference dir");
    let mut dirs: Vec<_> = fs::read_dir(&refs)
        .unwrap()
        .map(|e| e.unwrap().path())
        .filter(|p| p.join("staffs.json").exists())
        .collect();
    dirs.sort();
    let (mut same, mut diff) = (0, 0);
    for d in dirs {
        let name = d.file_name().unwrap().to_string_lossy().to_string();
        let pd = std::path::Path::new(&pages).join(&name);
        let page: Value =
            serde_json::from_str(&fs::read_to_string(pd.join("page.json")).unwrap()).unwrap();
        let ps: Vec<usize> = page["preprocessed_shape"]
            .as_array()
            .unwrap()
            .iter()
            .map(|v| v.as_u64().unwrap() as usize)
            .collect();
        let image = Gray {
            w: ps[1],
            h: ps[0],
            data: fs::read(pd.join("preprocessed.bin")).unwrap(),
        };
        let calls: Value =
            serde_json::from_str(&fs::read_to_string(d.join("staffs.json")).unwrap()).unwrap();
        for (k, c) in calls.as_array().unwrap().iter().enumerate() {
            let grid: Vec<StaffPoint> = c["grid"]
                .as_array()
                .unwrap()
                .iter()
                .map(|p| {
                    let p = p.as_array().unwrap();
                    StaffPoint {
                        x: p[0].as_f64().unwrap(),
                        y: p[1]
                            .as_array()
                            .unwrap()
                            .iter()
                            .map(|v| v.as_f64().unwrap())
                            .collect(),
                        angle: p[2].as_f64().unwrap(),
                    }
                })
                .collect();
            let staff = Staff::from_grid(grid).unwrap();
            let regions = Regions {
                centers: c["centers"]
                    .as_array()
                    .unwrap()
                    .iter()
                    .map(|v| (v[0].as_f64().unwrap(), v[1].as_f64().unwrap()))
                    .collect(),
            };
            let expected = fs::read(d.join(format!("canvas_{k}.bin"))).unwrap();
            match prepare_staff_image(&staff, &image, &regions) {
                Ok(canvas) if canvas.data == expected => same += 1,
                Ok(canvas) => {
                    diff += 1;
                    println!(
                        "DIFF {name} staff {k}: {} of {} canvas pixels differ",
                        diff_count(&canvas.data, &expected),
                        expected.len()
                    );
                }
                Err(e) => {
                    diff += 1;
                    println!("DIFF {name} staff {k}: {e}");
                }
            }
        }
    }
    println!("{same} staff images same, {diff} different");
}
