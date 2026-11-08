//! Instructions per stage of page analysis for one page (masks from tools/dump_page.py).
use homr_rs::braces::{brace_dot_image, multi_staffs};
use homr_rs::contours::find_contours;
use homr_rs::geom::{RBox, NO_FRAGMENT};
use homr_rs::staff::{detect_staff, Fragments};
use homr_rs::symbols::{bounding_ellipses, page_symbols, rotated_boxes, Masks};
use serde_json::Value;
use std::fs;

extern "C" {
    fn proc_pid_rusage(pid: i32, flavor: i32, buffer: *mut u64) -> i32;
    fn getpid() -> i32;
}
fn instr() -> u64 {
    let mut buf = [0u64; 96];
    unsafe { assert_eq!(proc_pid_rusage(getpid(), 4, buf.as_mut_ptr()), 0) };
    buf[2 + 29]
}
fn main() {
    let d = std::path::PathBuf::from(std::env::args().nth(1).unwrap());
    let page: Value =
        serde_json::from_str(&fs::read_to_string(d.join("page.json")).unwrap()).unwrap();
    let s: Vec<usize> = page["shape"]
        .as_array()
        .unwrap()
        .iter()
        .map(|v| v.as_u64().unwrap() as usize)
        .collect();
    let (h, w) = (s[0], s[1]);
    let rd = |n: &str| fs::read(d.join(format!("{n}.bin"))).unwrap();
    let (nh, sy, st, ck, sr) = (
        rd("notehead"),
        rd("symbols"),
        rd("staff"),
        rd("clefs_keys"),
        rd("stem_rests"),
    );
    let mut t = instr();
    let mut lap = |label: &str| {
        let n = instr();
        println!("{label:28} {:>8.1}M", (n - t) as f64 / 1e6);
        t = instr();
    };
    for (n, m) in [
        ("notehead", &nh),
        ("staff", &st),
        ("clefs_keys", &ck),
        ("stem_rests", &sr),
    ] {
        let c = find_contours(m, h, w);
        lap(&format!("contours {n} ({})", c.len()));
    }
    let _ = bounding_ellipses(&nh, h, w, (4.0, 4.0));
    lap("noteheads (ellipses+merge)");
    let _ = rotated_boxes(&st, h, w, true, Some((5.0, 1.0)), Some((10000.0, 100.0)));
    lap("staff fragments");
    let _ = rotated_boxes(&sr, h, w, false, None, None);
    lap("stems (merge)");
    let ps = page_symbols(&Masks {
        h,
        w,
        notehead: &nh,
        staff: &st,
        clefs_keys: &ck,
        stem_rests: &sr,
    })
    .unwrap();
    lap("page_symbols (all)");
    let boxes: Vec<RBox> = ps
        .fragments
        .iter()
        .enumerate()
        .map(|(i, f)| f.to_rbox(i))
        .collect();
    let contours: Vec<Vec<[i32; 2]>> = ps.fragments.iter().map(|f| f.contour.clone()).collect();
    let clefs: Vec<RBox> = ps
        .clefs_keys
        .iter()
        .map(|c| c.to_rbox(NO_FRAGMENT))
        .collect();
    let bars: Vec<RBox> = ps
        .bar_lines
        .iter()
        .map(|c| c.to_rbox(NO_FRAGMENT))
        .collect();
    let staffs = detect_staff(
        &st,
        Some(&sy),
        h,
        w,
        &Fragments {
            boxes: &boxes,
            contours: &contours,
        },
        &clefs,
        &bars,
    )
    .unwrap();
    lap("detect_staff");
    let _ = brace_dot_image(&sy, &st, h, w);
    lap("brace dot image");
    let multi = multi_staffs(&staffs, &sy, &st, h, w).unwrap();
    lap("multi_staffs (all)");
    let ps: Vec<usize> = page["preprocessed_shape"]
        .as_array()
        .unwrap()
        .iter()
        .map(|v| v.as_u64().unwrap() as usize)
        .collect();
    let image = homr_rs::warp::Gray {
        w: ps[1],
        h: ps[0],
        data: rd("preprocessed"),
    };
    let regions = homr_rs::staff_image::Regions {
        centers: multi
            .iter()
            .flat_map(|m| m.staffs.iter().map(|s| (s.min_y, s.max_y)))
            .collect(),
    };
    for m in &multi {
        for s in &m.staffs {
            let _ = homr_rs::staff_image::prepare_staff(s, &image, &regions);
            lap("one staff image");
        }
    }
}
