//! autocrop against homr's autocrop_with_offset: the images and crops dumped by
//! tools/dump_autocrop.py. Crop size, offset and every pixel must match; the exit
//! status is 1 if one doesn't.   check_autocrop <cases dir> <reference dir>
use homr_rs::preprocess::{autocrop, Bgr};
use serde_json::Value;
use std::fs;

fn main() {
    let cases = std::env::args().nth(1).expect("cases dir");
    let refs = std::env::args().nth(2).expect("reference dir");
    let mut names: Vec<String> = fs::read_dir(&refs)
        .unwrap()
        .map(|e| e.unwrap().file_name().to_string_lossy().to_string())
        .filter(|n| {
            std::path::Path::new(&refs)
                .join(n)
                .join("ref.json")
                .exists()
        })
        .collect();
    names.sort();
    let (mut same, mut diff, mut cropped, mut painted) = (0, 0, 0, 0);
    for name in names {
        let c = std::path::Path::new(&cases).join(&name);
        let r = std::path::Path::new(&refs).join(&name);
        let meta: Value =
            serde_json::from_str(&fs::read_to_string(c.join("meta.json")).unwrap()).unwrap();
        let shape: Vec<usize> = meta["shape"]
            .as_array()
            .unwrap()
            .iter()
            .map(|v| v.as_u64().unwrap() as usize)
            .collect();
        let image = Bgr {
            h: shape[0],
            w: shape[1],
            data: fs::read(c.join("image.bin")).unwrap(),
        };
        let want: Value =
            serde_json::from_str(&fs::read_to_string(r.join("ref.json")).unwrap()).unwrap();
        let want_shape: Vec<usize> = want["shape"]
            .as_array()
            .unwrap()
            .iter()
            .map(|v| v.as_u64().unwrap() as usize)
            .collect();
        let want_offset: Vec<usize> = want["offset"]
            .as_array()
            .unwrap()
            .iter()
            .map(|v| v.as_u64().unwrap() as usize)
            .collect();
        let want_data = fs::read(r.join("crop.bin")).unwrap();
        let (crop, (x, y)) = autocrop(&image);
        if (crop.h, crop.w) != (want_shape[0], want_shape[1])
            || (x, y) != (want_offset[0], want_offset[1])
        {
            diff += 1;
            println!(
                "DIFF {name}: crop {}x{} at ({x}, {y}), homr {}x{} at ({}, {})",
                crop.w, crop.h, want_shape[1], want_shape[0], want_offset[0], want_offset[1]
            );
            continue;
        }
        let wrong = crop
            .data
            .iter()
            .zip(&want_data)
            .filter(|(a, b)| a != b)
            .count();
        if wrong > 0 {
            diff += 1;
            println!("DIFF {name}: {wrong} of {} bytes differ", want_data.len());
        } else {
            same += 1;
            if (x, y) != (0, 0) || crop.w != image.w || crop.h != image.h {
                cropped += 1;
            }
            // pixels painted over: differ from the same region of the input
            let from_input = (0..crop.h).any(|row| {
                let src = ((y + row) * image.w + x) * 3;
                crop.data[row * crop.w * 3..(row + 1) * crop.w * 3]
                    != image.data[src..src + crop.w * 3]
            });
            if from_input {
                painted += 1;
            }
        }
    }
    println!("{same} same, {diff} different; {cropped} of the same were cropped, {painted} had surroundings painted over");
    if diff > 0 {
        std::process::exit(1);
    }
}
