//! Whole pages against homr: from the decoded image to the MusicXML, all in Rust (autocrop,
//! resize, CLAHE, tiles, masks, page analysis, system repair, staff images, the Rust encoder
//! and decoder, voices, MusicXML), except the segmentation network, whose class maps of the
//! tiles come from native homr (tools/dump_segnet.py: that network runs elsewhere in an app).
//! Compared with native homr's masks and with the MusicXML it wrote (tools/run_corpus.py); a page
//! homr fails on must fail here. Exit status 1 if a page differs.
//!
//! The contrast-adjusted page (CLAHE) differs from native homr's in a few pixels, by one level:
//! native OpenCV fuses multiply-adds. The staff images are cut from that page, so for the pass or
//! fail the steps after it read homr's page; how many pixels our own CLAHE differs by, and by how
//! much, is reported (OWN_CLAHE=1 also runs the page with our own).
//!   check_pages <segnet dumps> <homr output dir> <staff image dumps> <encoder.onnx> <decoder.onnx> [pages...]
//! (SAVE_DIR=<dir> keeps the MusicXML of pages which differ)
//!
//! The staff images the transformer reads are cut from the page by arithmetic which native
//! homr (fused multiply-adds) does one grey level differently from here in about 1% of the pixels,
//! enough to move a symbol's position by a fraction of a pixel. So the encoder reads homr's own
//! staff images (tools/dump_canvases.py) for the pass or fail, and how far ours are is reported
//! (OWN_CANVAS=1 reads ours).
use homr_rs::decoder::Decoder;
use homr_rs::encoder::Encoder;
use homr_rs::page::{analyse_page, PageMasks};
use homr_rs::preprocess::{
    autocrop, bgr_to_gray, clahe, masks, merge_tiles, postprocess, resize_image, tile_origins, Bgr,
};
use homr_rs::score::{page_xml, PageMapping};
use homr_rs::staff_image::to_model_input;
use homr_rs::tokens::{Symbol, Vocab};
use homr_rs::warp::Gray;
use serde_json::Value;
use std::fs;
use std::path::Path;

fn differs(label: &str, ours: &[u8], theirs: &[u8], problems: &mut Vec<String>) {
    if ours.len() != theirs.len() {
        problems.push(format!("{label}: {} vs {} bytes", ours.len(), theirs.len()));
    } else {
        let n = ours.iter().zip(theirs).filter(|(a, b)| a != b).count();
        if n > 0 {
            problems.push(format!("{label}: {n} of {} differ", ours.len()));
        }
    }
}

fn main() {
    let args: Vec<String> = std::env::args().collect();
    let (seg, refs) = (Path::new(&args[1]), Path::new(&args[2]));
    let canvases = Path::new(&args[3]);
    let encoder = Encoder::from_onnx(&fs::read(&args[4]).unwrap()).unwrap();
    let decoder = Decoder::from_onnx(&fs::read(&args[5]).unwrap()).unwrap();
    let vocab = Vocab::default();
    let mut names: Vec<String> = if args.len() > 6 {
        args[6..].to_vec()
    } else {
        fs::read_dir(seg)
            .unwrap()
            .map(|e| e.unwrap().file_name().to_string_lossy().to_string())
            .collect()
    };
    names.sort();
    let (mut same, mut same_err, mut diff, mut skipped, mut staffs) = (0, 0, 0, 0, 0usize);
    let (mut noise_pixels, mut noise_total, mut noise_max) = (0usize, 0usize, 0u8);
    for name in names {
        let d = seg.join(&name);
        if !d.join("segnet.json").exists() || !refs.join(format!("{name}.json")).exists() {
            skipped += 1;
            continue;
        }
        let meta: Value =
            serde_json::from_str(&fs::read_to_string(d.join("segnet.json")).unwrap()).unwrap();
        let shape: Vec<usize> = meta["image_shape"]
            .as_array()
            .unwrap()
            .iter()
            .map(|v| v.as_u64().unwrap() as usize)
            .collect();
        let image = Bgr {
            h: shape[0],
            w: shape[1],
            data: fs::read(d.join("image.bin")).unwrap(),
        };
        let homr: Value =
            serde_json::from_str(&fs::read_to_string(refs.join(format!("{name}.json"))).unwrap())
                .unwrap();
        let want_xml = fs::read_to_string(refs.join(format!("{name}.musicxml"))).ok();

        let mut problems: Vec<String> = Vec::new();
        let (cropped, crop) = autocrop(&image);
        let pre = bgr_to_gray(&resize_image(&cropped));
        let mut gray = clahe(&pre);
        let (h, w) = (gray.h, gray.w);
        let n = meta["tiles"].as_u64().unwrap() as usize;
        if tile_origins(h, w).len() != n {
            problems.push(format!("{} tiles, homr {n}", tile_origins(h, w).len()));
        }
        let tiles = fs::read(d.join("tiles.bin")).unwrap();
        let classes: Vec<Vec<u8>> = tiles
            .as_chunks::<{ 320 * 320 }>()
            .0
            .iter()
            .map(|c| c.to_vec())
            .collect();
        let m = postprocess(masks(&merge_tiles(&classes, h, w), h, w), &mut gray);
        for (label, ours) in [
            ("staff", &m.staff),
            ("symbols", &m.symbols),
            ("notehead", &m.notehead),
            ("clefs_keys", &m.clefs_keys),
            ("stems_rest", &m.stems_rest),
        ] {
            differs(
                label,
                ours,
                &fs::read(d.join(format!("final_{label}.bin"))).unwrap(),
                &mut problems,
            );
        }
        // our CLAHE against homr's (native, fused multiply-adds): how many pixels, by how much
        let homr_page = fs::read(d.join("final_preprocessed.bin")).unwrap();
        if homr_page.len() != gray.data.len() {
            problems.push(format!(
                "preprocessed: {} vs {} bytes",
                gray.data.len(),
                homr_page.len()
            ));
        }
        let off: Vec<u8> = gray
            .data
            .iter()
            .zip(&homr_page)
            .map(|(a, b)| a.abs_diff(*b))
            .filter(|d| *d > 0)
            .collect();
        let clahe_note = format!(
            "our CLAHE differs in {} pixels, by at most {}",
            off.len(),
            off.iter().max().unwrap_or(&0)
        );
        if off.iter().any(|d| *d > 1) {
            problems.push(format!("preprocessed: {clahe_note}"));
        }
        if std::env::var_os("OWN_CLAHE").is_none() {
            gray.data = homr_page;
        }
        let pm = PageMasks {
            h,
            w,
            notehead: &m.notehead,
            symbols: &m.symbols,
            staff: &m.staff,
            clefs_keys: &m.clefs_keys,
            stem_rests: &m.stems_rest,
        };
        let mapping = PageMapping {
            cropped: (cropped.h, cropped.w),
            preprocessed: (h, w),
            crop: (crop.0 as f64, crop.1 as f64),
        };
        let mut read = 0;
        let got: Result<String, String> = analyse_page(&pm).and_then(|multi| {
            if multi.is_empty() {
                return Err("No staffs found".into());
            }
            let image: &Gray = &gray;
            page_xml(&multi, image, &mapping, &mut |canvas: &Gray| -> Result<
                Vec<Symbol>,
                String,
            > {
                // homr's staff image for this staff, if dumped: ours is compared with it
                let theirs = fs::read(canvases.join(&name).join(format!("canvas-{read}.bin"))).ok();
                read += 1;
                let mut used = Gray {
                    w: canvas.w,
                    h: canvas.h,
                    data: canvas.data.clone(),
                };
                if let Some(theirs) = theirs.filter(|t| t.len() == canvas.data.len()) {
                    for (a, b) in canvas.data.iter().zip(&theirs) {
                        noise_pixels += (a != b) as usize;
                        noise_max = noise_max.max(a.abs_diff(*b));
                    }
                    noise_total += theirs.len();
                    if std::env::var_os("OWN_CANVAS").is_none() {
                        used.data = theirs;
                    }
                }
                let context = encoder.encode(&to_model_input(&used))?;
                let steps = decoder.generate(&context)?;
                Ok(steps.iter().filter_map(|s| s.symbol(&vocab)).collect())
            })
        });
        staffs += read;
        match (&want_xml, &got) {
            (Some(want), Ok(xml)) if xml == want => {}
            (Some(want), Ok(xml)) => {
                let at = xml
                    .bytes()
                    .zip(want.bytes())
                    .position(|(a, b)| a != b)
                    .unwrap_or(xml.len().min(want.len()));
                problems.push(format!(
                    "MusicXML differs from byte {at} (rust {} bytes, homr {})",
                    xml.len(),
                    want.len()
                ));
                if let Some(dir) = std::env::var_os("SAVE_DIR") {
                    fs::create_dir_all(&dir).unwrap();
                    fs::write(Path::new(&dir).join(format!("{name}.rust.musicxml")), xml).unwrap();
                }
            }
            (Some(_), Err(e)) => problems.push(format!("the Rust failed ({e}), homr read it")),
            (None, Ok(_)) => {
                problems.push(format!("homr failed ({}), the Rust read it", homr["error"]))
            }
            (None, Err(_)) => {}
        }
        if problems.is_empty() {
            if want_xml.is_some() {
                same += 1;
            } else {
                same_err += 1;
            }
            println!("SAME {name} ({clahe_note})");
        } else {
            diff += 1;
            println!("DIFF {name}: {}", problems.join("; "));
        }
    }
    println!("{same} pages identical, {same_err} fail the same way, {diff} different, {skipped} skipped; {staffs} staffs read");
    println!("our staff images differ from homr's in {noise_pixels} of {noise_total} pixels, by at most {noise_max} grey levels");
    if diff > 0 {
        std::process::exit(1);
    }
}
