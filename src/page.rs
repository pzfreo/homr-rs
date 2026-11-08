//! homr's page analysis (main.detect_staffs_in_image) from segmentation masks
//! to MultiStaffs.

use crate::braces::{multi_staffs, MultiStaff};
use crate::geom::{RBox, NO_FRAGMENT};
use crate::staff::{detect_staff, Fragments};
use crate::symbols::{page_symbols, Masks};

pub struct PageMasks<'a> {
    pub h: usize,
    pub w: usize,
    pub notehead: &'a [u8],
    pub symbols: &'a [u8],
    pub staff: &'a [u8],
    pub clefs_keys: &'a [u8],
    pub stem_rests: &'a [u8],
}

pub fn analyse_page(m: &PageMasks) -> Result<Vec<MultiStaff>, String> {
    let s = page_symbols(&Masks {
        h: m.h,
        w: m.w,
        notehead: m.notehead,
        staff: m.staff,
        clefs_keys: m.clefs_keys,
        stem_rests: m.stem_rests,
    })?;
    let boxes: Vec<RBox> = s
        .fragments
        .iter()
        .enumerate()
        .map(|(i, f)| f.to_rbox(i))
        .collect();
    let contours: Vec<Vec<[i32; 2]>> = s.fragments.iter().map(|f| f.contour.clone()).collect();
    let clefs: Vec<RBox> = s
        .clefs_keys
        .iter()
        .map(|c| c.to_rbox(NO_FRAGMENT))
        .collect();
    let bars: Vec<RBox> = s.bar_lines.iter().map(|c| c.to_rbox(NO_FRAGMENT)).collect();
    let staffs = detect_staff(
        m.staff,
        Some(m.symbols),
        m.h,
        m.w,
        &Fragments {
            boxes: &boxes,
            contours: &contours,
        },
        &clefs,
        &bars,
    )?;
    if staffs.is_empty() {
        return Err("No staffs found".into());
    }
    multi_staffs(&staffs, m.symbols, m.staff, m.h, m.w)
}
