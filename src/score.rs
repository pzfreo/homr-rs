//! From a page's MultiStaffs to homr's MusicXML (staff_parsing.parse_staffs,
//! parse_staff_image, predict_best; main.process_image), with the transformer
//! supplied by the caller.

use crate::braces::MultiStaff;
use crate::musicxml::generate_xml;
use crate::staff::Staff;
use crate::staff_image::{prepare_staff, Regions};
use crate::tokens::{is_lower, remove_duplicated_symbols, Symbol};
use crate::warp::Gray;

/// load_and_preprocess_predictions' mapping from the preprocessed page to the
/// image homr was given: undo the resize to 1920 wide, then the autocrop.
pub struct PageMapping {
    pub cropped: (usize, usize),      // (h, w) after autocrop
    pub preprocessed: (usize, usize), // (h, w) of the page homr analysed
    pub crop: (f64, f64),             // autocrop's top-left corner
}

impl PageMapping {
    pub fn apply(&self, p: (f64, f64)) -> (f64, f64) {
        let sx = self.preprocessed.1 as f64 / self.cropped.1 as f64;
        let sy = self.preprocessed.0 as f64 / self.cropped.0 as f64;
        let p = (p.0 / sx, p.1 / sy);
        (p.0 + self.crop.0, p.1 + self.crop.1)
    }
}

#[derive(Clone)]
pub struct Row {
    pub staffs: Vec<(Staff, bool)>,
}

pub fn sorted_row(mut staffs: Vec<(Staff, bool)>) -> Row {
    staffs.sort_by(|a, b| a.0.min_y.partial_cmp(&b.0.min_y).unwrap());
    Row { staffs }
}

/// _find_periodic_core
pub fn find_periodic_core(layout: &[bool]) -> Option<(usize, usize, usize)> {
    let n = layout.len();
    let mut best: Option<(usize, usize, usize, usize)> = None;
    for period in 1..=n / 2 {
        for front in 0..=period {
            for back in 0..=period {
                if front + back > n {
                    continue;
                }
                let core = &layout[front..n - back];
                if core.len() < 2 * period || !core.len().is_multiple_of(period) {
                    continue;
                }
                let first = &core[..period];
                if !core.chunks(period).all(|r| r == first) {
                    continue;
                }
                let cand = (front + back, period, front, back);
                if best.is_none_or(|b| (cand.0, cand.1) < (b.0, b.1)) {
                    best = Some(cand);
                }
            }
        }
    }
    best.map(|b| (b.1, b.2, b.3))
}

/// _ensure_same_number_of_staffs
pub fn ensure_same_number_of_staffs(rows: Vec<Row>) -> Vec<Row> {
    let lens: std::collections::BTreeSet<usize> = rows.iter().map(|r| r.staffs.len()).collect();
    if lens.len() == 1 && *lens.iter().next().unwrap() > 1 {
        return rows;
    }
    let flat: Vec<(Staff, bool)> = rows.iter().flat_map(|r| r.staffs.clone()).collect();
    let layout: Vec<bool> = flat.iter().map(|s| s.1).collect();
    if let Some((period, front, back)) = find_periodic_core(&layout) {
        let core = &flat[front..flat.len() - back];
        return core
            .chunks(period)
            .map(|c| sorted_row(c.to_vec()))
            .collect();
    }
    let mut out: Vec<Row> = rows
        .into_iter()
        .flat_map(|r| r.staffs.into_iter().map(|s| Row { staffs: vec![s] }))
        .collect();
    out.sort_by(|a, b| {
        a.staffs[0]
            .0
            .min_y
            .partial_cmp(&b.staffs[0].0.min_y)
            .unwrap()
    });
    out
}

pub fn symbol_center(s: &Symbol) -> Option<(f64, f64)> {
    if s.rhythm.starts_with("chord") {
        return None;
    }
    let c = s.coordinates.as_ref()?;
    if c.len() < 2 || c[0].is_nan() || c[1].is_nan() {
        return None;
    }
    Some((c[0], c[1]))
}

/// The page's MusicXML from its MultiStaffs (homr's process_image after page
/// analysis). `predict` is the transformer: canvas in, symbols out.
pub fn page_xml(
    multi: &[MultiStaff],
    image: &Gray,
    page: &PageMapping,
    predict: &mut dyn FnMut(&Gray) -> Result<Vec<Symbol>, String>,
) -> Result<String, String> {
    let rows: Vec<Row> = multi
        .iter()
        .map(|m| Row {
            staffs: m
                .staffs
                .iter()
                .cloned()
                .zip(m.grand.iter().copied())
                .collect(),
        })
        .collect();
    let rows = ensure_same_number_of_staffs(rows);
    let voices_n = rows[0].staffs.len();
    let regions = Regions {
        centers: rows
            .iter()
            .flat_map(|r| r.staffs.iter().map(|s| (s.0.min_y, s.0.max_y)))
            .collect(),
    };
    let mut voices = Vec::new();
    for voice in 0..voices_n {
        let mut result_for_voice: Vec<Symbol> = Vec::new();
        for row in &rows {
            let (staff, grand) = row
                .staffs
                .get(voice)
                .ok_or("IndexError: list index out of range")?;
            let prepared = prepare_staff(staff, image, &regions)?;
            let mut result: Vec<Symbol> = predict(&prepared.canvas)?;
            if !grand {
                result.retain(|s| !is_lower(&s.position));
            }
            for s in &mut result {
                if let Some(c) = symbol_center(s) {
                    s.image_coordinates = Some(page.apply(prepared.mapping.to_page(c)));
                }
            }
            if result.is_empty() {
                continue;
            }
            result.push(Symbol::new("newline"));
            result_for_voice.extend(result);
        }
        voices.push(remove_duplicated_symbols(&result_for_voice));
    }
    generate_xml(&voices, "")
}
