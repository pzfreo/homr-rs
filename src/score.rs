//! From a page's MultiStaffs to homr's MusicXML (staff_parsing.parse_staffs,
//! parse_staff_image, predict_best; main.process_image), with the transformer
//! supplied by the caller.

use crate::braces::MultiStaff;
use crate::musicxml::generate_xml;
use crate::staff::Staff;
use crate::staff_image::{prepare_staff, Regions};
use crate::systems::repair_systems;
use crate::tokens::{is_lower, remove_duplicated_symbols, Symbol};
use crate::triplets::restore_unmarked_triplets;
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

/// _clean_up_voice: the symbols of a voice with duplicates removed and the
/// triplets which were printed without their number restored.
pub fn clean_up_voice(symbols: &[Symbol]) -> Vec<Symbol> {
    restore_unmarked_triplets(&remove_duplicated_symbols(symbols))
}

/// _slots_in_layout: the staffs of the layout which the staffs of a system are, None if
/// that isn't unique, e.g. a system shows only the piano as the voice is hidden.
fn slots_in_layout(row: &[bool], layout: &[bool]) -> Option<Vec<usize>> {
    // itertools.combinations(range(len(layout)), len(row)) keeping those which match, in
    // its order; only whether there is exactly one matters, so stop at the second
    fn walk(
        row: &[bool],
        layout: &[bool],
        from: usize,
        now: &mut Vec<usize>,
        found: &mut Vec<Vec<usize>>,
    ) {
        if found.len() > 1 {
            return;
        }
        if now.len() == row.len() {
            found.push(now.clone());
            return;
        }
        for slot in from..layout.len() {
            if layout[slot] == row[now.len()] {
                now.push(slot);
                walk(row, layout, slot + 1, now, found);
                now.pop();
            }
        }
    }
    let mut found = Vec::new();
    walk(row, layout, 0, &mut Vec::new(), &mut found);
    (found.len() == 1).then(|| found.remove(0))
}

/// _assign_parts: for every system the parts which its staffs belong to; the parts are
/// the staffs of the largest system. None if a system doesn't fit into it unambiguously.
fn assign_parts(rows: &[MultiStaff]) -> Option<Vec<Vec<usize>>> {
    let mut layouts: Vec<&[bool]> = Vec::new();
    for r in rows {
        if !layouts.contains(&r.grand.as_slice()) {
            layouts.push(&r.grand);
        }
    }
    let largest = layouts.iter().copied().max_by_key(|l| l.len())?;
    if layouts
        .iter()
        .any(|l| l.len() == largest.len() && *l != largest)
    {
        return None;
    }
    rows.iter()
        .map(|r| slots_in_layout(&r.grand, largest))
        .collect()
}

/// _measure_rests_like: rests for a staff which is hidden in a system: one per measure of
/// the symbols another staff of the system has, with the same barlines.
pub fn measure_rests_like(symbols: &[Symbol], is_grandstaff: bool) -> Vec<Symbol> {
    let positions: &[&str] = if is_grandstaff {
        &["upper", "lower"]
    } else {
        &["upper"]
    };
    let mut result = Vec::new();
    let mut in_measure = false;
    for symbol in symbols {
        let (r, in_notes) = (
            &symbol.rhythm,
            symbol.rhythm.starts_with("note") || symbol.rhythm.starts_with("rest"),
        );
        if in_notes && !in_measure {
            for (i, position) in positions.iter().enumerate() {
                if i > 0 {
                    result.push(Symbol::new("chord"));
                }
                let mut rest = Symbol::new("measureRest");
                rest.position = position.to_string();
                result.push(rest);
            }
            in_measure = true;
        } else if r.contains("barline") || r.contains("repeat") {
            result.push(Symbol::new(r));
            in_measure = false;
        }
    }
    result
}

/// Reads one staff of parse_staffs: its index, the staff, if it's a grand staff, the regions
/// of the page; gives its symbols.
pub type StaffSymbols<'a> =
    dyn FnMut(usize, &Staff, bool, &Regions) -> Result<Vec<Symbol>, String> + 'a;

/// parse_staffs: the voices of a page (without homr's selected_staff option, which reads one
/// staff for debugging). `staff_symbols(index, staff, is_grandstaff, regions)`
/// reads one staff (the transformer); it is called in homr's order, a voice at a time. A
/// system with fewer staffs than the largest one hides some (a voice which only shows in
/// some systems): its staffs are matched to the parts of the largest, and the hidden ones
/// get a measure rest for every measure of a staff which is shown.
pub fn parse_staffs_with(
    multi: Vec<MultiStaff>,
    staff_symbols: &mut StaffSymbols,
) -> Result<Vec<Vec<Symbol>>, String> {
    let mut staffs: Vec<MultiStaff> = multi;
    let mut parts: Vec<Vec<usize>> = staffs
        .iter()
        .map(|r| (0..r.staffs.len()).collect())
        .collect();
    let counts: std::collections::HashSet<usize> = staffs.iter().map(|r| r.staffs.len()).collect();
    if counts.len() > 1 {
        // The voices are read across systems, a system with fewer staffs hides some
        match assign_parts(&staffs) {
            None => {
                // The layout changes on the page, reading every staff on its own
                staffs = staffs
                    .iter()
                    .flat_map(|r| {
                        r.staffs.iter().zip(&r.grand).map(|(s, &g)| MultiStaff {
                            staffs: vec![s.clone()],
                            grand: vec![g],
                        })
                    })
                    .collect();
                parts = vec![vec![0]; staffs.len()];
            }
            Some(assigned) => parts = assigned,
        }
    }
    // For simplicity we call every staff in a multi staff a voice, even if it's part of a
    // grand staff.
    let number_of_voices = staffs
        .iter()
        .map(|r| r.staffs.len())
        .max()
        .ok_or("ValueError: max() arg is an empty sequence")?;
    let layout = staffs
        .iter()
        .find(|r| r.staffs.len() == number_of_voices)
        .map(|r| r.grand.clone())
        .ok_or("StopIteration")?;
    let regions = Regions {
        centers: staffs
            .iter()
            .flat_map(|r| r.staffs.iter().map(|s| (s.min_y, s.max_y)))
            .collect(),
    };
    let mut i = 0;
    let mut results: std::collections::HashMap<(usize, usize), Vec<Symbol>> =
        std::collections::HashMap::new();
    for voice in 0..number_of_voices {
        for (staff_index, row) in staffs.iter().enumerate() {
            let Some(at) = parts[staff_index].iter().position(|&p| p == voice) else {
                continue;
            };
            let result = staff_symbols(i, &row.staffs[at], row.grand[at], &regions)?;
            i += 1;
            if result.is_empty() {
                continue; // Skipping empty staff
            }
            results.insert((staff_index, voice), result);
        }
    }
    let mut voices = Vec::new();
    for (voice, &voice_is_grandstaff) in layout.iter().enumerate() {
        let mut result_for_voice: Vec<Symbol> = Vec::new();
        for (staff_index, slots) in parts.iter().enumerate() {
            let symbols = if slots.contains(&voice) {
                results.get(&(staff_index, voice)).cloned()
            } else {
                // a part this system hides: a measure rest for each measure of the first
                // staff of the system which was read
                slots
                    .iter()
                    .find_map(|v| results.get(&(staff_index, *v)))
                    .map(|shown| measure_rests_like(shown, voice_is_grandstaff))
            };
            let Some(symbols) = symbols else { continue };
            result_for_voice.extend(symbols);
            result_for_voice.push(Symbol::new("newline"));
        }
        voices.push(clean_up_voice(&result_for_voice));
    }
    Ok(voices)
}

/// The page's MusicXML from its MultiStaffs (homr's process_image after page analysis:
/// repair_systems, parse_staffs, generate_xml). `predict` is the transformer: canvas in,
/// symbols out.
pub fn page_xml(
    multi: &[MultiStaff],
    image: &Gray,
    page: &PageMapping,
    predict: &mut dyn FnMut(&Gray) -> Result<Vec<Symbol>, String>,
) -> Result<String, String> {
    let multi = repair_systems(multi.to_vec())?;
    let voices = parse_staffs_with(multi, &mut |_, staff, grand, regions| {
        let prepared = prepare_staff(staff, image, regions)?;
        let mut result: Vec<Symbol> = predict(&prepared.canvas)?;
        if !grand {
            result.retain(|s| !is_lower(&s.position));
        }
        for s in &mut result {
            if let Some(c) = symbol_center(s) {
                s.image_coordinates = Some(page.apply(prepared.mapping.to_page(c)));
            }
        }
        Ok(result)
    })?;
    generate_xml(&voices, "")
}
