//! homr's system_repair.py: repairs the grouping of staffs into systems before the
//! transcription (a grand staff whose brace was missed, systems joined by a crease, a
//! system which repeats the layout, a single staff below the last system...).
//!
//! homr tells staffs apart by object identity; here every staff carries an id, new
//! for each staff made by a merge or a split.

use crate::braces::{merge_staffs, split_staff, unit_size, MultiStaff};
use crate::score::find_periodic_core;
use crate::staff::{Staff, StaffPoint, LINES};
use std::collections::{HashMap, HashSet};

type Layout = Vec<bool>;

#[derive(Clone)]
struct Node {
    id: usize,
    staff: Staff,
    grand: bool,
}

/// A MultiStaff.
#[derive(Clone)]
struct System {
    nodes: Vec<Node>,
}

impl System {
    /// _layout: is_grandstaff of each staff.
    fn layout(&self) -> Layout {
        self.nodes.iter().map(|n| n.grand).collect()
    }
    fn ids(&self) -> HashSet<usize> {
        self.nodes.iter().map(|n| n.id).collect()
    }
}

struct Ids(usize);

impl Ids {
    fn next(&mut self) -> usize {
        self.0 += 1;
        self.0 - 1
    }
}

fn flatten(rows: &[System]) -> Vec<Node> {
    rows.iter().flat_map(|r| r.nodes.clone()).collect()
}

fn is_uniform(rows: &[System]) -> bool {
    rows.iter()
        .map(|r| r.layout())
        .collect::<HashSet<_>>()
        .len()
        == 1
}

/// np.median of floats
fn median(mut v: Vec<f64>) -> f64 {
    v.sort_by(|a, b| a.partial_cmp(b).unwrap());
    let n = v.len();
    if n % 2 == 1 {
        v[n / 2]
    } else {
        (v[n / 2 - 1] + v[n / 2]) / 2.0
    }
}

/// A Counter, in the order its keys first appeared.
fn count(layouts: impl Iterator<Item = Layout>) -> Vec<(Layout, usize)> {
    let mut counts: Vec<(Layout, usize)> = Vec::new();
    for l in layouts {
        match counts.iter_mut().find(|(k, _)| *k == l) {
            Some((_, n)) => *n += 1,
            None => counts.push((l, 1)),
        }
    }
    counts
}

fn count_of(counts: &[(Layout, usize)], layout: &Layout) -> usize {
    counts.iter().find(|(k, _)| k == layout).map_or(0, |c| c.1)
}

/// _most_common: the most common layout, None if there is none or two are equally common.
fn most_common(counts: &[(Layout, usize)]) -> Option<Layout> {
    let max = counts.iter().map(|c| c.1).max()?;
    let mut best = counts.iter().filter(|c| c.1 == max);
    let first = best.next()?;
    if best.next().is_some() {
        return None;
    }
    Some(first.0.clone())
}

/// _vertical_gap: the gap between two staffs where both have points, None if there is
/// no such x.
fn vertical_gap(upper: &Staff, lower: &Staff) -> Option<f64> {
    let mut upper_points: HashMap<i64, &StaffPoint> = HashMap::new();
    for p in &upper.grid {
        upper_points.insert(crate::npmath::py_round(p.x), p);
    }
    let gaps: Vec<f64> = lower
        .grid
        .iter()
        .filter_map(|p| {
            upper_points
                .get(&crate::npmath::py_round(p.x))
                .map(|u| p.y[0] - u.y[u.y.len() - 1])
        })
        .collect();
    (!gaps.is_empty()).then(|| median(gaps))
}

/// _possible_grandstaffs: pairs of neighboring single staffs (by id) which can form a
/// grand staff: they overlap and, in unit sizes, are not further apart than the grand
/// staffs on the page. If strict, staffs of different systems must also be closer to
/// each other than to single staffs right above and below, else the pairing is a guess.
/// Without guess_lone_staffs, this also applies to two staffs which are systems of their
/// own, as evenly spaced single staffs don't show which belong together.
fn possible_grandstaffs(
    rows: &[System],
    strict: bool,
    guess_lone_staffs: bool,
) -> HashSet<(usize, usize)> {
    let flat = flatten(rows);
    let system_of: HashMap<usize, usize> = rows
        .iter()
        .enumerate()
        .flat_map(|(i, r)| r.nodes.iter().map(move |n| (n.id, i)))
        .collect();
    let lines = LINES;
    let grandstaff_gaps: Vec<f64> = flat
        .iter()
        .filter(|n| n.grand)
        .flat_map(|n| n.staff.grid.iter())
        .filter(|p| p.y.len() == 2 * lines)
        .map(|p| (p.y[lines] - p.y[lines - 1]) * (lines - 1) as f64 / (p.y[lines - 1] - p.y[0]))
        .collect();
    let max_gap = if grandstaff_gaps.is_empty() {
        f64::INFINITY
    } else {
        1.5 * median(grandstaff_gaps)
    };
    let measured: Vec<Option<f64>> = flat
        .windows(2)
        .map(|p| vertical_gap(&p[0].staff, &p[1].staff))
        .collect();
    // Without overlap, the distance of the bounding boxes is good enough for a comparison
    let gaps: Vec<f64> = flat
        .windows(2)
        .zip(&measured)
        .map(|(p, gap)| gap.unwrap_or(p[1].staff.min_y - p[0].staff.max_y))
        .collect();
    let mut result = HashSet::new();
    for (i, pair) in flat.windows(2).enumerate() {
        let (upper, lower) = (&pair[0], &pair[1]);
        let Some(gap) = measured[i] else { continue };
        let unit = unit_size(&upper.staff).min(unit_size(&lower.staff));
        if upper.grand || lower.grand || gap > max_gap * unit {
            continue;
        }
        let neighbors: Vec<f64> = [i.checked_sub(1), Some(i + 1)]
            .into_iter()
            .flatten()
            .filter(|&j| j < gaps.len() && !flat[j].grand && !flat[j + 1].grand)
            .map(|j| gaps[j])
            .collect();
        let connected = system_of[&upper.id] == system_of[&lower.id];
        let lone = [upper, lower]
            .iter()
            .all(|n| rows[system_of[&n.id]].nodes.len() == 1);
        let guess = !strict && (guess_lone_staffs || !lone);
        if guess || connected || neighbors.iter().all(|n| gap < 0.8 * n) {
            result.insert((upper.id, lower.id));
        }
    }
    result
}

/// _fit_to_layout: fits the staffs to the layout. Two single staffs in grandstaffs can
/// form a grand staff, if its brace was missed. None if the staffs don't fit.
fn fit_to_layout(
    staffs: &[Node],
    layout: &[bool],
    grandstaffs: &HashSet<(usize, usize)>,
    ids: &mut Ids,
) -> Result<Option<Vec<Node>>, String> {
    let mut result: Vec<Node> = Vec::new();
    let mut i = 0;
    for &is_grandstaff in layout {
        if i < staffs.len() && staffs[i].grand == is_grandstaff {
            result.push(staffs[i].clone());
            i += 1;
        } else if is_grandstaff
            && i + 1 < staffs.len()
            && !staffs[i].grand
            && !staffs[i + 1].grand
            && grandstaffs.contains(&(staffs[i].id, staffs[i + 1].id))
        {
            let staff = merge_staffs(&staffs[i].staff, &staffs[i + 1].staff)?;
            result.push(Node {
                id: ids.next(),
                staff,
                grand: true,
            });
            i += 2;
        } else {
            return Ok(None);
        }
    }
    Ok((i == staffs.len()).then_some(result))
}

/// _fit_neighbors: fits the systems from start on to the layout. Gives the staffs and
/// the next index. Without a grand staff in the layout, only systems with a connected
/// system among them are joined, else nothing tells which single staffs belong together.
fn fit_neighbors(
    rows: &[System],
    start: usize,
    layout: &[bool],
    grandstaffs: &HashSet<(usize, usize)>,
    ids: &mut Ids,
) -> Result<Option<(Vec<Node>, usize)>, String> {
    for end in start + 1..=rows.len() {
        let group = flatten(&rows[start..end]);
        if group.len() > 2 * layout.len() {
            return Ok(None);
        }
        if !layout.iter().any(|&g| g) && group.len() == end - start && end - start > 1 {
            continue;
        }
        if let Some(fitted) = fit_to_layout(&group, layout, grandstaffs, ids)? {
            return Ok(Some((fitted, end)));
        }
    }
    Ok(None)
}

/// _repair_systems: splits systems which repeat the layout and joins neighbor systems
/// which together have it. Other systems are kept, as the layout can change on a page,
/// e.g. a piano intro before the voice enters. Two single staffs become a grand staff
/// only if they pair up, see possible_grandstaffs.
fn repair_systems_to(
    rows: &[System],
    layout: &[bool],
    strict: bool,
    guess_lone_staffs: bool,
    ids: &mut Ids,
) -> Result<Vec<System>, String> {
    let grandstaffs = possible_grandstaffs(rows, strict, guess_lone_staffs);
    let mut result: Vec<System> = Vec::new();
    let mut i = 0;
    while i < rows.len() {
        if rows[i].layout() == layout {
            result.push(rows[i].clone());
            i += 1;
            continue;
        }
        let row = &rows[i].nodes;
        let repeats = row.len() / layout.len();
        if repeats > 1 && rows[i].layout() == layout.repeat(repeats) {
            result.extend(
                row.chunks(layout.len())
                    .map(|c| System { nodes: c.to_vec() }),
            );
            i += 1;
            continue;
        }
        match fit_neighbors(rows, i, layout, &grandstaffs, ids)? {
            None => {
                result.push(rows[i].clone());
                i += 1;
            }
            Some((fitted, end)) => {
                result.push(System { nodes: fitted });
                i = end;
            }
        }
    }
    Ok(result)
}

/// _repair_score: the staves in systems with the layout after the repair. Systems which
/// split a detected system or merge single staffs into a grand staff count half, as
/// they contradict the detection. Joined neighbor systems count fully, as connections
/// are often missed.
fn repair_score(rows: &[System], layout: &[bool], ids: &mut Ids) -> Result<f64, String> {
    let system_of: HashMap<usize, usize> = rows
        .iter()
        .enumerate()
        .flat_map(|(i, r)| r.nodes.iter().map(move |n| (n.id, i)))
        .collect();
    let grand = layout.iter().filter(|&&g| g).count();
    let mut score = 0.0;
    for row in repair_systems_to(rows, layout, true, true, ids)? {
        if row.layout() != layout {
            continue;
        }
        let own = row.ids();
        let contradicts = row.nodes.iter().any(|n| match system_of.get(&n.id) {
            None => true,
            Some(&source) => !rows[source].ids().is_subset(&own),
        });
        score += (row.nodes.len() + grand) as f64 * if contradicts { 0.5 } else { 1.0 };
    }
    Ok(score)
}

/// _mixed_layouts: layouts of systems and neighbor systems which mix single and grand
/// staffs.
fn mixed_layouts(rows: &[System]) -> Vec<(Layout, usize)> {
    let mut mixed: Vec<Layout> = Vec::new();
    for row in rows {
        let layout = row.layout();
        if layout.iter().any(|&g| g) && layout.iter().any(|&g| !g) {
            mixed.push(layout);
        }
    }
    for pair in rows.windows(2) {
        let layout: Layout = pair[0]
            .layout()
            .into_iter()
            .chain(pair[1].layout())
            .collect();
        if layout == [false, true] || layout == [true, false] {
            mixed.push(layout);
        }
    }
    count(mixed.into_iter())
}

fn is_shown_by_mixed_systems(rows: &[System], layout: &Layout) -> bool {
    let mixed = mixed_layouts(rows);
    most_common(&mixed).as_ref() == Some(layout) && count_of(&mixed, layout) > 1
}

/// _common_layout: the layout most systems share. Systems and neighbor systems which mix
/// single and grand staffs decide, if a layout of them occurs at least twice, as they
/// show how a system is built. Otherwise the layout which repairs the most staves
/// decides, e.g. two single staffs as close as a grand staff whose brace was missed.
/// Layouts of several grand staffs need to be detected twice, as a crease can connect
/// them.
fn common_layout(rows: &[System], ids: &mut Ids) -> Result<Option<Layout>, String> {
    let mixed = mixed_layouts(rows);
    if let Some(layout) = most_common(&mixed) {
        if count_of(&mixed, &layout) > 1 {
            return Ok(Some(layout));
        }
    }
    let counts = count(rows.iter().map(|r| r.layout()));
    let mut candidates: Vec<Layout> = counts.iter().chain(&mixed).map(|c| c.0.clone()).collect();
    candidates.sort();
    candidates.dedup();
    candidates.retain(|l| !(l.len() > 1 && l.iter().all(|&g| g) && count_of(&counts, l) < 2));
    let mut best: Option<(Layout, f64, usize)> = None;
    for layout in candidates {
        let score = repair_score(rows, &layout, ids)?;
        let n = count_of(&counts, &layout);
        if best.as_ref().is_none_or(|b| (score, n) > (b.1, b.2)) {
            best = Some((layout, score, n));
        }
    }
    Ok(best.map(|b| b.0))
}

/// _drop_single_staff_at_end: on a page of grand staffs, a single staff below the last
/// system is most likely a falsely detected staff. Drops it if it is the only single staff.
fn drop_single_staff_at_end(mut rows: Vec<System>, layout: &[bool]) -> Vec<System> {
    let single: Vec<usize> = (0..rows.len())
        .filter(|&i| rows[i].layout() == [false])
        .collect();
    let only_grandstaffs = !layout.contains(&false);
    if !only_grandstaffs || single != [rows.len().wrapping_sub(1)] || rows.len() < 3 {
        return rows;
    }
    rows.pop();
    rows
}

/// _regroup_by_pattern: groups the staffs by a repeating pattern of single and grand
/// staffs. Staffs at the edges which break the pattern stay systems of their own, but
/// only if nothing was grouped before. None if there is no such pattern.
fn regroup_by_pattern(rows: &[System]) -> Option<Vec<System>> {
    let flat = flatten(rows);
    let layout: Vec<bool> = flat.iter().map(|n| n.grand).collect();
    let (period, front, back) = find_periodic_core(&layout)?;
    let end = flat.len() - back;
    let pattern: HashSet<bool> = flat[front..front + period]
        .iter()
        .map(|n| n.grand)
        .collect();
    let mixes = pattern.len() == 2;
    let trimmed = front > 0 || back > 0;
    let nothing_grouped = rows.iter().all(|r| r.nodes.len() == 1);
    if !mixes || (trimmed && !nothing_grouped) {
        return None;
    }
    let single = |n: &Node| System {
        nodes: vec![n.clone()],
    };
    let mut result: Vec<System> = flat[..front].iter().map(single).collect();
    result.extend((front..end).step_by(period).map(|i| System {
        nodes: flat[i..(i + period).min(flat.len())].to_vec(),
    }));
    result.extend(flat[end..].iter().map(single));
    Some(result)
}

/// _convert_to_layout: merges or splits grand staffs so that the row has the layout.
/// None if a grand staff can't be split or two staffs can't be merged.
fn convert_to_layout(
    row: &System,
    layout: &[bool],
    ids: &mut Ids,
) -> Result<Option<System>, String> {
    if row.layout() == layout {
        return Ok(Some(row.clone()));
    }
    let mut singles: Vec<Staff> = Vec::new();
    for n in &row.nodes {
        if n.grand {
            let Some((upper, lower)) = split_staff(&n.staff) else {
                return Ok(None);
            };
            singles.push(upper);
            singles.push(lower);
        } else {
            singles.push(n.staff.clone());
        }
    }
    let mut remaining = singles.into_iter();
    let mut nodes = Vec::new();
    for &is_grandstaff in layout {
        let mut staff = remaining.next().ok_or("StopIteration")?;
        if is_grandstaff {
            let lower = remaining.next().ok_or("StopIteration")?;
            if vertical_gap(&staff, &lower).is_none() {
                return Ok(None);
            }
            staff = merge_staffs(&staff, &lower)?;
        }
        nodes.push(Node {
            id: ids.next(),
            staff,
            grand: is_grandstaff,
        });
    }
    Ok(Some(System { nodes }))
}

/// _unify_grand_staffs: if every system has the same number of staffs, but in some
/// systems a brace was missed or wrongly found, converts them to the most common layout.
fn unify_grand_staffs(rows: Vec<System>, ids: &mut Ids) -> Result<Vec<System>, String> {
    let per_system: HashSet<usize> = rows
        .iter()
        .map(|r| r.nodes.iter().map(|n| if n.grand { 2 } else { 1 }).sum())
        .collect();
    if per_system.len() != 1 {
        return Ok(rows);
    }
    let Some(layout) = most_common(&count(rows.iter().map(|r| r.layout()))) else {
        return Ok(rows);
    };
    let mut result = Vec::new();
    for row in &rows {
        match convert_to_layout(row, &layout, ids)? {
            Some(converted) => result.push(converted),
            None => return Ok(rows),
        }
    }
    Ok(result)
}

/// _ensure_same_number_of_staffs: repairs the grouping of staffs into systems towards
/// the layout most systems share. Systems with another layout are kept, as the layout
/// can change on a page.
fn ensure_same_number_of_staffs(rows: Vec<System>, ids: &mut Ids) -> Result<Vec<System>, String> {
    let rows = unify_grand_staffs(rows, ids)?;
    if is_uniform(&rows) && rows[0].nodes.len() > 1 {
        return Ok(rows);
    }
    let layout = common_layout(&rows, ids)?;
    let repaired = match &layout {
        None => rows.clone(),
        Some(layout) => {
            // Single staffs only pair up by a guess if systems show how the page is built
            let guess_lone_staffs = is_shown_by_mixed_systems(&rows, layout);
            repair_systems_to(&rows, layout, false, guess_lone_staffs, ids)?
        }
    };
    if !is_uniform(&repaired) {
        // Nothing grouped, or e.g. a crease connected staffs of different systems
        if let Some(regrouped) = regroup_by_pattern(&rows) {
            return Ok(regrouped);
        }
    }
    Ok(match layout {
        None => repaired,
        Some(layout) => drop_single_staff_at_end(repaired, &layout),
    })
}

/// repair_systems: repairs the grouping of staffs into systems, see
/// ensure_same_number_of_staffs.
pub fn repair_systems(multi: Vec<MultiStaff>) -> Result<Vec<MultiStaff>, String> {
    let mut next = 0;
    let rows: Vec<System> = multi
        .into_iter()
        .map(|m| System {
            nodes: m
                .staffs
                .into_iter()
                .zip(m.grand)
                .map(|(staff, grand)| {
                    next += 1;
                    Node {
                        id: next - 1,
                        staff,
                        grand,
                    }
                })
                .collect(),
        })
        .collect();
    let mut ids = Ids(next);
    Ok(ensure_same_number_of_staffs(rows, &mut ids)?
        .into_iter()
        .map(|r| MultiStaff {
            grand: r.nodes.iter().map(|n| n.grand).collect(),
            staffs: r.nodes.into_iter().map(|n| n.staff).collect(),
        })
        .collect())
}
