//! homr's triplet restoration (triplet_restoration.py): triplets which are printed
//! without their "3" are read as plain notes by the transformer, e.g. three eighths
//! which take a beat and a half instead of one beat.
//!
//! The symbols of a measure tell which notes start together, and the MusicXML
//! generator derives the timing from that: a chord starts when the earliest note
//! which is still sounding ends. So a measure can be checked without knowing the
//! answer: in music a voice carries on where its previous note ended. For every
//! measure which does not make sense in this way, three notes in a row with the
//! same plain note value are taken as a triplet if that leaves fewer places in the
//! measure where a voice does not carry on.

use crate::rational::Frac;
use crate::tokens::{sort_token_chord_indices, Symbol};
use std::collections::{BTreeMap, BTreeSet};

/// A note or rest of a measure: its voice, its duration and its index in the symbols.
type Note = (String, Frac, usize);
type Measure = Vec<Vec<Note>>;
type Indices = BTreeSet<usize>;
/// The notes of three chords in a row, and if the three start off the beat.
type Three = (Indices, bool);
/// Threes are known by the places of their chords in the measure and their note value.
type ThreesByPlace = BTreeMap<(Vec<usize>, String), Three>;

/// Six eighths against a half note only meet it if both threes are triplets, so
/// threes which follow each other directly are also tried together, up to this many.
const MAX_THREES_IN_A_ROW: usize = 4;
/// One or two measures which make more sense with a triplet are usually misread in
/// some other way: music which has triplets has several. A page (one voice of it)
/// which shows triplets of a note value in fewer places than this shows none.
const MIN_TRIPLETS_ON_A_PAGE: usize = 3;
/// A "measure" which is longer than this many whole notes is music without barlines,
/// or with barlines which were not recognized. There is no measure to make sense of.
const MAX_MEASURE_LENGTH: i64 = 2;

fn nr(r: &str) -> bool {
    r.starts_with("note") || r.starts_with("rest")
}

/// re.match(r"(note|rest)_(4|8|16|32|64)$", rhythm): the kind and the value.
fn plain(rhythm: &str) -> Option<(&str, &str)> {
    let (kind, value) = rhythm.split_once('_')?;
    if (kind == "note" || kind == "rest") && ["4", "8", "16", "32", "64"].contains(&value) {
        Some((kind, value))
    } else {
        None
    }
}

fn as_triplet(symbol: &Symbol) -> Symbol {
    let Some((kind, value)) = plain(&symbol.rhythm) else {
        return symbol.clone();
    };
    let mut s = symbol.clone();
    s.rhythm = format!("{kind}_{}", value.parse::<i64>().unwrap() * 3 / 2);
    s
}

fn split_into_measures(symbols: &[Symbol]) -> Vec<Measure> {
    let mut measures: Vec<Measure> = vec![vec![]];
    for chord in sort_token_chord_indices(symbols) {
        let rhythm = &symbols[chord[0]].rhythm;
        if rhythm.contains("barline") || rhythm.contains("repeat") {
            measures.push(vec![]);
        } else if nr(rhythm) && !rhythm.ends_with('m') {
            measures.last_mut().unwrap().push(
                chord
                    .iter()
                    .filter(|&&i| nr(&symbols[i].rhythm))
                    .map(|&i| (symbols[i].position.clone(), symbols[i].duration().fraction, i))
                    .collect(),
            );
        }
    }
    measures
}

/// Times the measure the way the MusicXML generator does it: a chord starts when
/// the earliest note which is still sounding ends. The notes in `triplets` are
/// played as members of a triplet. Gives the start of every chord together with
/// the voice and the end of each of its notes.
fn play(measure: &Measure, triplets: &Indices) -> Vec<(Frac, Vec<(String, Frac)>)> {
    let mut clock = Frac::ZERO;
    let mut sounding: Vec<Frac> = Vec::new();
    let mut result = Vec::new();
    for chord in measure {
        let mut notes = Vec::new();
        for (voice, duration, index) in chord {
            let played = if triplets.contains(index) {
                *duration * Frac::new(2, 3)
            } else {
                *duration
            };
            if played > Frac::ZERO {
                notes.push((voice.clone(), clock + played));
            }
        }
        let any = !notes.is_empty();
        result.push((clock, notes.clone()));
        if any {
            sounding.extend(notes.iter().map(|(_, end)| *end));
            clock = sounding.iter().filter(|&&e| e > clock).min().copied().unwrap();
            sounding.retain(|&e| e > clock);
        }
    }
    result
}

fn abs_diff(a: Frac, b: Frac) -> Frac {
    if a > b {
        a - b
    } else {
        b - a
    }
}

/// The places where a voice does not carry on exactly where its previous note ended,
/// and the voices which end before the measure does: their number and total size.
fn count_breaks(measure: &Measure, triplets: &Indices) -> (usize, Frac) {
    let mut end_of_voice: BTreeMap<String, Frac> = BTreeMap::new();
    let mut breaks = 0;
    let mut size = Frac::ZERO;
    for (start, notes) in play(measure, triplets) {
        let mut ends: BTreeMap<String, Frac> = BTreeMap::new();
        for (voice, end) in notes {
            let e = ends.entry(voice).or_insert(end);
            if end > *e {
                *e = end;
            }
        }
        for voice in ends.keys() {
            if let Some(&before) = end_of_voice.get(voice) {
                if before != start {
                    breaks += 1;
                    size = size + abs_diff(start, before);
                }
            }
        }
        for (voice, end) in ends {
            end_of_voice.insert(voice, end);
        }
    }
    if let Some(&end_of_measure) = end_of_voice.values().max() {
        for &end in end_of_voice.values() {
            if end != end_of_measure {
                breaks += 1;
                size = size + (end_of_measure - end);
            }
        }
    }
    (breaks, size)
}

fn get_length(measure: &Measure) -> Frac {
    play(measure, &Indices::new())
        .iter()
        .flat_map(|(_, notes)| notes.iter().map(|(_, end)| *end))
        .max()
        .unwrap_or(Frac::ZERO)
}

/// Three chords in a row in one voice which all have the same plain note value.
/// Voices which go through the same three chords with the same value form one three.
///
/// A three is off the beat if it does not start a whole number of threes into its
/// run of equal notes. Taking any three notes of a run as a triplet closes the same
/// gap, so the caller needs this to tell them apart.
fn find_single_threes(measure: &Measure, symbols: &[Symbol]) -> ThreesByPlace {
    let mut found: ThreesByPlace = BTreeMap::new();
    let voices: BTreeSet<&String> = measure.iter().flatten().map(|(v, _, _)| v).collect();
    for voice in voices {
        let chords: Vec<(usize, Vec<usize>)> = measure
            .iter()
            .enumerate()
            .map(|(place, chord)| {
                (
                    place,
                    chord.iter().filter(|(v, _, _)| v == voice).map(|(_, _, i)| *i).collect::<Vec<_>>(),
                )
            })
            .filter(|(_, notes)| !notes.is_empty())
            .collect();
        let values: Vec<Option<&str>> = chords
            .iter()
            .map(|(_, notes)| {
                let matches: Vec<Option<(&str, &str)>> =
                    notes.iter().map(|&i| plain(&symbols[i].rhythm)).collect();
                let values: BTreeSet<&str> = matches.iter().flatten().map(|m| m.1).collect();
                if matches.iter().all(|m| m.is_some()) && values.len() == 1 {
                    values.into_iter().next()
                } else {
                    None
                }
            })
            .collect();
        let mut start_of_run = 0;
        for (i, value) in values.iter().enumerate() {
            if i > 0 && (value.is_none() || *value != values[i - 1]) {
                start_of_run = i;
            }
            let Some(value) = value else { continue };
            if i + 3 > values.len() || values[i..i + 3].iter().any(|v| *v != Some(*value)) {
                continue;
            }
            let in_voice: Vec<usize> = chords[i..i + 3].iter().flat_map(|(_, n)| n.iter().copied()).collect();
            if !in_voice.iter().any(|&idx| symbols[idx].rhythm.starts_with("note")) {
                continue;
            }
            let key = (chords[i..i + 3].iter().map(|(p, _)| *p).collect::<Vec<_>>(), value.to_string());
            let entry = found.entry(key).or_insert((Indices::new(), false));
            entry.0.extend(in_voice);
            entry.1 = entry.1 || (i - start_of_run) % 3 != 0;
        }
    }
    found
}

/// Extends each run of threes by the three which follows it directly in the same voices.
fn add_following_three(
    in_a_row: &ThreesByPlace,
    single: &ThreesByPlace,
    measure: &Measure,
    symbols: &[Symbol],
) -> ThreesByPlace {
    let mut longer: ThreesByPlace = BTreeMap::new();
    for ((places, value), (indices, off_beat)) in in_a_row {
        let voices: BTreeSet<&str> = indices.iter().map(|&i| symbols[i].position.as_str()).collect();
        let next_place = (places[places.len() - 1] + 1..measure.len())
            .find(|&place| measure[place].iter().any(|(v, _, _)| voices.contains(v.as_str())));
        for ((next_places, next_value), (next_indices, _)) in single {
            if next_value == value
                && Some(next_places[0]) == next_place
                && indices.is_disjoint(next_indices)
                && next_indices.iter().map(|&i| symbols[i].position.as_str()).collect::<BTreeSet<_>>() == voices
            {
                let mut key = places.clone();
                key.extend(next_places);
                longer.insert((key, value.clone()), (indices.union(next_indices).copied().collect(), *off_beat));
            }
        }
    }
    longer
}

/// The threes of a measure, and the threes which follow each other directly as one.
fn find_threes(measure: &Measure, symbols: &[Symbol]) -> Vec<Three> {
    let single = find_single_threes(measure, symbols);
    let mut threes: BTreeMap<Indices, bool> = BTreeMap::new();
    for (indices, off_beat) in single.values() {
        threes.insert(indices.clone(), *off_beat);
    }
    let mut in_a_row = single.clone();
    for _ in 0..MAX_THREES_IN_A_ROW - 1 {
        in_a_row = add_following_three(&in_a_row, &single, measure, symbols);
        for (indices, off_beat) in in_a_row.values() {
            threes.entry(indices.clone()).or_insert(*off_beat);
        }
    }
    threes.into_iter().collect() // ordered by their sorted indices
}

/// Takes threes as triplets, the most helpful first, as long as each of them leaves
/// fewer places in the measure where a voice does not carry on.
fn choose_triplets(measure: &Measure, symbols: &[Symbol]) -> Vec<Indices> {
    let mut chosen = Vec::new();
    let mut triplets = Indices::new();
    let mut breaks = count_breaks(measure, &triplets);
    if breaks.0 == 0 || get_length(measure) > Frac::int(MAX_MEASURE_LENGTH) {
        return vec![];
    }
    let threes = find_threes(measure, symbols);
    loop {
        let mut best: Option<((usize, Frac), bool, Vec<usize>, &Indices)> = None;
        for (indices, off_beat) in &threes {
            if !indices.is_disjoint(&triplets) {
                continue;
            }
            let together: Indices = triplets.union(indices).copied().collect();
            let trial = (
                count_breaks(measure, &together),
                *off_beat,
                indices.iter().copied().collect::<Vec<_>>(),
                indices,
            );
            if best.as_ref().is_none_or(|b| (trial.0, trial.1, &trial.2) < (b.0, b.1, &b.2)) {
                best = Some(trial);
            }
        }
        let Some((best_breaks, _, _, indices)) = best else { break };
        if best_breaks.0 >= breaks.0 {
            break;
        }
        chosen.push(indices.clone());
        triplets = triplets.union(indices).copied().collect();
        breaks = best_breaks;
    }
    chosen
}

fn note_value<'a>(symbols: &'a [Symbol], indices: &Indices) -> &'a str {
    symbols[*indices.iter().next().unwrap()].rhythm.split('_').nth(1).unwrap_or("")
}

/// restore_unmarked_triplets: the symbols of one voice of a page with the triplets
/// which were read as plain notes restored.
pub fn restore_unmarked_triplets(symbols: &[Symbol]) -> Vec<Symbol> {
    let chosen: Vec<Vec<Indices>> = split_into_measures(symbols)
        .iter()
        .map(|m| choose_triplets(m, symbols))
        .collect();
    let mut shown: BTreeMap<&str, usize> = BTreeMap::new();
    for indices in chosen.iter().flatten() {
        *shown.entry(note_value(symbols, indices)).or_insert(0) += 1;
    }
    let triplets: Indices = chosen
        .iter()
        .flatten()
        .filter(|indices| shown[note_value(symbols, indices)] >= MIN_TRIPLETS_ON_A_PAGE)
        .flat_map(|indices| indices.iter().copied())
        .collect();
    symbols
        .iter()
        .enumerate()
        .map(|(i, s)| if triplets.contains(&i) { as_triplet(s) } else { s.clone() })
        .collect()
}
