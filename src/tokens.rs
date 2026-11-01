//! homr's transformer symbols (transformer/vocabulary.py): EncodedSymbol,
//! durations from **kern, and the clean-up homr applies to a voice.

use crate::rational::Frac;

pub const NONOTE: &str = ".";
pub const EMPTY: &str = "_";

#[derive(Clone, Debug)]
pub struct Symbol {
    pub rhythm: String,
    pub pitch: String,
    pub lift: String,
    pub articulation: String,
    pub slur: String,
    pub position: String,
    /// the transformer's coordinates on the staff image (from its attention)
    pub coordinates: Option<Vec<f64>>,
    /// mapped back to the image homr was given
    pub image_coordinates: Option<(f64, f64)>,
}

impl Symbol {
    pub fn new(rhythm: &str) -> Symbol {
        Symbol {
            rhythm: rhythm.into(),
            pitch: NONOTE.into(),
            lift: NONOTE.into(),
            articulation: NONOTE.into(),
            slur: NONOTE.into(),
            position: NONOTE.into(),
            coordinates: None,
            image_coordinates: None,
        }
    }
    /// str(symbol)
    pub fn key(&self) -> String {
        format!("{} {} {} {} {} {}", self.rhythm, self.pitch, self.lift, self.articulation, self.slur, self.position)
    }
    pub fn duration(&self) -> Duration {
        if !(self.rhythm.starts_with("note") || self.rhythm.starts_with("rest")) {
            return Duration::new(Frac::ZERO, 0, 1, 1, 1);
        }
        kern_duration(self.rhythm.split('_').nth(1).unwrap_or(""))
    }
    pub fn remove_tuplet(&self) -> Symbol {
        let r = &self.rhythm;
        let kind = if r.starts_with("note_") {
            "note"
        } else if r.starts_with("rest_") {
            "rest"
        } else {
            return self.clone();
        };
        let tail = &r[kind.len() + 1..];
        let digits: String = tail.chars().take_while(|c| c.is_ascii_digit()).collect();
        if digits.is_empty() {
            return self.clone();
        }
        let rest = &tail[digits.len()..];
        let mut d: i64 = digits.parse().unwrap();
        if d % 3 == 0 {
            d = d / 3 * 2;
        } else if d % 5 == 0 {
            d = d / 5 * 4;
        } else if d % 7 == 0 {
            d = d / 7 * 4;
        } else {
            return self.clone();
        }
        let mut s = self.clone();
        s.rhythm = format!("{kind}_{d}{rest}");
        s
    }
    pub fn to_upper_position(&self) -> Symbol {
        if !is_lower(&self.position) {
            return self.clone();
        }
        let mut s = self.clone();
        s.position = self.position.replace("lower", "upper");
        s
    }
}

pub fn is_lower(position: &str) -> bool {
    position.starts_with("lower")
}

#[derive(Clone, Copy, Debug)]
pub struct Duration {
    pub dots: i64,
    pub actual_notes: i64,
    pub normal_notes: i64,
    pub fraction: Frac,
    pub kern: i64,
}

impl Duration {
    fn new(base: Frac, dots: i64, actual: i64, normal: i64, kern: i64) -> Duration {
        let an = Frac::new(actual, normal);
        let (actual_notes, normal_notes) = (an.n, an.d);
        let mut dur = base;
        let mut add = dur / Frac::int(2);
        for _ in 0..dots {
            dur = dur + add;
            add = add / Frac::int(2);
        }
        if actual_notes != normal_notes {
            dur = dur * Frac::new(normal_notes, actual_notes);
        }
        Duration { dots, actual_notes, normal_notes, fraction: dur, kern }
    }
}

fn prior_power_of_two(n: i64) -> i64 {
    if n < 1 {
        return 1;
    }
    1 << (63 - n.leading_zeros() as i64)
}

/// kern_to_symbol_duration
pub fn kern_duration(kern: &str) -> Duration {
    if kern.ends_with('m') {
        return Duration::new(Frac::int(1), 0, 1, 1, 4);
    }
    let digits: String = kern.chars().take_while(|c| c.is_ascii_digit()).collect();
    let rest = &kern[digits.len()..];
    let base: i64 = if digits.is_empty() { 4 } else { digits.parse().unwrap() };
    let dots = rest.matches('.').count() as i64;
    if kern.contains('G') {
        return Duration::new(Frac::ZERO, dots, 1, 1, base);
    }
    if base == 0 {
        return Duration::new(Frac::int(1), dots, 1, 1, base);
    }
    if base & (base - 1) == 0 {
        Duration::new(Frac::new(1, base), dots, 1, 1, base)
    } else {
        let normal = prior_power_of_two(base);
        Duration::new(Frac::new(1, normal), dots, base, normal, normal)
    }
}

fn is_note_or_rest(r: &str) -> bool {
    r.starts_with("note") || r.starts_with("rest")
}

pub fn group_into_chords(symbols: &[Symbol]) -> Vec<Vec<Symbol>> {
    let mut chords: Vec<Vec<Symbol>> = Vec::new();
    let mut in_chord = false;
    for s in symbols {
        if s.rhythm == "chord" {
            in_chord = true;
        } else if in_chord && !chords.is_empty() {
            chords.last_mut().unwrap().push(s.clone());
            in_chord = false;
        } else {
            chords.push(vec![s.clone()]);
        }
    }
    chords
}

fn flatten_chords(chords: &[Vec<Symbol>]) -> Vec<Symbol> {
    let mut out = Vec::new();
    for chord in chords {
        let mut in_chord = false;
        for s in chord {
            if in_chord {
                out.push(Symbol::new("chord"));
            }
            out.push(s.clone());
            in_chord = true;
        }
    }
    out
}

fn group_into_measures(chords: Vec<Vec<Symbol>>) -> Vec<Vec<Vec<Symbol>>> {
    let mut measures = Vec::new();
    let mut current = Vec::new();
    for chord in chords {
        let ends = !chord.is_empty() && (chord[0].rhythm.contains("barline") || chord[0].rhythm.contains("repeat"));
        current.push(chord);
        if ends {
            measures.push(std::mem::take(&mut current));
        }
    }
    if !current.is_empty() {
        measures.push(current);
    }
    measures
}

fn measure_duration(measure: &[Vec<Symbol>]) -> Frac {
    let mut total = Frac::ZERO;
    for chord in measure {
        let mut d = Frac::ZERO;
        for s in chord {
            if is_note_or_rest(&s.rhythm) {
                let f = s.duration().fraction;
                if f > Frac::ZERO && (f < d || d == Frac::ZERO) {
                    d = f;
                }
            }
        }
        total = total + d;
    }
    total
}

fn fix_over_eager_tuplets(chords: Vec<Vec<Symbol>>) -> Vec<Vec<Symbol>> {
    let measures = group_into_measures(chords);
    let mut durations: Vec<Frac> = measures.iter().map(|m| measure_duration(m)).collect();
    let mean = if durations.is_empty() {
        Frac::ZERO
    } else {
        durations.sort();
        durations[durations.len() / 2]
    };
    let mut out = Vec::new();
    for m in measures {
        if measure_duration(&m) < mean {
            out.extend(m.iter().map(|c| c.iter().map(|s| s.remove_tuplet()).collect::<Vec<_>>()));
        } else {
            out.extend(m);
        }
    }
    out
}

fn only_keep_lower_staff_if_there_is_a_clef(chords: Vec<Vec<Symbol>>) -> Vec<Vec<Symbol>> {
    let mut has_lower_clef = false;
    let mut out = Vec::new();
    for (i, chord) in chords.into_iter().enumerate() {
        let mut r = Vec::new();
        for s in chord {
            if has_lower_clef {
                r.push(s);
            } else if i < 5 && s.rhythm.starts_with("clef") && is_lower(&s.position) {
                has_lower_clef = true;
                r.push(s);
            } else {
                r.push(s.to_upper_position());
            }
        }
        out.push(r);
    }
    out
}

fn remove_duplicated_pitches(chord: Vec<Symbol>) -> Vec<Symbol> {
    if chord.len() <= 1 || !is_note_or_rest(&chord[0].rhythm) {
        return chord;
    }
    // homr keeps the first of each pitch+position (its replacement of a longer
    // duplicate is stored under another key and never used)
    let mut seen: Vec<String> = Vec::new();
    let mut out = Vec::new();
    for s in chord {
        let key = format!("{} {}", s.pitch, s.position);
        if !seen.contains(&key) {
            seen.push(key);
            out.push(s);
        }
    }
    out
}

fn remove_redundant_clefs_keys_times(chords: Vec<Vec<Symbol>>) -> Vec<Vec<Symbol>> {
    let (mut clef_upper, mut clef_lower, mut key, mut time) = (String::new(), String::new(), String::new(), String::new());
    chords
        .into_iter()
        .map(|chord| {
            let mut r = Vec::new();
            for s in chord {
                if s.rhythm.starts_with("clef") {
                    if !is_lower(&s.position) {
                        if s.rhythm != clef_upper {
                            clef_upper = s.rhythm.clone();
                            r.push(s);
                        }
                    } else if s.rhythm != clef_lower {
                        clef_lower = s.rhythm.clone();
                        r.push(s);
                    }
                } else if s.rhythm.starts_with("keySignature") {
                    if s.rhythm != key {
                        key = s.rhythm.clone();
                        r.push(s);
                    }
                } else if s.rhythm.starts_with("timeSignature") {
                    if s.rhythm != time {
                        time = s.rhythm.clone();
                        r.push(s);
                    }
                } else {
                    r.push(s);
                }
            }
            r
        })
        .collect()
}

/// remove_duplicated_symbols(symbols, cleanup_tuplets=True)
pub fn remove_duplicated_symbols(symbols: &[Symbol]) -> Vec<Symbol> {
    let chords = group_into_chords(symbols);
    let chords = fix_over_eager_tuplets(chords);
    let chords = only_keep_lower_staff_if_there_is_a_clef(chords);
    let chords: Vec<Vec<Symbol>> = chords.into_iter().map(remove_duplicated_pitches).collect();
    let chords = remove_redundant_clefs_keys_times(chords);
    flatten_chords(&chords)
}

/// sort_token_chords: chords, each sorted as homr's EncodedSymbol.__lt__ orders them (by str, descending).
pub fn sort_token_chords(symbols: &[Symbol]) -> Vec<Vec<Symbol>> {
    let mut chords = group_into_chords(symbols);
    for c in &mut chords {
        c.sort_by(|a, b| b.key().cmp(&a.key()));
    }
    chords
}
