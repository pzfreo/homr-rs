//! homr's MusicXML writer (music_xml_generator.py), with a minimal element
//! tree that serialises exactly as Python's xml.etree.ElementTree does.

use crate::rational::Frac;
use crate::tokens::{is_lower, sort_token_chords, Duration, Symbol, EMPTY, NONOTE};
use std::collections::BTreeMap;

// ---------- a minimal ElementTree ----------

#[derive(Clone, Debug)]
pub enum Child {
    E(usize),
    Comment(String),
}

#[derive(Clone, Debug, Default)]
pub struct El {
    pub tag: String,
    pub attrs: Vec<(String, String)>,
    pub text: Option<String>,
    pub children: Vec<Child>,
}

#[derive(Default)]
pub struct Tree {
    pub els: Vec<El>,
}

impl Tree {
    pub fn el(&mut self, tag: &str, attrs: &[(&str, &str)]) -> usize {
        self.els.push(El {
            tag: tag.into(),
            attrs: attrs
                .iter()
                .map(|(k, v)| (k.to_string(), v.to_string()))
                .collect(),
            text: None,
            children: vec![],
        });
        self.els.len() - 1
    }
    pub fn sub(&mut self, parent: usize, tag: &str, attrs: &[(&str, &str)]) -> usize {
        let e = self.el(tag, attrs);
        self.els[parent].children.push(Child::E(e));
        e
    }
    pub fn sub_text(&mut self, parent: usize, tag: &str, text: &str) -> usize {
        let e = self.sub(parent, tag, &[]);
        self.els[e].text = Some(text.into());
        e
    }
    pub fn append(&mut self, parent: usize, child: usize) {
        self.els[parent].children.push(Child::E(child));
    }
    pub fn insert(&mut self, parent: usize, index: usize, child: usize) {
        let i = index.min(self.els[parent].children.len());
        self.els[parent].children.insert(i, Child::E(child));
    }
    pub fn remove(&mut self, parent: usize, child: usize) {
        let pos = self.els[parent]
            .children
            .iter()
            .position(|c| matches!(c, Child::E(e) if *e == child))
            .expect("remove: not a child");
        self.els[parent].children.remove(pos);
    }
    /// child elements (comments excluded: they don't match tags; but list(el) includes them)
    pub fn elems(&self, e: usize) -> Vec<usize> {
        self.els[e]
            .children
            .iter()
            .filter_map(|c| if let Child::E(i) = c { Some(*i) } else { None })
            .collect()
    }
    pub fn len(&self, e: usize) -> usize {
        self.els[e].children.len()
    }
    pub fn find(&self, e: usize, tag: &str) -> Option<usize> {
        self.elems(e).into_iter().find(|&c| self.els[c].tag == tag)
    }
    pub fn findall(&self, e: usize, tag: &str) -> Vec<usize> {
        self.elems(e)
            .into_iter()
            .filter(|&c| self.els[c].tag == tag)
            .collect()
    }
    pub fn findtext(&self, e: usize, tag: &str, default: &str) -> String {
        match self.find(e, tag) {
            Some(c) => self.els[c].text.clone().unwrap_or_default(),
            None => default.into(),
        }
    }
    pub fn get(&self, e: usize, attr: &str) -> Option<&str> {
        self.els[e]
            .attrs
            .iter()
            .find(|(k, _)| k == attr)
            .map(|(_, v)| v.as_str())
    }
    /// list(element).index(child)
    pub fn index_of(&self, parent: usize, child: usize) -> usize {
        self.els[parent]
            .children
            .iter()
            .position(|c| matches!(c, Child::E(e) if *e == child))
            .unwrap()
    }

    pub fn serialize(&self, e: usize, out: &mut String) {
        let el = &self.els[e];
        out.push('<');
        out.push_str(&el.tag);
        for (k, v) in &el.attrs {
            out.push_str(&format!(" {k}=\"{}\"", escape_attrib(v)));
        }
        let has_text = el.text.as_deref().is_some_and(|t| !t.is_empty());
        if has_text || !el.children.is_empty() {
            out.push('>');
            if has_text {
                out.push_str(&escape_cdata(el.text.as_deref().unwrap()));
            }
            for c in &el.children {
                match c {
                    Child::E(i) => self.serialize(*i, out),
                    Child::Comment(t) => {
                        out.push_str("<!--");
                        out.push_str(t);
                        out.push_str("-->");
                    }
                }
            }
            out.push_str("</");
            out.push_str(&el.tag);
            out.push('>');
        } else {
            out.push_str(" />");
        }
    }
}

fn escape_cdata(s: &str) -> String {
    s.replace('&', "&amp;")
        .replace('<', "&lt;")
        .replace('>', "&gt;")
}

fn escape_attrib(s: &str) -> String {
    s.replace('&', "&amp;")
        .replace('<', "&lt;")
        .replace('>', "&gt;")
        .replace('"', "&quot;")
        .replace('\r', "&#13;")
        .replace('\n', "&#10;")
        .replace('\t', "&#09;")
}

/// ET.ElementTree(root).write(f, encoding="unicode", xml_declaration=True)
pub fn to_document(tree: &Tree, root: usize) -> String {
    let mut s = String::from("<?xml version='1.0' encoding='utf-8'?>\n");
    tree.serialize(root, &mut s);
    s
}

// ---------- the generator ----------

pub type XmlResult<T> = Result<T, String>;

struct State {
    beats: i64,
    division: i64,
    nominator: Frac,
    tremolo_start: bool,
    volta_number: i64,
    last_volta_measure: i64,
    opts: XmlOptions,
}

/// Departures from homr's writer an app can ask for (all off: homr's output).
#[derive(Clone, Copy, Debug, Default)]
pub struct XmlOptions {
    /// Comments to write in each note after its image position.
    pub note_comments: Option<fn(&Symbol) -> Vec<String>>,
}

impl State {
    fn start_volta(&mut self, m: i64) -> i64 {
        if m == self.last_volta_measure + 1 {
            self.volta_number += 1;
        } else {
            self.volta_number = 1;
        }
        self.volta_number
    }
    fn stop_volta(&mut self, m: i64) -> i64 {
        self.last_volta_measure = m;
        self.volta_number
    }
    fn toggle_tremolo(&mut self) -> &'static str {
        self.tremolo_start = !self.tremolo_start;
        if self.tremolo_start {
            "start"
        } else {
            "stop"
        }
    }
}

#[derive(Clone)]
struct Chord {
    symbols: Vec<Symbol>,
    /// Tuplet start or stop by position (e.g. upper2): voices have their own tuplets
    tuplet_marks: BTreeMap<String, String>,
}

/// The attribute which carries a note's position (e.g. upper2) while a part is built.
const POSITION_ATTRIBUTE: &str = "homr-position";

fn nr(r: &str) -> bool {
    r.starts_with("note") || r.starts_with("rest")
}

impl Chord {
    fn new(symbols: Vec<Symbol>) -> Chord {
        Chord {
            symbols,
            tuplet_marks: BTreeMap::new(),
        }
    }
    fn mark_for(&self, s: &Symbol) -> &str {
        self.tuplet_marks
            .get(&s.position)
            .map(|m| m.as_str())
            .unwrap_or("")
    }
    fn is_barline(&self) -> bool {
        !self.symbols.is_empty()
            && (self.symbols[0].rhythm.contains("barline")
                || self.symbols[0].rhythm.contains("repeat"))
    }
    fn duration(&self) -> Frac {
        self.symbols
            .iter()
            .filter(|s| nr(&s.rhythm))
            .map(|s| s.duration().fraction)
            .min()
            .unwrap_or(Frac::ZERO)
    }
    fn to_positions(&self) -> Vec<Chord> {
        let mut keys: Vec<String> = Vec::new();
        let mut buckets: Vec<Vec<Symbol>> = Vec::new();
        for s in &self.symbols {
            match keys.iter().position(|k| *k == s.position) {
                Some(i) => buckets[i].push(s.clone()),
                None => {
                    keys.push(s.position.clone());
                    buckets.push(vec![s.clone()]);
                }
            }
        }
        let mut chords: Vec<Chord> = buckets
            .into_iter()
            .map(|b| Chord {
                symbols: b,
                tuplet_marks: self.tuplet_marks.clone(),
            })
            .collect();
        // sort(key=all rests, reverse=True), stable
        chords.sort_by_key(|c| !c.symbols.iter().all(|s| s.rhythm.starts_with("rest")));
        chords
    }
}

fn staff_of(s: &Symbol) -> i64 {
    if is_lower(&s.position) {
        2
    } else {
        1
    }
}

fn xml_voice(staff: i64, layer: i64) -> i64 {
    (staff - 1) * 4 + layer + 1
}

fn duration_name(k: i64) -> XmlResult<&'static str> {
    Ok(match k {
        0 => "breve",
        1 => "whole",
        2 => "half",
        4 => "quarter",
        8 => "eighth",
        16 => "16th",
        32 => "32nd",
        64 => "64th",
        128 => "128th",
        _ => return Err(format!("KeyError: {k}")),
    })
}

fn lift_alter(l: &str) -> XmlResult<i64> {
    Ok(match l {
        "N" => 0,
        "#" => 1,
        "##" => 2,
        "b" => -1,
        "bb" => -2,
        _ => return Err(format!("KeyError: {l}")),
    })
}

/// Python's round() of a float to an int (half to even), as for imgpos.
fn py_round_int(v: f64) -> i64 {
    v.round_ties_even() as i64
}

fn build_articulations(
    t: &mut Tree,
    note: usize,
    arts: &str,
    tuplet_mark: &str,
    st: &mut State,
) -> XmlResult<()> {
    let notation = t.sub(note, "notations", &[]);
    let mut xa: Vec<usize> = Vec::new();
    let mut xo: Vec<usize> = Vec::new();
    // A dynamic's modifier ("dynamic_dolce") isn't an articulation
    for a in arts.split('_').filter(|_| !arts.starts_with("dynamic")) {
        match a {
            "" => {}
            "." => {}
            "fermata" => {
                t.sub(notation, "fermata", &[]);
            }
            "arpeggiate" => {
                t.sub(notation, "arpeggiate", &[]);
            }
            "accent" | "mordent" | "staccato" | "staccatissimo" | "tenuto" | "caesura" | "doit" => {
                xa.push(t.el(a, &[]))
            }
            "accentStrong" => xa.push(t.el("strong-accent", &[])),
            "breathMark" => xa.push(t.el("breath-mark", &[])),
            "tremolo" => {
                let ty = st.toggle_tremolo();
                let e = t.el("tremolo", &[("type", ty)]);
                t.els[e].text = Some("3".into());
                xo.push(e);
            }
            "trill" => xo.push(t.el("trill-mark", &[])),
            "turn" => xo.push(t.el("inverted-turn", &[])),
            "slurStart" => {
                t.sub(notation, "slur", &[("type", "start")]);
            }
            "slurStop" => {
                t.sub(notation, "slur", &[("type", "stop")]);
            }
            "tieStart" => {
                t.sub(notation, "tied", &[("type", "start")]);
            }
            "tieStop" => {
                t.sub(notation, "tied", &[("type", "stop")]);
            }
            other => return Err(format!("ValueError: Unsupported articulation {other}")),
        }
    }
    for mark in tuplet_mark.split('_').filter(|m| !m.is_empty()) {
        t.sub(notation, "tuplet", &[("type", mark)]);
    }
    if !xa.is_empty() {
        let p = t.sub(notation, "articulations", &[]);
        for c in xa {
            t.append(p, c);
        }
    }
    if !xo.is_empty() {
        let p = t.sub(notation, "ornaments", &[]);
        for c in xo {
            t.append(p, c);
        }
    }
    Ok(())
}

fn build_slurs(t: &mut Tree, note: usize, slurs: &str, number: i64) -> XmlResult<()> {
    let notation = match t.find(note, "notations") {
        Some(n) => n,
        None => t.sub(note, "notations", &[]),
    };
    let n = number.to_string();
    match slurs {
        "_" | "" | "." => {}
        "slurStart" => {
            t.sub(notation, "slur", &[("type", "start"), ("number", &n)]);
        }
        "slurStop" => {
            t.sub(notation, "slur", &[("type", "stop"), ("number", &n)]);
        }
        "slurStart_slurStop" => {
            t.sub(notation, "slur", &[("type", "stop"), ("number", &n)]);
            t.sub(notation, "slur", &[("type", "start"), ("number", &n)]);
        }
        other => return Err(format!("ValueError: Unsupported slur {other}")),
    }
    Ok(())
}

fn build_note_or_rest(
    t: &mut Tree,
    s: &Symbol,
    layer: i64,
    is_chord: bool,
    st: &mut State,
    tuplet_mark: &str,
) -> XmlResult<usize> {
    let note = t.el("note", &[]);
    if is_chord {
        t.sub(note, "chord", &[]);
    }
    let d: Duration = s.duration();
    let grace = s.rhythm.contains('G');
    if grace {
        t.sub(note, "grace", &[]);
    }
    if s.pitch == EMPTY {
        if d.fraction.n == 0 {
            t.sub(note, "rest", &[("measure", "yes")]);
        } else {
            t.sub(note, "rest", &[]);
        }
    } else if s.pitch == NONOTE {
        t.sub(note, "rest", &[]);
    } else {
        let pitch = t.sub(note, "pitch", &[]);
        let chars: Vec<char> = s.pitch.chars().collect();
        t.sub_text(pitch, "step", &chars[0].to_string());
        t.sub_text(pitch, "octave", &chars[1].to_string());
        if s.lift != NONOTE && s.lift != EMPTY {
            t.sub_text(pitch, "alter", &lift_alter(&s.lift)?.to_string());
        }
    }
    if grace {
        t.sub_text(note, "type", duration_name(d.kern)?);
    } else if d.fraction.n > 0 {
        let base = if d.kern == 0 { 1 } else { d.kern };
        t.sub_text(
            note,
            "duration",
            &(d.fraction * Frac::int(st.division))
                .trunc()
                .max(1)
                .to_string(),
        );
        t.sub_text(note, "type", duration_name(base)?);
    } else {
        t.sub_text(note, "duration", &st.beats.to_string());
        t.sub_text(note, "type", duration_name(0)?);
    }
    for _ in 0..d.dots {
        t.sub(note, "dot", &[]);
    }
    if d.actual_notes != d.normal_notes {
        let tm = t.sub(note, "time-modification", &[]);
        t.sub_text(tm, "actual-notes", &d.actual_notes.to_string());
        t.sub_text(tm, "normal-notes", &d.normal_notes.to_string());
    }
    let staff = staff_of(s);
    t.sub_text(note, "voice", &xml_voice(staff, layer).to_string());
    t.sub_text(note, "staff", &staff.to_string());
    // Used for voices and stems, removed at the end of the part
    t.els[note]
        .attrs
        .push((POSITION_ATTRIBUTE.into(), s.position.clone()));
    // The other notes of a group, e.g. in the other staff, aren't part of the tuplet
    let is_tuplet = d.actual_notes != d.normal_notes;
    let mark = if is_tuplet { tuplet_mark } else { "" };
    build_articulations(t, note, &s.articulation, mark, st)?;
    build_slurs(t, note, &s.slur, staff)?;
    if let Some((x, y)) = s.image_coordinates {
        t.els[note].children.push(Child::Comment(format!(
            " imgpos: {}, {} ",
            py_round_int(x),
            py_round_int(y)
        )));
    }
    if let Some(f) = st.opts.note_comments {
        for c in f(s) {
            t.els[note].children.push(Child::Comment(c));
        }
    }
    Ok(note)
}

fn build_backup(t: &mut Tree, d: Frac, st: &State) -> XmlResult<usize> {
    if !(d > Frac::ZERO) {
        return Err("AssertionError: Backup duration must be positive".into());
    }
    let b = t.el("backup", &[]);
    t.sub_text(
        b,
        "duration",
        &(d * Frac::int(st.division)).trunc().max(1).to_string(),
    );
    Ok(b)
}

fn group_notes(notes: &[Symbol]) -> BTreeMap<Frac, Vec<Symbol>> {
    let max = notes.iter().map(|n| n.duration().fraction).max().unwrap();
    let mut g: BTreeMap<Frac, Vec<Symbol>> = BTreeMap::new();
    for n in notes {
        let d = n.duration();
        let f = if n.rhythm.contains('G') {
            Frac::ZERO
        } else if d.fraction.n == 0 {
            max
        } else {
            d.fraction
        };
        g.entry(f).or_default().push(n.clone());
    }
    g
}

fn build_note_chord(
    t: &mut Tree,
    chord: &Chord,
    st: &mut State,
    chord_duration: Frac,
) -> XmlResult<Vec<usize>> {
    let by = group_notes(&chord.symbols);
    let count = by.len();
    let mut out = Vec::new();
    for (i, (gd, gn)) in by.iter().enumerate() {
        let notes: Vec<&Symbol> = gn
            .iter()
            .filter(|n| n.pitch != EMPTY && n.pitch != NONOTE)
            .collect();
        let rests: Vec<&Symbol> = gn
            .iter()
            .filter(|n| n.pitch == EMPTY || n.pitch == NONOTE)
            .collect();
        let mut first = true;
        for n in &notes {
            out.push(build_note_or_rest(
                t,
                n,
                i as i64,
                !first,
                st,
                chord.mark_for(n),
            )?);
            first = false;
        }
        // A grace note without pitch would be a rest which takes no time
        if !rests.is_empty() && *gd > Frac::ZERO {
            if !notes.is_empty() {
                out.push(build_backup(t, *gd, st)?);
            }
            out.push(build_note_or_rest(
                t,
                rests[0],
                i as i64,
                false,
                st,
                chord.mark_for(rests[0]),
            )?);
        }
        if i != count - 1 && *gd > Frac::ZERO {
            out.push(build_backup(t, *gd, st)?);
        }
    }
    let max = *by.keys().next_back().unwrap();
    if chord_duration < max {
        out.push(build_backup(t, max - chord_duration, st)?);
    }
    Ok(out)
}

fn lcm(a: i64, b: i64) -> i64 {
    let g = {
        let (mut x, mut y) = (a.abs(), b.abs());
        while y != 0 {
            (x, y) = (y, x % y);
        }
        x
    };
    (a * b).abs() / g
}

/// np.median of fractions (the mean of the two middle ones for an even count).
fn median(mut v: Vec<Frac>) -> Frac {
    v.sort();
    let n = v.len();
    if n % 2 == 1 {
        v[n / 2]
    } else {
        (v[n / 2 - 1] + v[n / 2]) / Frac::int(2)
    }
}

/// find_division_and_time_signature_nominators: the division and one measure
/// length per time signature, estimated from the measures up to the next time
/// signature. Measures before the first one count towards it.
fn division_and_nominators(voice: &[Chord]) -> (i64, Vec<Frac>) {
    let mut durations = vec![Frac::new(1, 4)];
    let mut in_measure = Frac::ZERO;
    let mut segments: Vec<Vec<Frac>> = vec![vec![]];
    let mut has_time_signature = false;
    for c in voice {
        if !c.symbols.is_empty() && c.symbols[0].rhythm.starts_with("timeSignature") {
            if in_measure > Frac::ZERO {
                segments.last_mut().unwrap().push(in_measure);
                in_measure = Frac::ZERO;
            }
            if has_time_signature {
                segments.push(vec![]);
            }
            has_time_signature = true;
        } else if c.is_barline() && in_measure > Frac::ZERO {
            segments.last_mut().unwrap().push(in_measure);
            in_measure = Frac::ZERO;
        } else {
            for s in &c.symbols {
                if nr(&s.rhythm) {
                    let f = s.duration().fraction;
                    if f > Frac::ZERO {
                        durations.push(f);
                    }
                }
            }
            let d = c.duration();
            if d > Frac::ZERO {
                in_measure = in_measure + d;
            }
        }
    }
    if in_measure > Frac::ZERO {
        segments.last_mut().unwrap().push(in_measure);
    }
    let dens: Vec<i64> = durations
        .iter()
        .filter(|d| **d > Frac::ZERO)
        .map(|d| d.d)
        .collect();
    let division = dens[1..].iter().fold(dens[0], |c, &d| lcm(c, d));
    let all: Vec<Frac> = segments.iter().flatten().copied().collect();
    if all.is_empty() {
        return (division, vec![Frac::int(1); segments.len()]);
    }
    // A time signature without measures after it, e.g. a courtesy one at the end of a
    // system, belongs to the measures of the next one
    let mut nominators = Vec::new();
    let mut following = median(all);
    for segment in segments.iter().rev() {
        if !segment.is_empty() {
            following = median(segment.clone());
        }
        nominators.push(following);
    }
    nominators.reverse();
    (division, nominators)
}

/// When the groups of a measure start. Tokens say which notes start together, not
/// when each group starts. A group starts when the earliest still-sounding note ends
/// (that is how training data is grouped), but not before the earlier notes of its
/// own positions (voices) have ended.
struct GroupClock {
    time: Frac,
    sounding: Vec<Frac>,
    voice_free: BTreeMap<String, Frac>,
}

impl GroupClock {
    fn new() -> GroupClock {
        GroupClock {
            time: Frac::ZERO,
            sounding: Vec::new(),
            voice_free: BTreeMap::new(),
        }
    }
    /// Grace notes have no duration and take no time
    fn timed(group: &Chord) -> impl Iterator<Item = &Symbol> {
        group
            .symbols
            .iter()
            .filter(|s| nr(&s.rhythm) && s.duration().fraction > Frac::ZERO)
    }
    /// Moves the time to when the positions of the group are free, returns the gap.
    fn wait_for_voices(&mut self, group: &Chord) -> Frac {
        let free = Self::timed(group)
            .map(|s| self.voice_free.get(&s.position).copied().unwrap_or(Frac::ZERO))
            .max()
            .unwrap_or(Frac::ZERO);
        let gap = if free - self.time > Frac::ZERO {
            free - self.time
        } else {
            Frac::ZERO
        };
        self.time = self.time + gap;
        let time = self.time;
        self.sounding.retain(|e| *e > time);
        gap
    }
    /// Moves the time to the start of the next group and returns how far.
    fn advance(&mut self, group: &Chord) -> Frac {
        // A position can hold notes of different lengths, the shortest one frees it
        let mut shortest: BTreeMap<String, Frac> = BTreeMap::new();
        for s in Self::timed(group) {
            let d = s.duration().fraction;
            let e = shortest.entry(s.position.clone()).or_insert(d);
            if d < *e {
                *e = d;
            }
        }
        for (position, d) in shortest {
            self.voice_free.insert(position, self.time + d);
        }
        let advance = advance_to_next_group(group, self.time, &mut self.sounding);
        self.time = self.time + advance;
        let time = self.time;
        self.sounding.retain(|e| *e > time);
        advance
    }
}

fn is_power_of_two(v: i64) -> bool {
    v > 0 && v & (v - 1) == 0
}

/// A tuplet note or rest (not a grace note).
fn is_tuplet_symbol(s: &Symbol) -> bool {
    if !nr(&s.rhythm) || s.rhythm.contains('G') {
        return false;
    }
    let d = s.duration();
    d.normal_notes != d.actual_notes
}

/// The duration of the first tuplet note or rest at the position, otherwise of the
/// first timed one. None if the position has no timed symbol.
fn duration_at(group: &Chord, position: &str) -> Option<Duration> {
    let timed: Vec<&Symbol> = group
        .symbols
        .iter()
        .filter(|s| s.position == position && nr(&s.rhythm) && !s.rhythm.contains('G'))
        .collect();
    if let Some(s) = timed.iter().find(|s| is_tuplet_symbol(s)) {
        return Some(s.duration());
    }
    timed.first().map(|s| s.duration())
}

/// When each group of a measure starts, the same way as build_measures counts.
fn start_times(groups: &[Chord]) -> Vec<Frac> {
    let mut clock = GroupClock::new();
    let mut result = Vec::new();
    for group in groups {
        if group.symbols.iter().any(|s| nr(&s.rhythm)) {
            clock.wait_for_voices(group);
            result.push(clock.time);
            clock.advance(group);
        } else {
            result.push(clock.time);
        }
    }
    result
}

type Run = Vec<(usize, Frac, Frac)>; // group, start time and duration

/// The complete tuplets as (first, last) group and the groups left over: a tuplet
/// ends once what it covers (`by_length`: its own notes added up, otherwise where
/// it ends) has a power of two as denominator.
fn split_run(run: &Run, by_length: bool) -> (Vec<(usize, usize)>, Vec<usize>) {
    let mut tuplets = Vec::new();
    let mut pending: Vec<usize> = Vec::new();
    let mut total = Frac::ZERO;
    for &(index, start, duration) in run {
        pending.push(index);
        total = total + duration;
        let measured = if by_length { total } else { start + duration };
        if is_power_of_two(measured.d) {
            tuplets.push((pending[0], index));
            pending.clear();
            total = Frac::ZERO;
        }
    }
    (tuplets, pending)
}

/// Splits consecutive tuplet notes of one position into tuplets. A tuplet ends once
/// its own notes add up to a duration without a tuplet, e.g. three eighth triplets or
/// a quarter and an eighth triplet. That keeps working if an earlier note has a wrong
/// length. If that leaves notes over, e.g. if the tuplet starts with a rest in another
/// voice, a tuplet ends where it reaches the beat grid instead. Notes which are
/// left over form an incomplete tuplet.
fn mark_run(groups: &mut [Chord], position: &str, run: &Run) {
    let by_length = split_run(run, true);
    let by_grid = split_run(run, false);
    let (mut tuplets, left_over) = if by_length.1.len() <= by_grid.1.len() {
        by_length
    } else {
        by_grid
    };
    if !left_over.is_empty() {
        // An incomplete tuplet, e.g. one note too many, still has tuplet durations
        tuplets.push((left_over[0], *left_over.last().unwrap()));
    }
    for (first, last) in tuplets {
        let marks = |g: &mut Chord, m: &str| {
            g.tuplet_marks.insert(position.to_string(), m.to_string());
        };
        if first == last {
            marks(&mut groups[first], "start_stop");
        } else {
            marks(&mut groups[first], "start");
            marks(&mut groups[last], "stop");
        }
    }
}

fn add_tuplets(groups: &mut [Chord], position: &str) {
    let times = start_times(groups);
    let durations: Vec<Option<Duration>> = groups.iter().map(|g| duration_at(g, position)).collect();
    let mut run: Run = Vec::new();
    let mut format = (1, 1);
    for (index, duration) in durations.iter().enumerate() {
        let Some(d) = duration else { continue };
        let current = (d.actual_notes, d.normal_notes);
        if current != format && !run.is_empty() {
            mark_run(groups, position, &run);
            run.clear();
        }
        format = current;
        if current.0 != current.1 {
            run.push((index, times[index], d.fraction));
        }
    }
    if !run.is_empty() {
        mark_run(groups, position, &run);
    }
}

/// TupletParser.parse: marks the start and stop of tuplets, per measure and
/// position. Groups without a tuplet note at the position, e.g. a duplet in the
/// other staff, are skipped.
fn tuplet_parse(groups: &mut [Chord]) {
    let mut start = 0;
    let mut bounds = Vec::new();
    for (i, g) in groups.iter().enumerate() {
        if g.is_barline() {
            bounds.push((start, i + 1));
            start = i + 1;
        }
    }
    if start < groups.len() {
        bounds.push((start, groups.len()));
    }
    for (a, b) in bounds {
        let positions: std::collections::BTreeSet<String> = groups[a..b]
            .iter()
            .flat_map(|g| g.symbols.iter())
            .filter(|s| is_tuplet_symbol(s))
            .map(|s| s.position.clone())
            .collect();
        for position in positions {
            add_tuplets(&mut groups[a..b], &position);
        }
    }
}

const BEFORE_NOTES_ORDER: [&str; 3] = ["clef", "keySignature", "timeSignature"];

/// The transformer sometimes joins a clef, key, time signature or barline to a chord
/// of notes, e.g. a clef change in one staff on the beat where the other staff plays.
/// Split it: clefs, keys and time signatures go before the notes (in the order they
/// have at the start of a line), barlines and repeats after them.
fn split_mixed_chord(symbols: &[Symbol]) -> Vec<Vec<Symbol>> {
    let notes: Vec<Symbol> = symbols.iter().filter(|s| nr(&s.rhythm)).cloned().collect();
    if notes.is_empty() || notes.len() == symbols.len() {
        return vec![symbols.to_vec()];
    }
    let others: Vec<&Symbol> = symbols.iter().filter(|s| !nr(&s.rhythm)).collect();
    let is_after = |s: &Symbol| s.rhythm.contains("barline") || s.rhythm.contains("repeat");
    let mut before: Vec<(String, Vec<Symbol>)> = Vec::new();
    for s in &others {
        if is_after(s) {
            continue;
        }
        let kind = s.rhythm.split('_').next().unwrap().split('/').next().unwrap();
        match before.iter_mut().find(|(k, _)| k == kind) {
            Some((_, v)) => v.push((*s).clone()),
            None => before.push((kind.to_string(), vec![(*s).clone()])),
        }
    }
    let rank = |kind: &str| {
        BEFORE_NOTES_ORDER
            .iter()
            .position(|k| *k == kind)
            .unwrap_or(BEFORE_NOTES_ORDER.len())
    };
    before.sort_by_key(|(k, _)| rank(k)); // stable
    let mut parts: Vec<Vec<Symbol>> = before.into_iter().map(|(_, v)| v).collect();
    parts.push(notes);
    parts.extend(others.iter().filter(|s| is_after(s)).map(|s| vec![(*s).clone()]));
    parts
}

/// split_mixed_chords: applies split_mixed_chord to every group. Runs after the
/// tuplet marks are set, so the notes keep their mark and a split-off clef doesn't
/// interrupt a tuplet.
fn split_mixed_chords(groups: Vec<Chord>) -> Vec<Chord> {
    let mut result: Vec<Chord> = Vec::new();
    let mut split_barline = false;
    for group in groups {
        let parts = split_mixed_chord(&group.symbols);
        if split_barline && group.is_barline() && parts.len() == 1 {
            // The barline split off the previous chord and this one are the same barline:
            // keep one, preferring a repeat or double barline over a plain barline.
            if group.symbols[0].rhythm == "barline" && result.last().unwrap().symbols[0].rhythm != "barline" {
                split_barline = false;
                continue;
            }
            result.pop();
        }
        let n = parts.len();
        for part in parts {
            let is_notes = nr(&part[0].rhythm);
            result.push(Chord {
                symbols: part,
                tuplet_marks: if is_notes {
                    group.tuplet_marks.clone()
                } else {
                    BTreeMap::new()
                },
            });
        }
        split_barline = n > 1 && result.last().unwrap().is_barline();
    }
    result
}

fn advance_to_next_group(g: &Chord, clock: Frac, sounding: &mut Vec<Frac>) -> Frac {
    let timed: Vec<Frac> = g
        .symbols
        .iter()
        .filter(|s| nr(&s.rhythm))
        .map(|s| s.duration().fraction)
        .filter(|d| *d > Frac::ZERO)
        .collect();
    if timed.is_empty() {
        return Frac::ZERO;
    }
    sounding.extend(timed.iter().map(|d| clock + *d));
    sounding
        .iter()
        .filter(|e| **e > clock)
        .min()
        .copied()
        .unwrap()
        - clock
}

fn rebalance_measure_voices(t: &mut Tree, measure: usize) {
    struct Ev {
        staff: i64,
        start: i64,
        end: i64,
        notes: Vec<usize>,
    }
    let mut events: Vec<Ev> = Vec::new();
    let (mut current, mut last_start) = (0i64, 0i64);
    for c in t.elems(measure) {
        let tag = t.els[c].tag.clone();
        let dur_of = |t: &Tree, c: usize| {
            t.find(c, "duration").map(|d| {
                t.els[d]
                    .text
                    .clone()
                    .unwrap_or_default()
                    .parse::<i64>()
                    .unwrap()
            })
        };
        if tag == "backup" {
            if let Some(d) = dur_of(t, c) {
                current -= d;
            }
            continue;
        }
        if tag == "forward" {
            if let Some(d) = dur_of(t, c) {
                current += d;
            }
            continue;
        }
        if tag != "note" {
            continue;
        }
        let duration = dur_of(t, c).unwrap_or(0);
        let staff = t
            .find(c, "staff")
            .map(|s| t.els[s].text.clone().unwrap().parse::<i64>().unwrap())
            .unwrap_or(1);
        let chord_tone = t.find(c, "chord").is_some();
        let start = if chord_tone { last_start } else { current };
        let end = start + duration;
        if chord_tone
            && events
                .last()
                .is_some_and(|e| e.staff == staff && e.start == start && e.end == end)
        {
            events.last_mut().unwrap().notes.push(c);
        } else if chord_tone {
            events.push(Ev {
                staff,
                start,
                end,
                notes: vec![c],
            });
        } else {
            last_start = start;
            current += duration;
            events.push(Ev {
                staff,
                start,
                end,
                notes: vec![c],
            });
        }
    }
    let mut staff_order: Vec<i64> = Vec::new();
    for e in &events {
        if !staff_order.contains(&e.staff) {
            staff_order.push(e.staff);
        }
    }
    for staff in staff_order {
        let mut evs: Vec<&Ev> = events.iter().filter(|e| e.staff == staff).collect();
        evs.sort_by_key(|e| (e.start, e.end));
        let mut active: Vec<(i64, i64)> = Vec::new();
        for e in evs {
            active.retain(|(end, _)| *end > e.start);
            // The second voice of the model (e.g. upper2) prefers the second voice
            let mut v = if is_second_voice(t, e.notes[0]) { 2 } else { 1 };
            if active.iter().any(|(_, u)| *u == v) {
                v = 1;
                while active.iter().any(|(_, u)| *u == v) {
                    v += 1;
                }
            }
            active.push((e.end, v));
            let xv = xml_voice(staff, v - 1).to_string();
            for &n in &e.notes {
                if let Some(ve) = t.find(n, "voice") {
                    t.els[ve].text = Some(xv.clone());
                }
            }
        }
    }
}

fn is_second_voice(t: &Tree, note: usize) -> bool {
    t.get(note, POSITION_ATTRIBUTE)
        .unwrap_or("")
        .ends_with('2')
}

fn remove_position_attributes(t: &mut Tree, part: usize) {
    for measure in t.findall(part, "measure") {
        for note in t.findall(measure, "note") {
            t.els[note].attrs.retain(|(k, _)| k != POSITION_ATTRIBUTE);
        }
    }
}

/// Where a staff has two voices in a measure, the first voice has its stems up
/// and the second one down. Other stems are left to the reader.
fn add_stems(t: &mut Tree, part: usize) {
    for measure in t.findall(part, "measure") {
        let notes: Vec<usize> = t
            .findall(measure, "note")
            .into_iter()
            .filter(|&n| t.find(n, "rest").is_none())
            .collect();
        let mut staffs: Vec<String> = Vec::new();
        for &n in &notes {
            let staff = t.findtext(n, "staff", "1");
            if !staffs.contains(&staff) {
                staffs.push(staff);
            }
        }
        for staff in staffs {
            let on_staff: Vec<usize> = notes
                .iter()
                .copied()
                .filter(|&n| t.findtext(n, "staff", "1") == staff)
                .collect();
            let voices: std::collections::BTreeSet<bool> = on_staff
                .iter()
                .filter(|&&n| !t.get(n, POSITION_ATTRIBUTE).unwrap_or("").is_empty())
                .map(|&n| is_second_voice(t, n))
                .collect();
            if voices.len() < 2 {
                continue;
            }
            for note in on_staff {
                let stem = t.el("stem", &[]);
                t.els[stem].text = Some(if is_second_voice(t, note) { "down" } else { "up" }.into());
                let position = match t.find(note, "staff") {
                    Some(staff_el) => t.index_of(note, staff_el),
                    None => t.len(note),
                };
                t.insert(note, position, stem);
            }
        }
    }
}

fn build_dynamic(t: &mut Tree, s: &Symbol, measure: usize) {
    let mut marks: Vec<usize> = Vec::new();
    if s.rhythm.starts_with("dynamic") {
        let name = s.rhythm.split_once('_').map(|x| x.1).unwrap_or("");
        // `dynamic_empty` is the internal placeholder used for a modifier-only
        // marking (for example, "più"). It is not a MusicXML dynamics element.
        if name != "empty" {
            marks.push(t.el(name, &[]));
        }
    }
    if s.articulation.starts_with("dynamic") {
        let text = t.el("other-dynamics", &[]);
        let modifier = s.articulation.split_once('_').map(|x| x.1).unwrap_or("");
        t.els[text].text = Some(modifier.replace("piu", "più"));
        marks.push(text);
    }
    if marks.is_empty() {
        return;
    }
    let direction = t.sub(measure, "direction", &[("placement", "below")]);
    let direction_type = t.sub(direction, "direction-type", &[]);
    let dynamics = t.sub(direction_type, "dynamics", &[]);
    for m in marks {
        t.append(dynamics, m);
    }
    t.sub_text(direction, "staff", &staff_of(s).to_string());
}

/// A rest which fills the measure, e.g. in a staff which is hidden in a system.
fn build_measure_rest(t: &mut Tree, staff: i64, st: &State) -> usize {
    let note = t.el("note", &[]);
    t.sub(note, "rest", &[("measure", "yes")]);
    let duration = (st.nominator * Frac::int(st.division)).trunc().max(1);
    t.sub_text(note, "duration", &duration.to_string());
    t.sub_text(note, "voice", "1");
    t.sub_text(note, "staff", &staff.to_string());
    note
}

fn build_forward(t: &mut Tree, d: Frac, st: &State) -> usize {
    let f = t.el("forward", &[]);
    t.sub_text(
        f,
        "duration",
        &(d * Frac::int(st.division)).trunc().max(1).to_string(),
    );
    f
}

/// One measure rest per staff of the part, the second one after a backup.
fn measure_rests(t: &mut Tree, measure: usize, has_two_staves: bool, st: &State) -> XmlResult<()> {
    let staffs: &[i64] = if has_two_staves { &[1, 2] } else { &[1] };
    for (i, &staff) in staffs.iter().enumerate() {
        if i > 0 {
            let b = build_backup(t, st.nominator, st)?;
            t.append(measure, b);
        }
        let r = build_measure_rest(t, staff, st);
        t.append(measure, r);
    }
    Ok(())
}

/// close_current_measure: the voices of the measure are balanced, and a
/// multi-measure rest is written as that many measures.
fn close_measure(
    t: &mut Tree,
    measures: &mut Vec<usize>,
    current: usize,
    measure_number: &mut i64,
    rest_measures: &mut i64,
    has_two_staves: bool,
    st: &State,
) -> XmlResult<()> {
    rebalance_measure_voices(t, current);
    measures.push(current);
    for _ in 0..*rest_measures {
        *measure_number += 1;
        let m = t.el("measure", &[("number", &measure_number.to_string())]);
        measure_rests(t, m, has_two_staves, st)?;
        measures.push(m);
    }
    *rest_measures = 0;
    Ok(())
}

fn note_pitch(t: &Tree, note: usize) -> Option<String> {
    let p = t.find(note, "pitch")?;
    Some(
        ["step", "alter", "octave"]
            .iter()
            .map(|k| t.findtext(p, k, ""))
            .collect::<Vec<_>>()
            .join("|"),
    )
}

fn get_slur(t: &Tree, note: usize, ty: &str) -> Option<(usize, usize)> {
    for n in t.findall(note, "notations") {
        for s in t.findall(n, "slur") {
            if t.get(s, "type") == Some(ty) {
                return Some((s, n));
            }
        }
    }
    None
}

fn add_tie(t: &mut Tree, note: usize, ty: &str, notation: usize) {
    let tie = t.el("tie", &[("type", ty)]);
    let pos = match t.find(note, "duration") {
        Some(d) => t.index_of(note, d) + 1,
        None => 0,
    };
    t.insert(note, pos, tie);
    let tied = t.el("tied", &[("type", ty)]);
    t.insert(notation, 0, tied);
}

fn convert_ties(t: &mut Tree, part: usize) {
    let mut previous: Vec<((String, String), Vec<usize>)> = Vec::new();
    for measure in t.findall(part, "measure") {
        let mut events: Vec<Vec<usize>> = Vec::new();
        for note in t.findall(measure, "note") {
            if t.find(note, "chord").is_some() && !events.is_empty() {
                events.last_mut().unwrap().push(note);
            } else {
                events.push(vec![note]);
            }
        }
        for ev in events {
            let first = ev[0];
            let key = (
                t.findtext(first, "staff", "1"),
                t.findtext(first, "voice", "1"),
            );
            let before = previous
                .iter()
                .position(|(k, _)| *k == key)
                .map(|i| previous[i].1.clone());
            match previous.iter_mut().find(|(k, _)| *k == key) {
                Some(p) => p.1 = ev.clone(),
                None => previous.push((key, ev.clone())),
            }
            let Some(before) = before else { continue };
            for &start_note in &before {
                let Some(_) = get_slur(t, start_note, "start") else {
                    continue;
                };
                let Some(pitch) = note_pitch(t, start_note) else {
                    continue;
                };
                for &stop_note in &ev {
                    if note_pitch(t, stop_note).as_deref() != Some(pitch.as_str()) {
                        continue;
                    }
                    let Some((end_slur, end_notation)) = get_slur(t, stop_note, "stop") else {
                        continue;
                    };
                    let (begin_slur, begin_notation) = get_slur(t, start_note, "start").unwrap();
                    t.remove(begin_notation, begin_slur);
                    t.remove(end_notation, end_slur);
                    add_tie(t, start_note, "start", begin_notation);
                    add_tie(t, stop_note, "stop", end_notation);
                    break;
                }
            }
        }
    }
}

fn build_measures(
    t: &mut Tree,
    voice: &[Symbol],
    has_two_staves: bool,
    opts: XmlOptions,
) -> XmlResult<Vec<usize>> {
    let mut clock = GroupClock::new();
    let mut measure_number = 1i64;
    // Measures a multi-measure rest still covers after the current one
    let mut rest_measures = 0i64;
    let mut groups: Vec<Chord> = sort_token_chords(voice)
        .into_iter()
        .map(Chord::new)
        .collect();
    tuplet_parse(&mut groups);
    let groups = split_mixed_chords(groups);
    let (division, nominators) = division_and_nominators(&groups);
    let mut st = State {
        beats: 4 * 16,
        division,
        nominator: nominators[0],
        tremolo_start: false,
        volta_number: 1,
        last_volta_measure: -10,
        opts,
    };
    let mut time_signature_no = 0usize;
    let mut measures: Vec<usize> = Vec::new();
    let mut current = t.el("measure", &[("number", &measure_number.to_string())]);
    let first_attributes = t.sub(current, "attributes", &[]);
    t.sub_text(
        first_attributes,
        "divisions",
        &division.div_euclid(4).to_string(),
    );
    if has_two_staves {
        t.sub_text(first_attributes, "staves", "2");
        t.sub_text(first_attributes, "part-symbol", "brace");
    }
    let mut attributes: Option<usize> = Some(first_attributes);
    let get_attrs = |t: &mut Tree, m: usize, last: Option<usize>, force: bool| -> usize {
        match last {
            Some(a) if !force => a,
            _ => t.sub(m, "attributes", &[]),
        }
    };
    let n_groups = groups.len();
    for (gno, group) in groups.iter().enumerate() {
        let symbol = group.symbols[0].clone();
        let rhythm = symbol.rhythm.clone();
        let last_attributes = attributes;
        attributes = None;
        if nr(&rhythm) {
            if group.symbols.len() == 1 && rhythm.ends_with('m') {
                let a = get_attrs(t, current, last_attributes, false);
                attributes = Some(a);
                if t.find(a, "measure-style").is_none() {
                    let d: i64 = rhythm
                        .split('_')
                        .nth(1)
                        .unwrap()
                        .replace('m', "")
                        .parse()
                        .unwrap();
                    let s = t.sub(a, "measure-style", &[]);
                    t.sub_text(s, "multiple-rest", &d.to_string());
                    if d > 0 {
                        measure_rests(t, current, has_two_staves, &st)?;
                        rest_measures = d - 1;
                    }
                }
            } else {
                let positions = group.to_positions();
                let gap = clock.wait_for_voices(group);
                if gap > Frac::ZERO {
                    let f = build_forward(t, gap, &st);
                    t.append(current, f);
                }
                let advance = clock.advance(group);
                let np = positions.len();
                for (pno, sp) in positions.iter().enumerate() {
                    let cd = if pno == np - 1 { advance } else { Frac::ZERO };
                    for x in build_note_chord(t, sp, &mut st, cd)? {
                        t.append(current, x);
                    }
                }
            }
            continue;
        }
        if rhythm == "measureRest" {
            let mut rests: Vec<&Symbol> = group.symbols.iter().collect();
            rests.sort_by_key(|s| staff_of(s)); // stable
            for (staff_no, rest) in rests.into_iter().enumerate() {
                if staff_no > 0 {
                    let b = build_backup(t, st.nominator, &st)?;
                    t.append(current, b);
                }
                let r = build_measure_rest(t, staff_of(rest), &st);
                t.append(current, r);
            }
        } else if rhythm == "newline" {
            if gno != n_groups - 1 {
                t.sub(current, "print", &[("new-system", "yes")]);
            }
        } else if rhythm.starts_with("clef") {
            let a = get_attrs(t, current, last_attributes, true);
            attributes = Some(a);
            for c in &group.symbols {
                if c.rhythm.starts_with("clef") {
                    let sl: Vec<char> = c.rhythm.split('_').nth(1).unwrap().chars().collect();
                    let clef = t.sub(a, "clef", &[("number", &staff_of(c).to_string())]);
                    t.sub_text(clef, "sign", &sl[0].to_string());
                    t.sub_text(clef, "line", &sl[1].to_string());
                }
            }
        } else if rhythm.starts_with("keySignature") {
            let a = get_attrs(t, current, last_attributes, false);
            attributes = Some(a);
            let k = t.sub(a, "key", &[]);
            t.sub_text(k, "fifths", rhythm.split('_').nth(1).unwrap());
        } else if rhythm.starts_with("timeSignature") {
            let a = get_attrs(t, current, last_attributes, false);
            attributes = Some(a);
            st.nominator = nominators[time_signature_no.min(nominators.len() - 1)];
            time_signature_no += 1;
            let time = t.sub(a, "time", &[]);
            let den = rhythm.split('/').nth(1).unwrap().to_string();
            let beats = (st.nominator * Frac::int(den.parse::<i64>().unwrap()))
                .trunc()
                .max(1);
            t.sub_text(time, "beats", &beats.to_string());
            t.sub_text(time, "beat-type", &den);
            st.beats = beats;
        } else if rhythm.contains("barline") {
            if rhythm != "barline" {
                let b = barline(t, current, "right");
                let style = if rhythm == "bolddoublebarline" {
                    "heavy-heavy"
                } else {
                    "light-light"
                };
                t.sub_text(b, "bar-style", style);
            }
            close_measure(t, &mut measures, current, &mut measure_number, &mut rest_measures, has_two_staves, &st)?;
            clock = GroupClock::new();
            measure_number += 1;
            current = t.el("measure", &[("number", &measure_number.to_string())]);
        } else if rhythm == "repeatStart" {
            close_measure(t, &mut measures, current, &mut measure_number, &mut rest_measures, has_two_staves, &st)?;
            clock = GroupClock::new();
            measure_number += 1;
            current = t.el("measure", &[("number", &measure_number.to_string())]);
            let b = barline(t, current, "right");
            repeat(t, b, "repeatStart");
        } else if rhythm == "repeatEnd" {
            let b = barline(t, current, "right");
            repeat(t, b, "repeatEnd");
            close_measure(t, &mut measures, current, &mut measure_number, &mut rest_measures, has_two_staves, &st)?;
            clock = GroupClock::new();
            measure_number += 1;
            current = t.el("measure", &[("number", &measure_number.to_string())]);
        } else if rhythm == "repeatEndStart" {
            let b = barline(t, current, "right");
            repeat(t, b, "repeatEnd");
            close_measure(t, &mut measures, current, &mut measure_number, &mut rest_measures, has_two_staves, &st)?;
            clock = GroupClock::new();
            measure_number += 1;
            current = t.el("measure", &[("number", &measure_number.to_string())]);
            let b = barline(t, current, "right");
            repeat(t, b, "repeatStart");
        } else if rhythm.starts_with("voltaStart") {
            let v = st.start_volta(measure_number);
            let b = barline(t, current, "left");
            t.sub(
                b,
                "ending",
                &[("type", "start"), ("number", &v.to_string())],
            );
        } else if rhythm.starts_with("voltaStop") || rhythm.starts_with("voltaDiscontinue") {
            let v = st.stop_volta(measure_number);
            let b = barline(t, current, "right");
            let ty = if rhythm.starts_with("voltaStop") {
                "stop"
            } else {
                "discontinue"
            };
            t.sub(b, "ending", &[("type", ty), ("number", &v.to_string())]);
        } else if rhythm.starts_with("dynamic") || symbol.articulation.starts_with("dynamic") {
            for dynamic in &group.symbols {
                build_dynamic(t, dynamic, current);
            }
        }
    }
    if t.len(current) > 0 {
        close_measure(t, &mut measures, current, &mut measure_number, &mut rest_measures, has_two_staves, &st)?;
    }
    // A clef opens a second attributes element, so look at the whole first measure
    let has_time = t
        .findall(measures[0], "attributes")
        .into_iter()
        .any(|a| t.find(a, "time").is_some());
    if !has_time {
        let time = t.sub(first_attributes, "time", &[]);
        t.sub_text(
            time,
            "beats",
            &(st.nominator * Frac::int(4)).trunc().max(1).to_string(),
        );
        t.sub_text(time, "beat-type", "4");
    }
    Ok(measures)
}

fn barline(t: &mut Tree, measure: usize, location: &str) -> usize {
    for c in t.elems(measure) {
        if t.els[c].tag == "barline" && t.get(c, "location") == Some(location) {
            return c;
        }
    }
    t.sub(measure, "barline", &[("location", location)])
}

fn repeat(t: &mut Tree, barline: usize, rhythm: &str) {
    if t.find(barline, "repeat").is_some() {
        return;
    }
    let dir = if rhythm == "repeatStart" {
        "forward"
    } else {
        "backward"
    };
    t.sub(barline, "repeat", &[("direction", dir)]);
}

/// generate_xml(XmlGeneratorArguments(), voices, title) as homr writes it to a file.
pub fn generate_xml(voices: &[Vec<Symbol>], title: &str) -> XmlResult<String> {
    generate_xml_with(voices, title, XmlOptions::default())
}

pub fn generate_xml_with(
    voices: &[Vec<Symbol>],
    title: &str,
    opts: XmlOptions,
) -> XmlResult<String> {
    let mut t = Tree::default();
    let root = t.el("score-partwise", &[("version", "4.0")]);
    let work = t.sub(root, "work", &[]);
    t.sub_text(work, "work-title", title);
    let ident = t.sub(root, "identification", &[]);
    let enc = t.sub(ident, "encoding", &[]);
    t.sub_text(enc, "software", "homr");
    t.sub(root, "defaults", &[]);
    let two: Vec<bool> = voices
        .iter()
        .map(|v| v.iter().any(|s| is_lower(&s.position)))
        .collect();
    let pl = t.sub(root, "part-list", &[]);
    for (i, &two_staves) in two.iter().enumerate() {
        let id = format!("P{}", i + 1);
        let (name, sound, program) = if two_staves {
            ("Piano", "keyboard.piano", 1)
        } else {
            ("Voice", "voice", 54)
        };
        let sp = t.sub(pl, "score-part", &[("id", &id)]);
        t.sub_text(sp, "part-name", name);
        let si = t.sub(sp, "score-instrument", &[("id", &format!("{id}-I1"))]);
        t.sub_text(si, "instrument-name", name);
        t.sub_text(si, "instrument-sound", sound);
        let mi = t.sub(sp, "midi-instrument", &[("id", &format!("{id}-I1"))]);
        t.sub_text(mi, "midi-channel", &(i + 1).to_string());
        t.sub_text(mi, "midi-program", &program.to_string());
        t.sub_text(mi, "volume", "100");
        t.sub_text(mi, "pan", "0");
    }
    for (i, v) in voices.iter().enumerate() {
        let part = t.sub(root, "part", &[("id", &format!("P{}", i + 1))]);
        for m in build_measures(&mut t, v, two[i], opts)? {
            t.append(part, m);
        }
        convert_ties(&mut t, part);
        add_stems(&mut t, part);
        remove_position_attributes(&mut t, part);
    }
    Ok(to_document(&t, root))
}
