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
        self.els.push(El { tag: tag.into(), attrs: attrs.iter().map(|(k, v)| (k.to_string(), v.to_string())).collect(), text: None, children: vec![] });
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
        let pos = self.els[parent].children.iter().position(|c| matches!(c, Child::E(e) if *e == child)).expect("remove: not a child");
        self.els[parent].children.remove(pos);
    }
    /// child elements (comments excluded: they don't match tags; but list(el) includes them)
    pub fn elems(&self, e: usize) -> Vec<usize> {
        self.els[e].children.iter().filter_map(|c| if let Child::E(i) = c { Some(*i) } else { None }).collect()
    }
    pub fn len(&self, e: usize) -> usize {
        self.els[e].children.len()
    }
    pub fn find(&self, e: usize, tag: &str) -> Option<usize> {
        self.elems(e).into_iter().find(|&c| self.els[c].tag == tag)
    }
    pub fn findall(&self, e: usize, tag: &str) -> Vec<usize> {
        self.elems(e).into_iter().filter(|&c| self.els[c].tag == tag).collect()
    }
    pub fn findtext(&self, e: usize, tag: &str, default: &str) -> String {
        match self.find(e, tag) {
            Some(c) => self.els[c].text.clone().unwrap_or_default(),
            None => default.into(),
        }
    }
    pub fn get(&self, e: usize, attr: &str) -> Option<&str> {
        self.els[e].attrs.iter().find(|(k, _)| k == attr).map(|(_, v)| v.as_str())
    }
    /// list(element).index(child)
    pub fn index_of(&self, parent: usize, child: usize) -> usize {
        self.els[parent].children.iter().position(|c| matches!(c, Child::E(e) if *e == child)).unwrap()
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
    s.replace('&', "&amp;").replace('<', "&lt;").replace('>', "&gt;")
}

fn escape_attrib(s: &str) -> String {
    s.replace('&', "&amp;").replace('<', "&lt;").replace('>', "&gt;").replace('"', "&quot;").replace('\r', "&#13;").replace('\n', "&#10;").replace('\t', "&#09;")
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
        if self.tremolo_start { "start" } else { "stop" }
    }
}

#[derive(Clone)]
struct Chord {
    symbols: Vec<Symbol>,
    tuplet_mark: String,
}

fn nr(r: &str) -> bool {
    r.starts_with("note") || r.starts_with("rest")
}

impl Chord {
    fn is_barline(&self) -> bool {
        !self.symbols.is_empty() && (self.symbols[0].rhythm.contains("barline") || self.symbols[0].rhythm.contains("repeat"))
    }
    fn duration(&self) -> Frac {
        self.symbols.iter().filter(|s| nr(&s.rhythm)).map(|s| s.duration().fraction).min().unwrap_or(Frac::ZERO)
    }
    fn into_positions(&self) -> Vec<Chord> {
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
        let mut chords: Vec<Chord> = buckets.into_iter().map(|b| Chord { symbols: b, tuplet_mark: self.tuplet_mark.clone() }).collect();
        // sort(key=all rests, reverse=True), stable
        chords.sort_by_key(|c| !c.symbols.iter().all(|s| s.rhythm.starts_with("rest")));
        chords
    }
}

fn staff_of(s: &Symbol) -> i64 {
    if is_lower(&s.position) { 2 } else { 1 }
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

fn build_articulations(t: &mut Tree, note: usize, arts: &str, tuplet_mark: &str, st: &mut State) -> XmlResult<()> {
    let notation = t.sub(note, "notations", &[]);
    let mut xa: Vec<usize> = Vec::new();
    let mut xo: Vec<usize> = Vec::new();
    for a in arts.split('_') {
        match a {
            "" => {}
            "." => {}
            "fermata" => {
                t.sub(notation, "fermata", &[]);
            }
            "arpeggiate" => {
                t.sub(notation, "arpeggiate", &[]);
            }
            "accent" | "mordent" | "staccato" | "staccatissimo" | "tenuto" | "caesura" | "doit" => xa.push(t.el(a, &[])),
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
    if !tuplet_mark.is_empty() {
        t.sub(notation, "tuplet", &[("type", tuplet_mark)]);
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

fn build_note_or_rest(t: &mut Tree, s: &Symbol, layer: i64, is_chord: bool, st: &mut State, tuplet_mark: &str) -> XmlResult<usize> {
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
        t.sub_text(note, "duration", &(d.fraction * Frac::int(st.division)).trunc().max(1).to_string());
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
    build_articulations(t, note, &s.articulation, tuplet_mark, st)?;
    build_slurs(t, note, &s.slur, staff)?;
    if let Some((x, y)) = s.image_coordinates {
        t.els[note].children.push(Child::Comment(format!(" imgpos: {}, {} ", py_round_int(x), py_round_int(y))));
    }
    Ok(note)
}

fn build_backup(t: &mut Tree, d: Frac, st: &State) -> XmlResult<usize> {
    if !(d > Frac::ZERO) {
        return Err("AssertionError: Backup duration must be positive".into());
    }
    let b = t.el("backup", &[]);
    t.sub_text(b, "duration", &(d * Frac::int(st.division)).trunc().max(1).to_string());
    Ok(b)
}

fn group_notes(notes: &[Symbol]) -> BTreeMap<Frac, Vec<Symbol>> {
    let max = notes.iter().map(|n| n.duration().fraction).max().unwrap();
    let mut g: BTreeMap<Frac, Vec<Symbol>> = BTreeMap::new();
    for n in notes {
        let d = n.duration();
        let f = if n.rhythm.contains('G') { Frac::ZERO } else if d.fraction.n == 0 { max } else { d.fraction };
        g.entry(f).or_default().push(n.clone());
    }
    g
}

fn build_note_chord(t: &mut Tree, chord: &Chord, st: &mut State, chord_duration: Frac) -> XmlResult<Vec<usize>> {
    let by = group_notes(&chord.symbols);
    let count = by.len();
    let mut out = Vec::new();
    for (i, (gd, gn)) in by.iter().enumerate() {
        let notes: Vec<&Symbol> = gn.iter().filter(|n| n.pitch != EMPTY && n.pitch != NONOTE).collect();
        let rests: Vec<&Symbol> = gn.iter().filter(|n| n.pitch == EMPTY || n.pitch == NONOTE).collect();
        let mut first = true;
        for n in &notes {
            out.push(build_note_or_rest(t, n, i as i64, !first, st, &chord.tuplet_mark)?);
            first = false;
        }
        if !rests.is_empty() {
            if !(*gd > Frac::ZERO) {
                return Err("AssertionError".into());
            }
            if !notes.is_empty() {
                out.push(build_backup(t, *gd, st)?);
            }
            out.push(build_note_or_rest(t, rests[0], i as i64, false, st, &chord.tuplet_mark)?);
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

fn division_and_nominator(voice: &[Chord]) -> (i64, Frac) {
    let mut durations = vec![Frac::new(1, 4)];
    let mut in_measure = Frac::ZERO;
    let mut measures: Vec<Frac> = Vec::new();
    for c in voice {
        if c.is_barline() && in_measure > Frac::ZERO {
            measures.push(in_measure);
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
        measures.push(in_measure);
    }
    let dens: Vec<i64> = durations.iter().filter(|d| **d > Frac::ZERO).map(|d| d.d).collect();
    let division = dens[1..].iter().fold(dens[0], |c, &d| lcm(c, d));
    if measures.is_empty() {
        return (division, Frac::int(1));
    }
    measures.sort();
    let n = measures.len();
    let median = if n % 2 == 1 { measures[n / 2] } else { (measures[n / 2 - 1] + measures[n / 2]) / Frac::int(2) };
    (division, median)
}

fn tuplet_duration(c: &Chord) -> Option<Duration> {
    c.symbols.iter().filter(|s| nr(&s.rhythm)).map(|s| s.duration()).find(|d| d.normal_notes != d.actual_notes)
}

fn add_tuplets(groups: &mut [Chord]) -> bool {
    let mut cursor = 0;
    while cursor < groups.len() {
        let Some(d) = tuplet_duration(&groups[cursor]) else {
            cursor += 1;
            continue;
        };
        let start = cursor;
        let format = (d.actual_notes, d.normal_notes);
        let size = d.actual_notes as usize;
        while cursor - start < size {
            if cursor >= groups.len() {
                return false;
            }
            match tuplet_duration(&groups[cursor]) {
                Some(cd) if (cd.actual_notes, cd.normal_notes) == format => cursor += 1,
                _ => return false,
            }
        }
        groups[start].tuplet_mark = "start".into();
        groups[cursor - 1].tuplet_mark = "stop".into();
    }
    true
}

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
        let saved: Vec<String> = groups[a..b].iter().map(|g| g.tuplet_mark.clone()).collect();
        if add_tuplets(&mut groups[a..b]) {
            continue;
        }
        for (g, m) in groups[a..b].iter_mut().zip(saved) {
            g.tuplet_mark = m;
        }
    }
}

fn advance_to_next_group(g: &Chord, clock: Frac, sounding: &mut Vec<Frac>) -> Frac {
    let timed: Vec<Frac> = g.symbols.iter().filter(|s| nr(&s.rhythm)).map(|s| s.duration().fraction).filter(|d| *d > Frac::ZERO).collect();
    if timed.is_empty() {
        return Frac::ZERO;
    }
    sounding.extend(timed.iter().map(|d| clock + *d));
    sounding.iter().filter(|e| **e > clock).min().copied().unwrap() - clock
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
        let dur_of = |t: &Tree, c: usize| t.find(c, "duration").map(|d| t.els[d].text.clone().unwrap_or_default().parse::<i64>().unwrap());
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
        let staff = t.find(c, "staff").map(|s| t.els[s].text.clone().unwrap().parse::<i64>().unwrap()).unwrap_or(1);
        let chord_tone = t.find(c, "chord").is_some();
        let start = if chord_tone { last_start } else { current };
        let end = start + duration;
        if chord_tone && events.last().is_some_and(|e| e.staff == staff && e.start == start && e.end == end) {
            events.last_mut().unwrap().notes.push(c);
        } else if chord_tone {
            events.push(Ev { staff, start, end, notes: vec![c] });
        } else {
            last_start = start;
            current += duration;
            events.push(Ev { staff, start, end, notes: vec![c] });
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
            let mut v = 1;
            while active.iter().any(|(_, u)| *u == v) {
                v += 1;
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

fn note_pitch(t: &Tree, note: usize) -> Option<String> {
    let p = t.find(note, "pitch")?;
    Some(["step", "alter", "octave"].iter().map(|k| t.findtext(p, k, "")).collect::<Vec<_>>().join("|"))
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
            let key = (t.findtext(first, "staff", "1"), t.findtext(first, "voice", "1"));
            let before = previous.iter().position(|(k, _)| *k == key).map(|i| previous[i].1.clone());
            match previous.iter_mut().find(|(k, _)| *k == key) {
                Some(p) => p.1 = ev.clone(),
                None => previous.push((key, ev.clone())),
            }
            let Some(before) = before else { continue };
            for &start_note in &before {
                let Some(_) = get_slur(t, start_note, "start") else { continue };
                let Some(pitch) = note_pitch(t, start_note) else { continue };
                for &stop_note in &ev {
                    if note_pitch(t, stop_note).as_deref() != Some(pitch.as_str()) {
                        continue;
                    }
                    let Some((end_slur, end_notation)) = get_slur(t, stop_note, "stop") else { continue };
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

fn build_measures(t: &mut Tree, voice: &[Symbol], has_two_staves: bool) -> XmlResult<Vec<usize>> {
    let mut clock = Frac::ZERO;
    let mut sounding: Vec<Frac> = Vec::new();
    let mut measure_number = 1i64;
    let mut groups: Vec<Chord> = sort_token_chords(voice).into_iter().map(|s| Chord { symbols: s, tuplet_mark: String::new() }).collect();
    tuplet_parse(&mut groups);
    let (division, nominator) = division_and_nominator(&groups);
    let mut st = State { beats: 4 * 16, division, nominator, tremolo_start: false, volta_number: 1, last_volta_measure: -10 };
    let mut measures: Vec<usize> = Vec::new();
    let mut current = t.el("measure", &[("number", &measure_number.to_string())]);
    let first_attributes = t.sub(current, "attributes", &[]);
    t.sub_text(first_attributes, "divisions", &division.div_euclid(4).to_string());
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
                    let d: i64 = rhythm.split('_').nth(1).unwrap().replace('m', "").parse().unwrap();
                    let s = t.sub(a, "measure-style", &[]);
                    t.sub_text(s, "multiple-rest", &d.to_string());
                }
            } else {
                let positions = group.into_positions();
                let advance = advance_to_next_group(group, clock, &mut sounding);
                clock = clock + advance;
                sounding.retain(|e| *e > clock);
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
        let close = |t: &mut Tree, measures: &mut Vec<usize>, current: usize| {
            rebalance_measure_voices(t, current);
            measures.push(current);
        };
        if rhythm == "newline" {
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
            let time = t.sub(a, "time", &[]);
            let den = rhythm.split('/').nth(1).unwrap().to_string();
            let beats = (st.nominator * Frac::int(den.parse::<i64>().unwrap())).trunc().max(1);
            t.sub_text(time, "beats", &beats.to_string());
            t.sub_text(time, "beat-type", &den);
            st.beats = beats;
        } else if rhythm.contains("barline") {
            if rhythm != "barline" {
                let b = barline(t, current, "right");
                let style = if rhythm == "bolddoublebarline" { "heavy-heavy" } else { "light-light" };
                t.sub_text(b, "bar-style", style);
            }
            close(t, &mut measures, current);
            clock = Frac::ZERO;
            sounding.clear();
            measure_number += 1;
            current = t.el("measure", &[("number", &measure_number.to_string())]);
        } else if rhythm == "repeatStart" {
            close(t, &mut measures, current);
            clock = Frac::ZERO;
            sounding.clear();
            measure_number += 1;
            current = t.el("measure", &[("number", &measure_number.to_string())]);
            let b = barline(t, current, "right");
            repeat(t, b, "repeatStart");
        } else if rhythm == "repeatEnd" {
            let b = barline(t, current, "right");
            repeat(t, b, "repeatEnd");
            close(t, &mut measures, current);
            clock = Frac::ZERO;
            sounding.clear();
            measure_number += 1;
            current = t.el("measure", &[("number", &measure_number.to_string())]);
        } else if rhythm == "repeatEndStart" {
            let b = barline(t, current, "right");
            repeat(t, b, "repeatEnd");
            close(t, &mut measures, current);
            clock = Frac::ZERO;
            sounding.clear();
            measure_number += 1;
            current = t.el("measure", &[("number", &measure_number.to_string())]);
            let b = barline(t, current, "right");
            repeat(t, b, "repeatStart");
        } else if rhythm.starts_with("voltaStart") {
            let v = st.start_volta(measure_number);
            let b = barline(t, current, "left");
            t.sub(b, "ending", &[("type", "start"), ("number", &v.to_string())]);
        } else if rhythm.starts_with("voltaStop") || rhythm.starts_with("voltaDiscontinue") {
            let v = st.stop_volta(measure_number);
            let b = barline(t, current, "right");
            let ty = if rhythm.starts_with("voltaStop") { "stop" } else { "discontinue" };
            t.sub(b, "ending", &[("type", ty), ("number", &v.to_string())]);
        }
    }
    if t.len(current) > 0 {
        rebalance_measure_voices(t, current);
        measures.push(current);
    }
    if t.find(first_attributes, "time").is_none() {
        let time = t.sub(first_attributes, "time", &[]);
        t.sub_text(time, "beats", &(st.nominator * Frac::int(4)).trunc().max(1).to_string());
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
    let dir = if rhythm == "repeatStart" { "forward" } else { "backward" };
    t.sub(barline, "repeat", &[("direction", dir)]);
}

/// generate_xml(XmlGeneratorArguments(), voices, title) as homr writes it to a file.
pub fn generate_xml(voices: &[Vec<Symbol>], title: &str) -> XmlResult<String> {
    let mut t = Tree::default();
    let root = t.el("score-partwise", &[("version", "4.0")]);
    let work = t.sub(root, "work", &[]);
    t.sub_text(work, "work-title", title);
    let ident = t.sub(root, "identification", &[]);
    let enc = t.sub(ident, "encoding", &[]);
    t.sub_text(enc, "software", "homr");
    t.sub(root, "defaults", &[]);
    let two: Vec<bool> = voices.iter().map(|v| v.iter().any(|s| is_lower(&s.position))).collect();
    let pl = t.sub(root, "part-list", &[]);
    for (i, &two_staves) in two.iter().enumerate() {
        let id = format!("P{}", i + 1);
        let (name, sound, program) = if two_staves { ("Piano", "keyboard.piano", 1) } else { ("Voice", "voice", 54) };
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
        for m in build_measures(&mut t, v, two[i])? {
            t.append(part, m);
        }
        convert_ties(&mut t, part);
    }
    Ok(to_document(&t, root))
}
