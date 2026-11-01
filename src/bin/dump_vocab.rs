//! Prints the vocabularies as JSON (to compare with homr's).
use homr_rs::tokens::*;
fn main() {
    let v = serde_json::json!({"rhythm": rhythm_vocab(), "pitch": pitch_vocab(), "lift": lift_vocab(), "position": position_vocab(), "articulation": articulation_vocab(), "slur": slur_vocab()});
    println!("{v}");
}
