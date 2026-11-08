"""python tools/fuzz_gen.py <homr dir> <out dir> <count> <seed>   (TRIPLETS=1: plain notes which are triplets)
(needs record_decoder.py's Python dependencies)
Random symbol lists in the shapes the transformer produces (tuplets with and without
their numbers, several voices and staffs, grace notes, dynamics, multi-measure rests,
joined clefs and barlines, changing time signatures...), run through homr's
remove_duplicated_symbols, restore_unmarked_triplets and generate_xml. Each case dir
gets voices.json (the input and what the functions gave) and <name>.musicxml or error.txt."""
import io, json, os, random, sys, traceback
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from homr_setup import load
main = load(os.path.abspath(sys.argv[1]))
import xml.etree.ElementTree as ET
from homr.transformer.vocabulary import EncodedSymbol, remove_duplicated_symbols
from homr.triplet_restoration import restore_unmarked_triplets
from homr.music_xml_generator import XmlGeneratorArguments, generate_xml
out, count, seed = sys.argv[2], int(sys.argv[3]), int(sys.argv[4])
os.makedirs(out, exist_ok=True)
N = "."; E = "_"
PITCHES = ["C4", "D4", "E4", "F4", "G4", "A4", "B4", "C5", "D5", "G3", "A3", "C3", "E2"]
LIFTS = [E, E, E, "#", "b", "N"]
ARTS = [E, E, E, E, "staccato", "accent", "accentStrong", "fermata", "tenuto", "accent_fermata", "accentStrong_staccato", "trill", "tremolo"]
SLURS = [E, E, E, E, "slurStart", "slurStop", "slurStart_slurStop"]
PLAIN = ["4", "8", "8", "8", "16", "2", "4.", "8.", "1"]
TUPLET = ["12", "6", "24", "20", "10", "7", "18", "11"]
DYN = ["p", "mp", "mf", "f", "ff", "pp", "sfz", "fp", "empty"]
MOD = ["piu", "dolce", "poco", "sempre"]

def sym(r, pitch=N, lift=N, art=N, slur=N, pos=N):
    return EncodedSymbol(r, pitch, lift, art, slur, pos)

def note(rng, dur, pos, rest=False, grace=False):
    r = ("rest_" if rest else "note_") + dur + ("G" if grace else "")
    if rest:
        s = sym(r, E, E, E, E, pos)
    else:
        s = sym(r, rng.choice(PITCHES), rng.choice(LIFTS), rng.choice(ARTS), rng.choice(SLURS), pos)
    if rng.random() < 0.5:
        s.image_coordinates = (rng.uniform(0, 3000), rng.uniform(0, 3000))
    return s

def chord(rng, symbols):
    out = []
    for i, s in enumerate(symbols):
        if i: out.append(EncodedSymbol("chord"))
        out.append(s)
    return out

def make(rng):
    grand = rng.random() < 0.5
    positions = ["upper", "upper2"] + (["lower", "lower2"] if grand else [])
    voice = []
    for line in range(rng.randint(1, 3)):
        voice += [sym("clef_G2", E, E, E, E, "upper")]
        if grand:
            voice += [EncodedSymbol("chord"), sym("clef_F4", E, E, E, E, "lower")]
        voice.append(EncodedSymbol(f"keySignature_{rng.randint(-3, 3)}"))
        if line == 0 or rng.random() < 0.3:
            voice.append(EncodedSymbol(f"timeSignature/{rng.choice([4, 4, 8, 2])}"))
        for m in range(rng.randint(1, 5)):
            r = rng.random()
            if r < 0.07:
                voice.append(EncodedSymbol(f"rest_{rng.randint(2, 6)}m"))
            elif r < 0.12 and grand:
                voice += [sym("measureRest", pos="upper"), EncodedSymbol("chord"), sym("measureRest", pos="lower")]
            else:
                for beat in range(rng.randint(1, 6)):
                    k = rng.random()
                    pos = rng.choice(positions[:2] if k < 0.8 else positions)
                    if k < 0.25:       # a triplet or other tuplet, numbered
                        d = rng.choice(TUPLET if rng.random() < 0.3 else TUPLET[:3])
                        n = {"12": 3, "6": 3, "24": 3, "20": 5, "10": 5, "7": 7, "18": 9, "11": 11}[d]
                        for _ in range(n if rng.random() < 0.85 else rng.randint(1, n + 1)):
                            voice += chord(rng, [note(rng, d, pos, rest=rng.random() < 0.1)])
                    elif k < 0.45:     # a triplet read as plain notes (no number)
                        d = rng.choice(["8", "8", "4", "16"])
                        for _ in range(3):
                            voice += chord(rng, [note(rng, d, pos)])
                    elif k < 0.55:     # a chord of several notes
                        voice += chord(rng, [note(rng, rng.choice(PLAIN), rng.choice(positions)) for _ in range(rng.randint(2, 4))])
                    elif k < 0.62:     # grace note(s) then a note
                        voice += chord(rng, [note(rng, rng.choice(["8", "16"]), pos, grace=True, rest=rng.random() < 0.3)])
                        voice += chord(rng, [note(rng, rng.choice(PLAIN), pos)])
                    elif k < 0.68:     # a dynamic
                        d = sym("dynamic_" + rng.choice(DYN), E, E, E if rng.random() < 0.7 else "dynamic_" + rng.choice(MOD), E, rng.choice(["upper", "lower"] if grand else ["upper"]))
                        voice += chord(rng, [d])
                    elif k < 0.72:     # a clef or key joined to a note
                        voice += chord(rng, [note(rng, "4", "upper"), sym("clef_G2", E, E, E, E, "upper")] if rng.random() < 0.5 else [note(rng, "4", "upper"), EncodedSymbol(rng.choice(["barline", "doublebarline", "repeatEnd"]))])
                    else:
                        voice += chord(rng, [note(rng, rng.choice(PLAIN), pos, rest=rng.random() < 0.15)])
            b = rng.choice(["barline", "barline", "barline", "doublebarline", "bolddoublebarline", "repeatStart", "repeatEnd", "repeatEndStart", "voltaStart", "voltaStop", "voltaDiscontinue"])
            voice.append(EncodedSymbol(b))
        voice.append(EncodedSymbol("newline"))
    return voice

def ser(v):
    return [{"rhythm": s.rhythm, "pitch": s.pitch, "lift": s.lift, "articulation": s.articulation, "slur": s.slur,
             "position": s.position, "image_coordinates": list(s.image_coordinates) if s.image_coordinates else None} for s in v]

def attempt(fn):
    try:
        return fn(), None
    except BaseException as e:  # noqa
        return None, f"{type(e).__name__}: {e}"

def make_triplets(rng):
    """Plain notes which are really triplets, against a longer note in another voice."""
    voice = [sym("clef_G2", E, E, E, E, "upper"), EncodedSymbol("chord"), sym("clef_F4", E, E, E, E, "lower"),
             EncodedSymbol("keySignature_0"), EncodedSymbol("timeSignature/4")]
    unit = rng.choice(["8", "8", "16", "4"])
    long_ = {"8": "4", "16": "8", "4": "2"}[unit]
    for m in range(rng.randint(3, 8)):
        for beat in range(rng.randint(1, 4)):
            if rng.random() < 0.75:
                for i in range(3):
                    top = note(rng, unit, rng.choice(["upper", "upper", "upper2"]), rest=rng.random() < 0.08)
                    if i == 0:
                        voice += chord(rng, [top, note(rng, long_, "lower")])
                    else:
                        voice += chord(rng, [top])
            else:
                voice += chord(rng, [note(rng, long_, "upper"), note(rng, long_, "lower")])
        voice.append(EncodedSymbol("barline"))
    voice.append(EncodedSymbol("newline"))
    return voice

rng = random.Random(seed)
stats = {"ok": 0, "error": 0}
for n in range(count):
    d = os.path.join(out, f"fuzz-{seed}-{n}"); os.makedirs(d, exist_ok=True)
    voices = [make_triplets(rng) if os.environ.get('TRIPLETS') else make(rng) for _ in range(rng.choice([1, 1, 2]))]
    case = {"title": "", "voices": [ser(v) for v in voices]}
    case["remove_duplicated_symbols"] = []; case["restore_unmarked_triplets"] = []
    for v in voices:
        for name, fn in (("remove_duplicated_symbols", remove_duplicated_symbols), ("restore_unmarked_triplets", restore_unmarked_triplets)):
            res, err = attempt(lambda: fn(v))
            case[name].append({"in": ser(v), "out": ser(res) if res is not None else None, "error": err})
    # generate_xml on the cleaned-up voices, as homr does, and on the raw ones
    cleaned = []
    for v in voices:
        r, e = attempt(lambda: restore_unmarked_triplets(remove_duplicated_symbols(v)))
        cleaned.append(r if r is not None else v)
    case["voices"] = [ser(v) for v in cleaned]
    def run():
        root = generate_xml(XmlGeneratorArguments(False, None, None), cleaned, "")
        buf = io.StringIO(); ET.ElementTree(root).write(buf, encoding="unicode", xml_declaration=True)
        return buf.getvalue()
    xml, err = attempt(run)
    json.dump(case, open(os.path.join(d, "voices.json"), "w"))
    if xml is not None:
        open(os.path.join(d, f"fuzz-{seed}-{n}.musicxml"), "w").write(xml); stats["ok"] += 1
    else:
        open(os.path.join(d, "error.txt"), "w").write(err); stats["error"] += 1
print(stats)
