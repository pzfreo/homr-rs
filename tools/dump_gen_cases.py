"""python tools/dump_gen_cases.py <homr dir> <out dir> <page images...>
(needs record_decoder.py's Python dependencies)
Runs homr on each page and, at generate_xml, saves the voices it was given, the title
and the MusicXML file it wrote, so the Rust generator can be compared on the same input."""
import json, os, shutil, sys
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from homr_setup import config, load
main = load(os.path.abspath(sys.argv[1]))
from homr.music_xml_generator import XmlGeneratorArguments
out = sys.argv[2]; os.makedirs(out, exist_ok=True)
captured = {}
def ser(v):
    return [{"rhythm": s.rhythm, "pitch": s.pitch, "lift": s.lift, "articulation": s.articulation,
             "slur": s.slur, "position": s.position,
             "image_coordinates": list(s.image_coordinates) if s.image_coordinates else None} for s in v]
import homr.staff_parsing as sp
for name in ("remove_duplicated_symbols", "restore_unmarked_triplets"):
    def wrap(fn, name=name):
        def w(symbols, *a, **k):
            result = fn(symbols, *a, **k)
            captured.setdefault(name, []).append({"in": ser(symbols), "out": ser(result)})
            return result
        return w
    setattr(sp, name, wrap(getattr(sp, name)))
orig = main.generate_xml
def spy(args, staffs, title):
    captured["voices"] = [ser(v) for v in staffs]
    captured["title"] = title
    return orig(args, staffs, title)
main.generate_xml = spy
cfg = config(main)
for img in sys.argv[3:]:
    name = os.path.splitext(os.path.basename(img))[0]
    d = os.path.join(out, name)
    if os.path.exists(os.path.join(d, "native.musicxml")): continue
    os.makedirs(d, exist_ok=True)
    tmp = os.path.join(d, os.path.basename(img)); shutil.copy(img, tmp)
    captured.clear()
    try:
        main.process_image(tmp, cfg, XmlGeneratorArguments(False, None, None))
    except Exception as e:
        print(name, "ERROR", e, flush=True); continue
    json.dump(captured, open(os.path.join(d, "voices.json"), "w"))
    os.remove(tmp)
    print(name, "ok", flush=True)
