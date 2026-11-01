"""Dumps homr's detect_staff inputs and result for each page image, and the
instructions homr's Python retires for it (second run, imports warm).

    python tools/dump_staff.py <homr dir> <cases dir> <page images...>
"""
import json, os, sys
sys.path.insert(0, os.path.dirname(__file__))
import numpy as np
from homr_setup import config, load
from instr import instructions

main = load(sys.argv[1])
from homr.music_xml_generator import XmlGeneratorArguments  # noqa: E402

out_root = sys.argv[2]
class Stop(Exception): pass
_orig = main.detect_staff
def box(b):
    (cx, cy), (w, h), a = b.box
    return [float(cx), float(cy), float(w), float(h), float(a)]
def spy(debug, image, fragments, clefs, bars):
    result = _orig(debug, image, fragments, clefs, bars)
    a = instructions(); _orig(debug, image, fragments, clefs, bars); b = instructions()
    os.makedirs(spy.dir, exist_ok=True)
    image = np.ascontiguousarray(image, dtype=np.uint8)
    image.tofile(os.path.join(spy.dir, "staff.bin"))
    json.dump({
        "shape": list(image.shape),
        "fragments": [{"box": box(f), "contour": np.asarray(f.contours).reshape(-1, 2).tolist()} for f in fragments],
        "clefs": [box(c) for c in clefs],
        "bars": [box(c) for c in bars],
        "expected": [[[float(p.x), [float(y) for y in p.y], float(p.angle)] for p in s.grid] for s in result],
        "python_instructions": b - a,
    }, open(os.path.join(spy.dir, "case.json"), "w"))
    raise Stop
main.detect_staff = spy
for p in sys.argv[3:]:
    spy.dir = os.path.join(out_root, os.path.splitext(os.path.basename(p))[0])
    try:
        main.process_image(p, config(main), XmlGeneratorArguments())
    except Stop:
        print("dumped", spy.dir, flush=True)
    except Exception as e:
        print("skip", p, type(e).__name__, e, flush=True)
