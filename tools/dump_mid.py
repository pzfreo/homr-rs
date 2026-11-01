"""Dumps homr's intermediate staff-detection results (anchors, raw staffs) next
to a case from dump_staff.py, for `cargo run --bin mid` when a case differs.

    python tools/dump_mid.py <homr dir> <cases dir> <page images...>
"""
import json, os, sys
sys.path.insert(0, os.path.dirname(__file__))
from homr_setup import config, load

main = load(sys.argv[1])
from homr import staff_detection as sd  # noqa: E402
from homr.music_xml_generator import XmlGeneratorArguments  # noqa: E402

mid = {}
def box(b): (cx, cy), (w, h), a = b.box; return [cx, cy, w, h, a]
_fua, _rds, _po = sd.filter_unusual_anchors, sd.remove_duplicate_staffs, sd.predict_other_anchors_from_clefs
def fua(anchors):
    r = _fua(anchors); mid["anchors"] = [[box(a.symbol), a.average_unit_size, a.zone.start, a.zone.stop] for a in r]; return r
def rds(staffs):
    r = _rds(staffs); mid["raw"] = [[s.min_x, s.max_x, s.min_y, s.max_y, len(s.anchors), list(s.box)] for s in r]; return r
def po(a, img):
    r = _po(a, img); mid["others"] = [box(b) for b in r]; return r
sd.filter_unusual_anchors, sd.remove_duplicate_staffs, sd.predict_other_anchors_from_clefs = fua, rds, po
class Stop(Exception): pass
_ds = main.detect_staff
def ds(*a):
    _ds(*a); raise Stop
main.detect_staff = ds
for p in sys.argv[3:]:
    mid.clear()
    try: main.process_image(p, config(main), XmlGeneratorArguments())
    except Stop: pass
    name = os.path.splitext(os.path.basename(p))[0]
    json.dump(mid, open(os.path.join(sys.argv[2], name, "mid.json"), "w"), default=float)
    print("mid", name, flush=True)
