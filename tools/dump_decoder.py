"""Runs homr natively on page images (the models only run here) and records,
in call order, what the transformer returned for every staff (each symbol's
six tokens and its coordinates), plus the crop/resize of the page and homr's
native MusicXML. tools/pyodide_replay.mjs replays these in Pyodide.

    python tools/dump_decoder.py <homr dir> <out dir> <page images...>
"""
import json, os, sys
sys.path.insert(0, os.path.dirname(__file__))
import numpy as np
from homr_setup import config, load

main = load(sys.argv[1])
from homr.music_xml_generator import XmlGeneratorArguments  # noqa: E402
from homr.transformer import staff2score  # noqa: E402

calls = []
_predict = staff2score.Staff2Score.predict
def predict_spy(self, image):
    out = _predict(self, image)
    calls.append([[s.rhythm, s.pitch, s.lift, s.articulation, s.slur, s.position,
                   None if s.coordinates is None else np.asarray(s.coordinates, dtype=np.float64).reshape(-1).tolist()] for s in out])
    return out
staff2score.Staff2Score.predict = predict_spy
crop = {}
_autocrop = main.autocrop_with_offset
def autocrop_spy(img):
    out, top_left = _autocrop(img)
    crop["top_left"] = [int(top_left[0]), int(top_left[1])]
    crop["cropped_shape"] = list(out.shape)
    return out, top_left
main.autocrop_with_offset = autocrop_spy

for p in sys.argv[3:]:
    name = os.path.splitext(os.path.basename(p))[0]
    d = os.path.join(sys.argv[2], name)
    os.makedirs(d, exist_ok=True)
    calls.clear(); crop.clear()
    try:
        xml_path = main.process_image(p, config(main), XmlGeneratorArguments())
    except AssertionError as e:
        # homr itself fails (e.g. a zero-duration backup in music_xml_generator):
        # keep what the transformer returned so the port can be checked to fail too
        json.dump({"crop": crop, "calls": calls, "error": "AssertionError"}, open(os.path.join(d, "decoder.json"), "w"))
        print("dumped", d, len(calls), "staffs; homr fails with AssertionError", flush=True)
        continue
    except Exception as e:
        print("skip", p, type(e).__name__, e, flush=True)
        continue
    json.dump({"crop": crop, "calls": calls}, open(os.path.join(d, "decoder.json"), "w"))
    os.replace(xml_path, os.path.join(d, "native.musicxml"))
    print("dumped", d, len(calls), "staffs", flush=True)
