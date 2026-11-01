"""Dumps homr's page analysis for each page image: the segmentation masks
after preprocessing (load_and_preprocess_predictions) and what OpenCV and
homr make of them, for checking the port stage by stage.

    python tools/dump_page.py <homr dir> <cases dir> <page images...>
"""
import json, os, sys
sys.path.insert(0, os.path.dirname(__file__))
import numpy as np
from homr_setup import config, load

main = load(sys.argv[1])
import cv2  # noqa: E402
from homr.music_xml_generator import XmlGeneratorArguments  # noqa: E402
from homr.bar_line_detection import prepare_bar_line_image  # noqa: E402

MASKS = ["notehead", "symbols", "staff", "clefs_keys", "stem_rests"]
out_root = sys.argv[2]
class Stop(Exception): pass
_load = main.load_and_preprocess_predictions
def spy(*args, **kwargs):
    predictions, debug, mapping = _load(*args, **kwargs)
    d = spy.dir; os.makedirs(d, exist_ok=True)
    masks = {
        "notehead": predictions.notehead, "symbols": predictions.symbols, "staff": predictions.staff,
        "clefs_keys": predictions.clefs_keys, "stem_rests": predictions.stems_rest,
    }
    masks["bar_line_img"] = prepare_bar_line_image(predictions.stems_rest)
    meta = {"shape": list(predictions.staff.shape), "contours": {}}
    for name, m in masks.items():
        m = np.ascontiguousarray(m, dtype=np.uint8)
        m.tofile(os.path.join(d, f"{name}.bin"))
        contours, _ = cv2.findContours(m, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
        meta["contours"][name] = [c.reshape(-1, 2).tolist() for c in contours]
    json.dump(meta, open(os.path.join(d, "page.json"), "w"))
    raise Stop
main.load_and_preprocess_predictions = spy
for p in sys.argv[3:]:
    spy.dir = os.path.join(out_root, os.path.splitext(os.path.basename(p))[0])
    try:
        main.process_image(p, config(main), XmlGeneratorArguments())
    except Stop:
        print("dumped", spy.dir, flush=True)
    except Exception as e:
        print("skip", p, type(e).__name__, e, flush=True)
