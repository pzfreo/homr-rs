"""Runs homr's segmentation natively (the model only runs here) and dumps,
per page: the image as decoded (BGR), and the class map of every tile
(argmax of segnet's output) in homr's order. tools/pyodide_pre.mjs turns these
into reference results with homr in Pyodide.

    python tools/dump_segnet.py <homr dir> <out dir> <page images...>
"""
import json, os, sys
sys.path.insert(0, os.path.dirname(__file__))
import numpy as np
from homr_setup import load

main = load(sys.argv[1])
import cv2  # noqa: E402
from homr.segmentation import inference_segnet  # noqa: E402

captured = {}
_merge = inference_segnet.merge_patches
def merge_spy(patches, image_shape, win_size, step_size):
    captured["tiles"] = np.stack(patches).astype(np.uint8)
    captured["shape"] = list(image_shape)
    return _merge(patches, image_shape, win_size, step_size)
inference_segnet.merge_patches = merge_spy
_inference = inference_segnet.inference
def inference_spy(image_org, *a, **k):
    captured["segnet_input"] = np.ascontiguousarray(image_org)
    return _inference(image_org, *a, **k)
inference_segnet.inference = inference_spy

for p in sys.argv[3:]:
    name = os.path.splitext(os.path.basename(p))[0]
    d = os.path.join(sys.argv[2], name)
    os.makedirs(d, exist_ok=True)
    captured.clear()
    img = cv2.imread(p)
    img.tofile(os.path.join(d, "image.bin"))
    try:
        main.load_and_preprocess_predictions(p, False, False, False)
    except Exception as e:
        print("skip", p, type(e).__name__, e, flush=True)
        continue
    captured["tiles"].tofile(os.path.join(d, "tiles.bin"))
    captured["segnet_input"].tofile(os.path.join(d, "segnet_input.bin"))
    json.dump({"image_shape": list(img.shape), "merged_shape": captured["shape"], "tiles": int(captured["tiles"].shape[0])},
              open(os.path.join(d, "segnet.json"), "w"))
    print("dumped", d, flush=True)
