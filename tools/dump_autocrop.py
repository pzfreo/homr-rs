"""python tools/dump_autocrop.py <homr dir> <out dir> <image files...>
homr's autocrop_with_offset on each image, natively: the decoded image (image.bin, BGR
bytes, meta.json), its crop (crop.bin) and offset (ref.json) in <out dir>/<name>;
compare with the `check_autocrop` binary (it takes the dumps as cases and as reference)."""
import json, os, sys
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import cv2
import numpy as np
from homr_setup import load

load(os.path.abspath(sys.argv[1]))
from homr.autocrop import autocrop_with_offset  # noqa: E402

out = sys.argv[2]
for path in sys.argv[3:]:
    img = cv2.imread(path)
    if img is None:
        print("skip", path, flush=True)
        continue
    d = os.path.join(out, os.path.splitext(os.path.basename(path))[0])
    os.makedirs(d, exist_ok=True)
    np.ascontiguousarray(img, dtype=np.uint8).tofile(os.path.join(d, "image.bin"))
    json.dump({"shape": list(img.shape)}, open(os.path.join(d, "meta.json"), "w"))
    crop, (x, y) = autocrop_with_offset(img)
    np.ascontiguousarray(crop, dtype=np.uint8).tofile(os.path.join(d, "crop.bin"))
    json.dump({"shape": list(crop.shape), "offset": [int(x), int(y)]}, open(os.path.join(d, "ref.json"), "w"))
print("done", len(sys.argv) - 3, flush=True)
