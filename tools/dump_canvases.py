"""python tools/dump_canvases.py <homr dir> <out dir> <page images...>
The staff images (the 256 x 1280 canvases the transformer reads) homr cuts from each page,
natively, in the order it reads them: <out dir>/<page>/canvas-<i>.bin (uint8) and canvases.json.
The transformer itself is replaced by a stand-in which reads nothing, so this is quick.
`check_pages` feeds these to the Rust encoder and compares its own staff images with them."""
import json, os, shutil, sys
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import numpy as np
from homr_setup import config, load

main = load(os.path.abspath(sys.argv[1]))
import homr.staff_parsing as sp  # noqa: E402
from homr.music_xml_generator import XmlGeneratorArguments  # noqa: E402

out = sys.argv[2]
seen = []


def stand_in(staff_image, staff, config):
    seen.append(np.ascontiguousarray(staff_image, dtype=np.uint8))
    return []


sp.parse_staff_tromr = stand_in
cfg = config(main)
for path in sys.argv[3:]:
    name = os.path.splitext(os.path.basename(path))[0]
    d = os.path.join(out, name)
    os.makedirs(d, exist_ok=True)
    del seen[:]
    # homr writes its MusicXML (and a teaser image) next to the image it is given: work on a copy
    copy = os.path.join(d, os.path.basename(path))
    shutil.copyfile(path, copy)
    try:
        main.process_image(copy, cfg, XmlGeneratorArguments(False, None, None))
    except Exception as e:  # nothing was read, or homr fails on the page: the canvases it did cut are kept
        print(name, "homr stopped:", str(e).splitlines()[-1] if str(e) else type(e).__name__, flush=True)
    for leftover in os.listdir(d):  # the image copy, homr's output, canvases of an earlier run
        os.remove(os.path.join(d, leftover))
    for i, canvas in enumerate(seen):
        canvas.tofile(os.path.join(d, f"canvas-{i}.bin"))
    json.dump({"canvases": [list(c.shape) for c in seen]}, open(os.path.join(d, "canvases.json"), "w"))
    print(name, len(seen), flush=True)
