"""run_corpus.py <homr dir> <out dir> <images...>: homr's MusicXML (or its error) per page."""
import os, shutil, sys, time, json, traceback
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from homr_setup import config, load
main = load(os.path.abspath(sys.argv[1]))
from homr.music_xml_generator import XmlGeneratorArguments
out = sys.argv[2]; os.makedirs(out, exist_ok=True)
cfg = config(main)
for img in sys.argv[3:]:
    name = os.path.splitext(os.path.basename(img))[0]
    if os.path.exists(os.path.join(out, name + ".json")): continue
    dst = os.path.join(out, os.path.basename(img)); shutil.copy(img, dst)
    t = time.time(); res = {}
    for stale in (os.path.join(out, name + ".musicxml"),):  # a failing page must not keep an old reference
        if os.path.exists(stale): os.remove(stale)
    try:
        main.process_image(dst, cfg, XmlGeneratorArguments(False, None, None))
        res["ok"] = True
    except Exception as e:
        res["error"] = str(e).splitlines()[-1] if str(e) else type(e).__name__
    res["secs"] = round(time.time() - t, 1)
    os.remove(dst)
    json.dump(res, open(os.path.join(out, name + ".json"), "w"))
    print(name, res, flush=True)
