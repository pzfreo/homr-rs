"""Record native homr model inputs without repeating the expensive decoder.

Use record_decoder.py's pinned Python dependencies, plus onnx if inspecting
graphs. Usage: record_models.py HOMR NATIVE_DECODER_REF OUT MUSIC_FILES_TXT
All output must be in scratch. Segnet inputs are losslessly stored as uint8
(homr casts integer pixels to float32); encoder inputs are little-endian f32.
Default native session options, including threading/optimization, are retained.
"""
import hashlib
import json
from pathlib import Path
import shutil
import sys

import numpy as np
from homr_setup import config, load
from instr import instructions

homr, refs, dest, corpus = map(Path, sys.argv[1:])
main = load(str(homr))
import cv2
import onnxruntime as ort
import PIL
from homr.segmentation.inference_segnet import Segnet
from homr.transformer.staff2score import Staff2Score, _transform
from homr.transformer.vocabulary import EncodedSymbol
from homr.music_xml_generator import XmlGeneratorArguments

original_run = Segnet.run


def run(self, data):
    assert data.dtype == np.float32 and np.array_equal(data, data.astype(np.uint8))
    index = len(batches)
    data.astype(np.uint8).tofile(page / f"seg-{index}.input.u8")
    before = instructions()
    result = original_run(self, data)
    retired = instructions() - before
    classes = result.argmax(axis=1).astype(np.uint8)
    classes.tofile(page / f"seg-{index}.native.u8")
    batches.append({"shape": list(data.shape), "output_shape": list(result.shape),
                    "instructions": retired})
    return result


def predict(self, image):
    global staff
    d = page / str(staff)
    d.mkdir(exist_ok=True)
    x = _transform(image=image)
    x.astype("<f4").tofile(d / "input.f32")
    image.astype(np.uint8).tofile(d / "image.u8")
    (d / "input.json").write_text(json.dumps({"shape": list(x.shape), "image_shape": list(image.shape)}))
    context = reference / str(staff) / "context.bin"
    assert context.stat().st_size == 1280 * 512 * 4
    shutil.copyfile(context, d / "context.bin")
    symbols = [EncodedSymbol(*s[:6], coordinates=tuple(s[6])) for s in meta["calls"][staff]]
    staff += 1
    return symbols


Segnet.run = run
# predict uses the native staff canvas and normalization, and replays symbols
# from the same native run. No inference session options are changed.
Staff2Score.__init__ = lambda self, *a, **k: None
Staff2Score.predict = predict
versions = {"onnxruntime": ort.__version__, "numpy": np.__version__,
            "opencv": cv2.__version__, "pillow": PIL.__version__}
assert ort.__version__ == "1.31.0" and cv2.__version__ == "4.11.0"
models = list(homr.glob("homr/segmentation/segnet*.onnx")) + list(homr.glob("homr/transformer/encoder*.onnx"))
hashes = {p.name: hashlib.sha256(p.read_bytes()).hexdigest() for p in models}
for image in Path(corpus).read_text().splitlines():
    page = dest / Path(image).stem
    if (page / "models.json").exists():
        continue
    page.mkdir(parents=True, exist_ok=True)
    reference = refs / page.name
    meta = json.loads((reference / "decoder.json").read_text())
    assert json.loads((reference / "reference.json").read_text())["onnxruntime"] == ort.__version__
    batches, staff = [], 0
    local = page / ("source" + Path(image).suffix)
    shutil.copyfile(image, local)
    error = None
    try:
        xml = main.process_image(str(local), config(main), XmlGeneratorArguments())
        assert Path(xml).read_bytes() == (reference / "native.musicxml").read_bytes()
        Path(xml).unlink()
    except AssertionError:
        if meta.get("error") != "AssertionError":
            raise
        error = "AssertionError"
    finally:
        local.unlink(missing_ok=True)
    assert error == meta.get("error") and staff == len(meta["calls"])
    (page / "models.json").write_text(json.dumps({"versions": versions, "models_sha256": hashes,
        "image": image, "batches": batches, "staffs": staff, "error": error}, indent=2))
    print(page.name, len(batches), "batches", staff, "staffs", flush=True)
