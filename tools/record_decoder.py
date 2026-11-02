"""Record homr's real native pipeline and ScoreDecoder.generate, including EOS.

uv run --no-project --with numpy --with onnxruntime==1.31.0 \
 --with opencv-python-headless==4.11.0.86 --with scipy --with scikit-image \
 --with pillow --with musicxml --with typing_extensions --with scikit-learn \
 python tools/record_decoder.py HOMR OUT IMAGE ...

Default InferenceSession options are preserved. Images run sequentially; no
staffs/steps are skipped. Output is resumable per completed page and includes
encoder float32 bytes, all six logits, tokens, attention, and decoder.json.
Keep recordings outside the repository. Record package versions with each page.
"""
import json
from pathlib import Path
import sys
import numpy as np
from homr_setup import config, load

homr, out, *images = sys.argv[1:]
main = load(homr)
from homr.transformer.decoder_inference import ScoreDecoder
from homr.transformer.staff2score import Staff2Score
from homr.music_xml_generator import XmlGeneratorArguments

original_generate = ScoreDecoder.generate
original_predict = Staff2Score.predict
original_crop = main.autocrop_with_offset
calls = []
crop = {}


def generate(self, *args, **kwargs):
    staff_dir = page_dir / str(len(calls))
    staff_dir.mkdir(exist_ok=True)
    np.asarray(kwargs["context"], dtype="<f4").tofile(staff_dir / "context.bin")
    binding = self.io_binding
    net = self.net
    steps = []

    class Binding:
        def __getattr__(self, name):
            return getattr(binding, name)

        def get_outputs(self):
            outputs = binding.get_outputs()
            logits = [v.numpy().reshape(-1).astype(np.float32) for v in outputs[:6]]
            steps.append({"tokens": [int(v.argmax()) for v in logits],
                          "attention": outputs[6].numpy().reshape(-1).tolist()})
            np.concatenate(logits).astype("<f4").tofile(logits_file)
            return outputs

    class Net:
        def run_with_iobinding(self, iobinding):
            return net.run_with_iobinding(binding)

    self.io_binding = Binding()
    self.net = Net()
    try:
        with (staff_dir / "logits.bin").open("wb") as logits_file:
            symbols = original_generate(self, *args, **kwargs)
    finally:
        self.io_binding = binding
        self.net = net
    (staff_dir / "steps.json").write_text(json.dumps(steps))
    print(page_dir.name, len(calls), len(steps), "steps", flush=True)
    return symbols


def predict(self, image):
    symbols = original_predict(self, image)
    calls.append([[s.rhythm, s.pitch, s.lift, s.articulation, s.slur, s.position,
                   np.asarray(s.coordinates).reshape(-1).tolist()] for s in symbols])
    return symbols


def autocrop(image):
    image, top_left = original_crop(image)
    crop.update(top_left=[int(v) for v in top_left], cropped_shape=list(image.shape))
    return image, top_left


ScoreDecoder.generate = generate
Staff2Score.predict = predict
main.autocrop_with_offset = autocrop
for image in images:
    page_dir = Path(out) / Path(image).stem
    if (page_dir / "decoder.json").exists():
        continue
    page_dir.mkdir(parents=True, exist_ok=True)
    calls.clear()
    crop.clear()
    error = {}
    import shutil
    local_image = page_dir / ("source" + Path(image).suffix)
    shutil.copyfile(image, local_image)
    try:
        xml = main.process_image(str(local_image), config(main), XmlGeneratorArguments())
        Path(xml).replace(page_dir / "native.musicxml")
    except AssertionError:
        error = {"error": "AssertionError"}
    finally:
        local_image.unlink(missing_ok=True)
    # Other errors abort: a partial corpus must not look complete.
    (page_dir / "decoder.json").write_text(json.dumps({"crop": crop, "calls": calls, **error}))
    import onnxruntime
    import cv2
    import PIL
    (page_dir / "reference.json").write_text(json.dumps({"onnxruntime": onnxruntime.__version__,
        "opencv": cv2.__version__, "numpy": np.__version__, "pillow": PIL.__version__, "image": image,
        "staffs": len(calls), **error}))
