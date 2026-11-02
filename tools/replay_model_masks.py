"""Replay browser class maps through native homr, isolating segmentation.

replay_model_masks.py HOMR NATIVE_MODEL_INPUTS NATIVE_DECODER_REF BROWSER_OUT OUT CORPUS
Native decoder symbols are replayed in call order (check_xml-style). Changed
staff canvases or call counts are reported explicitly, so this cannot pretend
to be a complete neural end-to-end run. Only differing-class pages need replay.
"""
import json
import os
from pathlib import Path
import shutil
import sys
import time
import xml.etree.ElementTree as ET

import numpy as np
from homr_setup import load, config

homr, native, refs, browser, output, corpus = map(Path, sys.argv[1:])
main = load(str(homr))
from homr.segmentation.inference_segnet import Segnet
from homr.transformer.staff2score import Staff2Score
from homr.transformer.vocabulary import EncodedSymbol
from homr.music_xml_generator import XmlGeneratorArguments


class ReplayMismatch(Exception):
    pass


def run(self, data):
    global batch
    shape = meta['batches'][batch]['output_shape']
    assert list(data.shape) == meta['batches'][batch]['shape']
    classes = np.fromfile(browser / name / f'seg-{batch}.classes.u8', dtype=np.uint8).reshape(shape[0], *shape[2:])
    batch += 1
    return (classes[:, None] == np.arange(shape[1])[None, :, None, None]).astype(np.float32)


def predict(self, image):
    global staff, different_canvases
    if staff >= len(dec['calls']):
        raise ReplayMismatch('additional staff requested')
    d = native / name / str(staff)
    shape = json.loads((d / 'input.json').read_text())['image_shape']
    canvas = np.fromfile(d / 'image.u8', dtype=np.uint8).reshape(shape)
    different_canvases += int(image.shape != canvas.shape or not np.array_equal(image, canvas))
    symbols = [EncodedSymbol(*s[:6], coordinates=tuple(s[6])) for s in dec['calls'][staff]]
    staff += 1
    return symbols


Segnet.__init__ = lambda self, *a, **k: None
Segnet.run = run
Staff2Score.__init__ = lambda self, *a, **k: None
Staff2Score.predict = predict
output.mkdir(parents=True, exist_ok=True)
rows = []
alternate_decoder = os.getenv('MODEL_DECODER_OUTPUT')
for source in corpus.read_text().splitlines():
    name = Path(source).stem
    metrics = browser / name / 'metrics.json'
    for attempt in range(7200):
        if metrics.exists():
            break
        time.sleep(1)
    else:
        raise RuntimeError(f'missing completed segmentation {name}')
    observed = json.loads(metrics.read_text())
    if not observed['segnet']['different']:
        rows.append({'page': name, 'class_maps_identical': True,
            'xml': 'see encoder replay; segmentation boundary identical' if alternate_decoder else 'identical by identical model boundary'})
        continue
    meta = json.loads((native / name / 'models.json').read_text())
    decoder_path = (Path(alternate_decoder) if alternate_decoder else refs) / name / 'decoder.json'
    for attempt in range(14400):
        if decoder_path.exists():
            break
        time.sleep(1)
    else:
        raise RuntimeError(f'decoder did not complete {name}')
    dec = json.loads(decoder_path.read_text())
    batch = staff = different_canvases = 0
    d = output / name
    d.mkdir(exist_ok=True)
    local = d / ('source' + Path(source).suffix)
    shutil.copyfile(source, local)
    status, error, without_comments_same = None, None, False
    try:
        xml = main.process_image(str(local), config(main), XmlGeneratorArguments())
        same = (refs / name / 'native.musicxml').exists() and Path(xml).read_bytes() == (refs / name / 'native.musicxml').read_bytes()
        expected_xml = refs / name / 'native.musicxml'
        without_comments_same = expected_xml.exists() and ET.tostring(ET.parse(xml).getroot()) == ET.tostring(ET.parse(expected_xml).getroot())
        status = 'identical' if same else 'different'
    except AssertionError:
        status = 'matching failure' if dec.get('error') == 'AssertionError' else 'different failure'
        without_comments_same = status == 'matching failure'
    except ReplayMismatch as e:
        status, error = 'unreplayable', str(e)
    finally:
        local.unlink(missing_ok=True)
    row = {'page': name, 'xml': status, 'staffs_called': staff, 'native_staffs': len(dec['calls']),
           'different_canvases': different_canvases, 'error': error, 'without_comments_same': without_comments_same}
    if staff != len(dec['calls']):
        row['xml'] = 'unreplayable'
    rows.append(row)
    print(row, flush=True)
report = {'decoder_source': alternate_decoder or str(refs), 'pages': rows, 'summary': {'pages': len(rows),
    'identical_class_maps': sum(r.get('class_maps_identical', False) for r in rows),
    'replayed': sum('class_maps_identical' not in r for r in rows),
    'xml_changed': sum(r['xml'] in {'different', 'different failure'} for r in rows),
    'unreplayable': sum(r['xml'] == 'unreplayable' for r in rows),
    'different_canvases': sum(r.get('different_canvases', 0) for r in rows)}}
(output / 'comparison.json').write_text(json.dumps(report, indent=2))
print(report['summary'], flush=True)
