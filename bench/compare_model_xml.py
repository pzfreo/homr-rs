"""Run unchanged byte-exact check_xml semantics, plus classify comment drift.

compare_model_xml.py PAGE_FIXTURES DECODER_OUTPUT XML_REFERENCES OUT
Build `check_xml` first; CHECK_XML_BIN optionally selects its executable.
All saved XML and JSON are scratch. Comment-stripped comparisons supplement,
never replace, the byte comparison. Failure transitions remain differences.
"""
import json
import os
from pathlib import Path
import subprocess
import sys
import xml.etree.ElementTree as ET

pages, decoder, references, output = map(Path, sys.argv[1:])
output.mkdir(parents=True, exist_ok=True)
with (output / 'check_xml.log').open('w') as log:
    subprocess.run([os.getenv('CHECK_XML_BIN', 'target/release/check_xml'), str(pages), str(decoder), str(references)],
                   env=dict(os.environ, CHECK_XML_OUTPUT=str(output)), stdout=log, stderr=subprocess.STDOUT, check=True)
rows = []
for d in sorted(decoder.iterdir()):
    if not (d / 'decoder.json').is_file():
        continue
    native_error = json.loads((d / 'decoder.json').read_text()).get('error')
    p = output / d.name
    calls_consumed = json.loads((p / 'status.json').read_text())['unused_calls'] == 0
    error = (p / 'error.txt').read_text() if (p / 'error.txt').exists() else None
    got = p / 'result.musicxml'
    ref = references / d.name / 'replay.musicxml'
    if native_error:
        rows.append({'page': d.name, 'native_error': native_error, 'error': error,
                     'byte_same': calls_consumed and error == native_error,
                     'without_comments_same': calls_consumed and error == native_error})
    else:
        rows.append({'page': d.name, 'error': error,
            'byte_same': calls_consumed and got.exists() and got.read_bytes() == ref.read_bytes(),
            'without_comments_same': calls_consumed and got.exists() and ET.tostring(ET.parse(got).getroot()) == ET.tostring(ET.parse(ref).getroot())})
summary = {'pages': len(rows), 'byte_different': sum(not r['byte_same'] for r in rows),
    'without_comments_different': sum(not r['without_comments_same'] for r in rows),
    'matching_failures': sum(bool(r.get('native_error')) and r['byte_same'] for r in rows)}
(output / 'comparison.json').write_text(json.dumps({'summary': summary, 'pages': rows}, indent=2))
print(summary)
