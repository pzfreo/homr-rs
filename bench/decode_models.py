"""Stream completed browser contexts into the exact Rust decoder, resumably.

decode_models.py DECODER_ONNX NATIVE_DECODER_REF BROWSER_OUT OUT CORPUS_TXT
Build `cargo build --release --bin decode_contexts` first. DECODER_CHECKER may
select another executable. Per-staff data/step comparisons remain in scratch.
This also consumes the output of models_chrome.mjs without any format change.
"""
import json
import os
from pathlib import Path
import subprocess
import sys
import time

model, refs, contexts, out, corpus = map(Path, sys.argv[1:])
out.mkdir(parents=True, exist_ok=True)
reports = out / 'reports'
reports.mkdir(exist_ok=True)
rows = []
for image in corpus.read_text().splitlines():
    name = Path(image).stem
    complete = reports / f'{name}.json'
    if not complete.exists():
        count = len(json.loads((refs / name / 'decoder.json').read_text())['calls'])
        for attempt in range(14400):
            try:
                meta = json.loads((contexts / name / 'metrics.json').read_text())
                if len(meta['encoder']) == count:
                    break
            except (FileNotFoundError, json.JSONDecodeError):
                pass
            time.sleep(1)
        else:
            raise RuntimeError(f'encoder did not complete {name}')
        subprocess.run([os.getenv('DECODER_CHECKER', 'target/release/decode_contexts'),
                        str(model), str(refs), str(contexts), str(out), name], check=True)
        (out / 'comparison.json').replace(complete)
    report = json.loads(complete.read_text())
    rows.extend(report['pages'])
    print(name, report['summary'], flush=True)
staffs = [s for row in rows for s in row['staffs']]
keys = ['steps', 'native_steps', 'overlapping_steps', 'differing_steps', 'differing_tokens', 'missing_steps', 'extra_steps']
summary = {key: sum(s[key] for s in staffs) for key in keys}
summary.update(pages=len(rows), staffs=len(staffs), differing_staffs=sum(s['different'] for s in staffs))
(out / 'comparison.json').write_text(json.dumps({'summary': summary, 'pages': rows}, indent=2))
print(summary, flush=True)
