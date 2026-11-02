"""Build separately labelled decoder WASM artifacts; no weights are embedded.
python bench/build_decoder_variants.py <scratch-output-dir>
The strict artifact works with SIMD128. The fused artifact additionally needs
relaxed SIMD and a validated fused host (its load-time probe rejects unfused).
"""
import os
from pathlib import Path
import subprocess
import sys
out=Path(sys.argv[1]).resolve()
out.mkdir(parents=True,exist_ok=True)
for name,features in [('simd128','+simd128'),('fused','+simd128,+relaxed-simd')]:
    target=out/(name+'-target')
    subprocess.run(['cargo','build','--release','--lib','--target','wasm32-unknown-unknown'],
                   env=dict(os.environ,CARGO_TARGET_DIR=str(target),RUSTFLAGS='-C target-feature='+features),check=True)
    artifact=target/'wasm32-unknown-unknown/release/homr_rs.wasm'
    (out/('decoder-'+name+'.wasm')).write_bytes(artifact.read_bytes())
    print(name,features,out/('decoder-'+name+'.wasm'))
