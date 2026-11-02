"""Build explicit strict/fused shared-memory artifacts, without wasm-bindgen.

python bench/build_encoder_threads.py SCRATCH_OUT [--toolchain nightly]
rustup toolchain install nightly --profile minimal --component rust-src
The regular single-thread builds still use build_decoder_variants.py.
"""
import argparse
import os
from pathlib import Path
import subprocess

p = argparse.ArgumentParser(description=__doc__)
p.add_argument('output', type=Path)
p.add_argument('--toolchain', default='nightly')
p.add_argument('--probes', action='store_true', help='include isolated-operator diagnostic exports')
p.add_argument('--chunk-bench', action='store_true', help='include scheduler tuning and sleeping-rank diagnostics, never ship')
a = p.parse_args()
a.output.mkdir(parents=True, exist_ok=True)
for name, simd in [('strict', '+simd128'), ('fused', '+simd128,+relaxed-simd')]:
    target = (a.output / (name + '-target')).resolve()
    flags = '-C target-feature=' + simd + ',+atomics,+bulk-memory,+mutable-globals'
    for arg in ['--shared-memory', '--import-memory', '--export-memory', '--max-memory=1073741824',
                '--export=__stack_pointer', '--export-if-defined=__wasm_init_tls',
                '--export-if-defined=__tls_size', '--export-if-defined=__tls_align']:
        flags += ' -C link-arg=' + arg
    subprocess.run(['cargo', '+' + a.toolchain, 'build', '-Z', 'build-std=std,panic_abort',
                    '--release', '--lib', '--target', 'wasm32-unknown-unknown',
                    '--features', 'encoder-threads' + (',encoder-probes' if a.probes else '') + (',encoder-chunk-bench' if a.chunk_bench else ''), '-j2'],
                   env=dict(os.environ, CARGO_TARGET_DIR=str(target), RUSTFLAGS=flags), check=True)
    artifact = a.output / ('encoder-threads-' + name + ('-probes' if a.probes else '') + ('-chunk-bench' if a.chunk_bench else '') + '.wasm')
    artifact.write_bytes((target / 'wasm32-unknown-unknown/release/homr_rs.wasm').read_bytes())
    print(artifact, flush=True)
