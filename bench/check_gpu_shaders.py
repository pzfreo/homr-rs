"""Offline WGSL validation/Metal translation; this does NOT execute a GPU.

python3 bench/check_gpu_shaders.py SCRATCH_OUT [--offline]
Naga is a scratch-only validator dependency, not a homr-rs dependency.
"""
import argparse
import os
from pathlib import Path
import subprocess

p = argparse.ArgumentParser(description=__doc__)
p.add_argument("output", type=Path)
p.add_argument("--offline", action="store_true")
a = p.parse_args()
out = a.output.resolve()
repo = Path(__file__).resolve().parent.parent
out.mkdir(parents=True, exist_ok=True)
subprocess.run(["node", "--input-type=module", "-e", """
import fs from 'node:fs';import {shaderSource} from './tools/encoder_gpu_kernels.mjs';
for(const legacy of [false,true])for(const guarded of [true,false])fs.writeFileSync(process.argv[1]+'/encoder-'+(legacy?'legacy-':'')+guarded+'.wgsl',shaderSource({diagnosticUnguarded:!guarded,legacy}));
""", str(out)], cwd=repo, check=True)
project = out / "validator"
(project / "src").mkdir(parents=True, exist_ok=True)
(project / "Cargo.toml").write_text('''[package]
name="encoder-wgsl-validator"
version="0.0.0"
edition="2021"
[dependencies]
naga={version="=27.0.3",features=["wgsl-in","msl-out"]}
''')
(project / "src/main.rs").write_text('''fn main() {
for path in std::env::args().skip(1) {
 let code=std::fs::read_to_string(&path).unwrap();
 let module=naga::front::wgsl::parse_str(&code).unwrap_or_else(|e|panic!("{}",e.emit_to_string(&code)));
 let info=naga::valid::Validator::new(naga::valid::ValidationFlags::all(),naga::valid::Capabilities::empty()).validate(&module).unwrap_or_else(|e|panic!("{}",e.emit_to_string(&code)));
 let (metal,_)=naga::back::msl::write_string(&module,&info,&naga::back::msl::Options::default(),&naga::back::msl::PipelineOptions::default()).unwrap();
 std::fs::write(path.clone()+".metal",metal).unwrap();println!("Validated {path}: {} entry points, no optional capabilities",module.entry_points.len());
}}
''')
subprocess.run(["cargo", "run", "--release", "-j1", "--manifest-path", str(project / "Cargo.toml"),
                *(["--offline"] if a.offline else []), "--",
                *[str(out / ("encoder-" + legacy + guarded + ".wgsl"))
                  for legacy in ["", "legacy-"] for guarded in ["true", "false"]]],
               env=dict(os.environ, CARGO_TARGET_DIR=str(out / "target"), RUSTC_WRAPPER=""), check=True)
