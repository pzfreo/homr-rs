"""Local-only Safari corpus runner and shared manifest generator.

python3 bench/gpu_encoder_server.py S WASM SCRATCH_OUT [--threads 4]
Open the printed URL in Safari. The static page POSTs JSON to /report.
--manifest-only emits {roots,manifest} for gpu_encoder_chrome.mjs instead.
Full audits are deliberately expensive CPU/GPU validation, not performance runs.
"""
import argparse
import http.server
import json
import subprocess
from pathlib import Path
import time
from urllib.parse import unquote, urlsplit


def prepare(args):
    scratch = args.scratch.resolve()
    repo = Path(__file__).resolve().parent.parent
    encoder = next((scratch / "models_files").glob("encoder_pytorch_model_465-*.onnx"))
    decoder = next((scratch / "plain/src/homr/transformer").glob("decoder_pytorch_model_465-*.onnx"))
    roots = {"bench": str(repo / "bench"), "tools": str(repo / "tools"),
             "artifact": str(args.wasm.resolve().parent), "encoder": str(encoder.parent),
             "decoder": str(decoder.parent), "inputs": str(scratch / "models_native"),
             "native": str(scratch / "decoder_native_ref"), "config": str(args.output.resolve())}
    staffs = []
    pages = []
    for source in (scratch / "music_files.txt").read_text().splitlines():
        page = Path(source).stem
        meta = json.loads((scratch / "models_native" / page / "models.json").read_text())
        pages.append(page)
        for staff in range(meta["staffs"]):
            for prefix, name in [("inputs", "input.f32"), ("native", "context.bin"), ("native", "steps.json")]:
                if not (Path(roots[prefix]) / page / str(staff) / name).is_file():
                    raise ValueError(f"Missing {prefix}/{page}/{staff}/{name}")
            steps = len(json.loads((Path(roots["native"]) / page / str(staff) / "steps.json").read_text()))
            staffs.append({"page": page, "staff": staff, "input": f"/inputs/{page}/{staff}/input.f32",
                           "context": f"/native/{page}/{staff}/context.bin", "steps": f"/native/{page}/{staff}/steps.json", "native_steps": steps})
    total = len(staffs)
    if total != 627 or len(set(pages)) != 88 or sum(s["native_steps"] for s in staffs) != 38076:
        raise ValueError(f"Corpus coverage changed: {total} staffs, {len(pages)} pages")
    if args.only:
        page, staff = args.only.rsplit("/", 1)
        staffs = [s for s in staffs if s["page"] == page and s["staff"] == int(staff)]
    if args.limit:
        staffs = staffs[:args.limit]
    if not staffs:
        raise ValueError("Empty staff selection")
    manifest = {"wasm": "/artifact/" + args.wasm.name, "encoder": "/encoder/" + encoder.name,
                "decoder": "/decoder/" + decoder.name, "threads": args.threads,
                "staffs": staffs, "pages": len({s["page"] for s in staffs}), "corpus_staffs": total,
                "audit": args.audit, "math_cases": args.math_cases, "post_report": not args.manifest_only,
                "main_thread": args.main_thread, "native_steps": sum(s["native_steps"] for s in staffs),
                "revision": subprocess.check_output(["git", "-C", str(repo), "rev-parse", "HEAD"], text=True).strip(),
                "wasm_bytes": args.wasm.stat().st_size, "wasm_mtime_ns": args.wasm.stat().st_mtime_ns}
    args.output.mkdir(parents=True, exist_ok=True)
    (args.output / "manifest.json").write_text(json.dumps(manifest, indent=2))
    return roots, manifest


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("scratch", type=Path)
    p.add_argument("wasm", type=Path)
    p.add_argument("output", type=Path)
    p.add_argument("--threads", type=int, default=4)
    p.add_argument("--port", type=int, default=8796)
    p.add_argument("--audit", choices=["full", "first", "none"], default="full")
    p.add_argument("--math-cases", type=int, default=1048576)
    p.add_argument("--limit", type=int)
    p.add_argument("--only", help="page/staff, e.g. angelus_0/0")
    p.add_argument("--main-thread", action="store_true", help="ordinary WASM only, if worker WebGPU is unavailable")
    p.add_argument("--manifest-only", action="store_true")
    args = p.parse_args()
    if not 1 <= args.threads <= 32 or args.math_cases < 1 or (args.limit is not None and args.limit < 1):
        p.error("Invalid thread/count arguments")
    roots, manifest = prepare(args)
    if args.manifest_only:
        print(json.dumps({"roots": roots, "manifest": manifest}))
        return

    class Handler(http.server.SimpleHTTPRequestHandler):
        extensions_map = {**http.server.SimpleHTTPRequestHandler.extensions_map, ".mjs": "text/javascript", ".wasm": "application/wasm"}

        def end_headers(self):
            for name, value in [("Cross-Origin-Opener-Policy", "same-origin"), ("Cross-Origin-Embedder-Policy", "require-corp"),
                                ("Cross-Origin-Resource-Policy", "same-origin"), ("Cache-Control", "no-store")]:
                self.send_header(name, value)
            super().end_headers()

        def translate_path(self, url):
            path = unquote(urlsplit(url).path)
            if path == "/":
                return str(Path(roots["bench"]) / "gpu_encoder.html")
            prefix, _, rest = path.lstrip("/").partition("/")
            root = roots.get(prefix)
            if not root:
                return "/nonexistent"
            result = (Path(root) / rest).resolve()
            return str(result) if result.is_relative_to(Path(root).resolve()) else "/nonexistent"

        def do_POST(self):
            size = int(self.headers.get("Content-Length", "0"))
            if self.path != "/report" or not 0 < size <= 32 * 1024 * 1024:
                self.send_error(400)
                return
            data = json.loads(self.rfile.read(size))
            dest = args.output / f"report-{time.time_ns()}.json"
            dest.write_text(json.dumps(data, indent=2))
            print(f"Saved {dest}; complete={data.get('complete')}; all_gpu_bit_exact={data.get('all_gpu_bit_exact')}", flush=True)
            self.send_response(204)
            self.end_headers()

        def log_message(self, *_):
            pass

    server = http.server.ThreadingHTTPServer(("127.0.0.1", args.port), Handler)
    print(f"Open http://127.0.0.1:{server.server_port}/ in Safari. {len(manifest['staffs'])} staffs, audit={args.audit}.", flush=True)
    server.serve_forever()


if __name__ == "__main__":
    main()
