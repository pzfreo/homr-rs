"""Imports plain homr from a directory (a checkout of github.com/liebharc/homr
with its model files), stubbing optional dependencies that staff detection
doesn't use (title OCR, PDF input, model download, and onnxruntime where it's
missing)."""
import sys
import types


def load(homr_dir):
    for name, attrs in {
        "rapidocr": {"RapidOCR": type("RapidOCR", (), {})},
        "pypdfium2": {},
        "requests": {"exceptions": types.SimpleNamespace(RequestException=Exception)},
    }.items():
        if name not in sys.modules:
            m = types.ModuleType(name)
            m.__dict__.update(attrs)
            sys.modules[name] = m
    try:
        import onnxruntime  # noqa: F401
    except ImportError:  # e.g. in Pyodide: staff detection doesn't run the models
        m = types.ModuleType("onnxruntime")
        m.get_available_providers = lambda: ["CPUExecutionProvider"]
        sys.modules["onnxruntime"] = m
    sys.path.insert(0, homr_dir)
    from homr import main

    return main


def config(main):
    return main.ProcessingConfig(
        enable_debug=False, enable_cache=False, write_staff_positions=False, read_staff_positions=False,
        selected_staff=-1, transformer_use_gpu=False, segnet_use_gpu=False, coreml_encoder=False, title_detection=False,
    )
