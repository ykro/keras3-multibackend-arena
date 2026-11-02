"""Experiment 05: Portable .keras Artifact Export, Cross-Backend Reload, & Local HTTP Serving.

Exports the trained `.keras` archive under `KERAS_BACKEND=jax`, verifies
cross-backend reload parity, and exposes a lightweight local HTTP JSON server
(the same handler could be containerized for Cloud Run; no Cloud Run numbers here).
Note: `model(x)` runs eagerly, op by op; use `model.predict` or `jax.jit` over
`stateless_call` for a compiled serving path.
"""

from __future__ import annotations

import importlib
import json
import os
import sys
import threading
import time
import urllib.request
from http.server import BaseHTTPRequestHandler, HTTPServer
from pathlib import Path

os.environ.setdefault("KERAS_BACKEND", "jax")

import keras
import numpy as np
from keras import ops

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
mod01 = importlib.import_module("experiments.01_backend_agnostic_model")

DEFAULT_ARTIFACT_PATH = (
    Path(__file__).resolve().parent.parent / "artifacts" / "multimodal_arena.keras"
)


def export_trained_artifact(artifact_path: Path = DEFAULT_ARTIFACT_PATH) -> Path:
    """Trains the multimodal arena model briefly and saves a .keras archive."""
    artifact_path.parent.mkdir(parents=True, exist_ok=True)
    keras.utils.set_random_seed(42)
    model = mod01.build_multimodal_arena_model()
    model.compile(
        optimizer=keras.optimizers.AdamW(learning_rate=2e-3),
        loss=keras.losses.CategoricalCrossentropy(),
    )
    (vis, tab), y = mod01.generate_synthetic_dataset(num_samples=256, seed=99)
    model.fit([vis, tab], y, batch_size=64, epochs=2, verbose=0)
    model.save(artifact_path)
    return artifact_path


def create_server(model: keras.Model, host: str = "127.0.0.1", port: int = 8080) -> HTTPServer:
    """Creates a lightweight stdlib HTTPServer wrapping the model (eager `model(x)` calls)."""
    # Warm-up call (fills per-op caches; model(x) is not jit-compiled)
    (vis_w, tab_w), _ = mod01.generate_synthetic_dataset(num_samples=1, seed=1)
    _ = ops.convert_to_numpy(model([vis_w, tab_w], training=False))

    class ArenaHandler(BaseHTTPRequestHandler):
        def log_message(self, format: str, *args) -> None:
            return

        def do_GET(self) -> None:
            if self.path == "/health":
                payload = {
                    "status": "ok",
                    "backend": keras.backend.backend(),
                    "model": model.name,
                }
                body = json.dumps(payload).encode("utf-8")
                self.send_response(200)
                self.send_header("Content-Type", "application/json")
                self.send_header("Content-Length", str(len(body)))
                self.end_headers()
                self.wfile.write(body)
                return
            self.send_error(404)

        def do_POST(self) -> None:
            if self.path != "/predict":
                self.send_error(404)
                return
            length = int(self.headers.get("Content-Length", "0"))
            raw = self.rfile.read(length)
            req = json.loads(raw.decode("utf-8"))
            vis_arr = np.asarray(req["vision_patches"], dtype=np.float32)
            tab_arr = np.asarray(req["tabular_features"], dtype=np.float32)
            t0 = time.perf_counter()
            probs = ops.convert_to_numpy(model([vis_arr, tab_arr], training=False))
            latency_ms = (time.perf_counter() - t0) * 1000.0
            resp = {
                "backend": keras.backend.backend(),
                "predicted_class": int(np.argmax(probs[0])),
                "confidence": round(float(np.max(probs[0])), 6),
                "probabilities": [round(float(x), 6) for x in probs[0]],
                "latency_ms": round(latency_ms, 3),
            }
            body = json.dumps(resp).encode("utf-8")
            self.send_response(200)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)

    return HTTPServer((host, port), ArenaHandler)


def run_smoke_test() -> dict:
    """Exports the .keras file, reloads it, and runs an end-to-end HTTP smoke test."""
    artifact_path = export_trained_artifact()
    reloaded = keras.saving.load_model(artifact_path)

    server = create_server(reloaded, host="127.0.0.1", port=0)
    port = server.server_address[1]
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()

    try:
        (vis, tab), _ = mod01.generate_synthetic_dataset(num_samples=1, seed=555)
        payload = json.dumps(
            {
                "vision_patches": vis.tolist(),
                "tabular_features": tab.tolist(),
            }
        ).encode("utf-8")

        req = urllib.request.Request(
            f"http://127.0.0.1:{port}/predict",
            data=payload,
            headers={"Content-Type": "application/json"},
            method="POST",
        )
        with urllib.request.urlopen(req, timeout=5.0) as resp:
            result = json.loads(resp.read().decode("utf-8"))
    finally:
        server.shutdown()
        server.server_close()

    result["artifact_size_kb"] = round(artifact_path.stat().st_size / 1024.0, 1)
    return result


def main() -> None:
    if "--serve" in sys.argv:
        port_idx = sys.argv.index("--port") + 1 if "--port" in sys.argv else -1
        port = int(sys.argv[port_idx]) if port_idx != -1 else int(os.environ.get("PORT", "8080"))
        if not DEFAULT_ARTIFACT_PATH.exists():
            export_trained_artifact(DEFAULT_ARTIFACT_PATH)
        loaded_model = keras.saving.load_model(DEFAULT_ARTIFACT_PATH)
        srv = create_server(loaded_model, host="0.0.0.0", port=port)
        print(f"Serving {loaded_model.name} on port {port} (backend={keras.backend.backend()})...")
        srv.serve_forever()
        return

    print("=== Experiment 05: Export .keras Artifact & Local HTTP Endpoint Smoke Test ===")
    res = run_smoke_test()
    print(f"Exported artifact size : {res['artifact_size_kb']} KB")
    print(f"Serving backend        : {res['backend']}")
    print(f"Predicted class        : {res['predicted_class']} (confidence={res['confidence']:.4f})")
    print(f"model() latency in HTTP: {res['latency_ms']:.3f} ms")


if __name__ == "__main__":
    main()
