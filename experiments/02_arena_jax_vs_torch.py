"""Experiment 02: Arena Benchmark — KERAS_BACKEND=jax vs KERAS_BACKEND=torch.

Executes isolated subprocess benchmarks training and evaluating the identical
Keras 3 model in three configurations:

- `jax`        : Keras-jax, `jit_compile=True` (the whole train step compiled by XLA), on the JAX CPU device.
- `torch`      : Keras-torch eager (no `torch.compile`) on the Keras default torch device
                 (MPS, the integrated GPU, on Apple Silicon; CUDA on NVIDIA; else CPU).
- `torch-cpu`  : Keras-torch eager pinned to the CPU with `KERAS_TORCH_DEVICE=cpu`,
                 the same processor JAX uses. This is the like-for-like row.

Measures the first step (tracing/compilation or warmup), and the median steady-state
step latency (ms) and throughput (samples/sec). Medians, because the laptop is shared.
"""

from __future__ import annotations

import json
import os
import subprocess
import sys
import time
from pathlib import Path

os.environ.setdefault("KERAS_BACKEND", "jax")

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))


def _run_worker(batch_size: int = 256, steps: int = 45) -> dict:
    import importlib

    import keras
    import numpy as np
    from keras import ops

    mod01 = importlib.import_module("experiments.01_backend_agnostic_model")
    build_multimodal_arena_model = mod01.build_multimodal_arena_model
    generate_synthetic_dataset = mod01.generate_synthetic_dataset

    keras.utils.set_random_seed(42)
    model = build_multimodal_arena_model(
        num_patches=16,
        patch_dim=64,
        tabular_dim=32,
        embed_dim=128,
        hidden_dim=256,
        num_blocks=6,
        num_classes=10,
    )
    model.compile(
        optimizer=keras.optimizers.AdamW(learning_rate=1e-3, weight_decay=1e-4),
        loss=keras.losses.CategoricalCrossentropy(),
        jit_compile=(keras.backend.backend() == "jax"),
    )

    (vis, tab), y = generate_synthetic_dataset(num_samples=batch_size, seed=101)

    # Warmup / compilation step (First step)
    t0 = time.perf_counter()
    logs = model.train_on_batch([vis, tab], y, return_dict=True)
    # Force synchronization across backends
    _ = float(ops.convert_to_numpy(next(iter(logs.values()))))
    warmup_ms = (time.perf_counter() - t0) * 1000.0

    # Extra 3 warmup iterations to stabilize caches
    for _ in range(3):
        logs = model.train_on_batch([vis, tab], y, return_dict=True)
        _ = float(ops.convert_to_numpy(next(iter(logs.values()))))

    # Steady-state timed loop
    step_times_ms: list[float] = []
    for _ in range(steps):
        ts = time.perf_counter()
        logs = model.train_on_batch([vis, tab], y, return_dict=True)
        _ = float(ops.convert_to_numpy(next(iter(logs.values()))))
        step_times_ms.append((time.perf_counter() - ts) * 1000.0)

    final_loss = float(ops.convert_to_numpy(next(iter(logs.values()))))
    mean_step_ms = float(np.mean(step_times_ms))
    p50_step_ms = float(np.percentile(step_times_ms, 50))
    p95_step_ms = float(np.percentile(step_times_ms, 95))
    throughput = float(batch_size / (p50_step_ms / 1000.0))

    # Inference latency measurement
    _ = model.test_on_batch([vis, tab], y, return_dict=True)
    infer_times_ms: list[float] = []
    for _ in range(steps):
        ts = time.perf_counter()
        eval_logs = model.test_on_batch([vis, tab], y, return_dict=True)
        _ = float(ops.convert_to_numpy(next(iter(eval_logs.values()))))
        infer_times_ms.append((time.perf_counter() - ts) * 1000.0)

    mean_infer_ms = float(np.mean(infer_times_ms))
    p50_infer_ms = float(np.percentile(infer_times_ms, 50))
    infer_throughput = float(batch_size / (p50_infer_ms / 1000.0))

    if keras.backend.backend() == "jax":
        import jax

        device = jax.devices()[0].platform
    else:
        from keras.src.backend.torch import core as torch_core

        device = str(torch_core.get_device())

    return {
        "backend": keras.backend.backend(),
        "device": device,
        "jit_compile": keras.backend.backend() == "jax",
        "batch_size": batch_size,
        "steps": steps,
        "params": int(model.count_params()),
        "warmup_compile_ms": round(warmup_ms, 2),
        "train_step_mean_ms": round(mean_step_ms, 3),
        "train_step_p50_ms": round(p50_step_ms, 3),
        "train_step_p95_ms": round(p95_step_ms, 3),
        "train_throughput_samples_sec": round(throughput, 1),
        "eval_step_mean_ms": round(mean_infer_ms, 3),
        "eval_step_p50_ms": round(p50_infer_ms, 3),
        "eval_throughput_samples_sec": round(infer_throughput, 1),
        "final_loss": round(final_loss, 5),
    }


# name -> (KERAS_BACKEND, KERAS_TORCH_DEVICE or None for the Keras default)
ARENA_CONFIGS: dict[str, tuple[str, str | None]] = {
    "jax": ("jax", None),
    "torch": ("torch", None),
    "torch-cpu": ("torch", "cpu"),
}


def run_arena_benchmarks(batch_size: int = 256, steps: int = 45) -> dict[str, dict]:
    """Runs one isolated subprocess per arena configuration and returns metrics."""
    results: dict[str, dict] = {}
    for name, (backend_name, torch_device) in ARENA_CONFIGS.items():
        env = os.environ.copy()
        env["KERAS_BACKEND"] = backend_name
        env.pop("KERAS_TORCH_DEVICE", None)
        if torch_device:
            env["KERAS_TORCH_DEVICE"] = torch_device
        cmd = [
            sys.executable,
            __file__,
            "--worker",
            "--batch-size",
            str(batch_size),
            "--steps",
            str(steps),
        ]
        proc = subprocess.run(cmd, env=env, capture_output=True, text=True, check=True)
        line = [ln for ln in proc.stdout.splitlines() if ln.startswith("ARENA_JSON:")][-1]
        payload = json.loads(line.removeprefix("ARENA_JSON:"))
        results[name] = payload
    return results


def main() -> None:
    if "--worker" in sys.argv:
        b_idx = sys.argv.index("--batch-size") + 1 if "--batch-size" in sys.argv else -1
        s_idx = sys.argv.index("--steps") + 1 if "--steps" in sys.argv else -1
        batch_size = int(sys.argv[b_idx]) if b_idx != -1 else 256
        steps = int(sys.argv[s_idx]) if s_idx != -1 else 45
        metrics = _run_worker(batch_size=batch_size, steps=steps)
        print("ARENA_JSON:" + json.dumps(metrics))
        return

    print("=== Experiment 02: Keras 3 Multi-Backend Arena (same model, three configurations) ===")
    results = run_arena_benchmarks(batch_size=256, steps=45)
    print(
        f"{'Config':<10} | {'Device':>6} | {'First step (ms)':>15} | {'Train p50 (ms)':>14} | "
        f"{'Train (samples/s)':>17} | {'Eval p50 (ms)':>13}"
    )
    print("-" * 92)
    for name, m in results.items():
        print(
            f"{name:<10} | {m['device']:>6} | {m['warmup_compile_ms']:>15.2f} | "
            f"{m['train_step_p50_ms']:>14.3f} | {m['train_throughput_samples_sec']:>17.1f} | "
            f"{m['eval_step_p50_ms']:>13.3f}"
        )
    print("-" * 92)
    jax_m = results["jax"]
    for name in ("torch", "torch-cpu"):
        m = results[name]
        print(
            f"Keras-jax jit (cpu) vs {name} eager ({m['device']}): "
            f"train {m['train_step_p50_ms'] / jax_m['train_step_p50_ms']:.2f}x | "
            f"eval {m['eval_step_p50_ms'] / jax_m['eval_step_p50_ms']:.2f}x"
        )

if __name__ == "__main__":
    main()
