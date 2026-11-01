"""Experiment 02: Arena Benchmark — KERAS_BACKEND=jax vs KERAS_BACKEND=torch.

Executes isolated subprocess benchmarks training and evaluating the identical
Keras 3 model under JAX (OpenXLA) and PyTorch backends. Measures first-step
compilation/warmup overhead, steady-state step latency (ms), and throughput
(samples/sec).
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
    throughput = float(batch_size / (mean_step_ms / 1000.0))

    # Inference latency measurement
    _ = model.test_on_batch([vis, tab], y, return_dict=True)
    infer_times_ms: list[float] = []
    for _ in range(steps):
        ts = time.perf_counter()
        eval_logs = model.test_on_batch([vis, tab], y, return_dict=True)
        _ = float(ops.convert_to_numpy(next(iter(eval_logs.values()))))
        infer_times_ms.append((time.perf_counter() - ts) * 1000.0)

    mean_infer_ms = float(np.mean(infer_times_ms))
    infer_throughput = float(batch_size / (mean_infer_ms / 1000.0))

    return {
        "backend": keras.backend.backend(),
        "batch_size": batch_size,
        "steps": steps,
        "params": int(model.count_params()),
        "warmup_compile_ms": round(warmup_ms, 2),
        "train_step_mean_ms": round(mean_step_ms, 3),
        "train_step_p50_ms": round(p50_step_ms, 3),
        "train_step_p95_ms": round(p95_step_ms, 3),
        "train_throughput_samples_sec": round(throughput, 1),
        "eval_step_mean_ms": round(mean_infer_ms, 3),
        "eval_throughput_samples_sec": round(infer_throughput, 1),
        "final_loss": round(final_loss, 5),
    }


def run_arena_benchmarks(batch_size: int = 256, steps: int = 45) -> dict[str, dict]:
    """Runs isolated subprocesses for jax and torch backends and returns metrics."""
    results: dict[str, dict] = {}
    for backend_name in ("jax", "torch"):
        env = os.environ.copy()
        env["KERAS_BACKEND"] = backend_name
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
        results[backend_name] = payload
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

    print("=== Experiment 02: Keras 3 Multi-Backend Arena (JAX vs PyTorch) ===")
    results = run_arena_benchmarks(batch_size=256, steps=45)
    jax_m = results["jax"]
    torch_m = results["torch"]
    speedup_train = jax_m["train_throughput_samples_sec"] / torch_m["train_throughput_samples_sec"]
    speedup_eval = jax_m["eval_throughput_samples_sec"] / torch_m["eval_throughput_samples_sec"]

    print(
        f"{'Backend':<10} | {'Warmup (ms)':>12} | {'Train Step (ms)':>15} | "
        f"{'Train (img/s)':>14} | {'Eval Step (ms)':>14} | {'Eval (img/s)':>13}"
    )
    print("-" * 92)
    for b_name in ("jax", "torch"):
        m = results[b_name]
        print(
            f"{m['backend']:<10} | {m['warmup_compile_ms']:>12.2f} | "
            f"{m['train_step_mean_ms']:>15.3f} | "
            f"{m['train_throughput_samples_sec']:>14.1f} | "
            f"{m['eval_step_mean_ms']:>14.3f} | "
            f"{m['eval_throughput_samples_sec']:>13.1f}"
        )
    print("-" * 92)
    print(
        f"Steady-state speedup (KERAS_BACKEND=jax vs torch): "
        f"Train = {speedup_train:.2f}x | Eval = {speedup_eval:.2f}x"
    )


if __name__ == "__main__":
    main()
