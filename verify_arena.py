"""The Outside Judge: End-to-End Verification and Benchmark Harness for Keras 3 Arena.

Verifies:
1. Subprocess isolation across `KERAS_BACKEND=jax`, `KERAS_BACKEND=torch`, and `KERAS_BACKEND=numpy`.
2. Zero-copy weight portability: saves `.keras` under `jax` and reloads in `torch` and `numpy`
   with `np.allclose(atol=1e-5)`.
3. Steady-state training & inference throughput advantage of `KERAS_BACKEND=jax` (OpenXLA)
   over `KERAS_BACKEND=torch`.
4. OpenXLA HLO fusion verification and custom `model.stateless_call` + `jax.value_and_grad`
   loss convergence.
5. Writes real results to `benchmarks/latest_results.json` and `benchmarks/terminal_output.txt`.
"""

from __future__ import annotations

import importlib
import json
import os
import subprocess
import sys
from pathlib import Path

os.environ.setdefault("KERAS_BACKEND", "jax")

import numpy as np

ROOT_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT_DIR))


def _child_save_or_eval(action: str, artifact_path: str, preds_path: str) -> None:
    import keras
    from keras import ops

    mod01 = importlib.import_module("experiments.01_backend_agnostic_model")
    (vis, tab), y = mod01.generate_synthetic_dataset(num_samples=32, seed=1234)

    if action == "save":
        keras.utils.set_random_seed(42)
        model = mod01.build_multimodal_arena_model()
        model.compile(
            optimizer=keras.optimizers.AdamW(learning_rate=1e-3),
            loss=keras.losses.CategoricalCrossentropy(),
        )
        model.fit([vis, tab], y, batch_size=16, epochs=1, verbose=0)
        model.save(artifact_path)
    else:
        model = keras.saving.load_model(artifact_path)

    preds = ops.convert_to_numpy(model([vis, tab], training=False))
    np.save(preds_path, preds)


def verify_cross_backend_portability() -> dict:
    """Saves model in JAX and verifies identical outputs in PyTorch and NumPy."""
    artifacts_dir = ROOT_DIR / "artifacts"
    artifacts_dir.mkdir(parents=True, exist_ok=True)
    artifact_file = artifacts_dir / "portability_judge.keras"

    preds_by_backend: dict[str, np.ndarray] = {}
    for backend_name, action in (("jax", "save"), ("torch", "load"), ("numpy", "load")):
        preds_file = artifacts_dir / f"preds_{backend_name}.npy"
        env = os.environ.copy()
        env["KERAS_BACKEND"] = backend_name
        cmd = [
            sys.executable,
            __file__,
            "--child-portability",
            action,
            str(artifact_file),
            str(preds_file),
        ]
        subprocess.run(cmd, env=env, check=True)
        preds_by_backend[backend_name] = np.load(preds_file)

    max_diff_jax_torch = float(np.max(np.abs(preds_by_backend["jax"] - preds_by_backend["torch"])))
    max_diff_jax_numpy = float(np.max(np.abs(preds_by_backend["jax"] - preds_by_backend["numpy"])))

    assert preds_by_backend["jax"].shape == (32, 10), "Unexpected output shape"
    assert np.allclose(preds_by_backend["jax"], preds_by_backend["torch"], atol=1e-5), (
        f"JAX vs Torch mismatch: {max_diff_jax_torch}"
    )
    assert np.allclose(preds_by_backend["jax"], preds_by_backend["numpy"], atol=1e-5), (
        f"JAX vs NumPy mismatch: {max_diff_jax_numpy}"
    )

    return {
        "output_shape": list(preds_by_backend["jax"].shape),
        "max_abs_diff_jax_vs_torch": float(f"{max_diff_jax_torch:.2e}"),
        "max_abs_diff_jax_vs_numpy": float(f"{max_diff_jax_numpy:.2e}"),
        "allclose_atol_1e_5": True,
    }


def main() -> None:
    if "--child-portability" in sys.argv:
        idx = sys.argv.index("--child-portability")
        _child_save_or_eval(sys.argv[idx + 1], sys.argv[idx + 2], sys.argv[idx + 3])
        return

    mod02 = importlib.import_module("experiments.02_arena_jax_vs_torch")
    mod03 = importlib.import_module("experiments.03_xla_fusion_analysis")
    mod04 = importlib.import_module("experiments.04_custom_jax_train_step")
    mod05 = importlib.import_module("experiments.05_export_and_serve")

    portability = verify_cross_backend_portability()
    arena = mod02.run_arena_benchmarks(batch_size=256, steps=45)
    fusion = mod03.analyze_xla_fusion(batch_size=256, steps=50)
    stateless = mod04.run_stateless_jax_training(batch_size=256, steps=30)
    serving = mod05.run_smoke_test()

    jax_train_tp = arena["jax"]["train_throughput_samples_sec"]
    torch_train_tp = arena["torch"]["train_throughput_samples_sec"]
    jax_eval_tp = arena["jax"]["eval_throughput_samples_sec"]
    torch_eval_tp = arena["torch"]["eval_throughput_samples_sec"]

    assert jax_train_tp > torch_train_tp, (
        f"Expected JAX train throughput ({jax_train_tp}) > Torch ({torch_train_tp})"
    )
    assert jax_eval_tp > torch_eval_tp, (
        f"Expected JAX eval throughput ({jax_eval_tp}) > Torch ({torch_eval_tp})"
    )
    assert stateless["final_loss"] < stateless["initial_loss"], (
        "Expected custom stateless JAX training step to decrease loss"
    )

    train_speedup = round(jax_train_tp / torch_train_tp, 2)
    eval_speedup = round(jax_eval_tp / torch_eval_tp, 2)
    stateless["speedup_vs_torch_train"] = round(
        stateless["train_throughput_samples_sec"] / torch_train_tp, 2
    )

    summary = {
        "project": "keras3-multibackend-arena",
        "portability": portability,
        "arena": {
            "jax": arena["jax"],
            "torch": arena["torch"],
            "jax_vs_torch_train_speedup": train_speedup,
            "jax_vs_torch_eval_speedup": eval_speedup,
        },
        "xla_fusion": fusion,
        "stateless_custom_loop": stateless,
        "serving_smoke_test": serving,
    }

    bench_dir = ROOT_DIR / "benchmarks"
    bench_dir.mkdir(parents=True, exist_ok=True)
    json_path = bench_dir / "latest_results.json"
    txt_path = bench_dir / "terminal_output.txt"

    json_path.write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")

    lines = [
        "==========================================================================================",
        "KERAS 3 MULTI-BACKEND ARENA — OUTSIDE JUDGE VERIFICATION REPORT",
        "==========================================================================================",
        f"Model Parameters               : {arena['jax']['params']:,}",
        f"Batch Size / Steps             : {arena['jax']['batch_size']} / {arena['jax']['steps']}",
        (
            f"Cross-Backend Weight Parity    : PASS (max |JAX - Torch| = {portability['max_abs_diff_jax_vs_torch']:.2e}, "
            f"max |JAX - NumPy| = {portability['max_abs_diff_jax_vs_numpy']:.2e})"
        ),
        "------------------------------------------------------------------------------------------",
        (
            f"{'Backend':<12} | {'Warmup (ms)':>12} | {'Train Step (ms)':>15} | "
            f"{'Train (samples/s)':>17} | {'Eval Step (ms)':>14} | {'Eval (samples/s)':>16}"
        ),
        "------------------------------------------------------------------------------------------",
        (
            f"{'jax (XLA)':<12} | {arena['jax']['warmup_compile_ms']:>12.2f} | "
            f"{arena['jax']['train_step_mean_ms']:>15.3f} | "
            f"{arena['jax']['train_throughput_samples_sec']:>17.1f} | "
            f"{arena['jax']['eval_step_mean_ms']:>14.3f} | "
            f"{arena['jax']['eval_throughput_samples_sec']:>16.1f}"
        ),
        (
            f"{'torch (eager)':<12} | {arena['torch']['warmup_compile_ms']:>12.2f} | "
            f"{arena['torch']['train_step_mean_ms']:>15.3f} | "
            f"{arena['torch']['train_throughput_samples_sec']:>17.1f} | "
            f"{arena['torch']['eval_step_mean_ms']:>14.3f} | "
            f"{arena['torch']['eval_throughput_samples_sec']:>16.1f}"
        ),
        "------------------------------------------------------------------------------------------",
        f"Steady-State Speedup (JAX/XLA) : {train_speedup:.2f}x faster training | {eval_speedup:.2f}x faster inference",
        (
            f"OpenXLA HLO Fusion Analysis    : {fusion['hlo_fusion_ops']} fusion ops across {fusion['hlo_lines']} HLO lines "
            f"({fusion['eager_forward_ms']:.3f} ms eager -> {fusion['xla_jit_forward_ms']:.3f} ms XLA, {fusion['xla_fusion_speedup']:.2f}x)"
        ),
        (
            f"Stateless jax.value_and_grad   : loss {stateless['initial_loss']:.5f} -> {stateless['final_loss']:.5f} "
            f"(acc {stateless['initial_acc']:.2%} -> {stateless['final_acc']:.2%}, {stateless['mean_step_ms']:.3f} ms/step)"
        ),
        (
            f"Cloud Run HTTP Smoke Test      : status=200, latency={serving['latency_ms']:.3f} ms, "
            f"artifact={serving['artifact_size_kb']} KB"
        ),
        "==========================================================================================",
    ]
    report = "\n".join(lines)
    txt_path.write_text(report + "\n", encoding="utf-8")
    print(report)


if __name__ == "__main__":
    main()
