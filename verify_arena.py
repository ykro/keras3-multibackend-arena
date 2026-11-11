"""The Outside Judge: End-to-End Verification and Benchmark Harness for Keras 3 Arena.

Verifies:
1. Subprocess isolation across `KERAS_BACKEND=jax`, `KERAS_BACKEND=torch`, and `KERAS_BACKEND=numpy`.
2. Weight portability through a `.keras` file: saves under `jax` and reloads in `torch` and `numpy`
   with `np.allclose(atol=1e-5)`.
3. Steady-state training & inference latency of `KERAS_BACKEND=jax` (jit-compiled, CPU) against
   `KERAS_BACKEND=torch` in eager mode (no `torch.compile`) on the Keras default torch device and,
   for a like-for-like row, pinned to the same CPU (`KERAS_TORCH_DEVICE=cpu`). Speedups are reported,
   never asserted: they depend on the machine and its load, which the report records.
4. Eager vs `jax.jit` on the same forward (real HLO fusion instructions) and the custom
   `model.stateless_call` + `jax.value_and_grad` loop (loss must decrease).
5. Writes real results to `benchmarks/latest_results.json` and `benchmarks/terminal_output.txt`.
"""

from __future__ import annotations

import importlib
import json
import os
import platform
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


def _load_average() -> float:
    return round(os.getloadavg()[0], 1)


def _cpu_name() -> str:
    try:
        out = subprocess.run(
            ["sysctl", "-n", "machdep.cpu.brand_string"], capture_output=True, text=True, check=True
        )
        return out.stdout.strip()
    except (OSError, subprocess.CalledProcessError):
        return platform.processor() or platform.machine()


def _backend_check_lines() -> list[str]:
    """Experiment 01 in the same run: one subprocess per backend builds the same class."""
    lines = []
    for backend_name in ("jax", "torch", "numpy"):
        env = os.environ.copy()
        env["KERAS_BACKEND"] = backend_name
        cmd = [sys.executable, str(ROOT_DIR / "experiments" / "01_backend_agnostic_model.py"), "--child"]
        res = subprocess.run(cmd, env=env, capture_output=True, text=True, check=True)
        lines.append(res.stdout.strip())
    return lines


def main() -> None:
    if "--child-portability" in sys.argv:
        idx = sys.argv.index("--child-portability")
        _child_save_or_eval(sys.argv[idx + 1], sys.argv[idx + 2], sys.argv[idx + 3])
        return

    import jax
    import keras
    import torch

    mod02 = importlib.import_module("experiments.02_arena_jax_vs_torch")
    mod03 = importlib.import_module("experiments.03_xla_fusion_analysis")
    mod04 = importlib.import_module("experiments.04_custom_jax_train_step")
    mod05 = importlib.import_module("experiments.05_export_and_serve")

    load_before = _load_average()
    backend_lines = _backend_check_lines()
    portability = verify_cross_backend_portability()
    arena = mod02.run_arena_benchmarks(batch_size=256, steps=45)
    fusion = mod03.analyze_xla_fusion(batch_size=256, steps=50)
    stateless = mod04.run_stateless_jax_training(batch_size=256, steps=30)
    serving = mod05.run_smoke_test()
    load_after = _load_average()

    # Correctness checks (these must hold on any machine). Speedups are measured, not asserted.
    assert len({ln.split("params=")[1] for ln in backend_lines}) == 1, "Backends built different models"
    assert stateless["final_loss"] < stateless["initial_loss"], (
        "Expected custom stateless JAX training step to decrease loss"
    )
    assert fusion["hlo_dot_ops"] == 20, "Expected one dot per matmul of the model (20)"
    assert serving["backend"] == "jax" and 0 <= serving["predicted_class"] < 10

    jax_m = arena["jax"]
    ratios = {
        name: {
            "train": round(arena[name]["train_step_p50_ms"] / jax_m["train_step_p50_ms"], 2),
            "eval": round(arena[name]["eval_step_p50_ms"] / jax_m["eval_step_p50_ms"], 2),
        }
        for name in ("torch", "torch-cpu")
    }

    env_info = {
        "cpu": _cpu_name(),
        "cores": os.cpu_count(),
        "load_average_before": load_before,
        "load_average_after": load_after,
        "versions": {"keras": keras.__version__, "jax": jax.__version__, "torch": torch.__version__},
    }
    summary = {
        "project": "keras3-multibackend-arena",
        "environment": env_info,
        "backend_check": backend_lines,
        "portability": portability,
        "arena": {
            **arena,
            "jax_vs_torch_default_device": ratios["torch"],
            "jax_vs_torch_cpu": ratios["torch-cpu"],
            "statistic": "median step latency after 4 warmup steps (45 timed steps)",
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

    rule = "-" * 90
    dev = {name: m["device"] for name, m in arena.items()}
    lines = [
        "=" * 90,
        "KERAS 3 MULTI-BACKEND ARENA — OUTSIDE JUDGE VERIFICATION REPORT",
        "=" * 90,
        f"CPU      : {env_info['cpu']}, {env_info['cores']} cores, load average {load_before} before, {load_after} after",
        f"Versions : keras {keras.__version__}, jax {jax.__version__}, torch {torch.__version__}",
        f"Model    : {jax_m['params']:,} parameters, batch {jax_m['batch_size']}, {jax_m['steps']} timed steps (median)",
        rule,
        "01. One class, three backends (each backend initializes its own random weights)",
        *[f"    {ln}" for ln in backend_lines],
        (
            f"    Weight parity via .keras (saved on jax): PASS, max |jax - torch| = "
            f"{portability['max_abs_diff_jax_vs_torch']:.2e}, max |jax - numpy| = "
            f"{portability['max_abs_diff_jax_vs_numpy']:.2e} (atol 1e-5)"
        ),
        rule,
        "02. Arena: same model, AdamW, train_on_batch / test_on_batch",
        f"    {'Config':<10} | {'Mode':<16} | {'Device':<6} | {'First step':>10} | {'Train p50':>10} | {'Eval p50':>9}",
        *[
            (
                f"    {name:<10} | {('jit (XLA)' if m['jit_compile'] else 'eager'):<16} | {m['device']:<6} | "
                f"{m['warmup_compile_ms']:>7.0f} ms | {m['train_step_p50_ms']:>7.2f} ms | {m['eval_step_p50_ms']:>6.2f} ms"
            )
            for name, m in arena.items()
        ],
        (
            f"    jax jit (cpu) vs torch eager ({dev['torch']}) : {ratios['torch']['train']:.2f}x train | "
            f"{ratios['torch']['eval']:.2f}x eval  (different processors)"
        ),
        (
            f"    jax jit (cpu) vs torch eager (cpu) : {ratios['torch-cpu']['train']:.2f}x train | "
            f"{ratios['torch-cpu']['eval']:.2f}x eval  (same processor, still jit vs eager)"
        ),
        rule,
        "03. Same backend (jax), same CPU, same forward: eager vs jax.jit",
        (
            f"    {fusion['jaxpr_primitive_ops']} primitive ops dispatched one by one in eager -> "
            f"{fusion['hlo_fusion_instructions']} fusion instructions + {fusion['hlo_dot_ops']} dots in the compiled HLO"
        ),
        (
            f"    forward {fusion['eager_forward_ms']:.2f} ms eager -> {fusion['xla_jit_forward_ms']:.2f} ms jit: "
            f"{fusion['jit_vs_eager_forward_speedup']:.2f}x"
        ),
        rule,
        "04. model.stateless_call + jax.value_and_grad, 30 steps on one synthetic batch (SGD + clipping)",
        (
            f"    loss {stateless['initial_loss']:.4f} -> {stateless['final_loss']:.4f}, "
            f"acc {stateless['initial_acc']:.2%} -> {stateless['final_acc']:.2%} (same batch it trains on), "
            f"{stateless['step_p50_ms']:.2f} ms/step"
        ),
        rule,
        "05. .keras export + local HTTP smoke test (127.0.0.1, not Cloud Run)",
        (
            f"    status=200, eager model() inside the handler = {serving['latency_ms']:.2f} ms, "
            f"artifact = {serving['artifact_size_kb']} KB (weights + AdamW state)"
        ),
        "=" * 90,
        "ALL CORRECTNESS CHECKS PASSED. Speedups are measurements of this machine and load.",
    ]
    report = "\n".join(lines)
    txt_path.write_text(report + "\n", encoding="utf-8")
    print(report)


if __name__ == "__main__":
    main()
