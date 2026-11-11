"""Experiment 03: XLA Kernel Fusion Analysis on Keras 3 JAX Backend (eager vs jit, same CPU).

Inspects the OpenXLA HLO lowered and compiled program for `GatedResidualBlock`
and measures the latency difference between eager op-by-op dispatch and
XLA-compiled fused execution in `KERAS_BACKEND=jax`.
"""

from __future__ import annotations

import importlib
import os
import re
import sys
import time
from pathlib import Path

os.environ.setdefault("KERAS_BACKEND", "jax")

import jax
import keras
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
mod01 = importlib.import_module("experiments.01_backend_agnostic_model")


def analyze_xla_fusion(batch_size: int = 256, steps: int = 50) -> dict:
    """Compares eager vs XLA-compiled execution and extracts HLO fusion stats."""
    keras.utils.set_random_seed(42)
    model = mod01.build_multimodal_arena_model()
    (vis, tab), _ = mod01.generate_synthetic_dataset(num_samples=batch_size, seed=77)
    vis_jax = jax.numpy.asarray(vis)
    tab_jax = jax.numpy.asarray(tab)

    # Build variables
    _ = model([vis_jax, tab_jax], training=False)
    trainable_vars = [v.value for v in model.trainable_variables]
    non_trainable_vars = [v.value for v in model.non_trainable_variables]

    def pure_forward(t_vars, nt_vars, v_in, t_in):
        preds, _ = model.stateless_call(t_vars, nt_vars, [v_in, t_in], training=False)
        return preds

    jit_forward = jax.jit(pure_forward)

    # Lower and compile to inspect OpenXLA HLO text
    lowered = jit_forward.lower(trainable_vars, non_trainable_vars, vis_jax, tab_jax)
    compiled = lowered.compile()
    hlo_text = compiled.as_text()

    # `hlo_text.count("fusion")` counts a substring that each fusion repeats in its name, its opcode and
    # every use; the real number of fused kernels is the number of `fusion(` instructions.
    fusion_substring_count = hlo_text.count("fusion")
    fusion_instructions = len(re.findall(r"=\s*\S+\s+fusion\(", hlo_text))
    custom_call_count = hlo_text.count("custom-call")
    dot_count = hlo_text.count("dot(")

    # Primitive operations that eager mode dispatches one by one (the jaxpr equations, nested ones included).
    def _count_eqns(jaxpr) -> int:
        n = 0
        for eqn in jaxpr.eqns:
            subs = [v for v in eqn.params.values() if hasattr(v, "jaxpr") or hasattr(v, "eqns")]
            if subs:
                n += sum(_count_eqns(getattr(v, "jaxpr", v)) for v in subs)
            else:
                n += 1
        return n

    closed = jax.make_jaxpr(pure_forward)(trainable_vars, non_trainable_vars, vis_jax, tab_jax)
    primitive_ops = _count_eqns(closed.jaxpr)

    # Warmup both paths
    _ = pure_forward(trainable_vars, non_trainable_vars, vis_jax, tab_jax).block_until_ready()
    _ = jit_forward(trainable_vars, non_trainable_vars, vis_jax, tab_jax).block_until_ready()

    eager_times: list[float] = []
    for _ in range(steps):
        t0 = time.perf_counter()
        out = pure_forward(trainable_vars, non_trainable_vars, vis_jax, tab_jax)
        out.block_until_ready()
        eager_times.append((time.perf_counter() - t0) * 1000.0)

    jit_times: list[float] = []
    for _ in range(steps):
        t0 = time.perf_counter()
        out = jit_forward(trainable_vars, non_trainable_vars, vis_jax, tab_jax)
        out.block_until_ready()
        jit_times.append((time.perf_counter() - t0) * 1000.0)

    eager_ms = float(np.median(eager_times))
    jit_ms = float(np.median(jit_times))
    fusion_speedup = eager_ms / jit_ms

    return {
        "hlo_lines": len(hlo_text.splitlines()),
        "jaxpr_primitive_ops": primitive_ops,
        "hlo_fusion_instructions": fusion_instructions,
        "hlo_fusion_substring_count": fusion_substring_count,
        "hlo_dot_ops": dot_count,
        "hlo_custom_calls": custom_call_count,
        "eager_forward_ms": round(eager_ms, 3),
        "xla_jit_forward_ms": round(jit_ms, 3),
        "jit_vs_eager_forward_speedup": round(fusion_speedup, 2),
    }


def main() -> None:
    print("=== Experiment 03: eager vs jax.jit on the same forward (KERAS_BACKEND=jax, same CPU) ===")
    stats = analyze_xla_fusion(batch_size=256, steps=50)
    print(f"Primitive ops (eager)    : {stats['jaxpr_primitive_ops']}")
    print(f"HLO fusion instructions  : {stats['hlo_fusion_instructions']}")
    print(f"HLO dot (matmul) ops     : {stats['hlo_dot_ops']}")
    print(f"Eager forward (median)   : {stats['eager_forward_ms']:.3f} ms")
    print(f"jax.jit forward (median) : {stats['xla_jit_forward_ms']:.3f} ms")
    print(f"jit vs eager, forward    : {stats['jit_vs_eager_forward_speedup']:.2f}x")


if __name__ == "__main__":
    main()
