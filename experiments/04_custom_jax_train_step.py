"""Experiment 04: Low-Level Custom JAX Training Step with model.stateless_call.

Demonstrates how a Keras 3 model becomes a pure functional JAX transformation
via `model.stateless_call(trainable_variables, non_trainable_variables, x)`
inside `jax.jit` and `jax.value_and_grad(has_aux=True)`.
"""

from __future__ import annotations

import importlib
import os
import sys
import time
from pathlib import Path

os.environ.setdefault("KERAS_BACKEND", "jax")

import jax
import jax.numpy as jnp
import keras
import numpy as np
from keras import ops

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
mod01 = importlib.import_module("experiments.01_backend_agnostic_model")


def run_stateless_jax_training(
    batch_size: int = 256,
    steps: int = 30,
    learning_rate: float = 5e-2,
) -> dict:
    """Trains the Keras 3 multimodal model using pure jax.value_and_grad."""
    keras.utils.set_random_seed(42)
    model = mod01.build_multimodal_arena_model()
    (vis, tab), y = mod01.generate_synthetic_dataset(num_samples=batch_size, seed=2026)

    vis_jnp = jnp.asarray(vis)
    tab_jnp = jnp.asarray(tab)
    y_jnp = jnp.asarray(y)

    # Materialize weights once
    _ = model([vis_jnp, tab_jnp], training=False)

    trainable_vars = [v.value for v in model.trainable_variables]
    non_trainable_vars = [v.value for v in model.non_trainable_variables]

    def compute_loss(t_vars, nt_vars, x_vis, x_tab, target):
        probs, updated_nt_vars = model.stateless_call(
            t_vars, nt_vars, [x_vis, x_tab], training=True
        )
        clipped = jnp.clip(probs, 1e-7, 1.0 - 1e-7)
        loss = -jnp.mean(jnp.sum(target * jnp.log(clipped), axis=-1))
        preds = jnp.argmax(probs, axis=-1)
        labels = jnp.argmax(target, axis=-1)
        acc = jnp.mean((preds == labels).astype(jnp.float32))
        return loss, (updated_nt_vars, acc)

    grad_fn = jax.value_and_grad(compute_loss, has_aux=True)

    @jax.jit
    def train_step(t_vars, nt_vars, x_vis, x_tab, target):
        (loss, (new_nt_vars, acc)), grads = grad_fn(
            t_vars, nt_vars, x_vis, x_tab, target
        )
        # Global gradient norm clipping in pure JAX
        grad_norm = jnp.sqrt(
            sum(jnp.sum(jnp.square(g)) for g in jax.tree_util.tree_leaves(grads))
        )
        scale = jnp.minimum(1.0, 1.0 / (grad_norm + 1e-6))
        new_t_vars = jax.tree_util.tree_map(
            lambda w, g: w - learning_rate * (g * scale), t_vars, grads
        )
        return new_t_vars, new_nt_vars, loss, acc, grad_norm

    # First call triggers XLA compilation
    t_warmup_start = time.perf_counter()
    t_vars, nt_vars, init_loss, init_acc, _ = train_step(
        trainable_vars, non_trainable_vars, vis_jnp, tab_jnp, y_jnp
    )
    init_loss.block_until_ready()
    warmup_compile_ms = (time.perf_counter() - t_warmup_start) * 1000.0

    step_latencies_ms: list[float] = []
    for step_idx in range(steps):
        t0 = time.perf_counter()
        t_vars, nt_vars, loss_val, acc_val, gnorm_val = train_step(
            t_vars, nt_vars, vis_jnp, tab_jnp, y_jnp
        )
        loss_val.block_until_ready()
        step_latencies_ms.append((time.perf_counter() - t0) * 1000.0)

    # Copy trained pure-JAX arrays back into the stateful Keras model variables
    for keras_var, new_val in zip(model.trainable_variables, t_vars):
        keras_var.assign(new_val)
    for keras_var, new_val in zip(model.non_trainable_variables, nt_vars):
        keras_var.assign(new_val)

    # Verify stateful model reflects the updated parameters
    verify_probs = ops.convert_to_numpy(model([vis_jnp, tab_jnp], training=False))
    verify_loss = float(
        -np.mean(np.sum(y * np.log(np.clip(verify_probs, 1e-7, 1.0 - 1e-7)), axis=-1))
    )

    mean_step_ms = float(np.mean(step_latencies_ms))
    train_throughput = float(batch_size / (mean_step_ms / 1000.0))

    return {
        "warmup_compile_ms": round(warmup_compile_ms, 2),
        "initial_loss": round(float(init_loss), 5),
        "final_loss": round(float(loss_val), 5),
        "stateful_verify_loss": round(verify_loss, 5),
        "initial_acc": round(float(init_acc), 4),
        "final_acc": round(float(acc_val), 4),
        "final_grad_norm": round(float(gnorm_val), 4),
        "mean_step_ms": round(mean_step_ms, 3),
        "train_throughput_samples_sec": round(train_throughput, 1),
    }


def main() -> None:
    print("=== Experiment 04: Custom JAX Train Step via model.stateless_call ===")
    metrics = run_stateless_jax_training(batch_size=256, steps=30)
    print(f"Initial loss -> Final loss : {metrics['initial_loss']:.5f} -> {metrics['final_loss']:.5f}")
    print(f"Stateful model verify loss : {metrics['stateful_verify_loss']:.5f}")
    print(f"Initial acc  -> Final acc  : {metrics['initial_acc']:.2%} -> {metrics['final_acc']:.2%}")
    print(f"Mean stateless step latency: {metrics['mean_step_ms']:.3f} ms")


if __name__ == "__main__":
    main()
