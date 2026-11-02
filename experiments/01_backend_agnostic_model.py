"""Experiment 01: Backend-Agnostic Vision + Tabular Residual Model with keras.ops.

Builds a multimodal Gated Residual Cross-Attention network using exclusively
`keras.ops` (`keras.ops.matmul`, `keras.ops.gelu`, `keras.ops.softmax`,
`keras.ops.rsqrt`, `keras.ops.mean`) so the exact same Python class runs
without modification on `jax`, `torch`, and `numpy` backends.
"""

from __future__ import annotations

import os
import subprocess
import sys

os.environ.setdefault("KERAS_BACKEND", "jax")

import keras
import numpy as np
from keras import ops


@keras.saving.register_keras_serializable(package="arena")
class GatedResidualBlock(keras.layers.Layer):
    """Residual block using pure keras.ops for elementwise and matrix fusion."""

    def __init__(self, hidden_dim: int, **kwargs) -> None:
        super().__init__(**kwargs)
        self.hidden_dim = hidden_dim

    def build(self, input_shape: tuple[int, ...]) -> None:
        dim = int(input_shape[-1])
        self.w_up = self.add_weight(
            shape=(dim, self.hidden_dim * 2),
            initializer="glorot_uniform",
            name="w_up",
        )
        self.b_up = self.add_weight(
            shape=(self.hidden_dim * 2,),
            initializer="zeros",
            name="b_up",
        )
        self.w_down = self.add_weight(
            shape=(self.hidden_dim, dim),
            initializer="glorot_uniform",
            name="w_down",
        )
        self.b_down = self.add_weight(
            shape=(dim,),
            initializer="zeros",
            name="b_down",
        )
        self.gamma = self.add_weight(
            shape=(dim,),
            initializer="ones",
            name="rms_gamma",
        )
        super().build(input_shape)

    def call(self, x: keras.KerasTensor) -> keras.KerasTensor:
        # RMS normalization via pure keras.ops
        variance = ops.mean(ops.square(x), axis=-1, keepdims=True)
        normed = x * ops.rsqrt(variance + 1e-6) * self.gamma

        # Gated projection gelu(u) * sigmoid(v) via keras.ops
        proj = ops.matmul(normed, self.w_up) + self.b_up
        u, v = ops.split(proj, 2, axis=-1)
        gated = ops.gelu(u) * ops.sigmoid(v)

        out = ops.matmul(gated, self.w_down) + self.b_down
        return x + out

    def get_config(self) -> dict:
        config = super().get_config()
        config.update({"hidden_dim": self.hidden_dim})
        return config


@keras.saving.register_keras_serializable(package="arena")
class MultimodalCrossAttention(keras.layers.Layer):
    """Cross-modal attention fusing vision patch tokens and tabular embeddings."""

    def __init__(self, embed_dim: int, **kwargs) -> None:
        super().__init__(**kwargs)
        self.embed_dim = embed_dim

    def build(self, input_shape: list[tuple[int, ...]]) -> None:
        vis_dim = int(input_shape[0][-1])
        tab_dim = int(input_shape[1][-1])
        self.w_q = self.add_weight(
            shape=(tab_dim, self.embed_dim),
            initializer="glorot_uniform",
            name="w_q",
        )
        self.w_k = self.add_weight(
            shape=(vis_dim, self.embed_dim),
            initializer="glorot_uniform",
            name="w_k",
        )
        self.w_v = self.add_weight(
            shape=(vis_dim, self.embed_dim),
            initializer="glorot_uniform",
            name="w_v",
        )
        super().build(input_shape)

    def call(self, inputs: list[keras.KerasTensor]) -> keras.KerasTensor:
        vision_tokens, tabular_vec = inputs
        # tabular_vec: (B, D) -> query: (B, 1, E)
        q = ops.expand_dims(ops.matmul(tabular_vec, self.w_q), axis=1)
        # vision_tokens: (B, N, D) -> key/value: (B, N, E)
        k = ops.matmul(vision_tokens, self.w_k)
        v = ops.matmul(vision_tokens, self.w_v)

        scale = ops.cast(1.0 / np.sqrt(float(self.embed_dim)), dtype=q.dtype)
        scores = ops.matmul(q, ops.transpose(k, axes=(0, 2, 1))) * scale
        attn_weights = ops.softmax(scores, axis=-1)
        context = ops.squeeze(ops.matmul(attn_weights, v), axis=1)
        return ops.gelu(context + tabular_vec)

    def get_config(self) -> dict:
        config = super().get_config()
        config.update({"embed_dim": self.embed_dim})
        return config


def build_multimodal_arena_model(
    num_patches: int = 16,
    patch_dim: int = 64,
    tabular_dim: int = 32,
    embed_dim: int = 128,
    hidden_dim: int = 256,
    num_blocks: int = 6,
    num_classes: int = 10,
) -> keras.Model:
    """Constructs the Vision + Tabular Multimodal Residual Arena model."""
    vision_in = keras.Input(shape=(num_patches, patch_dim), name="vision_patches")
    tabular_in = keras.Input(shape=(tabular_dim,), name="tabular_features")

    vis_proj = keras.layers.Dense(embed_dim, name="vision_proj")(vision_in)
    tab_proj = keras.layers.Dense(embed_dim, name="tabular_proj")(tabular_in)

    fused = MultimodalCrossAttention(embed_dim=embed_dim, name="cross_attn")(
        [vis_proj, tab_proj]
    )

    x = fused
    for idx in range(num_blocks):
        x = GatedResidualBlock(hidden_dim=hidden_dim, name=f"res_block_{idx}")(x)

    logits = keras.layers.Dense(num_classes, name="logits")(x)
    probs = keras.layers.Activation("softmax", name="probabilities")(logits)

    return keras.Model(
        inputs=[vision_in, tabular_in],
        outputs=probs,
        name="multimodal_arena_net",
    )


def generate_synthetic_dataset(
    num_samples: int = 2048,
    num_patches: int = 16,
    patch_dim: int = 64,
    tabular_dim: int = 32,
    num_classes: int = 10,
    seed: int = 42,
) -> tuple[tuple[np.ndarray, np.ndarray], np.ndarray]:
    """Generates deterministic synthetic multimodal Vision + Tabular batches."""
    rng = np.random.default_rng(seed)
    vision = rng.standard_normal((num_samples, num_patches, patch_dim), dtype=np.float32)
    tabular = rng.standard_normal((num_samples, tabular_dim), dtype=np.float32)

    # Create structured target dependent on both modalities
    signal = vision[:, 0, :num_classes] * 0.6 + tabular[:, :num_classes] * 0.4
    labels = np.argmax(signal, axis=-1).astype(np.int32)
    one_hot = np.eye(num_classes, dtype=np.float32)[labels]
    return (vision, tabular), one_hot


def _run_single_backend_check() -> None:
    backend_name = keras.backend.backend()
    model = build_multimodal_arena_model()
    (vis, tab), _ = generate_synthetic_dataset(num_samples=8, seed=7)
    out = model([vis, tab], training=False)
    out_np = ops.convert_to_numpy(out)
    print(
        f"[backend={backend_name:5s}] output_shape={out_np.shape} "
        f"row0_sum={float(out_np[0].sum()):.6f} "
        f"row0_max={float(out_np[0].max()):.6f} "
        f"params={model.count_params():,}"
    )


def main() -> None:
    if "--child" in sys.argv:
        _run_single_backend_check()
        return

    print("=== Experiment 01: Backend-Agnostic Multimodal Model (keras.ops) ===")
    for backend_name in ("jax", "torch", "numpy"):
        env = os.environ.copy()
        env["KERAS_BACKEND"] = backend_name
        cmd = [sys.executable, __file__, "--child"]
        res = subprocess.run(cmd, env=env, capture_output=True, text=True, check=True)
        print(res.stdout.strip())


if __name__ == "__main__":
    main()
