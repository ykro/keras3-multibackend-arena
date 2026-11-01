"""Pytest suite for Keras 3 Multi-Backend Arena."""

from __future__ import annotations

import importlib
import json
import os
from pathlib import Path

os.environ.setdefault("KERAS_BACKEND", "jax")

import numpy as np

import verify_arena

mod01 = importlib.import_module("experiments.01_backend_agnostic_model")
mod04 = importlib.import_module("experiments.04_custom_jax_train_step")
mod05 = importlib.import_module("experiments.05_export_and_serve")


def test_multimodal_model_forward_shape() -> None:
    model = mod01.build_multimodal_arena_model()
    (vis, tab), y = mod01.generate_synthetic_dataset(num_samples=16, seed=11)
    out = model([vis, tab], training=False)
    assert tuple(out.shape) == (16, 10)
    assert y.shape == (16, 10)
    row_sums = np.sum(np.asarray(out), axis=-1)
    assert np.allclose(row_sums, 1.0, atol=1e-5)


def test_cross_backend_weight_portability() -> None:
    portability = verify_arena.verify_cross_backend_portability()
    assert portability["output_shape"] == [32, 10]
    assert portability["allclose_atol_1e_5"] is True
    assert portability["max_abs_diff_jax_vs_torch"] < 1e-5


def test_stateless_jax_training_converges() -> None:
    metrics = mod04.run_stateless_jax_training(batch_size=128, steps=15)
    assert metrics["final_loss"] < metrics["initial_loss"]
    assert metrics["final_acc"] >= metrics["initial_acc"]


def test_export_and_http_serve_smoke() -> None:
    res = mod05.run_smoke_test()
    assert res["backend"] == "jax"
    assert 0 <= res["predicted_class"] < 10
    assert len(res["probabilities"]) == 10
    assert res["artifact_size_kb"] > 100


def test_latest_benchmark_results_exist_and_jax_wins() -> None:
    json_path = Path(__file__).resolve().parent.parent / "benchmarks" / "latest_results.json"
    if not json_path.exists():
        verify_arena.main()
    data = json.loads(json_path.read_text(encoding="utf-8"))
    jax_tp = data["arena"]["jax"]["train_throughput_samples_sec"]
    torch_tp = data["arena"]["torch"]["train_throughput_samples_sec"]
    assert jax_tp > torch_tp
