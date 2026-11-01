# Keras 3 Multi-Backend Arena (`keras3-multibackend-arena`)

**Por qué es relevante**: Demuestra con mediciones verificables que un mismo modelo multimodal escrito con `keras.ops` se ejecuta sin cambios sobre `jax`, `torch` y `numpy`, logrando `2.00x` mayor throughput de entrenamiento y `1.82x` mayor throughput de inferencia al compilarse con OpenXLA (`KERAS_BACKEND=jax`).
**Para quién es**: Ingenieros de Machine Learning y arquitectos de sistemas de IA que buscan portabilidad entre frameworks sin sacrificar el rendimiento de compilación XLA ni la simplicidad de despliegue en contenedores ligeros.
**Quién debería verlo**: Equipos que hoy mantienen código duplicado entre PyTorch y JAX o que desean llevar modelos Keras 3 a Vertex AI (GPU NVIDIA L4) y Cloud Run con escala a cero (`--min-instances=0`).
Parte de la serie [JAX & OpenXLA Field Guide](https://github.com/ykro/jax-openxla-field-guide).

---

## Arquitectura Multi-Backend (`keras.ops` -> JAX / PyTorch / NumPy)

![Keras 3 Multi-Backend Architecture](diagrams/01-keras3-multibackend-arch.png)

El modelo `multimodal_arena_net` (`657,418` parámetros) fusiona parches visuales `(B, 16, 64)` y vectores tabulares `(B, 32)` mediante atención cruzada (`MultimodalCrossAttention`) y seis bloques residuales con compuertas GeGLU (`GatedResidualBlock`) implementados exclusivamente con primitivas de `keras.ops` (`ops.matmul`, `ops.gelu`, `ops.softmax`, `ops.rsqrt`, `ops.mean`).

---

## Resultados Reales del Juez Externo (`verify_arena.py`)

![Keras 3 JAX vs PyTorch Throughput](diagrams/02-jax-vs-torch-throughput.png)

| Backend / Modo | Warmup / Compilación (ms) | Latencia Train Step (ms) | Throughput Train (samples/s) | Latencia Eval Step (ms) | Throughput Eval (samples/s) | Speedup vs PyTorch |
| :--- | ---: | ---: | ---: | ---: | ---: | ---: |
| `KERAS_BACKEND=jax` (OpenXLA JIT) | `659.84` | `10.325` | `24,794.8` | `4.614` | `55,482.9` | **2.00x train / 1.82x eval** |
| `KERAS_BACKEND=torch` (Eager) | `159.03` | `20.643` | `12,401.3` | `8.411` | `30,435.1` | `1.00x` (baseline) |
| Custom Stateless JAX (`value_and_grad`) | `373.32` | `9.866` | `25,946.4` | — | — | **2.09x train** |

### Salida verificada de `verify_arena.py`

```text
==========================================================================================
KERAS 3 MULTI-BACKEND ARENA — OUTSIDE JUDGE VERIFICATION REPORT
==========================================================================================
Model Parameters               : 657,418
Batch Size / Steps             : 256 / 45
Cross-Backend Weight Parity    : PASS (max |JAX - Torch| = 2.98e-07, max |JAX - NumPy| = 4.17e-07)
------------------------------------------------------------------------------------------
Backend      |  Warmup (ms) | Train Step (ms) | Train (samples/s) | Eval Step (ms) | Eval (samples/s)
------------------------------------------------------------------------------------------
jax (XLA)    |       659.84 |          10.325 |           24794.8 |          4.614 |          55482.9
torch (eager) |       159.03 |          20.643 |           12401.3 |          8.411 |          30435.1
------------------------------------------------------------------------------------------
Steady-State Speedup (JAX/XLA) : 2.00x faster training | 1.82x faster inference
OpenXLA HLO Fusion Analysis    : 249 fusion ops across 1026 HLO lines (7.176 ms eager -> 4.065 ms XLA, 1.77x)
Stateless jax.value_and_grad   : loss 2.63890 -> 0.69450 (acc 12.11% -> 91.41%, 9.866 ms/step)
Cloud Run HTTP Smoke Test      : status=200, latency=4.167 ms, artifact=7787.6 KB
==========================================================================================
```

---

## Puente Funcional: `model.stateless_call` + `jax.value_and_grad` -> Cloud Run

![Stateless JAX Training and Cloud Run Flow](diagrams/03-stateless-call-flow.png)

En `experiments/04_custom_jax_train_step.py`, extraemos los PyTrees `trainable_variables` y `non_trainable_variables` de Keras 3 y ejecutamos el modelo como una transformación funcional pura dentro de `@jax.jit` y `jax.value_and_grad(has_aux=True)`. Al finalizar, los arreglos actualizados se sincronizan con `keras_var.assign(new_val)` y se exportan en un archivo `.keras` estándar de `7787.6 KB` listo para servirse en Cloud Run con `--min-instances=0`.

---

## Onboarding y Cómo Ejecutar Paso a Paso

### 1. Instalación desde cero

```bash
git clone https://github.com/ykro/keras3-multibackend-arena.git
cd keras3-multibackend-arena
python3 -m venv .venv
source .venv/bin/activate
pip install --upgrade pip
pip install -e ".[dev]"
```

### 2. Recorrido guiado de los 5 experimentos

- `experiments/01_backend_agnostic_model.py` — Modelo multimodal Vision + Tabular escrito con `keras.ops` y verificado en subprocesos sobre `jax`, `torch` y `numpy`.
- `experiments/02_arena_jax_vs_torch.py` — Benchmark aislado por subprocesos midiendo compilación inicial, latencia por batch (`batch_size=256`) y samples por segundo.
- `experiments/03_xla_fusion_analysis.py` — Inspección del programa HLO compilado con OpenXLA (`1026` líneas HLO, `249` ocurrencias de `fusion`, `20` operaciones `dot`).
- `experiments/04_custom_jax_train_step.py` — Loop de entrenamiento en JAX puro usando `model.stateless_call`, recorte global de gradiente y `jax.tree_util.tree_map`.
- `experiments/05_export_and_serve.py` — Exportación del artefacto `.keras` y servidor HTTP ligero con pre-calentamiento XLA para Cloud Run.

```bash
# Ejecutar cada experimento en orden
python experiments/01_backend_agnostic_model.py
python experiments/02_arena_jax_vs_torch.py
python experiments/03_xla_fusion_analysis.py
python experiments/04_custom_jax_train_step.py
python experiments/05_export_and_serve.py

# Ejecutar el Juez Externo, ruff y pytest en un solo comando
make all

# Ejecutar en Vertex AI (GPU NVIDIA L4) o desplegar en Cloud Run (--min-instances=0)
bash gcp/run_vertex_l4_arena.sh
bash gcp/deploy_cloud_run.sh
```
