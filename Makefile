PYTHON ?= $(shell if [ -x ../../.venv-shared/bin/python ]; then echo ../../.venv-shared/bin/python; else command -v python3; fi)
PYTEST ?= $(shell if [ -x ../../.venv-shared/bin/pytest ]; then echo ../../.venv-shared/bin/pytest; else command -v pytest; fi)
RUFF ?= $(shell if [ -x ../../.venv-shared/bin/ruff ]; then echo ../../.venv-shared/bin/ruff; else command -v ruff; fi)

.PHONY: all report experiments verify test lint security clean

all: lint experiments verify test security

# One run for the recording: ruff, pytest and the judge (which runs the five experiments once and
# writes benchmarks/latest_results.json from that same run).
report: lint test verify

experiments:
	KERAS_BACKEND=jax $(PYTHON) experiments/01_backend_agnostic_model.py
	$(PYTHON) experiments/02_arena_jax_vs_torch.py
	KERAS_BACKEND=jax $(PYTHON) experiments/03_xla_fusion_analysis.py
	KERAS_BACKEND=jax $(PYTHON) experiments/04_custom_jax_train_step.py
	KERAS_BACKEND=jax $(PYTHON) experiments/05_export_and_serve.py

verify:
	$(PYTHON) verify_arena.py

test:
	$(PYTEST) -v -p no:warnings

lint:
	$(RUFF) check .

security:
	gitleaks detect --source . --no-git

clean:
	rm -rf __pycache__ .pytest_cache .ruff_cache artifacts/*.keras artifacts/*.npy
