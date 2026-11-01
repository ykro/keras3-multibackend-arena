#!/usr/bin/env bash
set -euo pipefail

export CLOUDSDK_METRICS_ENVIRONMENT="${CLOUDSDK_METRICS_ENVIRONMENT:+$CLOUDSDK_METRICS_ENVIRONMENT }datacloud.antigravity"
PROJECT_ID="${PROJECT_ID:-ai-experiments-487722}"
REGION="${REGION:-us-central1}"
JOB_NAME="keras3-multibackend-arena-$(date +%Y%m%d-%H%M%S)"
IMAGE_URI="us-docker.pkg.dev/vertex-ai/training/jax-gpu.0-4:latest"
JOB_ID=""

cleanup() {
  if [[ -n "${JOB_ID}" ]]; then
    echo "[cleanup] Cancelling Vertex AI Custom Job ${JOB_ID} if still running..."
    gcloud ai custom-jobs cancel "${JOB_ID}" \
      --project="${PROJECT_ID}" \
      --region="${REGION}" \
      --quiet >/dev/null 2>&1 || true
  fi
}
trap cleanup EXIT INT TERM

echo "Submitting Keras 3 Multi-Backend Arena benchmark to Vertex AI (${PROJECT_ID}, ${REGION}) on NVIDIA L4..."
BUNDLE_B64="$(tar czf - experiments verify_arena.py pyproject.toml | base64 | tr -d '\n')"

JOB_ID="$(gcloud ai custom-jobs create \
  --project="${PROJECT_ID}" \
  --region="${REGION}" \
  --display-name="${JOB_NAME}" \
  --worker-pool-spec="machine-type=g2-standard-4,replica-count=1,accelerator-type=NVIDIA_L4,accelerator-count=1,container-image-uri=${IMAGE_URI}" \
  --command="bash,-c,mkdir -p /tmp/arena && cd /tmp/arena && echo ${BUNDLE_B64} | base64 -d | tar xzf - && pip install -q keras torch && python3 experiments/02_arena_jax_vs_torch.py && python3 verify_arena.py" \
  --format="value(name)")"

gcloud ai custom-jobs stream-logs "${JOB_ID}" \
  --project="${PROJECT_ID}" \
  --region="${REGION}"
