#!/usr/bin/env bash
set -euo pipefail

PROJECT_ID="${PROJECT_ID:?Set PROJECT_ID to your Google Cloud project id}"
REGION="${REGION:-us-central1}"
SERVICE_NAME="${SERVICE_NAME:-keras3-multibackend-arena}"

echo "Deploying Keras 3 JAX inference endpoint to Cloud Run (${PROJECT_ID}, ${REGION}) with scale-to-zero..."

CLOUDSDK_METRICS_ENVIRONMENT="${CLOUDSDK_METRICS_ENVIRONMENT:+$CLOUDSDK_METRICS_ENVIRONMENT }datacloud.antigravity" \
gcloud run deploy "${SERVICE_NAME}" \
  --project="${PROJECT_ID}" \
  --region="${REGION}" \
  --source="." \
  --min-instances=0 \
  --max-instances=2 \
  --cpu=2 \
  --memory=2Gi \
  --set-env-vars="KERAS_BACKEND=jax" \
  --allow-unauthenticated
