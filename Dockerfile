FROM python:3.12-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    KERAS_BACKEND=jax \
    PORT=8080

WORKDIR /app

RUN pip install --no-cache-dir \
    "keras>=3.14.0" \
    "jax[cpu]>=0.9.0" \
    "numpy>=2.0.0"

COPY experiments/ /app/experiments/
COPY artifacts/ /app/artifacts/

EXPOSE 8080

CMD ["python", "experiments/05_export_and_serve.py", "--serve", "--port", "8080"]
