# CPU-only image for the ABSA FastAPI service (and, optionally, the Streamlit demo).
#
# NOTE: Docker is a separate, industry-standard deliverable. The *live demo* does
# NOT require Docker — use `make demo` on a plain CPU venv (see README). This image
# is for containerized serving.
FROM python:3.12-slim

ENV PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    HF_HUB_OFFLINE=1 \
    TRANSFORMERS_OFFLINE=1 \
    TOKENIZERS_PARALLELISM=false \
    ABSA_DEVICE=cpu

WORKDIR /app

RUN apt-get update \
    && apt-get install -y --no-install-recommends build-essential curl \
    && rm -rf /var/lib/apt/lists/*

# Dependencies (cached layer): CPU torch first (forces the +cpu wheel on Linux),
# then the project's cpu/serve/demo extras.
COPY pyproject.toml README.md ./
COPY src ./src
RUN pip install --upgrade pip wheel \
    && pip install --index-url https://download.pytorch.org/whl/cpu torch==2.5.1 \
    && pip install ".[cpu,serve,demo]" \
    && python -m spacy download en_core_web_sm

# Application files (artifacts/ are mounted at runtime — see docker-compose.yml).
COPY configs ./configs
COPY app ./app
COPY data/sample ./data/sample

EXPOSE 8000
HEALTHCHECK --interval=30s --timeout=5s --start-period=60s --retries=3 \
    CMD curl -fsS http://localhost:8000/health || exit 1

CMD ["uvicorn", "absa.serving.app:app", "--host", "0.0.0.0", "--port", "8000"]
