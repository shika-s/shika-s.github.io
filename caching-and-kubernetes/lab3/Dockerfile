# Dockerfile
# ─── Stage 1: Build ──────────────────────────────────────────────────

FROM python:3.13-slim AS build
ARG APP_DIR=/src

RUN apt-get update && apt-get install -y --no-install-recommends \
    curl build-essential libffi-dev libgomp1 \
    && rm -rf /var/lib/apt/lists/*

ENV POETRY_VERSION=2.3.1
RUN curl -sSL https://install.python-poetry.org | python3 -
ENV PATH=/root/.local/bin:$PATH

WORKDIR ${APP_DIR}
COPY pyproject.toml poetry.lock ./

RUN python -m venv --copies ${APP_DIR}/venv
ENV POETRY_VIRTUALENVS_CREATE=false
RUN . ${APP_DIR}/venv/bin/activate && poetry install --no-root --only main

# ─── Stage 2: Runtime ────────────────────────────────────────────────
FROM python:3.13-slim AS prod
ARG APP_DIR=/src

# Only need curl for healthcheck — no build tools needed
RUN apt-get update && apt-get install -y --no-install-recommends \
    curl build-essential libffi-dev libgomp1 \
    && rm -rf /var/lib/apt/lists/*

# Copy installed Python packages from builder
COPY --from=build ${APP_DIR}/venv ${APP_DIR}/venv/
ENV PATH=${APP_DIR}/venv/bin:$PATH

WORKDIR ${APP_DIR}
# Copy application code and model
COPY . ./


HEALTHCHECK --interval=30s --timeout=5s --start-period=10s --retries=3 \
    CMD curl -f http://localhost:8000/lab/health || exit 1

CMD ["uvicorn", "src.main:app", "--host", "0.0.0.0", "--port", "8000"]
