# ============================================================
# ML Pipeline Container
# ============================================================
# Runs the full ML pipeline: model_comparison, predict,
# district_extremes, error_analysis, SHAP, etc.
#
# Build:
#   docker build -f Dockerfile.ml -t wfp-ml-pipeline .
#
# Run full pipeline (all countries, strategy B, pop target):
#   docker run -v $(pwd)/data:/app/data -v $(pwd)/logs:/app/logs \
#     wfp-ml-pipeline \
#     poetry run python scripts/machine_learning/run_ml_pipeline.py \
#       --strategy B --target-type pop
#
# Run single script:
#   docker run -v $(pwd)/data:/app/data \
#     wfp-ml-pipeline \
#     poetry run python scripts/machine_learning/model_comparison.py \
#       --iso3 cmr --strategy B --target-type pop
#
# Run with reduced features:
#   docker run -v $(pwd)/data:/app/data \
#     -e FEATURE_SUFFIX=_top20 \
#     wfp-ml-pipeline \
#     poetry run python scripts/machine_learning/model_comparison.py \
#       --iso3 moz --strategy B --target-type pop
# ============================================================

FROM python:3.12-slim

# System dependencies for scientific packages
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    gcc \
    g++ \
    gfortran \
    libopenblas-dev \
    liblapack-dev \
    cmake \
    git \
    curl \
    unzip \
    && rm -rf /var/lib/apt/lists/*

# Install AWS CLI v2
RUN curl -s "https://awscli.amazonaws.com/awscli-exe-linux-x86_64.zip" -o /tmp/awscliv2.zip \
    && unzip -q /tmp/awscliv2.zip -d /tmp \
    && /tmp/aws/install \
    && rm -rf /tmp/aws /tmp/awscliv2.zip

# Install poetry
ENV POETRY_HOME="/opt/poetry"
ENV PATH="$POETRY_HOME/bin:$PATH"
RUN pip install --no-cache-dir poetry

WORKDIR /app

# Copy dependency files first (for layer caching)
COPY pyproject.toml poetry.lock ./

# Install dependencies (no dev, no interaction)
RUN poetry config virtualenvs.create true \
    && poetry config virtualenvs.in-project true \
    && poetry install --no-interaction --no-ansi --no-root

# Copy pipeline code and entrypoint
COPY scripts/machine_learning/ scripts/machine_learning/
COPY scripts/merge/ scripts/merge/
COPY scripts/entrypoint.sh scripts/entrypoint.sh
RUN chmod +x scripts/entrypoint.sh

# Data is mounted at runtime, but create directory structure
RUN mkdir -p data/feature_engineering/admin_level_2 \
    data/models data/predictions data/joined \
    logs

# Entrypoint syncs data from/to S3, then runs the given command
ENTRYPOINT ["scripts/entrypoint.sh"]
CMD ["poetry", "run", "python", "scripts/machine_learning/run_ml_pipeline.py", "--help"]
