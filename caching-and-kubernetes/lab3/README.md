# Lab 3: Caching and Kubernetes

A FastAPI-based ML inference service for California housing price prediction, deployed on Kubernetes with Redis caching.

---

## Overview

This lab extends a housing price prediction API with:
- **Redis-backed response caching** via `fastapi-simple-redis-cache`
- **Kubernetes deployment** with health probes, resource limits, and init containers
- **Multi-stage Docker build** for a minimal production image

---

## Project Structure

```
lab3/
├── src/
│   ├── main.py               # FastAPI app entry point; mounts sub-application at /lab
│   └── housing_predict.py    # Sub-application with prediction endpoints and Redis cache middleware
├── trainer/
│   ├── train.py              # Trains an SVR model via GridSearchCV; saves model_pipeline.pkl
│   └── predict.py            # Benchmarks vectorized vs. loop-based inference
├── tests/
│   └── test_src.py           # Pytest test suite covering all endpoints and cache behavior
├── infra/
│   ├── namespace.yaml             # Kubernetes namespace: w255
│   ├── deployment-redis.yaml      # Redis deployment (1 replica, redis:7)
│   ├── service-redis.yaml         # ClusterIP service for Redis on port 6379
│   ├── deployment-pythonapi.yaml  # Python API deployment (3 replicas) with health probes
│   └── service-prediction.yaml   # LoadBalancer service exposing port 8000
├── Dockerfile                # Two-stage Alpine build (build + prod)
├── pyproject.toml            # Poetry dependencies
├── model_pipeline.pkl        # Pre-trained serialized model
└── curl_calls.sh             # Example curl commands for manual testing
```

---

## ML Model

**File:** [trainer/train.py](trainer/train.py)

- Dataset: California Housing (`sklearn.datasets.fetch_california_housing`)
- Pipeline: `SimpleImputer` → `RobustScaler` → `SVR`
- Tuning: `GridSearchCV` with 5-fold CV over imputer strategy, scaler quantile range, SVR `C` and `gamma`
- Output: `model_pipeline.pkl` (saved only if not already present)

---

## API Endpoints

All routes are mounted under the `/lab` prefix.

| Method | Path | Description |
|--------|------|-------------|
| `GET` | `/lab/health` | Returns current server time in ISO 8601 format |
| `GET` | `/lab/hello?name=<name>` | Returns a greeting message |
| `POST` | `/lab/predict` | Single house price prediction |
| `POST` | `/lab/bulk-predict` | Batch house price predictions |
| `GET` | `/lab/docs` | Auto-generated Swagger UI |

### Input Schema (`House`)

| Field | Type | Validation |
|-------|------|------------|
| `MedInc` | float | >= 0 |
| `HouseAge` | float | >= 0 |
| `AveRooms` | float | >= 0 |
| `AveBedrms` | float | >= 0 |
| `Population` | float | >= 0 |
| `AveOccup` | float | >= 0 |
| `Latitude` | float | -90 to 90 |
| `Longitude` | float | -180 to 180 |

Extra fields are rejected (`extra="forbid"`).

---

## Redis Caching

**File:** [src/housing_predict.py](src/housing_predict.py)

The `NaiveCache` middleware from `fastapi-simple-redis-cache` caches responses for `/predict` and `/bulk-predict`. The `/lab/health` endpoint is excluded from caching.

- Cache key prefix: `w255-cache-prediction`
- Redis connection is configured via environment variables:
  - `REDIS_URL` (default: `redis://127.0.0.1`)
  - `REDIS_PORT` (default: `6379`)
- Cache hits are indicated by the `x-cache-hit: True` response header
- Cache can be bypassed with `Cache-Control: no-store`

---

## Docker

**File:** [Dockerfile](Dockerfile)

Two-stage Alpine build:

1. **Build stage** — installs Poetry and project dependencies into a virtual environment
2. **Production stage** — copies only the venv and application code; runs `uvicorn` on port 8000

```bash
# Build
docker build -t lab3:latest .

# Run locally (requires Redis at localhost:6379)
docker run -p 8000:8000 lab3:latest
```

---

## Kubernetes Infrastructure

**Directory:** [infra/](infra/)

All resources are deployed in the `w255` namespace.

### Deployments

| Resource | Replicas | Image |
|----------|----------|-------|
| `python-api-deployment` | 3 | `lab3:latest` |
| `redis-deployment` | 1 | `redis:7` |

### Services

| Resource | Type | Port |
|----------|------|------|
| `prediction-service` | LoadBalancer | 8000 |
| `redis-service` | ClusterIP | 6379 |

### Health Probes (Python API)

- **Startup probe** — checks `/lab/health`, delay 10s, period 5s
- **Readiness probe** — checks `/lab/health`, delay 10s, period 3s
- **Liveness probe** — checks `/lab/health`, delay 15s, period 5s

### Init Containers

Before the API pod starts, two init containers verify Redis is available:

1. `init-verify-redis-service-dns` — polls DNS resolution of `redis-service:6379`
2. `init-verify-redis-ready` — sends a `PING` and waits for a `PONG` response

### Apply all manifests

```bash
kubectl apply -f infra/namespace.yaml
kubectl apply -f infra/deployment-redis.yaml
kubectl apply -f infra/service-redis.yaml
kubectl apply -f infra/deployment-pythonapi.yaml
kubectl apply -f infra/service-prediction.yaml
```

---

## Running Tests

```bash
poetry install
poetry run pytest tests/
```

Test coverage includes:
- `/lab/hello` — valid name, missing name, spaces, wrong HTTP method
- `/lab/health` — status, response shape, ISO format, mocked time
- `/lab/predict` — basic prediction, cache miss/hit, different input is a miss, `no-store` bypass
- `/lab/bulk-predict` — basic batch, correct count, missing fields, wrong method, cache miss/hit

---

## Example curl Calls

```bash
# Health check
curl -X GET "http://127.0.0.1:8000/lab/health"

# Hello
curl -X GET "http://127.0.0.1:8000/lab/hello?name=World"

# Single prediction
curl -X POST "http://127.0.0.1:8000/lab/predict" \
  -H "Content-Type: application/json" \
  -d '{"MedInc": 8.3, "HouseAge": 41.0, "AveRooms": 6.98, "AveBedrms": 1.02, "Population": 322.0, "AveOccup": 2.55, "Latitude": 37.88, "Longitude": -122.23}'

# Bulk prediction
curl -X POST "http://127.0.0.1:8000/lab/bulk-predict" \
  -H "Content-Type: application/json" \
  -d '{"houses": [{"MedInc": 1, "HouseAge": 1, "AveRooms": 3, "AveBedrms": 3, "Population": 3, "AveOccup": 5, "Latitude": 1, "Longitude": 1}]}'
```

---

## Dependencies

| Package | Version | Purpose |
|---------|---------|---------|
| `fastapi[standard]` | ^0.128 | Web framework |
| `scikit-learn` | 1.8.0 | ML model training and inference |
| `fastapi-simple-redis-cache` | ^2.0 | Redis response caching middleware |
| `pandas` | ^3.0 | DataFrame construction for model input |
| `pytest` | ^9.0 | Testing |
| `ruff` | ^0.14 | Linting |
