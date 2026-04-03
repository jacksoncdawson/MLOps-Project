# trndly

`trndly` is the primary project workspace for the MLOps app and architecture docs.

## Project structure

- `backend/services/scheduleServer.py` - FastAPI service that predicts best listing timeframe for a user item
- `backend/services/.env` - local runtime configuration (not committed)
- `backend/services/.env.example` - example service + training configuration
- `pipelines/training/generate_synthetic_listing_data.py` - synthetic trend/user/training split generator
- `pipelines/training/feature_contract.py` - shared featurization contract used by both training and serving
- `pipelines/training/train_listing_timeline.py` - timeframe classifier training + MLflow registration
- `pipelines/training/synthetic_data/` - generated local synthetic data artifacts
- `Dockerfile` - container image for the FastAPI service
- `.dockerignore` - Docker build context exclusions
- `requirements.txt` - Python dependencies

## 1) Configure environment

Create `backend/services/.env` (or copy from `.env.example`) and set values:

```env
MLFLOW_TRACKING_URI=http://<private-mlflow-host>:5000
MLFLOW_MODEL_URI=models:/listing_timeline@champion
TREND_SIGNALS_PATH=../../pipelines/training/synthetic_data/trend_signals.csv

MLFLOW_EXPERIMENT_NAME=mlops-team-project
MLFLOW_REGISTERED_MODEL_NAME=listing_timeline
MLFLOW_MODEL_ALIAS=champion

TRAIN_DATA_URI=pipelines/training/synthetic_data/train.csv
VAL_DATA_URI=pipelines/training/synthetic_data/val.csv
TEST_DATA_URI=pipelines/training/synthetic_data/test.csv
TARGET_COLUMN=best_timeframe
DATA_VERSION=synthetic-v1
```

Notes:
- `MLFLOW_MODEL_URI` is the single model served by the API.
- `TREND_SIGNALS_PATH` points to the local trend table used to featurize user items.
- Keep the real MLflow host in local `.env` only; do not commit private tracker IPs/hosts.

## 2) Install dependencies

From `trndly/`:

```bash
python3 -m venv .venv
.venv/bin/python -m pip install --upgrade pip
.venv/bin/python -m pip install -r requirements.txt
```

## 3) Generate synthetic local data

From `trndly/`:

```bash
.venv/bin/python pipelines/training/generate_synthetic_listing_data.py
```

This generates:
- `pipelines/training/synthetic_data/trend_signals.csv`
- `pipelines/training/synthetic_data/train.csv`
- `pipelines/training/synthetic_data/val.csv`
- `pipelines/training/synthetic_data/test.csv`
- `pipelines/training/synthetic_data/user_upload_items.json`

## 4) Train and register model in MLflow

From `trndly/`:

```bash
set -a
source backend/services/.env
set +a

.venv/bin/python pipelines/training/train_listing_timeline.py
```

The script trains a timeframe classifier, logs classification metrics to MLflow,
registers a new version under `listing_timeline`, and updates alias `champion`.

## 5) Run API locally (without Docker)

From `trndly/`:

```bash
cd backend/services
../../.venv/bin/uvicorn scheduleServer:app --reload --host 0.0.0.0 --port 8000
```

### Endpoint checks

```bash
curl http://127.0.0.1:8000/
curl http://127.0.0.1:8000/health
curl -X POST http://127.0.0.1:8000/predict \
  -H "Content-Type: application/json" \
  -d '{
    "item_name":"Denim Skirt #demo",
    "color":"blue",
    "category":"skirt",
    "material":"denim"
  }'
curl -X POST http://127.0.0.1:8000/reload-model
```

Example `/predict` response format:

```json
{
  "item_name": "Denim Skirt #demo",
  "best_timeframe": "next_month",
  "timeframe_scores": {
    "current": 0.31,
    "next_week": 0.58,
    "next_month": 0.81,
    "three_months": 0.66,
    "six_months": 0.42
  },
  "model_loaded": true,
  "model_uri": "models:/listing_timeline@champion",
  "run_id": "abc123..."
}
```

## 6) Build and run Docker container

From `trndly/`:

```bash
docker build -t trndly-fastapi:local .
docker run --rm -p 8000:8000 --env-file backend/services/.env trndly-fastapi:local
```

Optional image publish workflow:

```bash
docker tag trndly-fastapi:local <registry-user>/trndly-fastapi:latest
docker push <registry-user>/trndly-fastapi:latest
```

## 7) Swapping synthetic data with real data later

Keep the API contract and `feature_contract.py` stable, and swap only data sources:
- Replace `trend_signals.csv` with real offline model output using the same schema:
  `feature_type`, `feature_value`, and timeframe columns.
- Produce train/val/test files with the same feature columns and `best_timeframe` target.
- Update `.env` paths (`TREND_SIGNALS_PATH`, `TRAIN_DATA_URI`, `VAL_DATA_URI`, `TEST_DATA_URI`) to the real files.
