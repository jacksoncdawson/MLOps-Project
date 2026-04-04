# trndly

`trndly` is the primary project workspace for the MLOps checkpoint.

Current scope in this repo:

- synthetic local data generation (`train/val/test` + user payloads),
- model training + MLflow registration,
- FastAPI serving with model loaded from MLflow Model Registry alias URI.

## Project structure

- `backend/services/scheduleServer.py` - FastAPI service (`/`, `/health`, `/predict`)
- `backend/services/.env` - local runtime config (not committed)
- `pipelines/training/generate_synthetic_listing_data.py` - synthetic dataset generator
- `pipelines/training/feature_contract.py` - shared featurization contract for training + serving
- `pipelines/training/train_listing_timeline.py` - classifier training + MLflow registration/alias assignment
- `pipelines/training/synthetic_data/` - generated local artifacts
- `scripts/run_mlflow_experiment.sh` - one-command experiment run
- `scripts/run_api.sh` - one-command API start
- `scripts/kill_api.sh` - stop API processes on configured port
- `Dockerfile` - API container build

## Project Setup

### 1) Install dependencies

From `trndly/`:

```bash
python3 -m venv .venv
.venv/bin/python -m pip install --upgrade pip
.venv/bin/python -m pip install -r requirements.txt
```

### 2) Create local env file

Create `backend/services/.env` with values like:

```env
MLFLOW_TRACKING_URI=http://<private-mlflow-host>:5000
MLFLOW_MODEL_URI=models:/listing_timeline_experiments@candidate
MLFLOW_EXPERIMENT_NAME=checkpoint_fastapi
TREND_SIGNALS_PATH=../../pipelines/training/synthetic_data/trend_signals.csv
MLFLOW_MODEL_ARTIFACT_PATH=model
```

Notes:

- `MLFLOW_MODEL_URI` is the model the API serves.
- Keep private tracking host/IP in local `.env` only.
- `MLFLOW_MODEL_ARTIFACT_PATH` should be `model` (artifact path in a run), not a filesystem path.

### 3) Run training experiment (logs to MLflow)

From `trndly/`:

```bash
./scripts/run_mlflow_experiment.sh
```

By default this script:

- regenerates synthetic data,
- trains classifier using synthetic splits,
- logs params/metrics/artifacts to MLflow,
- registers to `listing_timeline_experiments`,
- sets alias `candidate`.

Example override:

```bash
N_ESTIMATORS=400 MODEL_ALIAS=candidate DATA_VERSION=synthetic-exp2 ./scripts/run_mlflow_experiment.sh
```

---

## Running the API

### Without Docker

From `trndly/`:

```bash
./scripts/run_api.sh
```

Equivalent direct command (without helper script):

```bash
cd backend/services
../../.venv/bin/uvicorn scheduleServer:app --host 127.0.0.1 --port 8000
```

Stop API:

```bash
./scripts/kill_api.sh
```

### With Docker

From `trndly/`:

```bash
# Format: <dockerhub-namespace>/<image-repo>:<tag>

# Example used for this checkpoint:
IMAGE_NAME=jacksoncdawson/trndly-fastapi:checkpoint-v1

# Build
docker build -t $IMAGE_NAME .

# Push
docker push $IMAGE_NAME

# Run
docker run --rm -p 8000:8000 --env-file backend/services/.env $IMAGE_NAME
```

> For this submission, our team published the image to Docker Hub namespace jacksoncdawson, so the registry screenshot shows jacksoncdawson/trndly-fastapi:checkpoint-v1.

---

## Example `/predict` request (input + expected output format)

Input payload format:

```json
{
  "item_name": "string",
  "color": "string",
  "category": "string",
  "material": "string"
}
```

Example request:

```bash
curl http://127.0.0.1:8000/
curl http://127.0.0.1:8000/health
curl -X POST http://127.0.0.1:8000/predict \
  -H "Content-Type: application/json" \
  -d '{
    "item_name": "Denim Skirt #demo",
    "color": "blue",
    "category": "skirt",
    "material": "denim"
  }'
```

Expected response format:

```json
{
  "item_name": "string",
  "best_timeframe": "current | next_week | next_month | three_months | six_months",
  "timeframe_scores": {
    "current": 0.0,
    "next_week": 0.0,
    "next_month": 0.0,
    "three_months": 0.0,
    "six_months": 0.0
  },
  "model_loaded": true,
  "model_uri": "models:/<registered_model>@<alias>",
  "run_id": "string"
}
```
