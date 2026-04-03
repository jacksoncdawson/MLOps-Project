# trndly

`trndly` is the main project folder for the MLOps app and supporting design docs.

## Project structure

- `backend/services/scheduleServer.py` - FastAPI prediction service
- `backend/services/.env` - local environment variables for the service
- `docs/architecture.md` - system architecture notes
- `docs/rationale.md` - key technical decision rationale
- `docker-compose.yml` - local container orchestration config
- `requirements.txt` - Python dependencies

## Backend quick start

1) Install dependencies from `trndly/`:

```bash
python -m pip install -r requirements.txt
```

2) Set service configuration in `backend/services/.env`:

```env
MLFLOW_TRACKING_URI=http://34.169.170.34:5000
MLFLOW_EXPERIMENT_NAME=mlops-team-project
MLFLOW_MODEL_ARTIFACT_PATH=~/mlflow-data/artifacts
MLFLOW_PRIMARY_METRIC=rmse
MLFLOW_METRIC_DIRECTION=min
```

3) Start the API from `trndly/backend/services/`:

```bash
uvicorn scheduleServer:app --reload --port 8000
```

4) Test endpoints:

```bash
curl http://127.0.0.1:8000/health
curl -X POST http://127.0.0.1:8000/predict \
  -H "Content-Type: application/json" \
  -d '{"features":[1.0,2.0,3.0]}'
curl -X POST http://127.0.0.1:8000/reload-model
```
