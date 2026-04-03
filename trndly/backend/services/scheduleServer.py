import logging
import math
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Optional

import mlflow
import pandas as pd
from dotenv import load_dotenv
from fastapi import FastAPI, HTTPException, status
from pydantic import BaseModel

app = FastAPI(title="MLflow-backed Prediction Service")
logger = logging.getLogger(__name__)

SERVICE_DIR = Path(__file__).resolve().parent
ENV_PATH = SERVICE_DIR / ".env"
load_dotenv(dotenv_path=ENV_PATH)

MLFLOW_TRACKING_URI = os.getenv("MLFLOW_TRACKING_URI", "http://34.169.170.34:5000")
MLFLOW_EXPERIMENT_NAME = os.getenv("MLFLOW_EXPERIMENT_NAME", "mlops-team-project")
MLFLOW_MODEL_ARTIFACT_PATH = os.getenv(
    "MLFLOW_MODEL_ARTIFACT_PATH",
    "~/mlflow-data/artifacts",
)
MLFLOW_PRIMARY_METRIC = os.getenv("MLFLOW_PRIMARY_METRIC", "rmse")
MLFLOW_METRIC_DIRECTION = os.getenv("MLFLOW_METRIC_DIRECTION", "min").lower()
if MLFLOW_METRIC_DIRECTION not in {"min", "max"}:
    logger.warning(
        "Invalid MLFLOW_METRIC_DIRECTION '%s'; defaulting to 'min'.",
        MLFLOW_METRIC_DIRECTION,
    )
    MLFLOW_METRIC_DIRECTION = "min"


@dataclass
class ModelState:
    model: Optional[Any] = None
    run_id: Optional[str] = None
    metric_value: Optional[float] = None
    error: Optional[str] = None

    @property
    def loaded(self) -> bool:
        return self.model is not None


MODEL_STATE = ModelState()


class RootResponse(BaseModel):
    message: str


class HealthResponse(BaseModel):
    status: str
    model_loaded: bool
    tracking_uri: str
    experiment_name: str
    metric_name: str
    run_id: Optional[str]
    metric_value: Optional[float]
    error: Optional[str]


class PredictRequest(BaseModel):
    features: list[float]


class PredictResponse(BaseModel):
    prediction: float
    model_loaded: bool
    run_id: Optional[str]


class ReloadModelResponse(BaseModel):
    loaded: bool
    run_id: Optional[str]
    metric_name: str
    metric_value: Optional[float]
    error: Optional[str]


def _order_by_clause() -> str:
    direction = "DESC" if MLFLOW_METRIC_DIRECTION == "max" else "ASC"
    return f"metrics.{MLFLOW_PRIMARY_METRIC} {direction}"


def _load_best_model_from_mlflow() -> ModelState:
    try:
        mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)
        runs = mlflow.search_runs(
            experiment_names=[MLFLOW_EXPERIMENT_NAME],
            filter_string="attributes.status = 'FINISHED'",
            order_by=[_order_by_clause()],
            max_results=50,
        )

        metric_column = f"metrics.{MLFLOW_PRIMARY_METRIC}"
        if runs.empty:
            return ModelState(
                error=(
                    f"No finished runs found in experiment "
                    f"'{MLFLOW_EXPERIMENT_NAME}'."
                )
            )

        if metric_column not in runs.columns:
            return ModelState(
                error=(
                    f"Metric '{MLFLOW_PRIMARY_METRIC}' was not logged in "
                    f"experiment '{MLFLOW_EXPERIMENT_NAME}'."
                )
            )

        ranked_runs = runs.dropna(subset=[metric_column])
        if ranked_runs.empty:
            return ModelState(
                error=(
                    f"No runs in experiment '{MLFLOW_EXPERIMENT_NAME}' include "
                    f"metric '{MLFLOW_PRIMARY_METRIC}'."
                )
            )

        best_run = ranked_runs.iloc[0]
        run_id = str(best_run["run_id"])
        metric_value = float(best_run[metric_column])
        model_uri = f"runs:/{run_id}/{MLFLOW_MODEL_ARTIFACT_PATH}"
        model = mlflow.pyfunc.load_model(model_uri=model_uri)

        logger.info(
            "Loaded model from run_id=%s with %s=%s",
            run_id,
            MLFLOW_PRIMARY_METRIC,
            metric_value,
        )
        return ModelState(model=model, run_id=run_id, metric_value=metric_value)
    except Exception as exc:  # noqa: BLE001
        logger.exception("Failed to load model from MLflow")
        return ModelState(error=str(exc))


def reload_model() -> ModelState:
    global MODEL_STATE
    MODEL_STATE = _load_best_model_from_mlflow()
    return MODEL_STATE


def _predict(features: list[float]) -> float:
    feature_names = [f"f{index}" for index in range(len(features))]
    inference_frame = pd.DataFrame([features], columns=feature_names)
    predictions = MODEL_STATE.model.predict(inference_frame)
    return float(predictions[0])


@app.on_event("startup")
def load_model_on_startup() -> None:
    reload_model()


@app.get("/", response_model=RootResponse)
def root() -> RootResponse:
    return RootResponse(message="Welcome to the MLflow-backed prediction server.")


@app.get("/health", response_model=HealthResponse)
def health() -> HealthResponse:
    current_status = "healthy" if MODEL_STATE.loaded else "degraded"
    return HealthResponse(
        status=current_status,
        model_loaded=MODEL_STATE.loaded,
        tracking_uri=MLFLOW_TRACKING_URI,
        experiment_name=MLFLOW_EXPERIMENT_NAME,
        metric_name=MLFLOW_PRIMARY_METRIC,
        run_id=MODEL_STATE.run_id,
        metric_value=MODEL_STATE.metric_value,
        error=MODEL_STATE.error,
    )


@app.post("/reload-model", response_model=ReloadModelResponse)
def reload_model_endpoint() -> ReloadModelResponse:
    state = reload_model()
    return ReloadModelResponse(
        loaded=state.loaded,
        run_id=state.run_id,
        metric_name=MLFLOW_PRIMARY_METRIC,
        metric_value=state.metric_value,
        error=state.error,
    )


@app.post("/predict", response_model=PredictResponse)
def predict(payload: PredictRequest) -> PredictResponse:
    if not MODEL_STATE.loaded:
        detail = MODEL_STATE.error or "Model is not loaded."
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail=detail,
        )

    if len(payload.features) == 0:
        raise HTTPException(
            status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
            detail="`features` must include at least one numeric value.",
        )

    if any(not math.isfinite(value) for value in payload.features):
        raise HTTPException(
            status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
            detail="All feature values must be finite numbers.",
        )

    prediction = _predict(payload.features)
    return PredictResponse(
        prediction=prediction,
        model_loaded=MODEL_STATE.loaded,
        run_id=MODEL_STATE.run_id,
    )
