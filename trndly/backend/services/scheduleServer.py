from __future__ import annotations

import logging
import os
import sys
from contextlib import asynccontextmanager
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Optional

import mlflow
from dotenv import load_dotenv
from fastapi import FastAPI, HTTPException, status
from mlflow.tracking import MlflowClient
from pydantic import BaseModel, Field, field_validator

# Add project root to sys.path
SERVICE_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = SERVICE_DIR.parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.append(str(PROJECT_ROOT))


# Local imports
from pipelines.training.feature_contract import (
    TIMEFRAMES,
    TrendLookup,
    build_feature_frame,
    compute_alignment_scores,
    load_trend_lookup,
    normalize_token,
)

# ENV VARIABLES
ENV_PATH = SERVICE_DIR / ".env"
load_dotenv(dotenv_path=ENV_PATH)

MLFLOW_TRACKING_URI = os.getenv("MLFLOW_TRACKING_URI")
MLFLOW_MODEL_URI = os.getenv("MLFLOW_MODEL_URI")
DEFAULT_TREND_SIGNALS_PATH = (
    PROJECT_ROOT / "pipelines" / "training" / "synthetic_data" / "trend_signals.csv"
)
_configured_trend_signals_path = Path(
    os.getenv("TREND_SIGNALS_PATH", str(DEFAULT_TREND_SIGNALS_PATH))
).expanduser()
TREND_SIGNALS_PATH = (
    _configured_trend_signals_path
    if _configured_trend_signals_path.is_absolute()
    else (SERVICE_DIR / _configured_trend_signals_path).resolve()
)


# Initialize logger
logger = logging.getLogger(__name__)


# DATA CLASSES

@dataclass
class ModelState:
    model: Optional[Any] = None
    model_uri: Optional[str] = None
    model_version: Optional[str] = None
    run_id: Optional[str] = None
    error: Optional[str] = None

    @property
    def loaded(self) -> bool:
        return self.model is not None


@dataclass
class TrendState:
    lookup: Optional[TrendLookup] = None
    source_path: Optional[str] = None
    error: Optional[str] = None

    @property
    def loaded(self) -> bool:
        return self.lookup is not None


MODEL_STATE = ModelState()
TREND_STATE = TrendState(source_path=str(TREND_SIGNALS_PATH))


# --- FASTAPI APP ---

@asynccontextmanager
async def lifespan(_: FastAPI):
    reload_trend_data()
    reload_model()
    yield


# Initialize FastAPI app
app = FastAPI(
    title="MLflow-backed Timeframe Recommendation Service",
    lifespan=lifespan,
)


class RootResponse(BaseModel):
    message: str


class HealthResponse(BaseModel):
    status: str
    model_loaded: bool
    trend_data_loaded: bool
    tracking_uri: str
    configured_model_uri: str
    configured_trend_data_path: str
    active_model_uri: Optional[str]
    model_version: Optional[str]
    run_id: Optional[str]
    error: Optional[str]
    trend_error: Optional[str]


class PredictRequest(BaseModel):
    item_name: str = Field(min_length=1, max_length=120)
    color: str = Field(min_length=1, max_length=40)
    category: str = Field(min_length=1, max_length=40)
    material: str = Field(min_length=1, max_length=40)

    @field_validator("item_name", "color", "category", "material")
    @classmethod
    def strip_non_empty(cls, value: str) -> str:
        trimmed = value.strip()
        if not trimmed:
            raise ValueError("Field must not be empty.")
        return trimmed


class PredictResponse(BaseModel):
    item_name: str
    best_timeframe: str
    timeframe_scores: dict[str, float]
    model_loaded: bool
    model_uri: Optional[str]
    run_id: Optional[str]


class ReloadModelResponse(BaseModel):
    loaded: bool
    trend_data_loaded: bool
    configured_model_uri: str
    configured_trend_data_path: str
    active_model_uri: Optional[str]
    model_version: Optional[str]
    run_id: Optional[str]
    error: Optional[str]
    trend_error: Optional[str]


def _parse_registry_alias_uri(model_uri: str) -> tuple[Optional[str], Optional[str]]:
    if not model_uri.startswith("models:/"):
        return None, None

    locator = model_uri.removeprefix("models:/")
    if "@" not in locator:
        return None, None

    model_name, alias = locator.split("@", maxsplit=1)
    return model_name, alias


def _resolve_registry_metadata(model_uri: str) -> tuple[Optional[str], Optional[str]]:
    model_name, alias = _parse_registry_alias_uri(model_uri)
    if not model_name or not alias:
        return None, None

    try:
        client = MlflowClient(tracking_uri=MLFLOW_TRACKING_URI)
        model_version = client.get_model_version_by_alias(name=model_name, alias=alias)
        return model_version.version, model_version.run_id
    except Exception: 
        # Metadata lookup should not block serving if model loading succeeds.
        logger.exception(
            "Loaded model, but failed resolving registry alias metadata for '%s'.",
            model_uri,
        )
        return None, None


def _load_model_from_mlflow(model_uri: str) -> ModelState:
    try:
        mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)
        model = mlflow.pyfunc.load_model(model_uri=model_uri)
        resolved_version, resolved_run_id = _resolve_registry_metadata(model_uri)
        metadata_run_id = getattr(getattr(model, "metadata", None), "run_id", None)
        run_id = resolved_run_id or metadata_run_id

        logger.info(
            "Loaded model from MLflow using model_uri=%s, run_id=%s, version=%s",
            model_uri,
            run_id,
            resolved_version,
        )
        return ModelState(
            model=model,
            model_uri=model_uri,
            model_version=resolved_version,
            run_id=run_id,
        )
    except Exception as exc:
        logger.exception("Failed to load model from MLflow using model_uri=%s", model_uri)
        return ModelState(error=str(exc))


def reload_model() -> ModelState:
    global MODEL_STATE

    primary_state = _load_model_from_mlflow(MLFLOW_MODEL_URI)
    MODEL_STATE = primary_state
    return MODEL_STATE


def reload_trend_data() -> TrendState:
    global TREND_STATE

    try:
        lookup = load_trend_lookup(TREND_SIGNALS_PATH)
        TREND_STATE = TrendState(
            lookup=lookup,
            source_path=str(TREND_SIGNALS_PATH),
        )
        logger.info("Loaded trend signals from %s", TREND_SIGNALS_PATH)
        return TREND_STATE
    except Exception as exc:
        logger.exception("Failed to load trend signals from %s", TREND_SIGNALS_PATH)
        TREND_STATE = TrendState(
            source_path=str(TREND_SIGNALS_PATH),
            error=str(exc),
        )
        return TREND_STATE


def _predict_timeframe(payload: PredictRequest) -> tuple[str, dict[str, float]]:
    if TREND_STATE.lookup is None:
        raise RuntimeError("Trend signals are not loaded.")

    item = {
        "item_name": payload.item_name.strip(),
        "color": normalize_token(payload.color),
        "category": normalize_token(payload.category),
        "material": normalize_token(payload.material),
    }

    inference_frame = build_feature_frame([item], TREND_STATE.lookup)
    predictions = MODEL_STATE.model.predict(inference_frame)
    model_prediction = str(predictions[0])

    alignment_scores = compute_alignment_scores(item=item, lookup=TREND_STATE.lookup)
    if model_prediction not in TIMEFRAMES:
        logger.warning(
            "Model returned unexpected timeframe '%s'; falling back to strongest "
            "alignment score.",
            model_prediction,
        )
        best_timeframe = max(TIMEFRAMES, key=lambda timeframe: alignment_scores[timeframe])
    else:
        best_timeframe = model_prediction

    rounded_scores = {
        timeframe: round(float(alignment_scores[timeframe]), 6) for timeframe in TIMEFRAMES
    }
    return best_timeframe, rounded_scores

@app.get("/", response_model=RootResponse)
def root() -> RootResponse:
    return RootResponse(message="Welcome to the MLflow-backed timeframe recommendation server.")


@app.get("/health", response_model=HealthResponse)
def health() -> HealthResponse:
    current_status = "healthy" if MODEL_STATE.loaded and TREND_STATE.loaded else "degraded"
    return HealthResponse(
        status=current_status,
        model_loaded=MODEL_STATE.loaded,
        trend_data_loaded=TREND_STATE.loaded,
        tracking_uri=MLFLOW_TRACKING_URI,
        configured_model_uri=MLFLOW_MODEL_URI,
        configured_trend_data_path=str(TREND_SIGNALS_PATH),
        active_model_uri=MODEL_STATE.model_uri,
        model_version=MODEL_STATE.model_version,
        run_id=MODEL_STATE.run_id,
        error=MODEL_STATE.error,
        trend_error=TREND_STATE.error,
    )


@app.post("/predict", response_model=PredictResponse)
def predict(payload: PredictRequest) -> PredictResponse:
    if not MODEL_STATE.loaded:
        detail = MODEL_STATE.error or "Model is not loaded."
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail=detail,
        )

    if not TREND_STATE.loaded:
        detail = TREND_STATE.error or "Trend signal data is not loaded."
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail=detail,
        )

    best_timeframe, timeframe_scores = _predict_timeframe(payload)
    return PredictResponse(
        item_name=payload.item_name,
        best_timeframe=best_timeframe,
        timeframe_scores=timeframe_scores,
        model_loaded=MODEL_STATE.loaded,
        model_uri=MODEL_STATE.model_uri,
        run_id=MODEL_STATE.run_id,
    )
