#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT_DIR="$(cd "${SCRIPT_DIR}/.." && pwd)"

PYTHON_BIN="${PYTHON_BIN:-${ROOT_DIR}/.venv/bin/python}"
ENV_FILE="${ENV_FILE:-${ROOT_DIR}/backend/services/.env}"

if [[ ! -x "${PYTHON_BIN}" ]]; then
  echo "Python interpreter not found at: ${PYTHON_BIN}"
  echo "Create/install venv first, e.g.:"
  echo "  cd \"${ROOT_DIR}\" && python3 -m venv .venv && .venv/bin/python -m pip install -r requirements.txt"
  exit 1
fi

if [[ -f "${ENV_FILE}" ]]; then
  # Export all sourced variables so training script can read them.
  set -a
  # shellcheck disable=SC1090
  source "${ENV_FILE}"
  set +a
else
  echo "Warning: ENV_FILE not found (${ENV_FILE}). Continuing with current shell env."
fi

GENERATE_SYNTHETIC="${GENERATE_SYNTHETIC:-1}"
if [[ "${GENERATE_SYNTHETIC}" == "1" ]]; then
  echo "Generating synthetic data..."
  "${PYTHON_BIN}" "${ROOT_DIR}/pipelines/training/generate_synthetic_listing_data.py"
fi

TIMESTAMP="$(date +%Y%m%d-%H%M%S)"
TRACKING_URI="${MLFLOW_TRACKING_URI:-http://127.0.0.1:5000}"
EXPERIMENT_NAME="${MLFLOW_EXPERIMENT_NAME:-mlops-team-project}"
REGISTERED_MODEL_NAME="${MLFLOW_REGISTERED_MODEL_NAME:-listing_timeline_experiments}"
MODEL_ALIAS="${MLFLOW_MODEL_ALIAS:-candidate}"
MODEL_ARTIFACT_PATH="${MLFLOW_MODEL_ARTIFACT_PATH:-model}"
TRAIN_DATA_URI="${TRAIN_DATA_URI:-${ROOT_DIR}/pipelines/training/data/train.csv}"
VAL_DATA_URI="${VAL_DATA_URI:-${ROOT_DIR}/pipelines/training/data/val.csv}"
TEST_DATA_URI="${TEST_DATA_URI:-${ROOT_DIR}/pipelines/training/data/test.csv}"
TARGET_COLUMN="${TARGET_COLUMN:-best_timeframe}"
DATA_VERSION="${DATA_VERSION:-synthetic-${TIMESTAMP}}"
N_ESTIMATORS="${N_ESTIMATORS:-100}"
RANDOM_STATE="${RANDOM_STATE:-42}"

echo "Starting MLflow experiment..."
echo "  tracking_uri          = ${TRACKING_URI}"
echo "  experiment_name       = ${EXPERIMENT_NAME}"
echo "  registered_model_name = ${REGISTERED_MODEL_NAME}"
echo "  model_alias           = ${MODEL_ALIAS}"
echo "  data_version          = ${DATA_VERSION}"

"${PYTHON_BIN}" "${ROOT_DIR}/pipelines/training/train_listing_timeline.py" \
  --train-data-uri "${TRAIN_DATA_URI}" \
  --val-data-uri "${VAL_DATA_URI}" \
  --test-data-uri "${TEST_DATA_URI}" \
  --target-column "${TARGET_COLUMN}" \
  --n-estimators "${N_ESTIMATORS}" \
  --random-state "${RANDOM_STATE}" \
  --tracking-uri "${TRACKING_URI}" \
  --experiment-name "${EXPERIMENT_NAME}" \
  --registered-model-name "${REGISTERED_MODEL_NAME}" \
  --model-alias "${MODEL_ALIAS}" \
  --model-artifact-path "${MODEL_ARTIFACT_PATH}" \
  --data-version "${DATA_VERSION}" \
  "$@"

echo "Done."
