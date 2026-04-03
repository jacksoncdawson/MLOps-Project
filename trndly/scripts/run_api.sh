#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT_DIR="$(cd "${SCRIPT_DIR}/.." && pwd)"

PYTHON_BIN="${PYTHON_BIN:-${ROOT_DIR}/.venv/bin/python}"
ENV_FILE="${ENV_FILE:-${ROOT_DIR}/backend/services/.env}"
HOST="${API_HOST:-127.0.0.1}"
PORT="${API_PORT:-8000}"
AUTO_RELOAD="${AUTO_RELOAD:-0}"

if [[ ! -x "${PYTHON_BIN}" ]]; then
  echo "Python interpreter not found at: ${PYTHON_BIN}"
  echo "Create/install venv first, e.g.:"
  echo "  cd \"${ROOT_DIR}\" && python3 -m venv .venv && .venv/bin/python -m pip install -r requirements.txt"
  exit 1
fi

if [[ -f "${ENV_FILE}" ]]; then
  set -a
  source "${ENV_FILE}"
  set +a
else
  echo "Warning: ENV_FILE not found (${ENV_FILE}). Continuing with current shell env."
fi

cd "${ROOT_DIR}/backend/services"

echo "Starting API at http://${HOST}:${PORT}"
if [[ "${AUTO_RELOAD}" == "1" ]]; then
  exec "${PYTHON_BIN}" -m uvicorn scheduleServer:app --reload --reload-dir . --host "${HOST}" --port "${PORT}"
fi

exec "${PYTHON_BIN}" -m uvicorn scheduleServer:app --host "${HOST}" --port "${PORT}"
