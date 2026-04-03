#!/usr/bin/env bash
set -euo pipefail

PORT="${API_PORT:-8000}"

pids=""

add_pid() {
  local candidate="${1:-}"

  [[ -n "${candidate}" ]] || return 0

  case " ${pids} " in
    *" ${candidate} "*) return 0 ;;
    *) pids="${pids} ${candidate}" ;;
  esac
}

if command -v lsof >/dev/null 2>&1; then
  while IFS= read -r pid; do
    add_pid "${pid}"
  done < <(lsof -tiTCP:"${PORT}" -sTCP:LISTEN 2>/dev/null || true)
fi

while IFS= read -r pid; do
  add_pid "${pid}"
done < <(pgrep -f "uvicorn scheduleServer:app|fastapi dev .*scheduleServer.py" 2>/dev/null || true)

if [[ -z "${pids// }" ]]; then
  echo "No API processes found."
  exit 0
fi

for pid in ${pids}; do
  echo "Stopping PID ${pid}"
  kill "${pid}" 2>/dev/null || true
done

sleep 1

for pid in ${pids}; do
  if kill -0 "${pid}" 2>/dev/null; then
    echo "Force killing PID ${pid}"
    kill -9 "${pid}" 2>/dev/null || true
  fi
done

if command -v lsof >/dev/null 2>&1 && lsof -iTCP:"${PORT}" -sTCP:LISTEN -n -P >/dev/null 2>&1; then
  echo "Warning: something is still listening on ${PORT}"
  lsof -iTCP:"${PORT}" -sTCP:LISTEN -n -P
  exit 1
fi

echo "API processes stopped. Port ${PORT} is free."
