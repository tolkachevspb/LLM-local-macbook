#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$ROOT_DIR"

stopped=0

for pid_file in run/proxy.pid run/backend.pid; do
  if [[ ! -f "${pid_file}" ]]; then
    continue
  fi
  pid="$(cat "${pid_file}")"
  if [[ -z "${pid}" ]]; then
    rm -f "${pid_file}"
    continue
  fi
  if ! kill -0 "${pid}" 2>/dev/null; then
    rm -f "${pid_file}"
    continue
  fi

  # Graceful SIGTERM, then wait up to 10s, then SIGKILL
  kill "${pid}" 2>/dev/null || true
  for _ in $(seq 1 20); do
    if ! kill -0 "${pid}" 2>/dev/null; then
      break
    fi
    sleep 0.5
  done
  if kill -0 "${pid}" 2>/dev/null; then
    kill -9 "${pid}" 2>/dev/null || true
  fi
  rm -f "${pid_file}"
  stopped=1
done

if [[ "${stopped}" == "1" ]]; then
  echo "Сервер остановлен."
else
  echo "Сервер не был запущен."
fi
