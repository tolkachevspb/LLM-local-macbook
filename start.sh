#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$ROOT_DIR"

if [[ ! -f "config.env" ]]; then
  cp config.env.example config.env
fi
source ./config.env

mkdir -p logs run

# --- Validate prerequisites ---
if [[ ! -x "bin/llamafile" ]]; then
  echo "Не найден bin/llamafile. Запустите: ./bootstrap.sh" >&2
  exit 1
fi
if [[ ! -f "${MODEL_FILE}" ]]; then
  echo "Не найдена модель ${MODEL_FILE}. Запустите: ./bootstrap.sh" >&2
  exit 1
fi

# --- Check if already running ---
for pf in run/proxy.pid run/backend.pid; do
  if [[ -f "${pf}" ]]; then
    pid="$(cat "${pf}")"
    if [[ -n "${pid}" ]] && kill -0 "${pid}" 2>/dev/null; then
      echo "Уже запущен (PID ${pid}). Используйте ./llm.sh restart для перезапуска."
      exit 0
    fi
    rm -f "${pf}"
  fi
done

# --- Build backend command ---
jinja_flag="--jinja"
if [[ "${JINJA}" == "0" ]]; then
  jinja_flag="--no-jinja"
fi

cmd=(
  ./bin/llamafile --server
  -m "${MODEL_FILE}"
  --host "${HOST}"
  --port "${BACKEND_PORT:-8092}"
  --alias "${MODEL_ALIAS}"
  -c "${CTX_SIZE}"
  -t "${THREADS}"
  -tb "${THREADS_BATCH}"
  -b "${BATCH_SIZE}"
  -ub "${UBATCH_SIZE}"
  -np "${PARALLEL}"
  -ctk "${CACHE_TYPE_K}"
  -ctv "${CACHE_TYPE_V}"
  -fa "${FLASH_ATTN}"
  -ngl "${GPU_LAYERS}"
  "${jinja_flag}"
)

if [[ -n "${CHAT_TEMPLATE:-}" ]]; then
  cmd+=(--chat-template "${CHAT_TEMPLATE}")
fi
if [[ "${CONT_BATCHING}" == "1" ]]; then
  cmd+=(--cont-batching)
fi
if [[ -n "${EXTRA_ARGS:-}" ]]; then
  # shellcheck disable=SC2206
  cmd+=(${EXTRA_ARGS})
fi

# --- Start backend ---
echo "Запуск backend (llamafile) ..."
nohup "${cmd[@]}" </dev/null > logs/backend.log 2>&1 &
echo $! > run/backend.pid
disown "$(cat run/backend.pid)" 2>/dev/null || true

# --- Wait for backend readiness ---
backend_pid="$(cat run/backend.pid)"
echo -n "Ожидание backend (PID ${backend_pid})"
ready=0
for i in $(seq 1 60); do
  if ! kill -0 "${backend_pid}" 2>/dev/null; then
    echo
    echo "Backend упал. Смотрите: tail -50 logs/backend.log" >&2
    rm -f run/backend.pid
    exit 1
  fi
  if curl -sf "http://${HOST}:${BACKEND_PORT:-8092}/v1/models" >/dev/null 2>&1; then
    ready=1
    break
  fi
  echo -n "."
  sleep 2
done
echo

if [[ "${ready}" != "1" ]]; then
  echo "Backend не ответил за 120 секунд. Смотрите: tail -50 logs/backend.log" >&2
  exit 1
fi
echo "Backend готов."

# --- Start proxy ---
echo "Запуск proxy ..."
nohup python3 ./proxy.py </dev/null > logs/proxy.log 2>&1 &
echo $! > run/proxy.pid
disown "$(cat run/proxy.pid)" 2>/dev/null || true
sleep 1

proxy_pid="$(cat run/proxy.pid)"
if ! kill -0 "${proxy_pid}" 2>/dev/null; then
  echo "Proxy не стартовал. Смотрите: tail -20 logs/proxy.log" >&2
  exit 1
fi

echo
echo "Готово!"
echo "  Backend: PID ${backend_pid} (порт ${BACKEND_PORT:-8092})"
echo "  Proxy:   PID ${proxy_pid}"
echo "  UI:      http://${HOST}:${PORT}"
echo "  API:     http://${HOST}:${PORT}/v1"
