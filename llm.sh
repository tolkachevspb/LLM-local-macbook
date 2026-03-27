#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$ROOT_DIR"

if [[ ! -f "config.env" ]]; then
  if [[ -f "config.env.example" ]]; then
    cp config.env.example config.env
  fi
fi

if [[ -f "config.env" ]]; then
  source ./config.env
fi

HOST="${HOST:-127.0.0.1}"
PORT="${PORT:-8091}"

cmd="${1:-help}"

case "${cmd}" in
  start)
    ./start.sh
    ;;
  stop)
    ./stop.sh
    ;;
  restart)
    ./stop.sh || true
    sleep 1
    ./start.sh
    ;;
  health)
    ./healthcheck.sh
    ;;
  status)
    running=0
    if [[ -f "run/backend.pid" ]]; then
      pid="$(cat run/backend.pid)"
      if [[ -n "${pid}" ]] && kill -0 "${pid}" 2>/dev/null; then
        echo "Backend: PID ${pid}"
        running=1
      fi
    fi
    if [[ -f "run/proxy.pid" ]]; then
      pid="$(cat run/proxy.pid)"
      if [[ -n "${pid}" ]] && kill -0 "${pid}" 2>/dev/null; then
        echo "Proxy:   PID ${pid}"
        running=1
      fi
    fi
    if [[ "${running}" == "1" ]]; then
      echo "UI:  http://${HOST}:${PORT}"
      echo "API: http://${HOST}:${PORT}/v1"
    else
      echo "Сервер не запущен."
      exit 1
    fi
    ;;
  test)
    shift || true
    PROMPT="${*:-}" MODEL_ALIAS="${MODEL_ALIAS:-gigachat3-lightning}" ./prompt-test.sh "${*:-}"
    ;;
  open)
    open "http://${HOST}:${PORT}"
    ;;
  ui)
    echo "http://${HOST}:${PORT}"
    ;;
  api)
    echo "http://${HOST}:${PORT}/v1"
    ;;
  logs)
    echo "=== backend.log ==="
    tail -n 30 logs/backend.log 2>/dev/null || echo "(пусто)"
    echo
    echo "=== proxy.log ==="
    tail -n 30 logs/proxy.log 2>/dev/null || echo "(пусто)"
    ;;
  benchmark)
    python3 ./benchmark.py
    ;;
  bootstrap)
    shift || true
    ./bootstrap.sh "${@:-bootstrap}"
    ;;
  help|--help|-h|"")
    cat <<'USAGE'
GigaChat Localhost — управление локальным LLM стеком

Использование: ./llm.sh <команда>

Команды:
  bootstrap [q6|q8]  Скачать llamafile + модель
  start               Запустить backend + proxy
  stop                Остановить все
  restart             Перезапустить
  status              Показать статус процессов
  health              Проверить /healthz и /v1/models
  test "prompt"       Тестовый запрос к API
  open                Открыть UI в браузере
  ui                  Показать URL интерфейса
  api                 Показать URL API
  logs                Последние строки логов
  benchmark           Запустить benchmark
  help                Эта справка
USAGE
    ;;
  *)
    echo "Неизвестная команда: ${cmd}" >&2
    echo "Используйте: ./llm.sh help" >&2
    exit 1
    ;;
esac
