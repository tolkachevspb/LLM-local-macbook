#!/usr/bin/env bash
set -euo pipefail

PROJECT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
BIN_DIR="${PROJECT_DIR}/bin"
LLAMAFILE_BIN="${BIN_DIR}/llamafile"
LLAMAFILE_VERSION="0.10.0"
PROFILE="${1:-bootstrap}"

for cmd in curl python3 uname; do
  if ! command -v "${cmd}" >/dev/null 2>&1; then
    echo "Не найдена команда: ${cmd}" >&2
    exit 1
  fi
done

os_name="$(uname -s)"
arch_name="$(uname -m)"

if [[ "${os_name}" != "Darwin" ]]; then
  echo "Этот проект поддерживает только macOS." >&2
  exit 1
fi
if [[ "${arch_name}" != "arm64" ]]; then
  echo "Требуется Apple Silicon (arm64). Текущая: ${arch_name}" >&2
  exit 1
fi

mkdir -p "${PROJECT_DIR}"/{bin,models,logs,run,downloads}

if [[ ! -f "${PROJECT_DIR}/config.env" ]]; then
  cp "${PROJECT_DIR}/config.env.example" "${PROJECT_DIR}/config.env"
  echo "Создан config.env"
fi

if [[ ! -x "${LLAMAFILE_BIN}" ]]; then
  echo "Скачиваю llamafile v${LLAMAFILE_VERSION} ..."
  curl -fL -C - --progress-bar -o "${LLAMAFILE_BIN}" \
    "https://github.com/mozilla-ai/llamafile/releases/download/${LLAMAFILE_VERSION}/llamafile-${LLAMAFILE_VERSION}"
  chmod +x "${LLAMAFILE_BIN}"
  echo "llamafile установлен"
else
  echo "llamafile уже установлен"
fi

case "${PROFILE}" in
  bootstrap|q6|recommended)
    "${PROJECT_DIR}/download-model.sh" q6
    ;;
  q8)
    "${PROJECT_DIR}/download-model.sh" q8
    ;;
  *)
    echo "Использование: $0 {bootstrap|q6|q8}" >&2
    exit 1
    ;;
esac

echo
echo "Bootstrap завершен. Дальше:"
echo "  ./llm.sh start"
echo "  ./llm.sh open"
