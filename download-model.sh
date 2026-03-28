#!/usr/bin/env bash
set -euo pipefail

PROJECT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
MODELS_DIR="${PROJECT_DIR}/models"
PROFILE="${1:-q6}"
HF_REVISION="a7bdb997a7c39391b8da2115682a89e209563ade"

mkdir -p "${MODELS_DIR}"

download() {
  local url="$1" dest="$2"
  echo "Скачиваю $(basename "${dest}") ..."
  # -C - возобновляет прерванную загрузку; если файл уже полный — curl завершится сразу
  curl -L -C - --progress-bar -o "${dest}" "${url}"
  echo "Готово: $(basename "${dest}")"
}

case "${PROFILE}" in
  q6|bootstrap|recommended|gigachat-q6)
    download \
      "https://huggingface.co/ai-sage/GigaChat3-10B-A1.8B-GGUF/resolve/${HF_REVISION}/GigaChat3-10B-A1.8B-q6_k.gguf?download=true" \
      "${MODELS_DIR}/GigaChat3-10B-A1.8B-q6_k.gguf"
    ;;
  gigachat-q8|q8)
    download \
      "https://huggingface.co/ai-sage/GigaChat3-10B-A1.8B-GGUF/resolve/${HF_REVISION}/GigaChat3-10B-A1.8B-q8_0.gguf?download=true" \
      "${MODELS_DIR}/GigaChat3-10B-A1.8B-q8_0.gguf"
    ;;
  qwen-7b)
    # Qwen2.5-7B-Instruct Q4_K_M — 4.7 GB, ~90 tok/s на M5, отличный баланс
    download \
      "https://huggingface.co/bartowski/Qwen2.5-7B-Instruct-GGUF/resolve/main/Qwen2.5-7B-Instruct-Q4_K_M.gguf" \
      "${MODELS_DIR}/Qwen2.5-7B-Instruct-Q4_K_M.gguf"
    ;;
  qwen-14b)
    # Qwen2.5-14B-Instruct Q4_K_M — 9 GB, ~45 tok/s на M5, лучшее качество в диапазоне
    download \
      "https://huggingface.co/bartowski/Qwen2.5-14B-Instruct-GGUF/resolve/main/Qwen2.5-14B-Instruct-Q4_K_M.gguf" \
      "${MODELS_DIR}/Qwen2.5-14B-Instruct-Q4_K_M.gguf"
    ;;
  phi3.5-mini|phi-mini)
    # Phi-3.5-mini-instruct Q4_K_M — 2.2 GB, ~140 tok/s, код и логика, публичная
    download \
      "https://huggingface.co/bartowski/Phi-3.5-mini-instruct-GGUF/resolve/main/Phi-3.5-mini-instruct-Q4_K_M.gguf" \
      "${MODELS_DIR}/Phi-3.5-mini-instruct-Q4_K_M.gguf"
    ;;
  gemma3-12b)
    # Gemma-3-12B-IT Q4_K_M — 8 GB, ~55 tok/s, Google, хорош на многих задачах
    download \
      "https://huggingface.co/bartowski/gemma-3-12b-it-GGUF/resolve/main/gemma-3-12b-it-Q4_K_M.gguf" \
      "${MODELS_DIR}/gemma-3-12b-it-Q4_K_M.gguf"
    ;;
  glm4-9b)
    # GLM-4-9B-Chat Q4_K_M — 5.5 GB, ~75 tok/s, Zhipu AI, сильный в китайском и коде
    download \
      "https://huggingface.co/bartowski/glm-4-9b-chat-GGUF/resolve/main/glm-4-9b-chat-Q4_K_M.gguf" \
      "${MODELS_DIR}/glm-4-9b-chat-Q4_K_M.gguf"
    ;;
  list)
    echo ""
    echo "Доступные профили (оптимально для Apple M5 24GB):"
    echo ""
    echo "  gigachat-q6   GigaChat3-10B Q6_K      ~8.8 GB  ~60 tok/s  (уже есть)"
    echo "  gigachat-q8   GigaChat3-10B Q8_0       ~11 GB   ~40 tok/s  качество++"
    echo "  qwen-7b       Qwen2.5-7B Q4_K_M        ~4.7 GB  ~90 tok/s  быстрый, универсальный"
    echo "  qwen-14b      Qwen2.5-14B Q4_K_M       ~9 GB    ~45 tok/s  лучшее качество"
    echo "  phi3.5-mini   Phi-3.5-mini Q4_K_M       ~2.2 GB  ~140 tok/s код и логика (публичная)"
    echo "  gemma3-12b    Gemma-3-12B Q4_K_M        ~8 GB    ~55 tok/s  Google, разносторонний"
    echo "  glm4-9b       GLM-4-9B-Chat Q4_K_M      ~5.5 GB  ~75 tok/s  Zhipu AI, китайский + код"
    echo ""
    exit 0
    ;;
  *)
    echo "Использование: $0 {gigachat-q6|gigachat-q8|qwen-7b|qwen-14b|phi3.5-mini|gemma3-12b|glm4-9b|list}"
    echo "Список с описанием: $0 list"
    exit 1
    ;;
esac
