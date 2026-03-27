# GigaChat Local

Локальный стек для запуска моделей семейства **GigaChat** и других GGUF-совместимых LLM на macOS Apple Silicon — без Docker, без Python-окружений, без облака.

Включает:
- Веб-интерфейс с историей сессий, Markdown-рендерингом и подсветкой кода
- OpenAI-совместимый REST API (`/v1/chat/completions`)
- Переключение моделей прямо из UI без перезапуска прокси
- Поддержку нескольких GGUF-моделей с возможностью докачки

---

## Быстрый старт

```bash
git clone https://github.com/<user>/gigachat-local
cd gigachat-local
./llm.sh bootstrap       # скачать llamafile + GigaChat3-10B Q6
./llm.sh start           # запустить backend + proxy
./llm.sh open            # открыть UI в браузере
```

Через ~30 секунд после `start` откроется `http://127.0.0.1:8091`.

---

## Архитектура

```
┌─────────────────────────────────────────────────────────┐
│                    Браузер / клиент                      │
└───────────────────────┬─────────────────────────────────┘
                        │ HTTP  :8091
          ┌─────────────▼──────────────┐
          │          proxy.py          │
          │  • Web UI (история, MD)    │
          │  • /v1/chat/completions    │
          │  • Переключение моделей    │
          │  • GigaChat prompt render  │
          └─────────────┬──────────────┘
                        │ HTTP  :8092
          ┌─────────────▼──────────────┐
          │    llamafile (backend)     │
          │  • /v1/completions         │
          │  • Apple Metal GPU         │
          └────────────────────────────┘
```

Прокси необходим для GigaChat-моделей: llamafile не форматирует промпты в нативном формате GigaChat (`<s>developer system<|role_sep|>...`) и пропускает служебные токены в ответе. Прокси рендерит промпт вручную и очищает ответ.

---

## Возможности UI

| Функция | Описание |
|---|---|
| История сессий | Сохраняется в localStorage, переключается в сайдбаре |
| Переименование чатов | Кнопка ✏️ рядом с сессией |
| Удаление с подтверждением | Двухшаговое удаление, защита от случайного нажатия |
| Markdown | Заголовки, таблицы, списки, цитаты, ссылки |
| Подсветка кода | Встроенный синтаксис для Python, JS, Go, Rust и других |
| Копирование кода | Кнопка "Копировать" в каждом блоке кода |
| Переключение модели | Клик на имя модели в топбаре → выбор из доступных GGUF |
| Температура | Слайдер 0–1 в сайдбаре, сохраняется между сессиями |
| Системный промпт | Редактируемый, скрываемый блок в сайдбаре |
| Стриминг | Ответ отображается токен за токеном |

---

## Команды

```bash
./llm.sh bootstrap [q6|q8]   # скачать llamafile + модель
./llm.sh start                # запустить stack
./llm.sh stop                 # остановить
./llm.sh restart              # перезапустить
./llm.sh status               # статус процессов и URL
./llm.sh health               # проверка /healthz
./llm.sh test "промпт"        # тестовый запрос
./llm.sh open                 # открыть UI в браузере
./llm.sh logs                 # последние строки логов
./llm.sh benchmark            # замер производительности
```

### Скачать дополнительные модели

```bash
./download-model.sh list        # список доступных профилей

./download-model.sh gigachat-q6  # GigaChat3-10B Q6_K   ~8.8 GB (по умолчанию)
./download-model.sh gigachat-q8  # GigaChat3-10B Q8_0   ~11 GB  (качество++)
./download-model.sh qwen-7b      # Qwen2.5-7B Q4_K_M    ~4.7 GB (быстрый)
./download-model.sh qwen-14b     # Qwen2.5-14B Q4_K_M   ~9 GB   (лучшее качество)
./download-model.sh phi3.5-mini  # Phi-3.5-mini Q4_K_M  ~2.2 GB (код и логика)
./download-model.sh gemma3-12b   # Gemma-3-12B Q4_K_M   ~8 GB   (Google)
```

После скачивания модель автоматически появится в панели выбора в UI.

---

## API

Прокси предоставляет OpenAI-совместимый эндпоинт:

```bash
curl http://127.0.0.1:8091/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{
    "model": "gigachat3-lightning",
    "messages": [{"role": "user", "content": "Привет!"}],
    "temperature": 0.2,
    "max_tokens": 512,
    "stream": false
  }'
```

Совместим с любым клиентом OpenAI API: `openai` Python SDK, LangChain, LlamaIndex, Continue и т.д.

```python
from openai import OpenAI

client = OpenAI(base_url="http://127.0.0.1:8091/v1", api_key="none")
resp = client.chat.completions.create(
    model="gigachat3-lightning",
    messages=[{"role": "user", "content": "Привет!"}],
)
print(resp.choices[0].message.content)
```

Или встроенный Python-клиент (без зависимостей):

```bash
python3 client.py "Напиши функцию на Python для сортировки пузырьком"
python3 client.py --temp 0.7 --max-tokens 1024 "Придумай историю"
```

---

## Конфигурация

Настройки хранятся в `config.env` (создаётся из `config.env.example` при первом запуске).

| Параметр | По умолчанию | Описание |
|---|---|---|
| `HOST` | `127.0.0.1` | Адрес прокси |
| `PORT` | `8091` | Порт веб-интерфейса и API |
| `BACKEND_PORT` | `8092` | Порт llamafile backend |
| `MODEL_FILE` | `./models/GigaChat3-10B-A1.8B-q6_k.gguf` | Путь к модели |
| `MODEL_ALIAS` | `gigachat3-lightning` | Имя модели в API |
| `CTX_SIZE` | `4096` | Размер контекста в токенах |
| `THREADS` | `6` | CPU потоков |
| `GPU_LAYERS` | `all` | Слоёв на GPU (Metal) |
| `FLASH_ATTN` | `on` | Flash Attention |
| `CHAT_TEMPLATE` | `gigachat` | Шаблон чата (`gigachat` / пусто) |

---

## Рекомендации по моделям (Apple M5 24 GB)

| Модель | Размер | Скорость | Подходит для |
|---|---|---|---|
| GigaChat3-10B Q6_K | 8.8 GB | ~60 tok/s | Русский язык, инструкции |
| Qwen2.5-7B Q4_K_M | 4.7 GB | ~90 tok/s | Универсальный, быстрый |
| Qwen2.5-14B Q4_K_M | 9 GB | ~45 tok/s | Лучшее качество |
| Phi-3.5-mini Q4_K_M | 2.2 GB | ~140 tok/s | Код, логика, скорость |
| Gemma-3-12B Q4_K_M | 8 GB | ~55 tok/s | Разносторонние задачи |

Модели до ~18 GB работают полностью на Apple Metal GPU без своппинга.

---

## Системные требования

| | Минимум | Рекомендуется |
|---|---|---|
| Платформа | macOS Apple Silicon | macOS Apple Silicon |
| RAM | 16 GB | 24+ GB |
| Диск | 10 GB | 20+ GB |
| Инструменты | `curl`, `python3` | `curl`, `python3` |

> Linux (x86_64) поддерживается llamafile, но скрипты тестировались на macOS.

---

## Структура проекта

```
gigachat-local/
├── proxy.py              # Прокси: Web UI + API + переключение моделей
├── client.py             # CLI-клиент (stdlib only, без зависимостей)
├── benchmark.py          # Замер производительности
├── llm.sh                # Точка входа: start/stop/status/...
├── start.sh              # Запуск backend + proxy
├── stop.sh               # Остановка
├── bootstrap.sh          # Первичная установка
├── download-model.sh     # Скачивание GGUF-моделей
├── healthcheck.sh        # Проверка /healthz и /v1/models
├── prompt-test.sh        # Тестовый запрос через curl
├── config.env.example    # Шаблон конфига (скопируйте в config.env)
└── models/               # GGUF-файлы (не в репозитории, см. download-model.sh)
```

---

## Upstream

- [mozilla/llamafile](https://github.com/Mozilla-Ocho/llamafile) — runtime
- [salute-developers/gigachat3](https://github.com/salute-developers/gigachat3) — архитектура модели
- [ai-sage/GigaChat3-10B-A1.8B-GGUF](https://huggingface.co/ai-sage/GigaChat3-10B-A1.8B-GGUF) — GGUF-веса

---

## Лицензия

MIT © 2026
