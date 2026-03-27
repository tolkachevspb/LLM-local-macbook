#!/usr/bin/env python3
"""
Model evaluation suite for gigachat-local.

Тестирует все доступные GGUF-модели на качество ответов и скорость.
Запуск: python3 eval/suite.py [--model <alias>] [--skip-switch]

Результаты сохраняются в eval/results/<timestamp>_<model>.json
и в сводный eval/results/report_<timestamp>.md
"""

import argparse
import ast
import json
import os
import re
import sys
import time
import urllib.error
import urllib.request
from datetime import datetime

# ── Config ─────────────────────────────────────────────────────────────────
PROXY_URL = "http://127.0.0.1:8091"
RESULTS_DIR = os.path.join(os.path.dirname(__file__), "results")
DEFAULT_TEMPERATURE = 0.1   # низкая для воспроизводимости
DEFAULT_MAX_TOKENS  = 512
SWITCH_TIMEOUT_S    = 180   # максимум ждём смены модели


# ── Test cases ──────────────────────────────────────────────────────────────
# Каждый кейс:
#   id, category, name, system (опц.), prompt, checks, weight (важность 1-3)
#
# checks — список правил:
#   ("contains_all",  [w1, w2])     все слова есть в ответе (lower)
#   ("contains_any",  [w1, w2])     хоть одно слово есть
#   ("not_contains",  [w1, w2])     ни одного нет (антислова)
#   ("length_min",    N)            ответ длиннее N символов
#   ("length_max",    N)            ответ короче N символов
#   ("exact_number",  N)            число N встречается в ответе
#   ("valid_json",    None)         ответ парсится как JSON
#   ("valid_python",  None)         в ответе есть валидный Python-блок
#   ("count_lines",   (min, max))   кол-во строк с текстом в диапазоне
#   ("has_cyrillic",  None)         есть кириллица
#   ("no_cyrillic",   None)         нет кириллицы
#   ("regex",         pattern)      паттерн найден в ответе
#   ("not_refusal",   None)         нет фраз отказа

TEST_CASES = [

    # ── 1. Русский язык ─────────────────────────────────────────────────────
    {
        "id": "ru_01",
        "category": "Русский язык",
        "name": "Объяснение термина простыми словами",
        "prompt": "Объясни, что такое квантование нейронных сетей, простыми словами за 2-3 предложения.",
        "checks": [
            ("contains_any", ["квантов", "точност", "размер", "весов", "памят"]),
            ("length_min", 80),
            ("length_max", 600),
            ("has_cyrillic", None),
        ],
        "weight": 2,
    },
    {
        "id": "ru_02",
        "category": "Русский язык",
        "name": "Исправление ошибок в тексте",
        "prompt": (
            "Найди и исправь ошибки в следующем тексте. "
            "Выведи исправленный вариант:\n\n"
            "«Вчера я пошёл в магазин и купил хлеб, молоко и йогурт. "
            "Продавец дала мне здачу не правильно — вместо десяти рублей она дала восем.»"
        ),
        "checks": [
            ("contains_any", ["сдачу", "восемь", "неправильно"]),
            ("length_min", 50),
            ("has_cyrillic", None),
            ("not_refusal", None),
        ],
        "weight": 2,
    },
    {
        "id": "ru_03",
        "category": "Русский язык",
        "name": "Перефразирование в официальный стиль",
        "prompt": (
            "Перепиши следующий текст в официально-деловом стиле:\n\n"
            "«Привет! Мы хотим, чтобы ты пришёл на наше мероприятие в пятницу. "
            "Будет круто, приходи обязательно!»"
        ),
        "checks": [
            ("not_contains", ["привет", "круто", "приходи"]),
            ("contains_any", ["уважаем", "приглашаем", "мероприятие", "просим"]),
            ("has_cyrillic", None),
            ("length_min", 60),
        ],
        "weight": 2,
    },
    {
        "id": "ru_04",
        "category": "Русский язык",
        "name": "Краткое резюме текста",
        "prompt": (
            "Сделай краткое резюме (1-2 предложения) следующего текста:\n\n"
            "«Искусственный интеллект — это область компьютерных наук, занимающаяся "
            "созданием систем, способных выполнять задачи, требующие человеческого "
            "интеллекта: распознавание речи, принятие решений, перевод текста. "
            "Современные системы ИИ основаны на нейронных сетях и обучаются "
            "на огромных массивах данных, что позволяет им достигать и превосходить "
            "человеческий уровень в узких задачах.»"
        ),
        "checks": [
            ("contains_any", ["интеллект", "нейрон", "задач", "компьютер", "систем"]),
            ("length_min", 40),
            ("length_max", 300),
            ("has_cyrillic", None),
        ],
        "weight": 2,
    },

    # ── 2. Код ──────────────────────────────────────────────────────────────
    {
        "id": "code_01",
        "category": "Код",
        "name": "Функция Python: бинарный поиск",
        "prompt": (
            "Напиши функцию на Python `binary_search(arr, target)`, которая возвращает "
            "индекс элемента в отсортированном списке или -1 если не найден. "
            "Только код, без пояснений."
        ),
        "checks": [
            ("valid_python", None),
            ("contains_all", ["def binary_search"]),
            ("contains_any", ["return -1", "return mid", "return left"]),
            ("not_refusal", None),
        ],
        "weight": 3,
    },
    {
        "id": "code_02",
        "category": "Код",
        "name": "SQL-запрос с агрегацией",
        "prompt": (
            "Напиши SQL-запрос: из таблицы `orders` выбери топ-5 клиентов "
            "(поле `customer_id`) по суммарному количеству заказов (поле `amount`), "
            "отсортированных по убыванию. Только код."
        ),
        "checks": [
            ("contains_all", ["SELECT", "orders"]),
            ("contains_any", ["GROUP BY", "group by"]),
            ("contains_any", ["ORDER BY", "order by"]),
            ("contains_any", ["LIMIT 5", "limit 5", "TOP 5", "top 5"]),
            ("not_refusal", None),
        ],
        "weight": 3,
    },
    {
        "id": "code_03",
        "category": "Код",
        "name": "Поиск бага в коде",
        "prompt": (
            "Найди баг в следующем Python-коде и объясни, в чём проблема:\n\n"
            "```python\n"
            "def average(numbers):\n"
            "    total = 0\n"
            "    for n in numbers:\n"
            "        total += n\n"
            "    return total / len(numbers)\n"
            "\n"
            "print(average([]))\n"
            "```"
        ),
        "checks": [
            ("contains_any", ["делени", "нуль", "zero", "division", "пустой", "empty", "ZeroDivision"]),
            ("length_min", 30),
            ("has_cyrillic", None),
            ("not_refusal", None),
        ],
        "weight": 2,
    },
    {
        "id": "code_04",
        "category": "Код",
        "name": "Функция Python: декоратор с таймером",
        "prompt": (
            "Напиши декоратор Python `timeit`, который выводит время выполнения "
            "декорируемой функции в секундах. Только код."
        ),
        "checks": [
            ("valid_python", None),
            ("contains_all", ["def timeit"]),
            ("contains_any", ["time.time()", "time.perf_counter()", "perf_counter"]),
            ("contains_any", ["wrapper", "inner", "decorated"]),
        ],
        "weight": 3,
    },
    {
        "id": "code_05",
        "category": "Код",
        "name": "Объяснение кода",
        "prompt": (
            "Объясни, что делает этот код на Python:\n\n"
            "```python\n"
            "result = [x**2 for x in range(10) if x % 2 == 0]\n"
            "```"
        ),
        "checks": [
            ("contains_any", ["квадрат", "чётн", "список", "list", "comprehension", "нечётн"]),
            ("contains_any", ["0", "4", "16", "36", "64"]),
            ("length_min", 40),
            ("has_cyrillic", None),
        ],
        "weight": 2,
    },

    # ── 3. Логика и математика ──────────────────────────────────────────────
    {
        "id": "math_01",
        "category": "Логика/Математика",
        "name": "Простая арифметика",
        "prompt": "Сколько будет 347 × 28? Ответь только числом.",
        "checks": [
            ("exact_number", 9716),
            ("length_max", 50),
        ],
        "weight": 3,
    },
    {
        "id": "math_02",
        "category": "Логика/Математика",
        "name": "Задача на проценты",
        "prompt": (
            "Товар стоил 2400 рублей. Его цена снизилась на 15%. "
            "Сколько стоит товар теперь? Ответь только числом в рублях."
        ),
        "checks": [
            ("exact_number", 2040),
            ("length_max", 80),
        ],
        "weight": 3,
    },
    {
        "id": "math_03",
        "category": "Логика/Математика",
        "name": "Последовательность чисел",
        "prompt": (
            "Найди следующее число в последовательности: 2, 6, 18, 54, ___\n"
            "Ответь только числом."
        ),
        "checks": [
            ("exact_number", 162),
            ("length_max", 50),
        ],
        "weight": 2,
    },
    {
        "id": "math_04",
        "category": "Логика/Математика",
        "name": "Логическая задача",
        "prompt": (
            "В комнате 3 кошки. Каждая кошка видит 2 кошки. "
            "Сколько кошек в комнате? Ответь только числом."
        ),
        "checks": [
            ("exact_number", 3),
            ("length_max", 80),
        ],
        "weight": 2,
    },
    {
        "id": "math_05",
        "category": "Логика/Математика",
        "name": "Задача на время",
        "prompt": (
            "Поезд едет из А в Б со скоростью 120 км/ч. "
            "Расстояние 360 км. Сколько часов займёт поездка? Ответь только числом."
        ),
        "checks": [
            ("exact_number", 3),
            ("length_max", 60),
        ],
        "weight": 2,
    },

    # ── 4. Следование инструкциям ────────────────────────────────────────────
    {
        "id": "inst_01",
        "category": "Инструкции",
        "name": "Вывод валидного JSON",
        "prompt": (
            "Верни JSON-объект с полями: name (строка), age (число), skills (массив из 3 строк). "
            "Только JSON, никакого другого текста."
        ),
        "checks": [
            ("valid_json", None),
            ("not_refusal", None),
        ],
        "weight": 3,
    },
    {
        "id": "inst_02",
        "category": "Инструкции",
        "name": "Список ровно из 5 пунктов",
        "prompt": (
            "Напиши список ровно из 5 советов по здоровому образу жизни. "
            "Каждый совет — отдельная строка, начинающаяся с цифры и точки."
        ),
        "checks": [
            ("regex", r"[1-5]\.\s"),
            ("count_lines", (4, 8)),
            ("has_cyrillic", None),
            ("not_refusal", None),
        ],
        "weight": 2,
    },
    {
        "id": "inst_03",
        "category": "Инструкции",
        "name": "Ответ только на английском",
        "prompt": "Answer in English only: What is the capital of France?",
        "checks": [
            ("contains_any", ["paris", "Paris"]),
            ("no_cyrillic", None),
            ("length_max", 100),
        ],
        "weight": 2,
    },
    {
        "id": "inst_04",
        "category": "Инструкции",
        "name": "Ответ одним словом",
        "prompt": "Столица Японии? Ответь одним словом.",
        "checks": [
            ("contains_any", ["Токио", "токио", "Tokyo", "tokyo"]),
            ("length_max", 30),
        ],
        "weight": 2,
    },

    # ── 5. Знания ────────────────────────────────────────────────────────────
    {
        "id": "know_01",
        "category": "Знания",
        "name": "Столица страны",
        "prompt": "Какая столица Австралии? Ответь одним словом.",
        "checks": [
            ("contains_any", ["Канберра", "канберра", "Canberra", "canberra"]),
            ("length_max", 50),
        ],
        "weight": 2,
    },
    {
        "id": "know_02",
        "category": "Знания",
        "name": "Исторический факт",
        "prompt": "В каком году был запущен первый искусственный спутник Земли? Ответь только числом.",
        "checks": [
            ("exact_number", 1957),
            ("length_max", 50),
        ],
        "weight": 2,
    },
    {
        "id": "know_03",
        "category": "Знания",
        "name": "Определение понятия",
        "prompt": "Что такое REST API? Объясни в 1-2 предложениях.",
        "checks": [
            ("contains_any", ["http", "HTTP", "запрос", "интерфейс", "ресурс", "протокол"]),
            ("length_min", 50),
            ("length_max", 400),
            ("not_refusal", None),
        ],
        "weight": 2,
    },

    # ── 6. Работа с текстом ──────────────────────────────────────────────────
    {
        "id": "text_01",
        "category": "Работа с текстом",
        "name": "Суммаризация: короче входного",
        "prompt": (
            "Сократи следующий текст до одного предложения, сохранив главную мысль:\n\n"
            "«Машинное обучение — это подраздел искусственного интеллекта, который "
            "изучает методы построения алгоритмов, способных обучаться на данных и "
            "делать предсказания или принимать решения без явного программирования "
            "каждого шага. Алгоритмы машинного обучения используются в рекомендательных "
            "системах, распознавании изображений, обработке естественного языка и других "
            "областях.»"
        ),
        "checks": [
            ("length_max", 250),
            ("contains_any", ["машинн", "обучен", "данных", "алгоритм", "предсказ"]),
            ("has_cyrillic", None),
        ],
        "weight": 2,
    },
    {
        "id": "text_02",
        "category": "Работа с текстом",
        "name": "Определение тональности",
        "prompt": (
            "Определи тональность отзыва: позитивная, негативная или нейтральная. "
            "Ответь одним словом.\n\n"
            "Отзыв: «Доставка пришла вовремя, товар соответствует описанию, упаковка целая. "
            "Всё как обещали.»"
        ),
        "checks": [
            ("contains_any", ["позитивн", "положительн", "нейтральн", "positiv", "neutral"]),
            ("length_max", 50),
        ],
        "weight": 2,
    },

    # ── 7. Многоязычность ────────────────────────────────────────────────────
    {
        "id": "ml_01",
        "category": "Многоязычность",
        "name": "Русский вопрос → русский ответ",
        "prompt": "Как называется самое глубокое озеро в мире?",
        "checks": [
            ("contains_any", ["Байкал", "байкал", "baikal", "Baikal"]),
            ("has_cyrillic", None),
        ],
        "weight": 2,
    },
    {
        "id": "ml_02",
        "category": "Многоязычность",
        "name": "Переключение на английский по запросу",
        "prompt": "Please respond in English only. What is photosynthesis?",
        "checks": [
            ("contains_any", ["light", "plant", "chloro", "energy", "sun", "oxygen"]),
            ("no_cyrillic", None),
            ("length_min", 30),
        ],
        "weight": 2,
    },

    # ── 8. Стресс-тест ──────────────────────────────────────────────────────
    {
        "id": "stress_01",
        "category": "Стресс",
        "name": "Не отказывает на технический запрос",
        "prompt": (
            "Объясни принцип работы алгоритма Дейкстры для поиска кратчайшего пути "
            "в графе. Используй пример с 4 вершинами."
        ),
        "checks": [
            ("not_refusal", None),
            ("contains_any", ["вершин", "граф", "путь", "вес", "ребр", "dijkstra", "Дейкстр"]),
            ("length_min", 100),
            ("has_cyrillic", None),
        ],
        "weight": 2,
    },
]

REFUSAL_PHRASES = [
    "не могу", "не в состоянии", "отказываюсь", "невозможно выполнить",
    "i cannot", "i can't", "i'm unable", "i refuse", "as an ai",
    "as a language model", "я языковая модель",
]


# ── Checker ─────────────────────────────────────────────────────────────────
def run_check(check_type, arg, response: str) -> bool:
    r = response.lower()
    if check_type == "contains_all":
        return all(w.lower() in r for w in arg)
    if check_type == "contains_any":
        return any(w.lower() in r for w in arg)
    if check_type == "not_contains":
        return not any(w.lower() in r for w in arg)
    if check_type == "length_min":
        return len(response.strip()) >= arg
    if check_type == "length_max":
        return len(response.strip()) <= arg
    if check_type == "exact_number":
        return str(arg) in response
    if check_type == "valid_json":
        text = re.sub(r"```(?:json)?\s*|\s*```", "", response).strip()
        # Find first {...} or [...]
        m = re.search(r'[\[{].*[\]}]', text, re.DOTALL)
        if m:
            try:
                json.loads(m.group(0))
                return True
            except Exception:
                pass
        try:
            json.loads(text)
            return True
        except Exception:
            return False
    if check_type == "valid_python":
        blocks = re.findall(r"```(?:python)?\s*(.*?)```", response, re.DOTALL)
        code = "\n".join(blocks) if blocks else response
        try:
            ast.parse(code)
            return True
        except SyntaxError:
            return False
    if check_type == "count_lines":
        lo, hi = arg
        lines = [l for l in response.splitlines() if l.strip()]
        return lo <= len(lines) <= hi
    if check_type == "has_cyrillic":
        return bool(re.search(r'[а-яёА-ЯЁ]', response))
    if check_type == "no_cyrillic":
        return not bool(re.search(r'[а-яёА-ЯЁ]', response))
    if check_type == "regex":
        return bool(re.search(arg, response))
    if check_type == "not_refusal":
        return not any(p in r for p in REFUSAL_PHRASES)
    return False


def score_checks(checks, response) -> tuple:
    """Returns (passed, total, details)."""
    passed, total = 0, len(checks)
    details = []
    for check_type, arg in checks:
        ok = run_check(check_type, arg, response)
        passed += int(ok)
        details.append({"check": check_type, "arg": str(arg)[:60], "ok": ok})
    return passed, total, details


# ── HTTP helpers ─────────────────────────────────────────────────────────────
def api_post(path, body, timeout=120):
    url = PROXY_URL + path
    data = json.dumps(body).encode()
    req = urllib.request.Request(url, data=data,
                                 headers={"Content-Type": "application/json"})
    with urllib.request.urlopen(req, timeout=timeout) as r:
        return json.loads(r.read())


def api_get(path, timeout=10):
    with urllib.request.urlopen(PROXY_URL + path, timeout=timeout) as r:
        return json.loads(r.read())


def wait_for_proxy(timeout=10):
    deadline = time.time() + timeout
    while time.time() < deadline:
        try:
            api_get("/healthz", timeout=2)
            return True
        except Exception:
            time.sleep(1)
    return False


# ── Model switching ───────────────────────────────────────────────────────────
def switch_model(model_file, alias, timeout=SWITCH_TIMEOUT_S):
    print(f"    Переключаю на {alias}...", end="", flush=True)
    try:
        api_post("/admin/switch", {"model_file": model_file, "alias": alias})
    except Exception as e:
        print(f" ОШИБКА switch: {e}")
        return False

    deadline = time.time() + timeout
    while time.time() < deadline:
        time.sleep(3)
        try:
            s = api_get("/admin/status")
            if s.get("phase") == "ready" and s.get("model") == alias:
                print(" готово.")
                return True
            if s.get("phase") == "error":
                print(f" ОШИБКА: {s.get('message')}")
                return False
            print(".", end="", flush=True)
        except Exception:
            print(".", end="", flush=True)
    print(" таймаут.")
    return False


# ── Run one test case ─────────────────────────────────────────────────────────
def run_case(model_alias, case) -> dict:
    messages = []
    if "system" in case:
        messages.append({"role": "system", "content": case["system"]})
    messages.append({"role": "user", "content": case["prompt"]})

    t0 = time.perf_counter()
    try:
        resp = api_post("/v1/chat/completions", {
            "model": model_alias,
            "messages": messages,
            "temperature": DEFAULT_TEMPERATURE,
            "max_tokens": DEFAULT_MAX_TOKENS,
            "stream": False,
        }, timeout=120)
        elapsed = time.perf_counter() - t0

        content = resp["choices"][0]["message"]["content"]
        usage = resp.get("usage", {})
        completion_tokens = usage.get("completion_tokens", 0)
        tok_per_sec = round(completion_tokens / elapsed, 1) if elapsed > 0 else 0

        passed, total, details = score_checks(case["checks"], content)
        score = round(100 * passed / total) if total else 0

        return {
            "id": case["id"],
            "category": case["category"],
            "name": case["name"],
            "status": "ok",
            "score": score,
            "passed": passed,
            "total": total,
            "elapsed_s": round(elapsed, 2),
            "tokens": completion_tokens,
            "tok_per_sec": tok_per_sec,
            "response": content[:500],
            "checks": details,
        }
    except Exception as e:
        elapsed = time.perf_counter() - t0
        return {
            "id": case["id"],
            "category": case["category"],
            "name": case["name"],
            "status": "error",
            "error": str(e),
            "score": 0,
            "passed": 0,
            "total": len(case["checks"]),
            "elapsed_s": round(elapsed, 2),
            "tokens": 0,
            "tok_per_sec": 0,
            "response": "",
            "checks": [],
        }


# ── Run full suite for one model ──────────────────────────────────────────────
def run_suite(model_alias, cases):
    results = []
    total_cases = len(cases)
    for i, case in enumerate(cases, 1):
        label = f"[{i:02d}/{total_cases}] {case['id']}: {case['name']}"
        print(f"  {label}", end="", flush=True)
        r = run_case(model_alias, case)
        status_icon = "✓" if r["score"] == 100 else ("~" if r["score"] >= 50 else "✗")
        speed = f"{r['tok_per_sec']} tok/s" if r["tok_per_sec"] else "err"
        print(f"\r  {status_icon} {label:<60} {r['score']:3d}%  {speed}")
        results.append(r)
    return results


# ── Report ────────────────────────────────────────────────────────────────────
def make_report(all_model_results: list, ts: str) -> str:
    lines = [
        f"# Отчёт тестирования моделей",
        f"",
        f"Дата: {ts}  |  Кейсов: {len(TEST_CASES)}  |  Моделей: {len(all_model_results)}",
        f"",
        f"---",
        f"",
    ]

    # Summary table
    lines += ["## Сводная таблица", ""]
    header = "| Модель | Средний балл | Код | Логика | Инструкции | tok/s |"
    sep    = "|---|---|---|---|---|---|"
    lines += [header, sep]

    for mr in all_model_results:
        alias = mr["alias"]
        results = mr["results"]
        ok = [r for r in results if r["status"] == "ok"]
        avg = round(sum(r["score"] for r in ok) / len(ok)) if ok else 0
        tps = round(sum(r["tok_per_sec"] for r in ok) / len(ok), 1) if ok else 0

        def cat_score(cat):
            cr = [r for r in ok if r["category"] == cat]
            if not cr:
                return "—"
            return f"{round(sum(r['score'] for r in cr) / len(cr))}%"

        lines.append(
            f"| {alias} | **{avg}%** | {cat_score('Код')} | {cat_score('Логика/Математика')} "
            f"| {cat_score('Инструкции')} | {tps} |"
        )
    lines += [""]

    # Per-model details
    for mr in all_model_results:
        alias = mr["alias"]
        results = mr["results"]
        lines += [f"---", f"", f"## Модель: `{alias}`", ""]

        categories = {}
        for r in results:
            categories.setdefault(r["category"], []).append(r)

        for cat, cat_results in categories.items():
            lines += [f"### {cat}", ""]
            lines += ["| Кейс | Балл | Время | tok/s | Детали |", "|---|---|---|---|---|"]
            for r in cat_results:
                score_icon = "✅" if r["score"] == 100 else ("⚠️" if r["score"] >= 50 else "❌")
                failed = [c["check"] for c in r.get("checks", []) if not c["ok"]]
                detail = ", ".join(failed) if failed else "всё прошло"
                err = f"ERROR: {r.get('error','')[:40]}" if r["status"] == "error" else detail
                lines.append(
                    f"| {score_icon} {r['name']} | {r['score']}% "
                    f"| {r['elapsed_s']}s | {r['tok_per_sec']} | {err} |"
                )
            lines += [""]

        # Worst answers
        worst = sorted([r for r in results if r["status"] == "ok" and r["score"] < 100],
                       key=lambda x: x["score"])[:3]
        if worst:
            lines += ["### Примеры провальных ответов", ""]
            for r in worst:
                preview = r["response"][:200].replace("\n", " ")
                lines += [
                    f"**{r['name']}** (балл: {r['score']}%)",
                    f"> {preview}",
                    "",
                ]

    return "\n".join(lines)


# ── Main ──────────────────────────────────────────────────────────────────────
def main():
    parser = argparse.ArgumentParser(description="Model evaluation suite")
    parser.add_argument("--model", help="Протестировать только эту модель (alias)")
    parser.add_argument("--skip-switch", action="store_true",
                        help="Не переключать модели, тестировать текущую")
    parser.add_argument("--categories", help="Через запятую: Код,Логика/Математика,...")
    args = parser.parse_args()

    # Filter test cases
    cases = TEST_CASES
    if args.categories:
        cats = {c.strip() for c in args.categories.split(",")}
        cases = [c for c in cases if c["category"] in cats]
        print(f"Фильтр категорий: {cats} → {len(cases)} кейсов")

    print("Проверяю доступность прокси...", end=" ")
    if not wait_for_proxy():
        print("ОШИБКА: прокси недоступен на", PROXY_URL)
        print("Запустите: ./llm.sh start")
        sys.exit(1)
    print("ОК")

    # Get models
    try:
        models_data = api_get("/admin/models")
    except Exception as e:
        print(f"Ошибка получения списка моделей: {e}")
        sys.exit(1)

    available = models_data.get("models", [])
    if not available:
        print("Нет доступных моделей в папке models/")
        sys.exit(1)

    if args.model:
        available = [m for m in available if args.model in (m["alias"], m["filename"])]
        if not available:
            print(f"Модель '{args.model}' не найдена")
            sys.exit(1)

    if args.skip_switch:
        status = api_get("/admin/status")
        current = status.get("model", "unknown")
        available = [m for m in available if m["alias"] == current or m["active"]]
        if not available:
            available = [{"alias": current, "file": "", "filename": current}]

    os.makedirs(RESULTS_DIR, exist_ok=True)
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")

    all_model_results = []
    total_models = len(available)
    print(f"\nТестирую {total_models} модел(и), {len(cases)} кейсов каждую\n")
    print("=" * 70)

    for mi, m in enumerate(available, 1):
        alias = m["alias"]
        print(f"\n[{mi}/{total_models}] Модель: {alias}")
        print(f"  Файл: {m.get('filename', '?')}  ({m.get('size_mb',0)} MB)")

        if not args.skip_switch and total_models > 1:
            if not switch_model(m["file"], alias):
                print(f"  Пропускаю {alias} — не удалось переключить")
                continue

        print()
        results = run_suite(alias, cases)

        # Save per-model JSON
        model_file = os.path.join(RESULTS_DIR, f"{ts}_{alias}.json")
        with open(model_file, "w", encoding="utf-8") as f:
            json.dump({
                "model": alias,
                "timestamp": ts,
                "total_cases": len(results),
                "results": results,
            }, f, ensure_ascii=False, indent=2)

        ok = [r for r in results if r["status"] == "ok"]
        avg = round(sum(r["score"] for r in ok) / len(ok)) if ok else 0
        tps = round(sum(r["tok_per_sec"] for r in ok) / len(ok), 1) if ok else 0
        print(f"\n  Итог: {avg}% | Скорость: {tps} tok/s | Файл: {model_file}")

        all_model_results.append({"alias": alias, "results": results})

    # Summary report
    if all_model_results:
        report = make_report(all_model_results, ts)
        report_file = os.path.join(RESULTS_DIR, f"report_{ts}.md")
        with open(report_file, "w", encoding="utf-8") as f:
            f.write(report)
        print(f"\n{'=' * 70}")
        print(f"Отчёт сохранён: {report_file}")

        # Print summary table to stdout
        print("\n## Итоги\n")
        for mr in all_model_results:
            ok = [r for r in mr["results"] if r["status"] == "ok"]
            avg = round(sum(r["score"] for r in ok) / len(ok)) if ok else 0
            tps = round(sum(r["tok_per_sec"] for r in ok) / len(ok), 1) if ok else 0
            bars = "█" * (avg // 10) + "░" * (10 - avg // 10)
            print(f"  {mr['alias']:<45} {bars} {avg:3d}%  {tps} tok/s")


if __name__ == "__main__":
    main()
