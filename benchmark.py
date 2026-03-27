#!/usr/bin/env python3
"""Local benchmark for GigaChat localhost stack."""

import json
import math
import os
import re
import statistics
import time
import urllib.error
import urllib.request


ROOT_DIR = os.path.dirname(os.path.abspath(__file__))
REPORT_PATH = os.path.join(ROOT_DIR, "BENCHMARK_REPORT.md")


def load_config(path: str) -> dict:
    config = {}
    if not os.path.exists(path):
        return config
    with open(path, "r", encoding="utf-8") as fh:
        for raw_line in fh:
            line = raw_line.strip()
            if not line or line.startswith("#") or "=" not in line:
                continue
            key, value = line.split("=", 1)
            config[key.strip()] = value.strip().strip('"')
    return config


CONFIG = load_config(os.path.join(ROOT_DIR, "config.env")) or load_config(
    os.path.join(ROOT_DIR, "config.env.example")
)
PORT = int(CONFIG.get("PORT", "8091"))
BACKEND_PORT = int(CONFIG.get("BACKEND_PORT", "8092"))
MODEL_ALIAS = CONFIG.get("MODEL_ALIAS", "gigachat3-lightning")
PROXY_BASE = f"http://127.0.0.1:{PORT}"
BACKEND_BASE = f"http://127.0.0.1:{BACKEND_PORT}"

DEVELOPER_SYSTEM = """Ты полезный помощник.
- Следуй системным и пользовательским инструкциям.
- Отвечай на языке пользователя.
- Будь кратким, если пользователь не просит развернутый ответ."""


def render_prompt(messages: list) -> str:
    parts = ["<s>developer system<|role_sep|>\n", DEVELOPER_SYSTEM, "\n<|message_sep|>\n\n"]
    for msg in messages:
        role = msg.get("role", "user")
        content = msg.get("content", "")
        if role not in {"system", "user", "assistant"}:
            role = "user"
        if not isinstance(content, str):
            content = json.dumps(content, ensure_ascii=False)
        parts.extend([f"{role}<|role_sep|>\n", content, "\n<|message_sep|>\n\n"])
    if not messages or messages[-1].get("role") != "assistant":
        parts.append("assistant<|role_sep|>\n")
    return "".join(parts)


def post_json(url: str, payload: dict, timeout: int = 300) -> dict:
    request = urllib.request.Request(
        url,
        data=json.dumps(payload).encode("utf-8"),
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    with urllib.request.urlopen(request, timeout=timeout) as response:
        return json.loads(response.read().decode("utf-8"))


def get_json(url: str, timeout: int = 60) -> dict:
    request = urllib.request.Request(url, method="GET")
    with urllib.request.urlopen(request, timeout=timeout) as response:
        return json.loads(response.read().decode("utf-8"))


def run_proxy_chat(messages, max_tokens, temperature=0.2):
    started = time.perf_counter()
    body = post_json(f"{PROXY_BASE}/v1/chat/completions", {
        "model": MODEL_ALIAS,
        "messages": messages,
        "temperature": temperature,
        "max_tokens": max_tokens,
    })
    elapsed = time.perf_counter() - started
    return body, elapsed


def run_backend_completion(messages, max_tokens, temperature=0.2):
    prompt = render_prompt(messages)
    return post_json(f"{BACKEND_BASE}/v1/completions", {
        "model": MODEL_ALIAS,
        "prompt": prompt,
        "temperature": temperature,
        "max_tokens": max_tokens,
        "stop": ["<|message_sep|>", "</s>"],
    })


QUALITY_CASES = [
    {
        "name": "RU knowledge",
        "messages": [{"role": "user", "content": "Назови столицу Бразилии одним словом."}],
        "max_tokens": 24,
        "check": lambda t: bool(re.search(r"\bбразилиа\b", t, re.I)),
    },
    {
        "name": "RU literature",
        "messages": [{"role": "user", "content": "Кто написал «Преступление и наказание»? Ответь фамилией."}],
        "max_tokens": 24,
        "check": lambda t: bool(re.search(r"\bдостоевск", t, re.I)),
    },
    {
        "name": "EN knowledge",
        "messages": [{"role": "user", "content": "Answer with one token: chemical symbol for gold."}],
        "max_tokens": 12,
        "check": lambda t: bool(re.search(r"\bau\b", t, re.I)),
    },
    {
        "name": "Arithmetic",
        "messages": [{"role": "user", "content": "Вычисли 17 * 19. Ответь только числом."}],
        "max_tokens": 16,
        "check": lambda t: bool(re.search(r"\b323\b", t)),
    },
    {
        "name": "Logic",
        "messages": [{"role": "user", "content": "Все розы — цветы. Некоторые цветы быстро вянут. Следует ли, что некоторые розы быстро вянут? Ответь: да или нет."}],
        "max_tokens": 48,
        "check": lambda t: bool(re.search(r"\bнет\b", t, re.I)),
    },
    {
        "name": "Instruction following",
        "messages": [{"role": "user", "content": "Ответь ровно одним словом: синий"}],
        "max_tokens": 12,
        "check": lambda t: re.sub(r"[^\wа-яА-ЯёЁ-]+", " ", t.strip().lower()).strip() == "синий",
    },
    {
        "name": "Python coding",
        "messages": [{"role": "user", "content": "Напиши Python-функцию square(x), которая возвращает квадрат числа."}],
        "max_tokens": 120,
        "check": lambda t: bool(re.search(r"def\s+square\s*\(\s*x\s*\).*return\s+.*(x\s*\*\*\s*2|x\s*\*\s*x)", t, re.S)),
    },
    {
        "name": "EN reasoning",
        "messages": [{"role": "user", "content": "Answer briefly: can a conclusion be certain if premises only say 'some members' have a property?"}],
        "max_tokens": 48,
        "check": lambda t: bool(re.search(r"\bno\b|\bnot necessarily\b|\bcannot\b", t, re.I)),
    },
]

SPEED_CASES = [
    {"name": "short_ru", "messages": [{"role": "user", "content": "Что такое локальный AI? Ответь в 2 предложениях."}], "max_tokens": 80},
    {"name": "medium_ru", "messages": [{"role": "user", "content": "Сравни локальный и облачный запуск LLM по приватности и скорости."}], "max_tokens": 140},
    {"name": "code_py", "messages": [{"role": "user", "content": "Напиши Python-функцию fibonacci(n) итеративно."}], "max_tokens": 120},
]


def benchmark_quality():
    results = []
    for case in QUALITY_CASES:
        body, wall = run_proxy_chat(case["messages"], case["max_tokens"], temperature=0.0)
        answer = body.get("choices", [{}])[0].get("message", {}).get("content", "").strip()
        passed = bool(case["check"](answer))
        results.append({"name": case["name"], "passed": passed, "latency_s": wall, "answer": answer, "prompt": case["messages"][0]["content"]})
    return results


def benchmark_speed():
    results = []
    for case in SPEED_CASES:
        proxy_body, wall = run_proxy_chat(case["messages"], case["max_tokens"])
        backend_body = run_backend_completion(case["messages"], case["max_tokens"])
        usage = backend_body.get("usage", {})
        timings = backend_body.get("timings", {})
        ct = usage.get("completion_tokens", 0)
        pt = usage.get("prompt_tokens", 0)
        tt = usage.get("total_tokens", 0)
        pred_ms = timings.get("predicted_ms", 0.0)
        prompt_ms = timings.get("prompt_ms", 0.0)
        gen_tps = ct / (pred_ms / 1000.0) if pred_ms and ct else 0.0
        prompt_tps = pt / (prompt_ms / 1000.0) if prompt_ms and pt else 0.0
        total_tps = tt / wall if wall and tt else 0.0
        results.append({
            "name": case["name"], "wall_s": wall,
            "prompt_tokens": pt, "completion_tokens": ct, "total_tokens": tt,
            "gen_tps": gen_tps, "prompt_tps": prompt_tps, "total_tps": total_tps,
        })
    return results


def render_report(speed, quality):
    p = sum(1 for q in quality if q["passed"])
    t = len(quality)
    avg_wall = statistics.mean(s["wall_s"] for s in speed)
    avg_gen = statistics.mean(s["gen_tps"] for s in speed)
    avg_total = statistics.mean(s["total_tps"] for s in speed)

    lines = [
        "# Benchmark Report", "",
        f"Дата: {time.strftime('%Y-%m-%d %H:%M:%S')}", "",
        f"Стек: GigaChat3-10B-A1.8B Q6_K + llamafile + proxy", "",
        "## Quality", "",
        f"Итог: **{p}/{t}**", "",
        "| Тест | Статус | Задержка |",
        "| --- | --- | ---: |",
    ]
    for q in quality:
        lines.append(f"| {q['name']} | {'PASS' if q['passed'] else 'FAIL'} | {q['latency_s']:.2f}s |")

    lines.extend(["", "## Speed", "",
        f"- Средняя e2e latency: **{avg_wall:.2f}s**",
        f"- Средняя generation: **{avg_gen:.2f} tok/s**",
        f"- Средняя total throughput: **{avg_total:.2f} tok/s**", "",
        "| Сценарий | wall | prompt tok | gen tok | gen tok/s | total tok/s |",
        "| --- | ---: | ---: | ---: | ---: | ---: |",
    ])
    for s in speed:
        lines.append(f"| {s['name']} | {s['wall_s']:.2f}s | {s['prompt_tokens']} | {s['completion_tokens']} | {s['gen_tps']:.2f} | {s['total_tps']:.2f} |")
    lines.append("")
    return "\n".join(lines)


def main() -> int:
    try:
        get_json(f"{PROXY_BASE}/healthz")
    except Exception as exc:
        print(f"Стек не отвечает на /healthz. Запустите: ./llm.sh start")
        return 1

    print("Запуск benchmark ...\n")
    quality = benchmark_quality()
    speed = benchmark_speed()
    report = render_report(speed, quality)
    with open(REPORT_PATH, "w", encoding="utf-8") as fh:
        fh.write(report)
    print(report)
    print(f"\nОтчет сохранен: {REPORT_PATH}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
