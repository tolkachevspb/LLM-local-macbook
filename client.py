#!/usr/bin/env python3
"""Minimal CLI client for GigaChat localhost proxy."""

import json
import os
import sys
import urllib.error
import urllib.request


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


def main() -> int:
    root_dir = os.path.dirname(os.path.abspath(__file__))
    config = load_config(os.path.join(root_dir, "config.env"))
    if not config:
        config = load_config(os.path.join(root_dir, "config.env.example"))

    host = config.get("HOST", "127.0.0.1")
    port = config.get("PORT", "8091")
    model = config.get("MODEL_ALIAS", "gigachat3-lightning")

    prompt = " ".join(sys.argv[1:]).strip() or "Сделай короткий ответ по-русски о пользе локальных LLM."

    payload = json.dumps({
        "model": model,
        "messages": [
            {"role": "system", "content": "Ты отвечаешь кратко и по-русски."},
            {"role": "user", "content": prompt},
        ],
        "temperature": 0.2,
        "max_tokens": 160,
    }).encode("utf-8")

    request = urllib.request.Request(
        f"http://{host}:{port}/v1/chat/completions",
        data=payload,
        headers={"Content-Type": "application/json"},
        method="POST",
    )

    try:
        with urllib.request.urlopen(request, timeout=120) as response:
            body = json.loads(response.read().decode("utf-8"))
    except urllib.error.URLError as exc:
        print(f"Ошибка: {exc}", file=sys.stderr)
        return 1

    message = body.get("choices", [{}])[0].get("message", {}).get("content", "")
    print(message.strip())
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
