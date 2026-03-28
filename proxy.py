#!/usr/bin/env python3
"""
GigaChat localhost proxy.

Renders prompts in GigaChat format, cleans service tokens,
and exposes an OpenAI-compatible /v1/chat/completions API + web UI.
"""

import json
import os
import re
import signal
import subprocess
import sys
import threading
import time
import uuid
import urllib.error
import urllib.request
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer


DEVELOPER_SYSTEM = """Ты полезный помощник.
- Следуй системным и пользовательским инструкциям.
- Отвечай на языке пользователя.
- Если пользователь пишет по-русски, отвечай по-русски.
- Будь кратким, если пользователь не просит развернутый ответ.
- Не выводи служебные разделители ролей и внутренние токены.
- Если запрос сформулирован слишком абстрактно, сначала постарайся интерпретировать его разумно, а не уходить в шаблонный отказ."""


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


ROOT_DIR = os.path.dirname(os.path.abspath(__file__))
CONFIG = load_config(os.path.join(ROOT_DIR, "config.env")) or load_config(
    os.path.join(ROOT_DIR, "config.env.example")
)
HOST = CONFIG.get("HOST", "127.0.0.1")
PORT = int(CONFIG.get("PORT", "8091"))
BACKEND_PORT = int(CONFIG.get("BACKEND_PORT", "8092"))
MODEL_ALIAS = CONFIG.get("MODEL_ALIAS", "gigachat3-lightning")
BACKEND_BASE = f"http://127.0.0.1:{BACKEND_PORT}"


# ── Model switching state ─────────────────────────────────
_backend_lock = threading.Lock()
_switch_status = {"phase": "ready", "model": MODEL_ALIAS, "message": ""}
_current_model_file = CONFIG.get("MODEL_FILE", "")
_current_model_alias = MODEL_ALIAS


def scan_models() -> list:
    models_dir = os.path.join(ROOT_DIR, "models")
    result = []
    if not os.path.isdir(models_dir):
        return result
    for fname in sorted(os.listdir(models_dir)):
        if not fname.lower().endswith(".gguf"):
            continue
        rel = os.path.join("./models", fname)
        full = os.path.join(models_dir, fname)
        real = os.path.realpath(full)
        size_mb = round(os.path.getsize(real) / 1024 / 1024) if os.path.exists(real) else 0
        alias = re.sub(r"[^a-z0-9]+", "-", fname[:-5].lower()).strip("-")
        active = (fname == os.path.basename(_current_model_file))
        result.append({
            "file": rel, "filename": fname, "alias": alias,
            "size_mb": size_mb, "active": active,
        })
    return result


def _build_backend_cmd(model_file: str, alias: str) -> list:
    cfg = CONFIG
    llamafile = os.path.join(ROOT_DIR, "bin", "llamafile")
    model_abs = model_file if os.path.isabs(model_file) else os.path.join(ROOT_DIR, model_file.lstrip("./"))
    flash = cfg.get("FLASH_ATTN", "on")
    fa_val = "1" if flash == "on" else "0"
    jinja_flag = "--no-jinja" if cfg.get("JINJA", "0") == "0" else "--jinja"
    cmd = [
        llamafile, "--server",
        "-m", model_abs,
        "--host", HOST, "--port", str(BACKEND_PORT),
        "--alias", alias,
        "-c",  cfg.get("CTX_SIZE", "4096"),
        "-t",  cfg.get("THREADS", "4"),
        "-tb", cfg.get("THREADS_BATCH", "4"),
        "-b",  cfg.get("BATCH_SIZE", "512"),
        "-ub", cfg.get("UBATCH_SIZE", "512"),
        "-np", cfg.get("PARALLEL", "1"),
        "-ctk", cfg.get("CACHE_TYPE_K", "q8_0"),
        "-ctv", cfg.get("CACHE_TYPE_V", "q8_0"),
        "-fa", fa_val,
        "-ngl", cfg.get("GPU_LAYERS", "999"),
        jinja_flag,
    ]
    # Chat template: только для GigaChat моделей (остальные используют встроенный шаблон)
    if "gigachat" in model_file.lower() or "gigachat" in alias.lower():
        tmpl = cfg.get("CHAT_TEMPLATE", "gigachat") or "gigachat"
        cmd += ["--chat-template", tmpl]
    return cmd


def _kill_backend() -> None:
    pid_file = os.path.join(ROOT_DIR, "run", "backend.pid")
    if not os.path.exists(pid_file):
        return
    try:
        pid = int(open(pid_file).read().strip())
        os.kill(pid, signal.SIGTERM)
        for _ in range(20):
            time.sleep(0.5)
            try:
                os.kill(pid, 0)
            except ProcessLookupError:
                break
        else:
            os.kill(pid, signal.SIGKILL)
    except Exception:
        pass
    finally:
        try:
            os.remove(pid_file)
        except Exception:
            pass


def switch_backend_async(model_file: str, alias: str) -> None:
    global _current_model_file, _current_model_alias, MODEL_ALIAS

    def run():
        global _current_model_file, _current_model_alias, MODEL_ALIAS
        with _backend_lock:
            _switch_status["phase"] = "stopping"
            _switch_status["message"] = "Останавливаю текущую модель..."
            _kill_backend()

            # Ждём освобождения порта (до 10 сек)
            import socket as _socket
            for _ in range(20):
                try:
                    s = _socket.create_connection(("127.0.0.1", BACKEND_PORT), timeout=0.3)
                    s.close()
                    time.sleep(0.5)
                except OSError:
                    break  # порт свободен
            else:
                time.sleep(1)

            _switch_status["phase"] = "starting"
            _switch_status["message"] = f"Запускаю {alias}..."
            cmd = _build_backend_cmd(model_file, alias)
            log_path = os.path.join(ROOT_DIR, "logs", "backend.log")
            os.makedirs(os.path.dirname(log_path), exist_ok=True)
            # llamafile — cosmopolitan binary, требует запуска через shell
            # (прямой execve() падает с ENOEXEC на macOS/ARM)
            import shlex as _shlex
            shell_cmd = _shlex.join(str(x) for x in cmd)
            with open(log_path, "a") as lf:
                proc = subprocess.Popen(
                    shell_cmd, shell=True,
                    stdin=subprocess.DEVNULL, stdout=lf, stderr=lf,
                    start_new_session=True
                )
            pid_file = os.path.join(ROOT_DIR, "run", "backend.pid")
            os.makedirs(os.path.dirname(pid_file), exist_ok=True)
            open(pid_file, "w").write(str(proc.pid))

            for i in range(90):
                time.sleep(2)
                if proc.poll() is not None:
                    _switch_status.update({"phase": "error", "message": f"Backend упал (код {proc.returncode})"})
                    return
                try:
                    urllib.request.urlopen(f"{BACKEND_BASE}/v1/models", timeout=2)
                    _current_model_file = model_file
                    _current_model_alias = alias
                    MODEL_ALIAS = alias
                    _switch_status.update({"phase": "ready", "model": alias, "message": ""})
                    return
                except Exception:
                    _switch_status["message"] = f"Загрузка модели... ({i*2}с)"
            _switch_status.update({"phase": "error", "message": "Таймаут — backend не ответил за 180с"})

    threading.Thread(target=run, daemon=True).start()


def _render_prompt_glm(messages: list) -> str:
    """GLM4 native format: [gMASK]<sop><|role|>\ncontent"""
    parts = []
    first = True
    for msg in messages:
        role = msg.get("role", "user")
        content = msg.get("content", "")
        if not isinstance(content, str):
            content = json.dumps(content, ensure_ascii=False)
        if role == "system":
            prefix = "[gMASK]<sop><|system|>\n" if first else "<|system|>\n"
            parts.append(prefix + content)
            first = False
        elif role in {"user", "human"}:
            prefix = "[gMASK]<sop><|user|>\n" if first else "<|user|>\n"
            parts.append(prefix + content)
            first = False
        elif role == "assistant":
            parts.append("<|assistant|>\n" + content)
    if not messages or messages[-1].get("role") != "assistant":
        parts.append("<|assistant|>\n")
    return "".join(parts)


def render_prompt(messages: list) -> str:
    alias = _current_model_alias.lower()
    if "glm" in alias:
        return _render_prompt_glm(messages)
    # GigaChat format
    parts = [
        "<s>developer system<|role_sep|>\n",
        DEVELOPER_SYSTEM,
        "\n<|message_sep|>\n\n",
    ]
    for message in messages:
        role = message.get("role", "user")
        content = message.get("content", "")
        if role == "tool":
            role = "function result"
        if role not in {"system", "user", "assistant", "function result"}:
            role = "user"
        if not isinstance(content, str):
            content = json.dumps(content, ensure_ascii=False)
        parts.extend([
            f"{role}<|role_sep|>\n",
            content,
            "\n<|message_sep|>\n\n",
        ])
    if not messages or messages[-1].get("role") != "assistant":
        parts.append("assistant<|role_sep|>\n")
    return "".join(parts)


def _stop_tokens() -> list:
    """Stop-токены в зависимости от активной модели."""
    alias = _current_model_alias.lower()
    if "gigachat" in alias:
        return ["<|message_sep|>", "</s>"]
    if "phi" in alias:
        # Phi генерирует "user" / "assistant" как plain text после ответа
        return ["<|end|>", "<|user|>", "<|assistant|>", "<|system|>", "<|endoftext|>",
                "\nuser", "\nassistant", "\nsystem"]
    if "qwen" in alias:
        return ["<|im_end|>", "<|im_start|>", "<|endoftext|>",
                "\nuser", "\nassistant"]
    if "gemma" in alias:
        return ["<end_of_turn>", "<start_of_turn>"]
    if "glm" in alias:
        return ["<|user|>", "<|observation|>", "</s>", "<eop>"]
    if "llama" in alias or "mistral" in alias:
        return ["[INST]", "[/INST]", "</s>", "<|eot_id|>"]
    # Универсальный fallback
    return ["</s>", "<|endoftext|>", "<|end|>", "<|im_end|>",
            "\nuser", "\nassistant"]


# Паттерны стоп-токенов для финальной очистки ответа
_STOP_PATTERNS = re.compile(
    r"(<\|message_sep\|>.*|<\|role_sep\|>.*|<\|end_of_assistant\|>.*"
    r"|<\|end\|>.*|<\|im_end\|>.*|<\|im_start\|>.*"
    r"|<end_of_turn>.*|<start_of_turn>.*"
    r"|<\|eot_id\|>.*|\[INST\].*|\[/INST\].*"
    r"|<\|endoftext\|>.*|<\|user\|>.*|<\|assistant\|>.*|<\|system\|>.*"
    r"|<\|observation\|>.*|<eop>.*|\[gMASK\].*"
    r"|\n+user\s*$|\n+assistant\s*$|\n+system\s*$)",
    re.DOTALL | re.IGNORECASE,
)


def clean_text(text: str) -> str:
    text = text.replace("\r\n", "\n").replace("\r", "\n")
    text = re.sub(r"^\s*(\[\]|<\|[^>]+?\|>|\n)+", "", text)
    text = _STOP_PATTERNS.sub("", text)
    return text.strip()


def backend_completion(prompt: str, temperature: float, max_tokens: int) -> tuple:
    payload = {
        "model": MODEL_ALIAS,
        "prompt": prompt,
        "temperature": temperature,
        "max_tokens": max_tokens,
        "stop": _stop_tokens(),
    }
    request = urllib.request.Request(
        f"{BACKEND_BASE}/v1/completions",
        data=json.dumps(payload).encode("utf-8"),
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    with urllib.request.urlopen(request, timeout=300) as response:
        body = json.loads(response.read().decode("utf-8"))
    choice = body.get("choices", [{}])[0]
    text = clean_text(choice.get("text", ""))
    finish_reason = choice.get("finish_reason", "stop")
    usage = body.get("usage", {})
    return text, finish_reason, usage


def backend_completion_stream(prompt: str, temperature: float, max_tokens: int):
    payload = {
        "model": MODEL_ALIAS,
        "prompt": prompt,
        "temperature": temperature,
        "max_tokens": max_tokens,
        "stop": _stop_tokens(),
        "stream": True,
    }
    request = urllib.request.Request(
        f"{BACKEND_BASE}/v1/completions",
        data=json.dumps(payload).encode("utf-8"),
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    return urllib.request.urlopen(request, timeout=300)


UI_HTML = r"""<!doctype html>
<html lang="ru">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <title>GigaChat · Local</title>
  <style>
    *, *::before, *::after { box-sizing: border-box; margin: 0; padding: 0; }
    :root {
      --bg:        #0d0f14;
      --sidebar:   #111318;
      --main:      #0d0f14;
      --surface:   #161a22;
      --surface2:  #1c2030;
      --border:    #252a38;
      --border2:   #1e2330;
      --ink:       #e8eaf0;
      --muted:     #6b7280;
      --accent:    #6366f1;
      --accent2:   #818cf8;
      --green:     #34d399;
      --red:       #f87171;
      --yellow:    #fbbf24;
      --code-bg:   #12141c;
      --radius:    12px;
      --sidebar-w: 260px;
    }
    html, body { height: 100%; }
    body {
      font-family: ui-sans-serif, -apple-system, BlinkMacSystemFont, "Segoe UI", sans-serif;
      font-size: 14px; line-height: 1.6;
      background: var(--bg); color: var(--ink);
      display: flex; overflow: hidden;
    }

    /* ── SIDEBAR ─────────────────────────────────── */
    #sidebar {
      width: var(--sidebar-w); min-width: var(--sidebar-w);
      background: var(--sidebar); border-right: 1px solid var(--border2);
      display: flex; flex-direction: column; height: 100vh;
      overflow: hidden;
    }
    .sidebar-head {
      padding: 18px 16px 14px;
      border-bottom: 1px solid var(--border2);
      display: flex; align-items: center; gap: 10px;
    }
    .logo {
      width: 32px; height: 32px; border-radius: 8px;
      background: linear-gradient(135deg, var(--accent), #a855f7);
      display: flex; align-items: center; justify-content: center;
      font-size: 16px; flex-shrink: 0;
    }
    .logo-text { font-size: 15px; font-weight: 700; color: var(--ink); }
    .logo-sub  { font-size: 11px; color: var(--muted); margin-top: 1px; }

    .sidebar-section { padding: 14px 12px 6px; }
    .sidebar-label {
      font-size: 10px; font-weight: 700; letter-spacing: .1em;
      text-transform: uppercase; color: var(--muted);
      padding: 0 4px; margin-bottom: 8px;
    }

    #sessionList { flex: 1; overflow-y: auto; padding: 0 8px 8px; }
    .sess-item {
      padding: 9px 10px; border-radius: 8px; cursor: pointer;
      display: flex; align-items: center; justify-content: space-between;
      gap: 6px; color: var(--muted); font-size: 13px;
      transition: background .15s, color .15s;
    }
    .sess-item:hover { background: var(--surface); color: var(--ink); }
    .sess-item.active { background: var(--surface2); color: var(--ink); }
    .sess-title { flex: 1; overflow: hidden; text-overflow: ellipsis; white-space: nowrap; }
    .sess-title-input {
      flex: 1; background: var(--surface2); border: 1px solid var(--accent);
      border-radius: 5px; padding: 2px 6px; color: var(--ink);
      font: inherit; font-size: 13px; outline: none; min-width: 0;
    }
    .sess-actions { display: flex; gap: 2px; opacity: 0; transition: opacity .15s; }
    .sess-item:hover .sess-actions { opacity: 1; }
    .sess-btn {
      font-size: 13px; color: var(--muted); background: none;
      border: none; cursor: pointer; padding: 2px 5px; border-radius: 4px;
      line-height: 1; transition: color .15s;
    }
    .sess-btn:hover { color: var(--ink); }
    .sess-btn.del:hover { color: var(--red); }
    .del-confirm {
      margin: 4px 8px 6px; padding: 10px 12px;
      background: var(--surface2); border: 1px solid rgba(248,113,113,.35);
      border-radius: 8px; font-size: 12px; color: var(--ink);
    }
    .del-confirm-text { margin-bottom: 8px; line-height: 1.4; }
    .del-confirm-text strong { color: var(--red); }
    .del-confirm-btns { display: flex; gap: 6px; }
    .del-confirm-btns button {
      flex: 1; padding: 5px 0; border-radius: 6px; border: none;
      font: 600 12px/1 inherit; cursor: pointer; transition: opacity .15s;
    }
    .del-confirm-btns button:hover { opacity: .85; }
    .del-yes { background: var(--red); color: #fff; }
    .del-no  { background: var(--surface); color: var(--ink); border: 1px solid var(--border) !important; }

    .sidebar-new {
      margin: 0 12px 10px; padding: 9px 14px; border-radius: 8px;
      background: var(--surface); border: 1px dashed var(--border);
      color: var(--muted); font-size: 13px; cursor: pointer;
      display: flex; align-items: center; gap: 8px;
      transition: background .15s, color .15s, border-color .15s;
    }
    .sidebar-new:hover { background: var(--surface2); color: var(--ink); border-color: var(--accent); }

    /* Settings panel */
    .sidebar-settings {
      border-top: 1px solid var(--border2); padding: 14px 16px;
    }
    .setting-row { margin-bottom: 12px; }
    .setting-row:last-child { margin-bottom: 0; }
    .setting-label {
      display: flex; justify-content: space-between; align-items: center;
      font-size: 12px; color: var(--muted); margin-bottom: 6px;
    }
    .setting-val { color: var(--ink); font-weight: 600; font-variant-numeric: tabular-nums; }
    input[type=range] {
      width: 100%; accent-color: var(--accent); cursor: pointer;
      height: 4px; appearance: none; background: var(--border);
      border-radius: 2px; outline: none;
    }
    input[type=range]::-webkit-slider-thumb {
      appearance: none; width: 14px; height: 14px; border-radius: 50%;
      background: var(--accent); cursor: pointer;
      box-shadow: 0 0 0 3px rgba(99,102,241,.25);
    }
    select {
      width: 100%; padding: 7px 10px; border-radius: 8px;
      background: var(--surface2); border: 1px solid var(--border);
      color: var(--ink); font: inherit; outline: none; font-size: 13px;
    }
    select:focus { border-color: var(--accent); }
    .sys-prompt-toggle {
      display: flex; align-items: center; gap: 8px;
      font-size: 12px; color: var(--muted); cursor: pointer;
      padding: 6px 0; user-select: none;
    }
    .sys-prompt-toggle:hover { color: var(--ink); }
    .arrow { transition: transform .2s; display: inline-block; }
    .arrow.open { transform: rotate(90deg); }
    textarea.sys-prompt {
      width: 100%; height: 80px; resize: none; margin-top: 8px;
      background: var(--surface2); border: 1px solid var(--border);
      border-radius: 8px; padding: 8px 10px;
      font: 12px/1.5 ui-monospace, "Cascadia Code", Menlo, monospace;
      color: var(--ink); outline: none;
    }
    textarea.sys-prompt:focus { border-color: var(--accent); }

    /* ── MAIN ────────────────────────────────────── */
    #main {
      flex: 1; display: flex; flex-direction: column;
      height: 100vh; overflow: hidden;
    }

    /* Top bar */
    #topbar {
      display: flex; align-items: center; justify-content: space-between;
      padding: 12px 20px; border-bottom: 1px solid var(--border2);
      background: var(--sidebar); flex-shrink: 0; gap: 12px;
    }
    #topbar-title { font-weight: 600; font-size: 14px; color: var(--ink); flex: 1; }
    .badge {
      display: inline-flex; align-items: center; gap: 5px;
      padding: 4px 10px; border-radius: 999px; font-size: 11px; font-weight: 600;
      letter-spacing: .04em; border: 1px solid;
    }
    .badge-local {
      background: rgba(52,211,153,.08); border-color: rgba(52,211,153,.25); color: var(--green);
    }
    .badge-model {
      background: rgba(99,102,241,.1); border-color: rgba(99,102,241,.3); color: var(--accent2);
    }
    .badge-dot { width: 6px; height: 6px; border-radius: 50%; background: currentColor; }

    /* Chat area */
    #chat {
      flex: 1; overflow-y: auto; padding: 24px 0;
      display: flex; flex-direction: column; gap: 0;
    }
    #chat::-webkit-scrollbar { width: 6px; }
    #chat::-webkit-scrollbar-track { background: transparent; }
    #chat::-webkit-scrollbar-thumb { background: var(--border); border-radius: 3px; }

    .msg-row {
      display: flex; padding: 6px 24px; gap: 14px;
      transition: background .1s;
    }
    .msg-row:hover { background: rgba(255,255,255,.02); }
    .msg-row.user { flex-direction: row-reverse; }

    .avatar {
      width: 30px; height: 30px; border-radius: 8px; flex-shrink: 0;
      display: flex; align-items: center; justify-content: center;
      font-size: 14px; font-weight: 700; margin-top: 2px;
    }
    .avatar-user {
      background: linear-gradient(135deg, #6366f1, #8b5cf6); color: #fff;
    }
    .avatar-assistant {
      background: linear-gradient(135deg, #0ea5e9, #6366f1); color: #fff;
    }
    /* Цвета аватаров для конкретных моделей */
    .avatar-gigachat  { background: linear-gradient(135deg, #7c3aed, #db2777); color: #fff; }
    .avatar-phi       { background: linear-gradient(135deg, #0369a1, #0ea5e9); color: #fff; }
    .avatar-qwen      { background: linear-gradient(135deg, #047857, #10b981); color: #fff; }
    .avatar-gemma     { background: linear-gradient(135deg, #b45309, #f59e0b); color: #fff; }
    .avatar-llama     { background: linear-gradient(135deg, #7f1d1d, #ef4444); color: #fff; }
    .avatar-mistral   { background: linear-gradient(135deg, #1e3a5f, #3b82f6); color: #fff; }
    .avatar-glm       { background: linear-gradient(135deg, #0f766e, #06b6d4); color: #fff; }

    .msg-body { flex: 1; min-width: 0; max-width: 780px; }
    .msg-row.user .msg-body { text-align: right; }

    .msg-meta {
      font-size: 11px; color: var(--muted); margin-bottom: 4px;
      display: flex; align-items: center; gap: 8px;
    }
    .msg-row.user .msg-meta { justify-content: flex-end; }
    .msg-name { font-weight: 600; color: var(--muted); }

    .msg-bubble {
      display: inline-block; max-width: 100%;
      padding: 12px 16px; border-radius: var(--radius);
      line-height: 1.7; word-break: break-word;
      text-align: left;
    }
    .bubble-user {
      background: var(--surface2); border: 1px solid var(--border);
      color: var(--ink);
    }
    .bubble-assistant {
      background: transparent; padding-left: 0; padding-right: 0;
    }

    /* Markdown rendering */
    .md p { margin: 0 0 .75em; }
    .md p:last-child { margin-bottom: 0; }
    .md strong { font-weight: 700; color: var(--ink); }
    .md em { font-style: italic; color: #c4b5fd; }
    .md h1,.md h2,.md h3,.md h4 {
      font-weight: 700; margin: 1em 0 .4em; color: var(--ink); line-height: 1.3;
    }
    .md h1 { font-size: 1.35em; } .md h2 { font-size: 1.2em; }
    .md h3 { font-size: 1.05em; } .md h4 { font-size: .95em; }
    .md ul,.md ol { padding-left: 1.4em; margin: .5em 0; }
    .md li { margin: .25em 0; }
    .md blockquote {
      border-left: 3px solid var(--accent); margin: .75em 0;
      padding: .4em 1em; color: var(--muted); font-style: italic;
      background: rgba(99,102,241,.06); border-radius: 0 6px 6px 0;
    }
    .md hr { border: none; border-top: 1px solid var(--border); margin: 1em 0; }
    .md a { color: var(--accent2); text-decoration: none; border-bottom: 1px solid rgba(129,140,248,.3); }
    .md a:hover { border-bottom-color: var(--accent2); }
    .md code:not(.block-code) {
      font: 12.5px/1.4 ui-monospace, "Cascadia Code", Menlo, monospace;
      background: var(--code-bg); color: #a5f3fc;
      padding: 2px 6px; border-radius: 5px; border: 1px solid var(--border);
    }
    .code-block {
      margin: .8em 0; border-radius: 10px; overflow: hidden;
      border: 1px solid var(--border); background: var(--code-bg);
    }
    .code-header {
      display: flex; align-items: center; justify-content: space-between;
      padding: 7px 14px; background: rgba(255,255,255,.04);
      border-bottom: 1px solid var(--border);
    }
    .code-lang { font-size: 11px; color: var(--muted); font-weight: 600; text-transform: uppercase; letter-spacing: .06em; }
    .copy-btn {
      font-size: 11px; padding: 3px 9px; border-radius: 5px;
      background: var(--surface); border: 1px solid var(--border);
      color: var(--muted); cursor: pointer; transition: all .15s;
    }
    .copy-btn:hover { background: var(--surface2); color: var(--ink); }
    .copy-btn.copied { color: var(--green); border-color: rgba(52,211,153,.3); }
    .code-block pre {
      margin: 0; padding: 14px 16px; overflow-x: auto;
      font: 13px/1.6 ui-monospace, "Cascadia Code", Menlo, monospace;
      color: #e2e8f0;
    }
    /* syntax highlight tokens */
    .tok-kw  { color: #c084fc; font-weight: 600; }
    .tok-str { color: #86efac; }
    .tok-num { color: #fb923c; }
    .tok-cmt { color: #4b5563; font-style: italic; }
    .tok-fn  { color: #7dd3fc; }
    .tok-op  { color: #f9a8d4; }
    .tok-cls { color: #fde68a; }

    .md table { border-collapse: collapse; width: 100%; margin: .75em 0; font-size: 13px; }
    .md th { background: var(--surface2); font-weight: 700; padding: 7px 12px; text-align: left; border: 1px solid var(--border); }
    .md td { padding: 6px 12px; border: 1px solid var(--border); }
    .md tr:nth-child(even) td { background: rgba(255,255,255,.02); }

    /* Cursor blink */
    .cursor {
      display: inline-block; width: 2px; height: 1em;
      background: var(--accent2); border-radius: 1px;
      animation: blink .9s steps(1) infinite; vertical-align: text-bottom;
      margin-left: 2px;
    }
    @keyframes blink { 50% { opacity: 0; } }

    /* Action row below message */
    .msg-actions {
      display: flex; gap: 6px; margin-top: 6px; opacity: 0; transition: opacity .15s;
    }
    .msg-row:hover .msg-actions { opacity: 1; }
    .msg-row.user .msg-actions { justify-content: flex-end; }
    .act-btn {
      font-size: 11px; padding: 3px 8px; border-radius: 5px;
      background: none; border: 1px solid var(--border);
      color: var(--muted); cursor: pointer; transition: all .15s;
    }
    .act-btn:hover { background: var(--surface); color: var(--ink); }

    /* Empty state */
    #empty {
      flex: 1; display: flex; flex-direction: column;
      align-items: center; justify-content: center;
      gap: 14px; padding: 40px 24px; text-align: center;
    }
    .empty-icon {
      width: 64px; height: 64px; border-radius: 16px;
      background: linear-gradient(135deg, rgba(99,102,241,.2), rgba(168,85,247,.2));
      border: 1px solid rgba(99,102,241,.3);
      display: flex; align-items: center; justify-content: center; font-size: 28px;
    }
    .empty-title { font-size: 20px; font-weight: 700; color: var(--ink); }
    .empty-sub { color: var(--muted); font-size: 14px; max-width: 320px; line-height: 1.6; }
    .suggestions { display: flex; flex-wrap: wrap; gap: 8px; justify-content: center; margin-top: 6px; }
    .sug {
      padding: 8px 14px; border-radius: 999px; border: 1px solid var(--border);
      background: var(--surface); color: var(--muted); font-size: 12px;
      cursor: pointer; transition: all .15s;
    }
    .sug:hover { background: var(--surface2); color: var(--ink); border-color: var(--accent); }

    /* Composer */
    #composer {
      flex-shrink: 0; padding: 12px 20px 16px;
      background: var(--sidebar); border-top: 1px solid var(--border2);
    }
    .composer-inner {
      display: flex; gap: 10px; align-items: flex-end;
      background: var(--surface); border: 1px solid var(--border);
      border-radius: 14px; padding: 10px 12px;
      transition: border-color .15s;
    }
    .composer-inner:focus-within { border-color: var(--accent); }
    #prompt {
      flex: 1; resize: none; border: none; outline: none;
      background: transparent; color: var(--ink); font: 14px/1.6 inherit;
      min-height: 24px; max-height: 160px; overflow-y: auto;
    }
    #prompt::placeholder { color: var(--muted); }
    #sendBtn {
      width: 36px; height: 36px; border-radius: 9px; flex-shrink: 0;
      background: var(--accent); border: none; cursor: pointer;
      display: flex; align-items: center; justify-content: center;
      transition: all .15s; color: #fff;
    }
    #sendBtn:hover:not(:disabled) { background: #4f46e5; transform: scale(1.05); }
    #sendBtn:disabled { opacity: .45; cursor: wait; transform: none; }
    #sendBtn svg { width: 16px; height: 16px; }
    .composer-footer {
      display: flex; align-items: center; justify-content: space-between;
      padding: 6px 2px 0; gap: 10px;
    }
    .composer-hint { font-size: 11px; color: var(--muted); }
    .composer-hint kbd {
      font: 10px ui-monospace, monospace; background: var(--surface);
      border: 1px solid var(--border); border-radius: 4px; padding: 1px 5px;
    }
    #genStatus {
      display: none; align-items: center; gap: 6px;
      font-size: 12px; color: var(--muted);
    }
    #genStatus.on { display: flex; }
    .gen-dots { display: flex; gap: 3px; }
    .gen-dot {
      width: 5px; height: 5px; border-radius: 50%;
      background: var(--accent2); opacity: .3;
      animation: dotpulse 1.2s ease-in-out infinite;
    }
    .gen-dot:nth-child(2) { animation-delay: .2s; }
    .gen-dot:nth-child(3) { animation-delay: .4s; }
    @keyframes dotpulse { 0%,80%,100% { opacity: .3; transform: scale(.8); } 40% { opacity: 1; transform: scale(1); } }

    /* Model picker */
    .model-picker-wrap { position: relative; }
    .badge-model { cursor: pointer; user-select: none; transition: background .15s; }
    .badge-model:hover { background: rgba(99,102,241,.2); }
    #modelPanel {
      display: none; position: absolute; top: calc(100% + 8px); right: 0;
      width: 340px; background: var(--surface); border: 1px solid var(--border);
      border-radius: 12px; box-shadow: 0 8px 32px rgba(0,0,0,.5);
      z-index: 100; overflow: hidden;
    }
    #modelPanel.open { display: block; }
    .mp-head {
      padding: 12px 14px 10px; border-bottom: 1px solid var(--border);
      font-size: 12px; font-weight: 700; color: var(--muted);
      text-transform: uppercase; letter-spacing: .07em;
      display: flex; align-items: center; justify-content: space-between;
    }
    .mp-close { background: none; border: none; color: var(--muted); cursor: pointer; font-size: 16px; padding: 0 4px; }
    .mp-close:hover { color: var(--ink); }
    .mp-list { max-height: 320px; overflow-y: auto; }
    .mp-item {
      padding: 10px 14px; display: flex; align-items: center; gap: 10px;
      cursor: pointer; transition: background .12s; border-bottom: 1px solid var(--border2);
    }
    .mp-item:last-child { border-bottom: none; }
    .mp-item:hover { background: var(--surface2); }
    .mp-item.active { background: rgba(99,102,241,.1); cursor: default; }
    .mp-dot {
      width: 8px; height: 8px; border-radius: 50%; flex-shrink: 0;
      background: var(--border);
    }
    .mp-item.active .mp-dot { background: var(--green); }
    .mp-info { flex: 1; min-width: 0; }
    .mp-name { font-size: 13px; color: var(--ink); white-space: nowrap; overflow: hidden; text-overflow: ellipsis; }
    .mp-size { font-size: 11px; color: var(--muted); margin-top: 2px; }
    .mp-badge {
      font-size: 10px; padding: 2px 7px; border-radius: 999px;
      background: rgba(52,211,153,.1); border: 1px solid rgba(52,211,153,.25);
      color: var(--green); font-weight: 600;
    }
    .mp-switch-btn {
      font-size: 11px; padding: 4px 10px; border-radius: 6px;
      background: var(--accent); border: none; color: #fff;
      cursor: pointer; white-space: nowrap; flex-shrink: 0;
    }
    .mp-switch-btn:hover { background: #4f46e5; }
    /* Switching overlay */
    #switchOverlay {
      display: none; position: fixed; inset: 0; z-index: 200;
      background: rgba(0,0,0,.65); backdrop-filter: blur(4px);
      align-items: center; justify-content: center;
    }
    #switchOverlay.on { display: flex; }
    .switch-card {
      background: var(--surface); border: 1px solid var(--border);
      border-radius: 16px; padding: 28px 32px; text-align: center;
      min-width: 280px;
    }
    .switch-spinner {
      width: 40px; height: 40px; border: 3px solid var(--border);
      border-top-color: var(--accent); border-radius: 50%;
      animation: spin 1s linear infinite; margin: 0 auto 16px;
    }
    @keyframes spin { to { transform: rotate(360deg); } }
    .switch-title { font-size: 15px; font-weight: 700; margin-bottom: 6px; }
    .switch-msg { font-size: 13px; color: var(--muted); }
    .switch-error { color: var(--red); }

    /* Scrollbar */
    #sessionList::-webkit-scrollbar { width: 4px; }
    #sessionList::-webkit-scrollbar-thumb { background: var(--border); border-radius: 2px; }

    /* Responsive */
    @media (max-width: 640px) {
      #sidebar { display: none; }
    }
  </style>
</head>
<body>

<!-- SIDEBAR -->
<aside id="sidebar">
  <div class="sidebar-head">
    <div class="logo">🤖</div>
    <div>
      <div class="logo-text">GigaChat</div>
      <div class="logo-sub">Local inference</div>
    </div>
  </div>

  <div class="sidebar-section">
    <div class="sidebar-label">Сессии</div>
  </div>
  <div id="sessionList"></div>
  <button class="sidebar-new" id="newSessBtn">
    <span style="font-size:16px">＋</span> Новый чат
  </button>

  <div class="sidebar-settings">
    <div class="setting-row">
      <div class="setting-label">
        <span>Температура</span>
        <span class="setting-val" id="tempVal">0.2</span>
      </div>
      <input type="range" id="tempRange" min="0" max="1" step="0.05" value="0.2">
    </div>
    <div class="setting-row">
      <div class="setting-label"><span>Макс. токенов</span></div>
      <select id="maxTokens">
        <option value="256">256</option>
        <option value="512">512</option>
        <option value="768" selected>768</option>
        <option value="1200">1200</option>
        <option value="1600">1600</option>
        <option value="2400">2400</option>
        <option value="3200">3200</option>
      </select>
    </div>
    <div class="setting-row">
      <div class="sys-prompt-toggle" id="sysToggle">
        <span class="arrow" id="sysArrow">▶</span>
        <span>Системный промпт</span>
      </div>
      <div id="sysPromptWrap" style="display:none">
        <textarea class="sys-prompt" id="sysPrompt" placeholder="Необязательно..."></textarea>
      </div>
    </div>
  </div>
</aside>

<!-- MAIN -->
<div id="main">
  <div id="topbar">
    <div id="topbar-title">Новый чат</div>
    <div class="model-picker-wrap">
      <span class="badge badge-model" id="modelBadge" title="Сменить модель">
        <span class="badge-dot"></span>
        <span id="modelBadgeLabel">__MODEL_ALIAS__</span>
        <span style="font-size:9px;margin-left:3px;opacity:.7">▼</span>
      </span>
      <div id="modelPanel">
        <div class="mp-head">
          <span>Доступные модели</span>
          <button class="mp-close" id="mpClose">✕</button>
        </div>
        <div class="mp-list" id="mpList">
          <div style="padding:14px;color:var(--muted);font-size:13px">Загрузка...</div>
        </div>
      </div>
    </div>
    <span class="badge badge-local">
      <span class="badge-dot"></span>
      Локально
    </span>

  </div>

  <div id="chat">
    <div id="empty">
      <div class="empty-icon">💬</div>
      <div class="empty-title">Готов к диалогу</div>
      <div class="empty-sub">GigaChat 3 Lightning запущен локально. Никаких запросов в облако.</div>
      <div class="suggestions">
        <span class="sug">Объясни квантование LLM</span>
        <span class="sug">Напиши Python-функцию</span>
        <span class="sug">Переведи текст</span>
        <span class="sug">Придумай идею проекта</span>
      </div>
    </div>
  </div>

  <div id="composer">
    <div class="composer-inner">
      <textarea id="prompt" rows="1" placeholder="Введите сообщение..."></textarea>
      <button id="sendBtn" title="Отправить (Enter)">
        <svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2.2" stroke-linecap="round" stroke-linejoin="round">
          <line x1="22" y1="2" x2="11" y2="13"></line>
          <polygon points="22 2 15 22 11 13 2 9 22 2"></polygon>
        </svg>
      </button>
      <button id="stopBtn" title="Остановить генерацию" style="display:none; width:36px; height:36px; border-radius:9px; flex-shrink:0; background:#ef4444; border:none; cursor:pointer; align-items:center; justify-content:center; color:#fff;">
        <svg viewBox="0 0 24 24" fill="currentColor" width="14" height="14"><rect x="4" y="4" width="16" height="16" rx="2"/></svg>
      </button>
    </div>
    <div class="composer-footer">
      <div id="genStatus">
        <div class="gen-dots"><div class="gen-dot"></div><div class="gen-dot"></div><div class="gen-dot"></div></div>
        <span>Генерация ответа...</span>
      </div>
      <div class="composer-hint"><kbd>Enter</kbd> — отправить &nbsp;·&nbsp; <kbd>Shift+Enter</kbd> — новая строка</div>
    </div>
  </div>
</div>

<!-- Switching overlay -->
<div id="switchOverlay">
  <div class="switch-card">
    <div class="switch-spinner"></div>
    <div class="switch-title" id="switchTitle">Переключение модели</div>
    <div class="switch-msg" id="switchMsg">Пожалуйста, подождите...</div>
  </div>
</div>

<script>
// ── State ─────────────────────────────────────────────────
let sessions = JSON.parse(localStorage.getItem('gc-sessions') || '[]');
let currentId = null;

function newSession(title) {
  const s = { id: Date.now().toString(36) + Math.random().toString(36).slice(2,6), title: title || 'Новый чат', messages: [] };
  sessions.unshift(s);
  saveSessions();
  return s;
}
function currentSession() { return sessions.find(s => s.id === currentId); }
function saveSessions() { localStorage.setItem('gc-sessions', JSON.stringify(sessions)); }

// ── Markdown renderer (inline, no CDN) ───────────────────
function escHtml(s) { return s.replace(/&/g,'&amp;').replace(/</g,'&lt;').replace(/>/g,'&gt;'); }

function highlightCode(code, lang) {
  let s = escHtml(code);
  const kw = /\b(def|class|return|import|from|if|else|elif|for|while|in|not|and|or|is|None|True|False|try|except|finally|with|as|pass|break|continue|lambda|yield|raise|async|await|const|let|var|function|=>|new|this|typeof|instanceof|export|default|extends|interface|type|enum|public|private|protected|static|void|int|str|bool|float|list|dict|tuple|set|fn|mut|use|mod|struct|impl|trait|pub|match|Some|Ok|Err)\b/g;
  const str = /(["'`])((?:\\.|(?!\1)[^\\])*?)\1/g;
  const num = /\b(\d+\.?\d*)\b/g;
  const cmt = /(\/\/[^\n]*|#[^\n]*|\/\*[\s\S]*?\*\/)/g;
  const fn  = /\b([a-zA-Z_]\w*)\s*(?=\()/g;
  // order matters
  s = s.replace(cmt, m => `<span class="tok-cmt">${m}</span>`);
  s = s.replace(str, (m,q,v) => `<span class="tok-str">${escHtml ? m : m}</span>`);
  s = s.replace(num, m => `<span class="tok-num">${m}</span>`);
  s = s.replace(kw, m => `<span class="tok-kw">${m}</span>`);
  s = s.replace(fn, (m,name) => `<span class="tok-fn">${name}</span>(`);
  return s;
}

function renderMd(text) {
  // Code blocks
  text = text.replace(/```(\w*)\n?([\s\S]*?)```/g, (_, lang, code) => {
    const id = 'cb' + Math.random().toString(36).slice(2,8);
    const hi = highlightCode(code.trimEnd(), lang);
    return `<div class="code-block"><div class="code-header"><span class="code-lang">${lang||'code'}</span><button class="copy-btn" onclick="copyCode('${id}',this)">Копировать</button></div><pre id="${id}"><code class="block-code">${hi}</code></pre></div>`;
  });
  // Inline code
  text = text.replace(/`([^`]+)`/g, '<code>$1</code>');
  // Tables
  text = text.replace(/^\|(.+)\|\s*\n\|[-| :]+\|\s*\n((?:\|.+\|\s*\n?)*)/gm, (_, head, body) => {
    const ths = head.split('|').filter(c=>c.trim()).map(c=>`<th>${c.trim()}</th>`).join('');
    const trs = body.trim().split('\n').map(row => {
      const tds = row.split('|').filter(c=>c.trim()).map(c=>`<td>${c.trim()}</td>`).join('');
      return `<tr>${tds}</tr>`;
    }).join('');
    return `<table><thead><tr>${ths}</tr></thead><tbody>${trs}</tbody></table>`;
  });
  // Headings
  text = text.replace(/^#{4} (.+)$/gm, '<h4>$1</h4>');
  text = text.replace(/^#{3} (.+)$/gm, '<h3>$1</h3>');
  text = text.replace(/^#{2} (.+)$/gm, '<h2>$1</h2>');
  text = text.replace(/^# (.+)$/gm, '<h1>$1</h1>');
  // Blockquote
  text = text.replace(/^> (.+)$/gm, '<blockquote>$1</blockquote>');
  // HR
  text = text.replace(/^---+$/gm, '<hr>');
  // Bold / italic
  text = text.replace(/\*\*\*(.+?)\*\*\*/g, '<strong><em>$1</em></strong>');
  text = text.replace(/\*\*(.+?)\*\*/g, '<strong>$1</strong>');
  text = text.replace(/\*(.+?)\*/g, '<em>$1</em>');
  // Links
  text = text.replace(/\[([^\]]+)\]\(([^)]+)\)/g, '<a href="$2" target="_blank" rel="noreferrer">$1</a>');
  // Lists
  text = text.replace(/^(\s*)[-*+] (.+)$/gm, '$1<li>$2</li>');
  text = text.replace(/(<li>.*<\/li>)/gs, m => `<ul>${m}</ul>`);
  text = text.replace(/^(\s*)\d+\. (.+)$/gm, '$1<li>$2</li>');
  // Paragraphs
  text = text.replace(/\n{2,}/g, '\n</p><p>');
  text = '<p>' + text + '</p>';
  text = text.replace(/<p>\s*(<(?:h[1-4]|ul|ol|blockquote|hr|div|table|pre)[^>]*>)/g, '$1');
  text = text.replace(/(<\/(?:h[1-4]|ul|ol|blockquote|hr|div|table|pre)>)\s*<\/p>/g, '$1');
  text = text.replace(/<p>\s*<\/p>/g, '');
  return text;
}

function copyCode(id, btn) {
  const el = document.getElementById(id);
  if (!el) return;
  navigator.clipboard.writeText(el.innerText).then(() => {
    btn.textContent = 'Скопировано ✓';
    btn.classList.add('copied');
    setTimeout(() => { btn.textContent = 'Копировать'; btn.classList.remove('copied'); }, 2000);
  });
}

// ── Session sidebar ───────────────────────────────────────
function renderSidebar() {
  const list = document.getElementById('sessionList');
  list.innerHTML = '';
  sessions.forEach(s => {
    const div = document.createElement('div');
    div.className = 'sess-item' + (s.id === currentId ? ' active' : '');
    div.innerHTML = `
      <span class="sess-title" title="${escHtml(s.title)}">${escHtml(s.title)}</span>
      <div class="sess-actions">
        <button class="sess-btn rename" title="Переименовать">✏️</button>
        <button class="sess-btn del" title="Удалить">✕</button>
      </div>`;

    div.querySelector('.rename').addEventListener('click', e => {
      e.stopPropagation();
      startRename(div, s);
    });
    div.querySelector('.del').addEventListener('click', e => {
      e.stopPropagation();
      showDeleteConfirm(div, s);
    });
    div.addEventListener('click', () => loadSession(s.id));
    list.appendChild(div);
  });

  function showDeleteConfirm(div, s) {
    // Remove any existing confirm dialogs
    document.querySelectorAll('.del-confirm').forEach(el => el.remove());

    const box = document.createElement('div');
    box.className = 'del-confirm';
    box.innerHTML = `
      <div class="del-confirm-text">Удалить чат <strong>"${escHtml(s.title.slice(0,30))}"</strong>?</div>
      <div class="del-confirm-btns">
        <button class="del-yes">Удалить</button>
        <button class="del-no">Отмена</button>
      </div>`;
    div.insertAdjacentElement('afterend', box);

    box.querySelector('.del-yes').addEventListener('click', () => {
      sessions = sessions.filter(x => x.id !== s.id);
      saveSessions();
      if (currentId === s.id) { currentId = sessions[0]?.id || null; loadSession(currentId); }
      renderSidebar();
    });
    box.querySelector('.del-no').addEventListener('click', () => box.remove());

    // Close on outside click
    setTimeout(() => {
      document.addEventListener('click', function handler(e) {
        if (!box.contains(e.target) && !div.contains(e.target)) {
          box.remove();
          document.removeEventListener('click', handler);
        }
      });
    }, 0);
  }

  function startRename(div, s) {
    const titleEl = div.querySelector('.sess-title');
    const input = document.createElement('input');
    input.className = 'sess-title-input';
    input.value = s.title;
    titleEl.replaceWith(input);
    input.focus();
    input.select();

    function commit() {
      const val = input.value.trim();
      if (val) { s.title = val; saveSessions(); }
      if (currentId === s.id) document.getElementById('topbar-title').textContent = s.title;
      renderSidebar();
    }
    input.addEventListener('blur', commit);
    input.addEventListener('keydown', e => {
      if (e.key === 'Enter') { e.preventDefault(); input.blur(); }
      if (e.key === 'Escape') { input.value = s.title; input.blur(); }
    });
  }
}

function loadSession(id) {
  currentId = id;
  const s = currentSession();
  document.getElementById('topbar-title').textContent = s ? s.title : 'Новый чат';
  renderSidebar();
  renderChat();
}

// ── Chat render ───────────────────────────────────────────
const emptyNode = document.getElementById('empty');

// ── Профили моделей ──────────────────────────────────────────
const MODEL_PROFILES = {
  gigachat: { label: 'GigaChat',    letter: 'G', cls: 'avatar-gigachat' },
  phi:      { label: 'Phi',         letter: 'Φ', cls: 'avatar-phi'      },
  qwen:     { label: 'Qwen',        letter: 'Q', cls: 'avatar-qwen'     },
  gemma:    { label: 'Gemma',       letter: 'Ge', cls: 'avatar-gemma'   },
  llama:    { label: 'Llama',       letter: 'L', cls: 'avatar-llama'    },
  mistral:  { label: 'Mistral',     letter: 'M', cls: 'avatar-mistral'  },
  glm:      { label: 'GLM',         letter: 'Z', cls: 'avatar-glm'      },
};

function modelProfile(alias) {
  if (!alias) alias = '';
  const a = alias.toLowerCase();
  for (const [key, p] of Object.entries(MODEL_PROFILES)) {
    if (a.includes(key)) return p;
  }
  // Fallback: первая буква алиаса
  const letter = alias.charAt(0).toUpperCase() || '?';
  return { label: alias || 'AI', letter, cls: 'avatar-assistant' };
}

function friendlyModelName(alias) {
  if (!alias) return 'AI';
  const p = modelProfile(alias);
  // Добавляем размер модели если есть в алиасе (7b, 14b, ...)
  const sizeMatch = alias.match(/(\d+\.?\d*b)/i);
  const size = sizeMatch ? ' ' + sizeMatch[1].toUpperCase() : '';
  // Добавляем квантизацию если есть
  const quantMatch = alias.match(/(q4|q6|q8)[\w]*/i);
  const quant = quantMatch ? ' · ' + quantMatch[0].toUpperCase() : '';
  return p.label + size + quant;
}

function msgRowHtml(m, i) {
  const isUser = m.role === 'user';
  const bubbleCls = isUser ? 'bubble-user' : 'bubble-assistant';
  const content = isUser
    ? `<div class="msg-bubble ${bubbleCls}">${escHtml(m.content)}</div>`
    : `<div class="msg-bubble ${bubbleCls} md" id="bubble-${i}">${renderMd(m.content)}</div>`;
  const time = m.ts ? new Date(m.ts).toLocaleTimeString('ru-RU',{hour:'2-digit',minute:'2-digit'}) : '';

  // Для сообщений ассистента берём профиль из сохранённого алиаса модели
  const profile  = isUser ? null : modelProfile(m.model || '');
  const avatarCls = isUser ? 'avatar-user' : (profile ? profile.cls : 'avatar-assistant');
  const avatarLetter = isUser ? 'В' : (profile ? profile.letter : '?');
  const senderName = isUser ? 'Вы' : friendlyModelName(m.model || '');

  return `
    <div class="msg-row ${m.role}" data-idx="${i}">
      <div class="avatar ${avatarCls}" title="${senderName}">${avatarLetter}</div>
      <div class="msg-body">
        <div class="msg-meta">
          <span class="msg-name">${senderName}</span>
          ${time ? `<span>${time}</span>` : ''}
        </div>
        ${content}
        <div class="msg-actions">
          <button class="act-btn" onclick="copyMsg(${i})">Копировать</button>
        </div>
      </div>
    </div>`;
}

function renderChat() {
  const chatEl = document.getElementById('chat');
  const s = currentSession();
  const msgs = s ? s.messages : [];

  if (!msgs.length) {
    chatEl.innerHTML = '';
    emptyNode.style.display = 'flex';
    chatEl.appendChild(emptyNode);
    return;
  }
  emptyNode.style.display = 'none';
  chatEl.innerHTML = msgs.map((m, i) => msgRowHtml(m, i)).join('');
  chatEl.scrollTop = chatEl.scrollHeight;
}

// Update only the streaming assistant bubble — no full re-render
function updateStreamingBubble(idx, content, withCursor) {
  const el = document.getElementById('bubble-' + idx);
  if (!el) return;
  el.innerHTML = renderMd(content) + (withCursor ? '<span class="cursor"></span>' : '');
  const chatEl = document.getElementById('chat');
  chatEl.scrollTop = chatEl.scrollHeight;
}

function copyMsg(idx) {
  const s = currentSession();
  if (!s) return;
  navigator.clipboard.writeText(s.messages[idx]?.content || '');
}

// ── Streaming send ────────────────────────────────────────
const promptEl   = document.getElementById('prompt');
const sendBtn    = document.getElementById('sendBtn');
const stopBtn    = document.getElementById('stopBtn');
const genStatus  = document.getElementById('genStatus');
let   activeAbort = null;
const tempRange  = document.getElementById('tempRange');
const tempVal    = document.getElementById('tempVal');
const maxTokens  = document.getElementById('maxTokens');
const sysPrompt  = document.getElementById('sysPrompt');

// Restore settings
tempRange.value = localStorage.getItem('gc-temp') || '0.2';
tempVal.textContent = tempRange.value;
const savedTok = localStorage.getItem('gc-max-tokens');
if (savedTok) maxTokens.value = savedTok;

tempRange.addEventListener('input', () => {
  tempVal.textContent = Number(tempRange.value).toFixed(2);
  localStorage.setItem('gc-temp', tempRange.value);
});
maxTokens.addEventListener('change', () => localStorage.setItem('gc-max-tokens', maxTokens.value));

// Auto-resize textarea
promptEl.addEventListener('input', () => {
  promptEl.style.height = 'auto';
  promptEl.style.height = Math.min(promptEl.scrollHeight, 160) + 'px';
});

async function send() {
  const text = promptEl.value.trim();
  if (!text || sendBtn.disabled) return;
  promptEl.value = '';
  promptEl.style.height = 'auto';

  if (!currentId) {
    const s = newSession(text.slice(0, 40));
    currentId = s.id;
  } else {
    const s = currentSession();
    if (s && s.title === 'Новый чат' && !s.messages.length) s.title = text.slice(0, 40);
  }

  const sess = currentSession();
  const sysTxt = sysPrompt.value.trim();
  const apiMsgs = sysTxt ? [{ role: 'system', content: sysTxt }, ...sess.messages] : sess.messages.slice();

  sess.messages.push({ role: 'user', content: text, ts: Date.now() });
  apiMsgs.push({ role: 'user', content: text });
  // Сохраняем алиас активной модели в каждом сообщении ассистента
  sess.messages.push({ role: 'assistant', content: '', ts: Date.now(), model: activeModelAlias, _streaming: true });
  const idx = sess.messages.length - 1;

  document.getElementById('topbar-title').textContent = sess.title;
  renderSidebar();
  renderChat();

  sendBtn.style.display = 'none';
  stopBtn.style.display  = 'flex';
  genStatus.classList.add('on');
  activeAbort = new AbortController();

  try {
    const resp = await fetch('/v1/chat/completions', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      signal: activeAbort.signal,
      body: JSON.stringify({
        model: '__MODEL_ALIAS__',
        messages: apiMsgs,
        temperature: Number(tempRange.value),
        max_tokens: Number(maxTokens.value || 768),
        stream: true
      })
    });

    if (!resp.ok) {
      throw new Error(`HTTP ${resp.status}`);
    }

    const reader = resp.body.getReader();
    const dec = new TextDecoder();
    let buf = '';
    let finished = false;
    outer: while (true) {
      const { value, done } = await reader.read();
      if (done) break;
      buf += dec.decode(value, { stream: true });
      const parts = buf.split('\n\n');
      buf = parts.pop() || '';
      for (const p of parts) {
        const line = p.trim();
        if (!line.startsWith('data:')) continue;
        const payload = line.slice(5).trim();
        if (payload === '[DONE]') { finished = true; break outer; }
        try {
          const d = JSON.parse(payload);
          const delta = d.choices?.[0]?.delta?.content || '';
          if (delta) {
            sess.messages[idx].content += delta;
            updateStreamingBubble(idx, sess.messages[idx].content, true);
          }
        } catch (_) {}
      }
    }
    reader.cancel().catch(() => {});
    if (!sess.messages[idx].content) sess.messages[idx].content = '(пустой ответ)';
  } catch (e) {
    sess.messages[idx].content = 'Ошибка: ' + e.message;
  } finally {
    delete sess.messages[idx]._streaming;
    saveSessions();
    // Final full render to apply markdown properly and remove cursor
    renderChat();
    sendBtn.style.display = 'flex';
    stopBtn.style.display  = 'none';
    genStatus.classList.remove('on');
    activeAbort = null;
    promptEl.focus();
  }
}

stopBtn.addEventListener('click', () => {
  if (activeAbort) activeAbort.abort();
});

sendBtn.addEventListener('click', send);
promptEl.addEventListener('keydown', e => {
  if (e.key === 'Enter' && !e.shiftKey) { e.preventDefault(); send(); }
});

// System prompt toggle
document.getElementById('sysToggle').addEventListener('click', () => {
  const wrap = document.getElementById('sysPromptWrap');
  const arrow = document.getElementById('sysArrow');
  const open = wrap.style.display === 'none';
  wrap.style.display = open ? 'block' : 'none';
  arrow.classList.toggle('open', open);
});

// New session
document.getElementById('newSessBtn').addEventListener('click', () => {
  const s = newSession('Новый чат');
  loadSession(s.id);
});

// Suggestions
document.querySelectorAll('.sug').forEach(el => {
  el.addEventListener('click', () => { promptEl.value = el.textContent; promptEl.focus(); });
});

// ── Model picker ─────────────────────────────────────────
const modelBadge  = document.getElementById('modelBadge');
const modelPanel  = document.getElementById('modelPanel');
const mpList      = document.getElementById('mpList');
const mpClose     = document.getElementById('mpClose');
const switchOverlay = document.getElementById('switchOverlay');
const switchTitle = document.getElementById('switchTitle');
const switchMsg   = document.getElementById('switchMsg');
const modelBadgeLabel = document.getElementById('modelBadgeLabel');

// Единый источник правды для активного алиаса модели
let activeModelAlias = modelBadgeLabel.textContent.trim();

let switchPollTimer = null;

modelBadge.addEventListener('click', async () => {
  const open = modelPanel.classList.toggle('open');
  if (open) await loadModelList();
});
mpClose.addEventListener('click', () => modelPanel.classList.remove('open'));
document.addEventListener('click', e => {
  if (!modelBadge.contains(e.target) && !modelPanel.contains(e.target))
    modelPanel.classList.remove('open');
});

async function loadModelList() {
  mpList.innerHTML = '<div style="padding:14px;color:var(--muted);font-size:13px">Загрузка...</div>';
  try {
    const data = await fetch('/admin/models').then(r => r.json());
    if (!data.models.length) {
      mpList.innerHTML = '<div style="padding:14px;color:var(--muted);font-size:13px">Нет файлов .gguf в папке models/</div>';
      return;
    }
    mpList.innerHTML = data.models.map(m => `
      <div class="mp-item ${m.active ? 'active' : ''}" data-file="${escHtml(m.file)}" data-alias="${escHtml(m.alias)}">
        <div class="mp-dot"></div>
        <div class="mp-info">
          <div class="mp-name" title="${escHtml(m.filename)}">${escHtml(m.filename.replace('.gguf',''))}</div>
          <div class="mp-size">${m.size_mb >= 1024 ? (m.size_mb/1024).toFixed(1)+' GB' : m.size_mb+' MB'}</div>
        </div>
        ${m.active
          ? '<span class="mp-badge">Активна</span>'
          : `<button class="mp-switch-btn">Выбрать</button>`}
      </div>`).join('');
    mpList.querySelectorAll('.mp-switch-btn').forEach(btn => {
      btn.addEventListener('click', () => {
        const item = btn.closest('.mp-item');
        switchModel(item.dataset.file, item.dataset.alias);
      });
    });
  } catch(e) {
    mpList.innerHTML = `<div style="padding:14px;color:var(--red);font-size:13px">Ошибка: ${escHtml(e.message)}</div>`;
  }
}

async function switchModel(file, alias) {
  modelPanel.classList.remove('open');
  switchTitle.textContent = `Переключение на ${alias}`;
  switchMsg.textContent = 'Останавливаю текущую модель...';
  switchOverlay.classList.add('on');

  try {
    await fetch('/admin/switch', {
      method: 'POST',
      headers: {'Content-Type': 'application/json'},
      body: JSON.stringify({model_file: file, alias})
    });
  } catch(e) {
    showSwitchError(e.message);
    return;
  }
  switchPollTimer = setInterval(pollSwitchStatus, 2000);
}

async function pollSwitchStatus() {
  try {
    const s = await fetch('/admin/status').then(r => r.json());
    switchMsg.textContent = s.message || '';
    if (s.phase === 'ready') {
      clearInterval(switchPollTimer);
      modelBadgeLabel.textContent = s.model;
      activeModelAlias = s.model;          // синхронизируем глобал
      switchOverlay.classList.remove('on');
    } else if (s.phase === 'error') {
      clearInterval(switchPollTimer);
      showSwitchError(s.message);
    }
  } catch(_) {}
}

function showSwitchError(msg) {
  switchOverlay.classList.remove('on');
  // Show error inline in badge area briefly
  const old = modelBadgeLabel.textContent;
  modelBadgeLabel.style.color = 'var(--red)';
  modelBadgeLabel.textContent = 'Ошибка!';
  setTimeout(() => { modelBadgeLabel.style.color=''; modelBadgeLabel.textContent = old; }, 3000);
}

// Init
if (!sessions.length) {
  const s = newSession('Новый чат');
  currentId = s.id;
} else {
  currentId = sessions[0].id;
}
renderSidebar();
renderChat();

// Синхронизируем activeModelAlias с сервером при старте
(async () => {
  try {
    const s = await fetch('/admin/status').then(r => r.json());
    if (s.model) {
      activeModelAlias = s.model;
      modelBadgeLabel.textContent = s.model;
    }
  } catch(e) {}
})();
</script>
</body>
</html>""".replace("__MODEL_ALIAS__", MODEL_ALIAS)


class Handler(BaseHTTPRequestHandler):
    def _send_json(self, payload: dict, status: int = 200) -> None:
        data = json.dumps(payload, ensure_ascii=False).encode("utf-8")
        self.send_response(status)
        self.send_header("Content-Type", "application/json; charset=utf-8")
        self.send_header("Content-Length", str(len(data)))
        self.end_headers()
        self.wfile.write(data)

    def _send_html(self, html: str, status: int = 200) -> None:
        data = html.encode("utf-8")
        self.send_response(status)
        self.send_header("Content-Type", "text/html; charset=utf-8")
        self.send_header("Content-Length", str(len(data)))
        self.end_headers()
        self.wfile.write(data)

    def _send_sse_headers(self) -> None:
        self.send_response(200)
        self.send_header("Content-Type", "text/event-stream; charset=utf-8")
        self.send_header("Cache-Control", "no-cache")
        self.send_header("Connection", "keep-alive")
        self.end_headers()

    def _write_sse(self, payload) -> None:
        data = payload if isinstance(payload, str) else json.dumps(payload, ensure_ascii=False)
        self.wfile.write(f"data: {data}\n\n".encode("utf-8"))
        self.wfile.flush()

    def do_GET(self) -> None:
        if self.path in {"/", "/index.html"}:
            self._send_html(UI_HTML)
        elif self.path == "/healthz":
            self._send_json({"status": "ok", "backend": BACKEND_BASE, "model": MODEL_ALIAS})
        elif self.path == "/v1/models":
            self._send_json({
                "object": "list",
                "data": [{"id": MODEL_ALIAS, "object": "model", "created": int(time.time()), "owned_by": "gigachat-localhost"}],
            })
        elif self.path == "/admin/models":
            self._send_json({"models": scan_models(), "current": _current_model_alias})
        elif self.path == "/admin/status":
            self._send_json({**_switch_status, "model": _current_model_alias})
        elif self.path == "/admin/reset":
            _switch_status.update({"phase": "ready", "message": ""})
            self._send_json({"status": "reset", "model": _current_model_alias})
        else:
            self._send_json({"error": "Not found"}, status=404)

    def do_HEAD(self) -> None:
        if self.path in {"/", "/index.html"}:
            data = UI_HTML.encode("utf-8")
            self.send_response(200)
            self.send_header("Content-Type", "text/html; charset=utf-8")
            self.send_header("Content-Length", str(len(data)))
            self.end_headers()
        elif self.path == "/healthz":
            self.send_response(200)
            self.send_header("Content-Type", "application/json; charset=utf-8")
            self.end_headers()
        else:
            self.send_response(404)
            self.end_headers()

    def do_POST(self) -> None:
        if self.path == "/admin/switch":
            length = int(self.headers.get("Content-Length", "0"))
            body = json.loads(self.rfile.read(length).decode("utf-8"))
            model_file = body.get("model_file", "")
            alias = body.get("alias", "")
            if not model_file or not alias:
                self._send_json({"error": "model_file and alias required"}, status=400)
                return
            if _switch_status["phase"] not in {"ready", "error"}:
                self._send_json({"error": "Switch already in progress"}, status=409)
                return
            # Помечаем СРАЗУ до запуска потока — иначе race condition на 409
            _switch_status.update({"phase": "switching", "message": f"Инициализация {alias}..."})
            switch_backend_async(model_file, alias)
            self._send_json({"status": "switching", "alias": alias})
            return

        if self.path != "/v1/chat/completions":
            self._send_json({"error": "Not found"}, status=404)
            return

        length = int(self.headers.get("Content-Length", "0"))
        raw_body = self.rfile.read(length)
        try:
            body = json.loads(raw_body.decode("utf-8"))
        except json.JSONDecodeError:
            self._send_json({"error": "Invalid JSON"}, status=400)
            return

        msgs = body.get("messages", [])
        if not isinstance(msgs, list):
            self._send_json({"error": "messages must be a list"}, status=400)
            return

        temperature = float(body.get("temperature", 0.2))
        max_tokens = int(body.get("max_tokens", 768))
        stream = bool(body.get("stream", False))

        if stream:
            self._handle_stream(msgs, temperature, max_tokens)
        else:
            self._handle_sync(msgs, temperature, max_tokens)

    def _handle_sync(self, msgs, temperature, max_tokens):
        try:
            prompt = render_prompt(msgs)
            text, finish_reason, usage = backend_completion(prompt, temperature, max_tokens)
        except urllib.error.URLError as exc:
            self._send_json({"error": f"Backend unavailable: {exc}"}, status=502)
            return
        except Exception as exc:
            self._send_json({"error": f"Proxy error: {exc}"}, status=500)
            return

        self._send_json({
            "id": f"chatcmpl-{uuid.uuid4().hex[:12]}",
            "object": "chat.completion",
            "created": int(time.time()),
            "model": MODEL_ALIAS,
            "choices": [{
                "index": 0,
                "finish_reason": finish_reason,
                "message": {"role": "assistant", "content": text},
            }],
            "usage": usage,
        })

    def _handle_stream(self, msgs, temperature, max_tokens):
        try:
            sys.stderr.write(f"[stream] msgs={len(msgs)} last_role={msgs[-1].get('role') if msgs else '-'} last_content={repr(msgs[-1].get('content','')[:60]) if msgs else '-'}\n")
            sys.stderr.flush()
            prompt = render_prompt(msgs)
            with backend_completion_stream(prompt, temperature, max_tokens) as response:
                self._send_sse_headers()
                raw_accum = ""
                clean_accum = ""
                chunk_id = f"chatcmpl-{uuid.uuid4().hex[:12]}"

                for raw_line in response:
                    line = raw_line.decode("utf-8", errors="ignore").strip()
                    if not line.startswith("data:"):
                        continue
                    payload = line[5:].strip()
                    if payload == "[DONE]":
                        self._write_sse("[DONE]")
                        return
                    try:
                        data = json.loads(payload)
                    except json.JSONDecodeError:
                        continue

                    choice = data.get("choices", [{}])[0]
                    text_piece = choice.get("text", "")
                    finish_reason = choice.get("finish_reason")
                    usage = data.get("usage", {})

                    raw_accum += text_piece
                    cleaned = clean_text(raw_accum)
                    delta = ""
                    if cleaned.startswith(clean_accum):
                        delta = cleaned[len(clean_accum):]
                    elif cleaned != clean_accum:
                        delta = cleaned
                    clean_accum = cleaned

                    if delta:
                        self._write_sse({
                            "id": chunk_id,
                            "object": "chat.completion.chunk",
                            "created": int(time.time()),
                            "model": MODEL_ALIAS,
                            "choices": [{"index": 0, "delta": {"role": "assistant", "content": delta}, "finish_reason": None}],
                        })

                    if finish_reason is not None:
                        self._write_sse({
                            "id": chunk_id,
                            "object": "chat.completion.chunk",
                            "created": int(time.time()),
                            "model": MODEL_ALIAS,
                            "choices": [{"index": 0, "delta": {}, "finish_reason": finish_reason}],
                            "usage": usage,
                        })
                        self._write_sse("[DONE]")
                        return
                # Backend closed stream without explicit [DONE] — send it now
                self._write_sse("[DONE]")
        except urllib.error.URLError as exc:
            self._send_json({"error": f"Backend unavailable: {exc}"}, status=502)
        except Exception as exc:
            self._send_json({"error": f"Proxy error: {exc}"}, status=500)

    def log_message(self, fmt: str, *args) -> None:
        sys.stderr.write("%s [%s] %s\n" % (self.address_string(), self.log_date_time_string(), fmt % args))


def main() -> int:
    server = ThreadingHTTPServer((HOST, PORT), Handler)
    print(f"proxy: http://{HOST}:{PORT}", flush=True)
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        pass
    finally:
        server.server_close()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
