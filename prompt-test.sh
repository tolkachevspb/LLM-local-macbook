#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$ROOT_DIR"

if [[ ! -f "config.env" ]]; then
  cp config.env.example config.env
fi
source ./config.env

prompt="${1:-Напиши короткое предложение о пользе локальных LLM.}"

payload="$(PROMPT="${prompt}" MODEL_ALIAS="${MODEL_ALIAS}" python3 -c "
import json, os
print(json.dumps({
    'model': os.environ['MODEL_ALIAS'],
    'messages': [
        {'role': 'system', 'content': 'Ты отвечаешь кратко и по-русски.'},
        {'role': 'user', 'content': os.environ['PROMPT']},
    ],
    'temperature': 0.2,
    'max_tokens': 120,
}, ensure_ascii=False))
")"

echo "Prompt: ${prompt}"
echo "---"
curl -sf "http://${HOST}:${PORT}/v1/chat/completions" \
  -H "Content-Type: application/json" \
  -d "${payload}" | python3 -c "
import json, sys
body = json.load(sys.stdin)
text = body.get('choices', [{}])[0].get('message', {}).get('content', '')
print(text.strip())
"
