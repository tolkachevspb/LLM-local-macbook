#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$ROOT_DIR"

if [[ ! -f "config.env" ]]; then
  cp config.env.example config.env
fi
source ./config.env

echo "--- /v1/models ---"
curl -sf "http://${HOST}:${PORT}/v1/models" | python3 -m json.tool 2>/dev/null || echo "(нет ответа)"
echo
echo "--- /healthz ---"
curl -sf "http://${HOST}:${PORT}/healthz" | python3 -m json.tool 2>/dev/null || echo "(нет ответа)"
