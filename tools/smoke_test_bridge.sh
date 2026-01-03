#!/usr/bin/env bash

set -euo pipefail

PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
LOG="${LOG:-/tmp/cosyvoice_bridge.log}"
WAV_OUT="${WAV_OUT:-/tmp/cosyvoice_smoke.wav}"
PORT="${PORT:-3000}"
PROMPT_AUDIO="${PROMPT_AUDIO:-asset/zero_shot_prompt.wav}"
PROMPT_TEXT="${PROMPT_TEXT:-You are a helpful assistant.<|endofprompt|>Greetings, how are you today?}"
TEXT="${TEXT:-Hello from the CosyVoice bridge server smoke test. This is a longer sample to validate end-to-end synthesis and WAV output integrity.}"

cd "$PROJECT_ROOT"

if [ -f "$PROJECT_ROOT/.env" ]; then
    set -a
    # shellcheck disable=SC1091
    . "$PROJECT_ROOT/.env"
    set +a
fi

if [ -n "${LD_LIBRARY_PATH_EXTRA:-}" ]; then
    LD_LIBRARY_PATH_EXTRA_ABS="$PROJECT_ROOT/${LD_LIBRARY_PATH_EXTRA}"
    case ":${LD_LIBRARY_PATH:-}:" in
        *":$LD_LIBRARY_PATH_EXTRA_ABS:"*) ;;
        *) export LD_LIBRARY_PATH="$LD_LIBRARY_PATH_EXTRA_ABS${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}" ;;
    esac
fi

rm -f "$LOG" "$WAV_OUT"

COSYVOICE_ORT_USE_TRT=1 ./rust/start-server.sh >"$LOG" 2>&1 &
SERVER_PID=$!

cleanup() {
    kill "$SERVER_PID" >/dev/null 2>&1 || true
}
trap cleanup EXIT

sleep 35

curl -s -o /tmp/cosyvoice_health.json -w "%{http_code}" \
    "http://127.0.0.1:${PORT}/health" | rg -q "^200$" \
    || { echo "Health check failed"; exit 1; }

curl -s -o "$WAV_OUT" -w "%{http_code}" \
    -H "Content-Type: application/json" \
    -d "{\"text\":\"${TEXT}\",\"prompt_audio\":\"${PROMPT_AUDIO}\",\"prompt_text\":\"${PROMPT_TEXT}\"}" \
    "http://127.0.0.1:${PORT}/synthesize" | rg -q "^200$" \
    || { echo "Synthesize request failed"; exit 1; }

pixi run python - <<PY
import wave
path = "${WAV_OUT}"
with wave.open(path, 'rb') as w:
    frames = w.getnframes()
    rate = w.getframerate()
    dur = frames / float(rate) if rate else 0
    print(f"channels={w.getnchannels()} rate={rate} frames={frames} dur_s={dur:.2f}")
PY

rg -n "tensorrt" "$LOG" || true

echo "Smoke test complete: ${WAV_OUT}"
