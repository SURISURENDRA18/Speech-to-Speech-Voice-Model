#!/usr/bin/env bash

set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
REPO_DIR="$(cd -- "$SCRIPT_DIR/.." && pwd)"
RUNTIME_DIR="${S2S_RUNTIME_DIR:-$REPO_DIR/.local-stack}"
VENV_BIN="${S2S_VENV_BIN:-$REPO_DIR/.venv/bin}"

if [[ -f "$REPO_DIR/.env" ]]; then
    set -a
    # shellcheck disable=SC1091
    source "$REPO_DIR/.env"
    set +a
fi

resolve_repo_path() {
    case "$1" in
        /*) printf '%s\n' "$1" ;;
        *) printf '%s/%s\n' "$REPO_DIR" "$1" ;;
    esac
}

LLM_MODEL="$(resolve_repo_path "${S2S_LLM_MODEL:-models/llm/Qwen3.5-4B-UD-Q4_K_XL.gguf}")"
TTS_TALKER="$(resolve_repo_path "${S2S_TTS_TALKER:-models/qwen3-tts/qwen-talker-1.7b-customvoice-Q8_0.gguf}")"
TTS_CODEC="$(resolve_repo_path "${S2S_TTS_CODEC:-models/qwen3-tts/qwen-tokenizer-12hz-Q8_0.gguf}")"
SMART_TURN_MODEL="$(resolve_repo_path "${S2S_SMART_TURN_MODEL:-models/smart-turn-v3/smart-turn-v3.2-cpu.onnx}")"

STT_DEVICE="${S2S_STT_DEVICE:-cuda}"
TTS_SPEAKER="${S2S_TTS_SPEAKER:-aiden}"
LLM_GPU_LAYERS="${S2S_LLM_GPU_LAYERS:-99}"
LLM_CONTEXT_SIZE="${S2S_LLM_CONTEXT_SIZE:-8192}"

LLM_HOST="${S2S_LLM_HOST:-127.0.0.1}"
LLM_PORT="${S2S_LLM_PORT:-8080}"
PIPELINE_HOST="${S2S_PIPELINE_HOST:-127.0.0.1}"
PIPELINE_PORT="${S2S_PIPELINE_PORT:-8765}"
DEMO_HOST="${S2S_DEMO_HOST:-127.0.0.1}"
DEMO_PORT="${S2S_DEMO_PORT:-7860}"

# Realtime URL the visitor's browser connects to. Set it when the stack is behind an HTTPS proxy, for
# example wss://voice.example.com/v1/realtime. The default works only on this machine.
PUBLIC_URL="${S2S_PUBLIC_URL:-ws://$PIPELINE_HOST:$PIPELINE_PORT/v1/realtime}"

# Speech backends: STT is parakeet-tdt or whisper, TTS is qwen3 or svara.
STT="${S2S_STT:-parakeet-tdt}"
WHISPER_MODEL="${S2S_WHISPER_MODEL:-openai/whisper-large-v3-turbo}"
WHISPER_LANGUAGES="${S2S_WHISPER_LANGUAGES:-hi,en,bn,mr,te,ta,gu,kn,ml,pa}"
TTS="${S2S_TTS:-qwen3}"
SVARA_MODEL="$(resolve_repo_path "${S2S_SVARA_MODEL:-models/svara/svara-tts-v1.Q4_K_M.gguf}")"
SVARA_HOST="${S2S_SVARA_HOST:-127.0.0.1}"
SVARA_PORT="${S2S_SVARA_PORT:-8081}"
SVARA_GENDER="${S2S_SVARA_GENDER:-female}"
SVARA_LANGUAGE="${S2S_SVARA_LANGUAGE:-hi}"
SVARA_NORMALIZER_LANGUAGES="${S2S_SVARA_NORMALIZER_LANGUAGES:-hi,en}"

# Seconds to wait for the speech pipeline. The first start may download models.
PIPELINE_START_TIMEOUT="${S2S_PIPELINE_START_TIMEOUT:-180}"

case "$STT" in parakeet-tdt | whisper) ;; *) echo "S2S_STT must be parakeet-tdt or whisper" >&2; exit 1 ;; esac
case "$TTS" in qwen3 | svara) ;; *) echo "S2S_TTS must be qwen3 or svara" >&2; exit 1 ;; esac

mkdir -p "$RUNTIME_DIR"

usage() {
    cat <<'EOF'
Usage: scripts/local_stack.sh {run|start|stop|restart|status|logs}

Commands:
  run      Start and supervise the stack in this terminal; Ctrl+C stops it.
  start    Start llama-server, speech-to-speech, and the browser demo.
  stop     Stop all services.
  restart  Stop and start the stack.
  status   Show process and endpoint status.
  logs     Follow all service logs (Ctrl+C stops following, not the services).

Model paths and ports can be overridden with S2S_* environment variables.
EOF
}

require_command() {
    if ! command -v "$1" >/dev/null 2>&1; then
        echo "Missing required command: $1" >&2
        exit 1
    fi
}

require_file() {
    if [[ ! -f "$1" ]]; then
        echo "Missing required model file: $1" >&2
        exit 1
    fi
}

pid_file() {
    printf '%s/%s.pid\n' "$RUNTIME_DIR" "$1"
}

log_file() {
    printf '%s/%s.log\n' "$RUNTIME_DIR" "$1"
}

read_pid() {
    local file
    file="$(pid_file "$1")"
    [[ -f "$file" ]] && tr -d '[:space:]' < "$file"
}

process_matches() {
    local service="$1"
    local pid
    pid="$(read_pid "$service")"
    [[ "$pid" =~ ^[0-9]+$ ]] || return 1
    [[ -r "/proc/$pid/cmdline" ]] || return 1

    local command_line
    command_line="$(tr '\0' ' ' < "/proc/$pid/cmdline")"
    case "$service" in
        llama) [[ "$command_line" == *"llama-server"*"$LLM_MODEL"* ]] ;;
        svara) [[ "$command_line" == *"llama-server"*"$SVARA_MODEL"* ]] ;;
        pipeline) [[ "$command_line" == *"speech-to-speech"*"serve"* ]] ;;
        demo) [[ "$command_line" == *"uvicorn"*"server:app"* ]] ;;
        *) return 1 ;;
    esac
}

wait_for_http() {
    local name="$1"
    local url="$2"
    local attempts="${3:-120}"

    for ((attempt = 1; attempt <= attempts; attempt++)); do
        if ! process_matches "$name"; then
            echo "$name exited before becoming ready" >&2
            tail -n 30 "$(log_file "$name")" >&2 || true
            return 1
        fi
        if curl --silent --show-error --max-time 1 --output /dev/null "$url" 2>/dev/null; then
            return 0
        fi
        sleep 1
    done

    echo "$name did not become ready at $url" >&2
    echo "Last log lines:" >&2
    tail -n 30 "$(log_file "$name")" >&2 || true
    return 1
}

start_llama() {
    if process_matches llama; then
        echo "llama-server already running (PID $(read_pid llama))"
        return
    fi

    rm -f "$(pid_file llama)"
    echo "Starting llama-server..."
    nohup llama-server \
        -m "$LLM_MODEL" \
        -ngl "$LLM_GPU_LAYERS" \
        -c "$LLM_CONTEXT_SIZE" \
        -np 1 \
        -fa on \
        --host "$LLM_HOST" \
        --port "$LLM_PORT" \
        >"$(log_file llama)" 2>&1 &
    echo "$!" > "$(pid_file llama)"
    wait_for_http llama "http://$LLM_HOST:$LLM_PORT/health"
    echo "llama-server ready on http://$LLM_HOST:$LLM_PORT"
}

start_svara() {
    if process_matches svara; then
        echo "Svara TTS server already running (PID $(read_pid svara))"
        return
    fi

    rm -f "$(pid_file svara)"
    echo "Starting Svara TTS server..."
    # The GGUF names the chat end-of-turn token as EOS, so generation would run past the end of speech.
    # GPU sampling keeps the 157k-token vocabulary from slowing speech below real time.
    nohup llama-server \
        -m "$SVARA_MODEL" \
        -ngl 99 \
        -c 4096 \
        -np 1 \
        -fa on \
        --backend-sampling \
        --override-kv tokenizer.ggml.eos_token_id=int:128258 \
        --host "$SVARA_HOST" \
        --port "$SVARA_PORT" \
        >"$(log_file svara)" 2>&1 &
    echo "$!" > "$(pid_file svara)"
    wait_for_http svara "http://$SVARA_HOST:$SVARA_PORT/health"
    echo "Svara TTS server ready on http://$SVARA_HOST:$SVARA_PORT"
}

start_pipeline() {
    if process_matches pipeline; then
        echo "speech pipeline already running (PID $(read_pid pipeline))"
        return
    fi

    rm -f "$(pid_file pipeline)"
    echo "Starting speech pipeline..."
    local stt_args tts_args
    if [[ "$STT" == whisper ]]; then
        # Live transcription works only with Parakeet. Asking for replies in the detected language
        # keeps the small language model from drifting into English.
        stt_args=(
            --stt whisper
            --stt_model_name "$WHISPER_MODEL"
            --stt_device "$STT_DEVICE"
            --language auto
            --stt_allowed_languages "$WHISPER_LANGUAGES"
            --enable_lang_prompt True
            --enable_live_transcription False
        )
    else
        stt_args=(
            --stt parakeet-tdt
            --parakeet_tdt_device "$STT_DEVICE"
            --enable_live_transcription
        )
    fi
    if [[ "$TTS" == svara ]]; then
        tts_args=(
            --tts svara
            --svara_tts_server_url "http://$SVARA_HOST:$SVARA_PORT"
            --svara_tts_gender "$SVARA_GENDER"
            --svara_tts_default_language "$SVARA_LANGUAGE"
            --svara_tts_normalizer_languages "$SVARA_NORMALIZER_LANGUAGES"
        )
        if [[ -n "${S2S_SVARA_NORMALIZER_CACHE:-}" ]]; then
            tts_args+=(--svara_tts_normalizer_cache_dir "$S2S_SVARA_NORMALIZER_CACHE")
        fi
    else
        tts_args=(
            --tts qwen3
            --qwen3_tts_model_name Qwen/Qwen3-TTS-12Hz-1.7B-CustomVoice
            --qwen3_tts_backend ggml
            --qwen3_tts_gguf_talker_path "$TTS_TALKER"
            --qwen3_tts_gguf_codec_path "$TTS_CODEC"
            --qwen3_tts_speaker "$TTS_SPEAKER"
        )
    fi
    nohup "$VENV_BIN/speech-to-speech" serve \
        --host "$PIPELINE_HOST" \
        --port "$PIPELINE_PORT" \
        "${stt_args[@]}" \
        --llm_backend chat-completions \
        --model_name local \
        --responses_api_base_url "http://$LLM_HOST:$LLM_PORT/v1" \
        --responses_api_api_key local \
        --responses_api_disable_thinking True \
        "${tts_args[@]}" \
        --smart_turn_model_path "$SMART_TURN_MODEL" \
        >"$(log_file pipeline)" 2>&1 &
    echo "$!" > "$(pid_file pipeline)"
    wait_for_http pipeline "http://$PIPELINE_HOST:$PIPELINE_PORT/" "$PIPELINE_START_TIMEOUT"
    echo "speech pipeline ready on ws://$PIPELINE_HOST:$PIPELINE_PORT/v1/realtime"
}

start_demo() {
    if process_matches demo; then
        echo "browser demo already running (PID $(read_pid demo))"
        return
    fi

    rm -f "$(pid_file demo)"
    echo "Starting browser demo..."
    nohup env SPEECH_TO_SPEECH_URL="$PUBLIC_URL" \
        "$VENV_BIN/uvicorn" --app-dir demo server:app \
        --host "$DEMO_HOST" \
        --port "$DEMO_PORT" \
        >"$(log_file demo)" 2>&1 &
    echo "$!" > "$(pid_file demo)"
    wait_for_http demo "http://$DEMO_HOST:$DEMO_PORT/"
    echo "browser demo ready on http://$DEMO_HOST:$DEMO_PORT"
}

start_stack() {
    require_command curl
    require_command llama-server
    require_file "$VENV_BIN/speech-to-speech"
    require_file "$VENV_BIN/uvicorn"
    require_file "$LLM_MODEL"
    if [[ "$TTS" == svara ]]; then
        require_file "$SVARA_MODEL"
    else
        require_file "$TTS_TALKER"
        require_file "$TTS_CODEC"
    fi
    require_file "$SMART_TURN_MODEL"

    cd "$REPO_DIR"
    start_llama
    [[ "$TTS" != svara ]] || start_svara
    start_pipeline
    start_demo
    echo
    echo "Stack is ready: http://localhost:$DEMO_PORT"
}

stop_service() {
    local service="$1"
    local pid

    if ! process_matches "$service"; then
        rm -f "$(pid_file "$service")"
        echo "$service is not running"
        return
    fi

    pid="$(read_pid "$service")"
    echo "Stopping $service (PID $pid)..."
    kill "$pid"
    for _ in {1..20}; do
        if ! kill -0 "$pid" 2>/dev/null; then
            rm -f "$(pid_file "$service")"
            return
        fi
        sleep 0.5
    done

    echo "$service did not stop cleanly; terminating it forcefully" >&2
    kill -KILL "$pid" 2>/dev/null || true
    rm -f "$(pid_file "$service")"
}

stop_stack() {
    stop_service demo
    stop_service pipeline
    stop_service svara
    stop_service llama
}

run_stack() {
    local shutdown_requested=0

    request_shutdown() {
        shutdown_requested=1
    }

    trap request_shutdown INT TERM
    trap stop_stack EXIT
    start_stack
    echo "Supervising the stack. Press Ctrl+C to stop everything."

    while ((shutdown_requested == 0)); do
        if ! process_matches llama || ! process_matches pipeline || ! process_matches demo ||
            { [[ "$TTS" == svara ]] && ! process_matches svara; }; then
            echo "A stack service exited unexpectedly." >&2
            return 1
        fi
        sleep 2
    done
}

show_service_status() {
    local service="$1"
    local endpoint="$2"
    if process_matches "$service"; then
        local pid
        pid="$(read_pid "$service")"
        if curl --silent --show-error --max-time 1 --output /dev/null "$endpoint" 2>/dev/null; then
            printf '%-10s running  PID %-8s %s\n' "$service" "$pid" "$endpoint"
        else
            printf '%-10s starting PID %-8s %s\n' "$service" "$pid" "$endpoint"
        fi
    else
        printf '%-10s stopped\n' "$service"
    fi
}

show_status() {
    show_service_status llama "http://$LLM_HOST:$LLM_PORT/health"
    [[ "$TTS" != svara ]] || show_service_status svara "http://$SVARA_HOST:$SVARA_PORT/health"
    show_service_status pipeline "http://$PIPELINE_HOST:$PIPELINE_PORT/"
    show_service_status demo "http://$DEMO_HOST:$DEMO_PORT/"
}

follow_logs() {
    local logs=("$(log_file llama)" "$(log_file pipeline)" "$(log_file demo)")
    [[ "$TTS" != svara ]] || logs+=("$(log_file svara)")
    touch "${logs[@]}"
    tail -n 40 -F "${logs[@]}"
}

case "${1:-}" in
    run) run_stack ;;
    start) start_stack ;;
    stop) stop_stack ;;
    restart) stop_stack; start_stack ;;
    status) show_status ;;
    logs) follow_logs ;;
    *) usage; exit 2 ;;
esac
