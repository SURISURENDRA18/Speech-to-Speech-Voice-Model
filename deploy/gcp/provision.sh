#!/usr/bin/env bash
# Install the voice agent on a fresh Ubuntu 24.04 VM with an NVIDIA driver, or update it.
# Run on the VM as root: sudo provision.sh DOMAIN APP_ARCHIVE
# APP_ARCHIVE is a .tar.gz of this repository that includes uv.lock (see README.md).

set -euo pipefail

DOMAIN="${1:?usage: provision.sh DOMAIN APP_ARCHIVE}"
ARCHIVE="$(realpath "${2:?usage: provision.sh DOMAIN APP_ARCHIVE}")"

LLAMA_TAG=b12722
LLAMA_SHA=98bd142911d8efa1da39819e74bc9443d99c8e0d7125fde435b75e26c253fee3
CUDART_SHA=0735c0748e48494694dc183a5009199eb949723e7a14968578c49f8415185c18
LANGUAGES="hi en bn mr te ta gu kn ml pa"
WHISPER_MODEL=openai/whisper-large-v3

nvidia-smi >/dev/null || { echo "No working NVIDIA driver." >&2; exit 1; }

echo "== Packages"
export DEBIAN_FRONTEND=noninteractive NEEDRESTART_MODE=a
apt-get update -qq
apt-get install -y -qq curl git caddy libportaudio2 libsndfile1 libgomp1 build-essential >/dev/null
[[ -x /usr/local/bin/uv ]] || curl -LsSf https://astral.sh/uv/install.sh | UV_INSTALL_DIR=/usr/local/bin INSTALLER_NO_MODIFY_PATH=1 sh >/dev/null
id s2s >/dev/null 2>&1 || useradd --system --home-dir /var/lib/s2s --shell /usr/sbin/nologin s2s
install -d -o s2s -g s2s -m 750 /var/lib/s2s
install -d /opt/s2s/llama /opt/s2s/models /etc/s2s

echo "== llama.cpp $LLAMA_TAG"
fetch_llama() {
    local name="$1" sha="$2"
    [[ -d "/opt/s2s/llama/$3" ]] && return
    curl -sfL "https://github.com/ggml-org/llama.cpp/releases/download/$LLAMA_TAG/$name.tar.gz" -o "/tmp/$name.tar.gz"
    echo "$sha  /tmp/$name.tar.gz" | sha256sum -c --quiet
    tar xzf "/tmp/$name.tar.gz" -C /opt/s2s/llama
    rm "/tmp/$name.tar.gz"
}
BUILD="llama-$LLAMA_TAG-bin-ubuntu-cuda-12.8-x64"
fetch_llama "$BUILD" "$LLAMA_SHA" "llama-$LLAMA_TAG"
fetch_llama "cudart-$BUILD" "$CUDART_SHA" "cudart-$BUILD"
# The library path is set only for llama-server, so its CUDA 12 does not reach PyTorch's CUDA 13.
cat > /usr/local/bin/llama-server <<EOF
#!/bin/sh
LD_LIBRARY_PATH=/opt/s2s/llama/llama-$LLAMA_TAG:/opt/s2s/llama/cudart-$BUILD exec /opt/s2s/llama/llama-$LLAMA_TAG/llama-server "\$@"
EOF
chmod 755 /usr/local/bin/llama-server

echo "== App"
systemctl stop s2s 2>/dev/null || true
mkdir -p /opt/s2s/app.new
tar xzf "$ARCHIVE" -C /opt/s2s/app.new --no-same-owner
[[ -f /opt/s2s/app.new/uv.lock ]] || { echo "The archive has no uv.lock." >&2; exit 1; }
[[ -d /opt/s2s/app/.venv ]] && mv /opt/s2s/app/.venv /opt/s2s/app.new/.venv
rm -rf /opt/s2s/app
mv /opt/s2s/app.new /opt/s2s/app
cd /opt/s2s/app
export UV_PYTHON_INSTALL_DIR=/opt/s2s/python
uv sync -q --frozen --python 3.11 --no-dev --extra svara --group svara-normalization
chown -R root:root /opt/s2s
chmod -R go-w,a+rX /opt/s2s

echo "== Models"
hf() { uvx -q --from 'huggingface_hub[cli]>=0.34' hf "$@" >/dev/null; }
hf download mradermacher/svara-tts-v1-GGUF svara-tts-v1.Q4_K_M.gguf --revision 97a29a804ce4e369efa7a16bb4cf9e309300aa3e --local-dir /opt/s2s/models/svara
hf download unsloth/Qwen3.5-4B-GGUF Qwen3.5-4B-UD-Q4_K_XL.gguf --revision e87f176479d0855a907a41277aca2f8ee7a09523 --local-dir /opt/s2s/models/llm
hf download pipecat-ai/smart-turn-v3 smart-turn-v3.2-cpu.onnx --revision f766f81d3cfdf7737ac64aad813d91bbfd56bf93 --local-dir /opt/s2s/models/smart-turn-v3
chmod -R a+rX /opt/s2s/models

cat > /etc/s2s/stack.env <<EOF
HOME=/var/lib/s2s
HF_HOME=/var/lib/s2s/huggingface
# Everything is downloaded below, so a restart never needs or picks up anything new.
HF_HUB_OFFLINE=1
TORCH_HOME=/var/lib/s2s/torch
NLTK_DATA=/var/lib/s2s/nltk
S2S_RUNTIME_DIR=/var/lib/s2s/run
S2S_LLM_MODEL=/opt/s2s/models/llm/Qwen3.5-4B-UD-Q4_K_XL.gguf
S2S_SMART_TURN_MODEL=/opt/s2s/models/smart-turn-v3/smart-turn-v3.2-cpu.onnx
S2S_STT=whisper
S2S_WHISPER_MODEL=$WHISPER_MODEL
S2S_WHISPER_LANGUAGES=${LANGUAGES// /,}
S2S_TTS=svara
S2S_SVARA_MODEL=/opt/s2s/models/svara/svara-tts-v1.Q4_K_M.gguf
S2S_SVARA_NORMALIZER_LANGUAGES=${LANGUAGES// /,}
S2S_SVARA_NORMALIZER_CACHE=/var/lib/s2s/normalization
S2S_VOICES=Female,Male
S2S_WEBRTC=0
S2S_PUBLIC_URL=wss://$DOMAIN/v1/realtime
S2S_PIPELINE_START_TIMEOUT=600
EOF

as_s2s() { runuser -u s2s -- env HOME=/var/lib/s2s HF_HOME=/var/lib/s2s/huggingface TORCH_HOME=/var/lib/s2s/torch NLTK_DATA=/var/lib/s2s/nltk "$@"; }
as_s2s .venv/bin/python - <<EOF
from huggingface_hub import snapshot_download

snapshot_download("$WHISPER_MODEL", allow_patterns=["*.json", "*.txt", "model.safetensors"])
snapshot_download("hubertsiuzdak/snac_24khz")
import nltk

for name in ("punkt_tab", "averaged_perceptron_tagger_eng"):
    nltk.download(name, download_dir="/var/lib/s2s/nltk", quiet=True)
import torch

torch.hub.load("snakers4/silero-vad:master", "silero_vad", trust_repo=True, skip_validation=True)
EOF

echo "== Text normalization grammars"
# About a minute each the first time.
for language in $LANGUAGES; do
    [[ -f /var/lib/s2s/normalization/$language/.compiled ]] || echo "$language"
done | as_s2s xargs -r -P "$(nproc)" -I{} .venv/bin/python -m speech_to_speech.TTS.svara_text /var/lib/s2s/normalization {} >/dev/null 2>&1

echo "== Services"
install -m 644 deploy/gcp/s2s.service /etc/systemd/system/s2s.service
sed "s/__DOMAIN__/$DOMAIN/" deploy/gcp/Caddyfile > /etc/caddy/Caddyfile
caddy validate --config /etc/caddy/Caddyfile >/dev/null 2>&1
systemctl daemon-reload
systemctl enable -q s2s caddy
systemctl reload-or-restart caddy
if [[ -f /etc/s2s/admin.env ]]; then
    systemctl restart s2s
    echo "Started. Follow it with: journalctl -fu s2s"
else
    echo "Set the admin login before starting: sudo /opt/s2s/app/deploy/gcp/update_admin.sh EMAIL && sudo systemctl start s2s"
fi
