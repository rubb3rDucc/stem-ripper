#!/usr/bin/env bash
# macOS / Linux equivalent of setup.ps1. Creates the two virtual environments used
# by pipeline.py and installs the pinned packages. Run from the repo folder:
#     ./setup.sh
# Requires Python 3.10 (python3.10 on PATH) and git. On macOS:
#     brew install python@3.10 ffmpeg git
set -euo pipefail
root="$(cd "$(dirname "$0")" && pwd)"

python="$(command -v python3.10 || true)"
if [ -z "$python" ]; then
    echo "Python 3.10 was not found. On macOS install it with: brew install python@3.10" >&2
    exit 1
fi
echo "Using Python 3.10 at $python"

command -v git >/dev/null || { echo "git was not found on PATH. It is needed to install ADTOF-pytorch from GitHub." >&2; exit 1; }
command -v ffmpeg >/dev/null || echo "WARNING: ffmpeg is not on PATH. pipeline.py will refuse to run until it is installed (brew install ffmpeg)." >&2

new_env() {
    local name="$1" requirements="$2"
    local env_dir="$root/$name"
    if [ ! -x "$env_dir/bin/python" ]; then
        echo "Creating $name"
        "$python" -m venv "$env_dir"
    else
        echo "$name already exists, updating packages"
    fi
    "$env_dir/bin/python" -m pip install --upgrade pip
    "$env_dir/bin/python" -m pip install -r "$root/$requirements"
}

new_env ".venv" "requirements-main.txt"
new_env ".venv-basicpitch" "requirements-basicpitch.txt"

echo
echo "Checking installs"
"$root/.venv/bin/python" -c "import torch, demucs, yt_dlp, librosa, adtof_pytorch, audio_separator; print('main env ok, torch', torch.__version__, 'cuda available:', torch.cuda.is_available())"
"$root/.venv-basicpitch/bin/python" -c "import basic_pitch, onnxruntime; print('basic pitch env ok')"
echo
echo "Setup complete. Try: python3 pipeline.py <youtube-url-or-audio-file>"
