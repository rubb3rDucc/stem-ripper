# Creates the two virtual environments used by pipeline.py and installs
# the pinned packages. Run from the stem-pipeline folder:
#     .\setup.ps1
# Requires Python 3.10 (py -3.10 must work) and git on PATH.

$ErrorActionPreference = "Stop"
$root = $PSScriptRoot

function Find-Python310 {
    $candidates = @(
        { & py -3.10 -c "import sys; print(sys.executable)" 2>$null },
        { & python3.10 -c "import sys; print(sys.executable)" 2>$null },
        { & python -c "import sys; assert sys.version_info[:2] == (3, 10); print(sys.executable)" 2>$null }
    )
    foreach ($candidate in $candidates) {
        try {
            $found = & $candidate
            if ($LASTEXITCODE -eq 0 -and $found) { return $found.Trim() }
        } catch { }
    }
    return $null
}

$python = Find-Python310
if (-not $python) {
    Write-Error "Python 3.10 was not found. Install it with: winget install Python.Python.3.10"
    exit 1
}
Write-Host "Using Python 3.10 at $python"

if (-not (Get-Command git -ErrorAction SilentlyContinue)) {
    Write-Error "git was not found on PATH. It is needed to install ADTOF-pytorch from GitHub."
    exit 1
}

if (-not (Get-Command ffmpeg -ErrorAction SilentlyContinue)) {
    Write-Warning "ffmpeg is not on PATH. pipeline.py will refuse to run until it is installed (winget install Gyan.FFmpeg)."
}

function New-Env($name, $requirements) {
    $envDir = Join-Path $root $name
    $envPython = Join-Path $envDir "Scripts\python.exe"
    if (-not (Test-Path $envPython)) {
        Write-Host "Creating $name"
        & $python -m venv $envDir
    } else {
        Write-Host "$name already exists, updating packages"
    }
    & $envPython -m pip install --upgrade pip
    & $envPython -m pip install -r (Join-Path $root $requirements)
    if ($LASTEXITCODE -ne 0) {
        Write-Error "pip install failed for $name"
        exit 1
    }
}

New-Env ".venv" "requirements-main.txt"
New-Env ".venv-basicpitch" "requirements-basicpitch.txt"

Write-Host ""
Write-Host "Checking installs"
& (Join-Path $root ".venv\Scripts\python.exe") -c "import torch, demucs, yt_dlp, librosa, adtof_pytorch, audio_separator; print('main env ok, torch', torch.__version__, 'cuda available:', torch.cuda.is_available())"
& (Join-Path $root ".venv-basicpitch\Scripts\python.exe") -c "import basic_pitch, onnxruntime; print('basic pitch env ok')"
Write-Host ""
Write-Host "Setup complete. Try: python pipeline.py <youtube-url-or-audio-file>"
