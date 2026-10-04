"""Song to stems and MIDI pipeline.

Usage:
    python pipeline.py <youtube-url-or-audio-file>

Stages:
    1. Fetch the audio (yt-dlp for URLs) and convert it to WAV with ffmpeg.
    2. Separate it into six stems. In "fine" mode (default) this is a chain:
         BS-Roformer takes the vocals out, Demucs htdemucs_ft takes drums and
         bass out of the instrumental, Demucs htdemucs_6s takes guitar and piano
         out of what is left. In "fast" mode a single htdemucs_6s pass is used.
    3. Split the drums stem into kick, snare, toms, hi-hat, ride and crash
       (MDX23C DrumSep, skipped with --no-drum-parts).
    4. Estimate BPM and key with librosa and write info.txt.
    5. Transcribe bass, guitar, piano and vocals to MIDI with Basic Pitch.
    6. Transcribe drums to MIDI with ADTOF (PyTorch port).

MIDI files are rewritten to carry the estimated BPM, so setting the FL Studio
project to the BPM in info.txt lines the notes up with the audio stems.

Every stage skips its work when its output files already exist, so a failed
run can be restarted with the same command and only the missing parts rerun.

The script re-launches itself inside the main virtual environment (.venv) when
it is started with some other Python, so `python pipeline.py` works from any
shell once setup.ps1 has been run.
"""

import argparse
import datetime
import hashlib
import json
import os
import re
import shutil
import subprocess
import sys
import time
import unicodedata
from pathlib import Path


ROOT = Path(__file__).resolve().parent
OUTPUT_ROOT = ROOT / "output"
MODELS_DIR = ROOT / "models"
MAIN_VENV = ROOT / ".venv"
BASIC_PITCH_VENV = ROOT / ".venv-basicpitch"

STEM_NAMES = ["drums", "bass", "vocals", "guitar", "piano", "other"]
BASIC_PITCH_STEMS = ["bass", "guitar", "piano", "vocals"]
TOTAL_STAGES = 6

# Separation models.
DEMUCS_6S = "htdemucs_6s"
DEMUCS_FT = "htdemucs_ft"
VOCAL_MODEL = "model_bs_roformer_ep_317_sdr_12.9755.ckpt"
DRUM_PARTS_MODEL = "MDX23C-DrumSep-aufr33-jarredou.ckpt"
# Stem names as the DrumSep model reports them, mapped to our file names.
DRUM_PARTS = {
    "Kick": "kick",
    "Snare": "snare",
    "Toms": "toms",
    "HH": "hihat",
    "Ride": "ride",
    "Crash": "crash",
}

# basic_pitch has no __main__ module, so call the same function its console
# script (basic-pitch) points at. sys.argv[0] is "-c" here, and argparse only
# uses it for the usage line. Same trick for audio-separator.
BASIC_PITCH_ENTRY = "from basic_pitch.predict import main; main()"
AUDIO_SEPARATOR_ENTRY = "from audio_separator.utils.cli import main; main()"

SOURCE_MARKER = ".source"
WINDOWS_RESERVED_NAMES = {
    "CON", "PRN", "AUX", "NUL",
    "COM1", "COM2", "COM3", "COM4", "COM5", "COM6", "COM7", "COM8", "COM9",
    "LPT1", "LPT2", "LPT3", "LPT4", "LPT5", "LPT6", "LPT7", "LPT8", "LPT9",
}


# ---------------------------------------------------------------------------
# Small helpers
# ---------------------------------------------------------------------------

class PipelineError(Exception):
    """Raised for errors that should stop the run with a clear message."""


def venv_python(venv_dir):
    """Return the Python executable inside a virtual environment."""
    if sys.platform == "win32":
        return venv_dir / "Scripts" / "python.exe"
    return venv_dir / "bin" / "python"


def print_stage(number, text):
    print("")
    print("=== [%d/%d] %s ===" % (number, TOTAL_STAGES, text), flush=True)


def print_info(text):
    print("    " + text, flush=True)


def print_warning(text):
    print("")
    print("WARNING: " + text, flush=True)
    print("")


def format_elapsed(started):
    seconds = time.time() - started
    return "%.1fs" % seconds


def run_command(command, description, env=None):
    """Run a subprocess, streaming its output. Raises PipelineError on failure."""
    print_info("running: " + " ".join(str(part) for part in command))
    result = subprocess.run([str(part) for part in command], env=env)
    if result.returncode != 0:
        raise PipelineError("%s failed with exit code %d" % (description, result.returncode))


def run_command_capture(command, description):
    """Run a subprocess and return its stdout. Raises PipelineError on failure."""
    result = subprocess.run(
        [str(part) for part in command],
        capture_output=True,
        text=True,
        encoding="utf-8",
        errors="replace",
    )
    if result.returncode != 0:
        details = result.stderr.strip()
        raise PipelineError("%s failed with exit code %d\n%s" % (description, result.returncode, details))
    return result.stdout


def finish_file(temp_path, final_path):
    """Move a finished file into place so partial files never look complete."""
    if final_path.exists():
        final_path.unlink()
    shutil.move(str(temp_path), str(final_path))


def all_exist(paths):
    for path in paths:
        if not path.exists():
            return False
    return True


# ---------------------------------------------------------------------------
# Startup checks
# ---------------------------------------------------------------------------

def ensure_running_in_main_venv():
    """Re-launch this script with the main venv's Python if needed."""
    main_python = venv_python(MAIN_VENV)
    if not main_python.exists():
        raise PipelineError(
            "The main virtual environment is missing (%s).\n"
            "Run setup.ps1 from the stem-pipeline folder first." % MAIN_VENV
        )
    current = Path(sys.executable).resolve()
    if current == main_python.resolve():
        return
    result = subprocess.run([str(main_python), str(Path(__file__).resolve())] + sys.argv[1:])
    sys.exit(result.returncode)


def check_ffmpeg():
    ffmpeg_path = shutil.which("ffmpeg")
    if ffmpeg_path is None:
        raise PipelineError(
            "ffmpeg was not found on PATH.\n"
            "Install it (for example: winget install Gyan.FFmpeg) and open a new terminal."
        )
    print_info("ffmpeg: " + ffmpeg_path)


def check_basic_pitch_env():
    basic_pitch_python = venv_python(BASIC_PITCH_VENV)
    if not basic_pitch_python.exists():
        raise PipelineError(
            "The Basic Pitch virtual environment is missing (%s).\n"
            "Run setup.ps1 from the stem-pipeline folder first." % BASIC_PITCH_VENV
        )
    return basic_pitch_python


def detect_device():
    """Return 'cuda' when PyTorch can see a CUDA GPU, otherwise 'cpu'."""
    try:
        import torch
    except ImportError:
        return "cpu"
    if torch.cuda.is_available():
        return "cuda"
    return "cpu"


def subprocess_env(device):
    """Environment for tools that pick the GPU on their own (audio-separator)."""
    env = dict(os.environ)
    if device == "cpu":
        env["CUDA_VISIBLE_DEVICES"] = ""
    return env


# ---------------------------------------------------------------------------
# Source handling and naming
# ---------------------------------------------------------------------------

def is_url(text):
    lowered = text.lower()
    return lowered.startswith("http://") or lowered.startswith("https://")


def sanitize_song_name(title):
    """Turn a song title into a safe folder name on Windows, macOS and Linux."""
    cleaned = unicodedata.normalize("NFKC", title)
    cleaned = re.sub(r'[<>:"/\\|?*\x00-\x1f]', " ", cleaned)
    cleaned = re.sub(r"\s+", " ", cleaned).strip(" .")
    cleaned = cleaned[:80].rstrip(" .")
    if cleaned == "":
        cleaned = "untitled"
    if cleaned.upper() in WINDOWS_RESERVED_NAMES:
        cleaned = cleaned + "_song"
    return cleaned


def fetch_url_title(url):
    """Ask yt-dlp for the video title without downloading anything."""
    output = run_command_capture(
        [sys.executable, "-m", "yt_dlp", "--no-playlist", "--skip-download", "--dump-single-json", url],
        "yt-dlp metadata fetch",
    )
    info = json.loads(output)
    title = info.get("title")
    if title is None or title.strip() == "":
        title = info.get("id", "untitled")
    return title


def resolve_source(argument):
    """Return (kind, value, title) for a URL or a local audio file."""
    if is_url(argument):
        title = fetch_url_title(argument)
        return "url", argument, title
    file_path = Path(argument).expanduser().resolve()
    if not file_path.is_file():
        raise PipelineError("Input file not found: %s" % file_path)
    return "file", str(file_path), file_path.stem


def choose_song_folder(song_name, source_value):
    """Pick output/<song-name>, adding a short hash if that folder belongs to another source."""
    folder = OUTPUT_ROOT / song_name
    marker = folder / SOURCE_MARKER
    if folder.exists() and marker.exists():
        recorded = marker.read_text(encoding="utf-8").strip()
        if recorded != source_value:
            digest = hashlib.sha1(source_value.encode("utf-8")).hexdigest()[:6]
            folder = OUTPUT_ROOT / (song_name + "-" + digest)
            marker = folder / SOURCE_MARKER
    folder.mkdir(parents=True, exist_ok=True)
    if not marker.exists():
        marker.write_text(source_value + "\n", encoding="utf-8")
    return folder


# ---------------------------------------------------------------------------
# Stage 1: original.wav
# ---------------------------------------------------------------------------

def download_audio(url, work_dir):
    """Download the best audio stream with yt-dlp and return the downloaded file."""
    for old in work_dir.glob("source.*"):
        old.unlink()
    template = str(work_dir / "source.%(ext)s")
    run_command(
        [sys.executable, "-m", "yt_dlp", "--no-playlist", "-f", "bestaudio/best", "-o", template, url],
        "yt-dlp download",
    )
    downloaded = sorted(work_dir.glob("source.*"))
    if len(downloaded) == 0:
        raise PipelineError("yt-dlp finished but no downloaded file was found in %s" % work_dir)
    return downloaded[0]


def convert_to_wav(source_path, wav_path):
    """Convert any audio file to 44.1 kHz stereo 16-bit WAV with ffmpeg."""
    temp_path = wav_path.with_name(wav_path.name + ".part.wav")
    run_command(
        [
            "ffmpeg", "-y", "-hide_banner", "-loglevel", "error", "-stats",
            "-i", source_path,
            "-vn", "-ac", "2", "-ar", "44100", "-c:a", "pcm_s16le",
            temp_path,
        ],
        "ffmpeg conversion",
    )
    finish_file(temp_path, wav_path)


def stage_prepare_audio(source_kind, source_value, song_dir, work_dir):
    print_stage(1, "Preparing original.wav")
    original = song_dir / "original.wav"
    if original.exists():
        print_info("original.wav already exists, skipping")
        return original
    started = time.time()
    if source_kind == "url":
        print_info("downloading audio with yt-dlp")
        source_path = download_audio(source_value, work_dir)
    else:
        source_path = Path(source_value)
    print_info("converting to WAV with ffmpeg")
    convert_to_wav(source_path, original)
    print_info("done in " + format_elapsed(started))
    return original


# ---------------------------------------------------------------------------
# Separation building blocks
# ---------------------------------------------------------------------------

def run_demucs(model, input_wav, out_dir, device, shifts):
    """Run one Demucs model. Returns the folder holding <stem>.wav files."""
    if out_dir.exists():
        shutil.rmtree(out_dir)
    run_command(
        [
            sys.executable, "-m", "demucs",
            "-n", model,
            "-d", device,
            "--shifts", shifts,
            "-o", out_dir,
            "--filename", "{stem}.{ext}",
            input_wav,
        ],
        "Demucs " + model,
    )
    produced_dir = out_dir / model
    for name in ["drums", "bass", "vocals", "other"]:
        if not (produced_dir / (name + ".wav")).exists():
            raise PipelineError("Demucs %s did not produce %s.wav" % (model, name))
    return produced_dir


def run_audio_separator(model, input_wav, out_dir, output_names, device, description):
    """Run one audio-separator model with fixed output file names (no extension)."""
    if out_dir.exists():
        shutil.rmtree(out_dir)
    out_dir.mkdir(parents=True)
    MODELS_DIR.mkdir(exist_ok=True)
    run_command(
        [
            sys.executable, "-c", AUDIO_SEPARATOR_ENTRY,
            input_wav,
            "--model_filename", model,
            "--model_file_dir", MODELS_DIR,
            "--output_dir", out_dir,
            "--output_format", "WAV",
            "--custom_output_names", json.dumps(output_names),
        ],
        description,
        env=subprocess_env(device),
    )
    for file_name in output_names.values():
        if not (out_dir / (file_name + ".wav")).exists():
            raise PipelineError("%s did not produce %s.wav" % (description, file_name))


def mix_wavs(inputs, output):
    """Sum several WAV files sample by sample and write the result."""
    import numpy as np
    import soundfile as sf

    total = None
    sample_rate = None
    for path in inputs:
        data, rate = sf.read(str(path), dtype="float32", always_2d=True)
        if sample_rate is None:
            sample_rate = rate
        if rate != sample_rate:
            raise PipelineError("Sample rate mismatch while mixing %s" % path)
        if total is None:
            total = data
        else:
            length = max(len(total), len(data))
            if len(total) < length:
                total = np.pad(total, ((0, length - len(total)), (0, 0)))
            if len(data) < length:
                data = np.pad(data, ((0, length - len(data)), (0, 0)))
            total = total + data
    total = np.clip(total, -1.0, 1.0)
    sf.write(str(output), total, sample_rate, subtype="PCM_16")


# ---------------------------------------------------------------------------
# Stage 2: stems
# ---------------------------------------------------------------------------

def separate_fast(original, stems_dir, work_dir, device, shifts):
    """One htdemucs_6s pass on the full mix."""
    produced_dir = run_demucs(DEMUCS_6S, original, work_dir / "demucs6s", device, shifts)
    for name in STEM_NAMES:
        if not (produced_dir / (name + ".wav")).exists():
            raise PipelineError("Demucs did not produce %s.wav" % name)
    for name in STEM_NAMES:
        finish_file(produced_dir / (name + ".wav"), stems_dir / (name + ".wav"))


def separate_fine(original, stems_dir, work_dir, device, shifts):
    """Vocals with BS-Roformer, then drums and bass with htdemucs_ft, then
    guitar and piano with htdemucs_6s on what is left. Each sub-step keeps its
    output in the work folder so a resumed run does not redo it."""

    # Step A: vocals out.
    roformer_dir = work_dir / "roformer"
    vocals_wav = roformer_dir / "vocals.wav"
    instrumental_wav = roformer_dir / "instrumental.wav"
    if all_exist([vocals_wav, instrumental_wav]):
        print_info("vocal split already done, reusing")
    else:
        print_info("step A: separating vocals with BS-Roformer")
        run_audio_separator(
            VOCAL_MODEL, original, roformer_dir,
            {"Vocals": "vocals", "Instrumental": "instrumental"},
            device, "BS-Roformer vocal separation",
        )

    # Step B: drums and bass out of the instrumental.
    ft_dir = work_dir / "demucs_ft"
    ft_other = work_dir / "other_after_ft.wav"
    ft_outputs = [ft_dir / DEMUCS_FT / "drums.wav", ft_dir / DEMUCS_FT / "bass.wav", ft_other]
    if all_exist(ft_outputs):
        print_info("drums and bass split already done, reusing")
    else:
        print_info("step B: separating drums and bass with Demucs " + DEMUCS_FT)
        produced = run_demucs(DEMUCS_FT, instrumental_wav, ft_dir, device, shifts)
        # Whatever this model still thinks is vocals goes back into "other".
        mix_wavs([produced / "other.wav", produced / "vocals.wav"], ft_other)

    # Step C: guitar and piano out of what is left.
    six_dir = work_dir / "demucs6s"
    six_outputs = [six_dir / DEMUCS_6S / "guitar.wav", six_dir / DEMUCS_6S / "piano.wav"]
    if all_exist(six_outputs):
        print_info("guitar and piano split already done, reusing")
    else:
        print_info("step C: separating guitar and piano with Demucs " + DEMUCS_6S)
        run_demucs(DEMUCS_6S, ft_other, six_dir, device, shifts)

    # Assemble the six stems. Residues from the last model fold into "other".
    six_produced = six_dir / DEMUCS_6S
    final_other = work_dir / "other_final.wav"
    mix_wavs(
        [six_produced / "other.wav", six_produced / "drums.wav",
         six_produced / "bass.wav", six_produced / "vocals.wav"],
        final_other,
    )
    finish_file(vocals_wav, stems_dir / "vocals.wav")
    finish_file(ft_dir / DEMUCS_FT / "drums.wav", stems_dir / "drums.wav")
    finish_file(ft_dir / DEMUCS_FT / "bass.wav", stems_dir / "bass.wav")
    finish_file(six_produced / "guitar.wav", stems_dir / "guitar.wav")
    finish_file(six_produced / "piano.wav", stems_dir / "piano.wav")
    finish_file(final_other, stems_dir / "other.wav")


def stage_separate_stems(original, stems_dir, work_dir, device, mode, shifts):
    print_stage(2, "Separating stems (%s mode, %s)" % (mode, device))
    stems_dir.mkdir(exist_ok=True)
    if all_exist([stems_dir / (name + ".wav") for name in STEM_NAMES]):
        print_info("all six stems already exist, skipping")
        return
    started = time.time()
    if mode == "fast":
        separate_fast(original, stems_dir, work_dir, device, shifts)
    else:
        separate_fine(original, stems_dir, work_dir, device, shifts)
    print_info("done in " + format_elapsed(started))


# ---------------------------------------------------------------------------
# Stage 3: drum parts
# ---------------------------------------------------------------------------

def stage_drum_parts(stems_dir, work_dir, device, enabled):
    """Split drums.wav into kick, snare, toms, hihat, ride and crash."""
    print_stage(3, "Splitting drums into parts (DrumSep)")
    if not enabled:
        print_info("drum parts disabled, skipping")
        return
    parts_dir = stems_dir / "drums"
    targets = [parts_dir / (name + ".wav") for name in DRUM_PARTS.values()]
    if all_exist(targets):
        print_info("drum parts already exist, skipping")
        return
    started = time.time()
    scratch = work_dir / "drumsep"
    run_audio_separator(
        DRUM_PARTS_MODEL, stems_dir / "drums.wav", scratch, DRUM_PARTS,
        device, "DrumSep drum part separation",
    )
    parts_dir.mkdir(exist_ok=True)
    for file_name in DRUM_PARTS.values():
        finish_file(scratch / (file_name + ".wav"), parts_dir / (file_name + ".wav"))
    shutil.rmtree(scratch, ignore_errors=True)
    print_info("done in " + format_elapsed(started))


# ---------------------------------------------------------------------------
# Stage 4: BPM, key and info.txt
# ---------------------------------------------------------------------------

# Krumhansl-Schmuckler key profiles, index 0 is the tonic.
MAJOR_PROFILE = [6.35, 2.23, 3.48, 2.33, 4.38, 4.09, 2.52, 5.19, 2.39, 3.66, 2.29, 2.88]
MINOR_PROFILE = [6.33, 2.68, 3.52, 5.38, 2.60, 3.53, 2.54, 4.75, 3.98, 2.69, 3.34, 3.17]
PITCH_CLASSES = ["C", "C#", "D", "D#", "E", "F", "F#", "G", "G#", "A", "A#", "B"]


def estimate_key(chroma_mean):
    """Return the best matching key name, for example 'A minor'."""
    import numpy as np

    best_name = "unknown"
    best_score = -2.0
    for tonic in range(12):
        for mode_name, profile in (("major", MAJOR_PROFILE), ("minor", MINOR_PROFILE)):
            rotated = np.roll(np.array(profile), tonic)
            score = np.corrcoef(chroma_mean, rotated)[0, 1]
            if score > best_score:
                best_score = score
                best_name = PITCH_CLASSES[tonic] + " " + mode_name
    return best_name


def analyze_audio(original):
    """Return (bpm, key_name) estimated with librosa."""
    import librosa
    import numpy as np

    audio, sample_rate = librosa.load(str(original), sr=22050, mono=True)
    tempo, _beats = librosa.beat.beat_track(y=audio, sr=sample_rate)
    bpm = float(np.atleast_1d(tempo)[0])
    chroma = librosa.feature.chroma_cqt(y=audio, sr=sample_rate)
    key_name = estimate_key(chroma.mean(axis=1))
    return bpm, key_name


def read_bpm_from_info(info_path):
    """Read the BPM back from an existing info.txt so a resumed run can use it."""
    for line in info_path.read_text(encoding="utf-8").splitlines():
        if line.startswith("BPM:"):
            return float(line.split(":", 1)[1].strip())
    raise PipelineError("info.txt exists but has no BPM line. Delete it and run again.")


def stage_info(original, song_dir, source_kind, source_value, song_title, device, mode, drum_parts):
    """Write info.txt and return the BPM for the MIDI stages."""
    print_stage(4, "Estimating BPM and key, writing info.txt")
    info_path = song_dir / "info.txt"
    if info_path.exists():
        bpm = read_bpm_from_info(info_path)
        print_info("info.txt already exists, skipping (BPM %.1f)" % bpm)
        return bpm
    started = time.time()
    bpm, key_name = analyze_audio(original)
    processed = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    if source_kind == "url":
        source_label = "Source URL"
    else:
        source_label = "Source path"
    if mode == "fast":
        separation_label = "fast (Demucs %s)" % DEMUCS_6S
    else:
        separation_label = "fine (BS-Roformer vocals, Demucs %s drums and bass, Demucs %s guitar and piano)" % (DEMUCS_FT, DEMUCS_6S)
    if drum_parts:
        drum_parts_label = "yes (" + DRUM_PARTS_MODEL + ")"
    else:
        drum_parts_label = "no"
    lines = [
        "Song: " + song_title,
        source_label + ": " + source_value,
        "Processed: " + processed,
        "BPM: %.1f" % bpm,
        "Key: " + key_name,
        "Separation: " + separation_label,
        "Drum parts: " + drum_parts_label,
        "Device: " + device,
    ]
    temp_path = info_path.with_name("info.part.txt")
    temp_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    finish_file(temp_path, info_path)
    print_info("BPM %.1f, key %s" % (bpm, key_name))
    print_info("done in " + format_elapsed(started))
    return bpm


# ---------------------------------------------------------------------------
# MIDI tempo
# ---------------------------------------------------------------------------

def set_midi_tempo(midi_path, bpm):
    """Rewrite a MIDI file so its tempo is bpm while note times stay in seconds.

    Basic Pitch and ADTOF write every file at 120 BPM. FL Studio places notes
    by beat, so without this the notes drift against the audio as soon as the
    project tempo is set to the real BPM."""
    import pretty_midi

    source = pretty_midi.PrettyMIDI(str(midi_path))
    rewritten = pretty_midi.PrettyMIDI(resolution=source.resolution, initial_tempo=bpm)
    rewritten.instruments = source.instruments
    rewritten.write(str(midi_path))


# ---------------------------------------------------------------------------
# Stage 5: Basic Pitch
# ---------------------------------------------------------------------------

def stage_basic_pitch(stems_dir, midi_dir, work_dir, basic_pitch_python, bpm):
    print_stage(5, "Transcribing melodic stems with Basic Pitch")
    midi_dir.mkdir(exist_ok=True)
    for name in BASIC_PITCH_STEMS:
        target = midi_dir / (name + ".mid")
        if target.exists():
            print_info(name + ".mid already exists, skipping")
            continue
        started = time.time()
        print_info("transcribing " + name)
        stem_wav = stems_dir / (name + ".wav")
        scratch = work_dir / ("basic_pitch_" + name)
        if scratch.exists():
            shutil.rmtree(scratch)
        scratch.mkdir(parents=True)
        run_command(
            [basic_pitch_python, "-c", BASIC_PITCH_ENTRY, scratch, stem_wav],
            "Basic Pitch on " + name,
        )
        produced = scratch / (stem_wav.stem + "_basic_pitch.mid")
        if not produced.exists():
            raise PipelineError("Basic Pitch did not produce %s" % produced)
        set_midi_tempo(produced, bpm)
        finish_file(produced, target)
        shutil.rmtree(scratch, ignore_errors=True)
        print_info("%s.mid done in %s" % (name, format_elapsed(started)))


# ---------------------------------------------------------------------------
# Stage 6: ADTOF drums
# ---------------------------------------------------------------------------

def stage_drums_midi(stems_dir, midi_dir, work_dir, device, bpm):
    """Transcribe drums. Returns True on success, False when skipped with a warning."""
    print_stage(6, "Transcribing drums with ADTOF (%s)" % device)
    target = midi_dir / "drums.mid"
    if target.exists():
        print_info("drums.mid already exists, skipping")
        return True
    started = time.time()
    temp_midi = work_dir / "drums.part.mid"
    if temp_midi.exists():
        temp_midi.unlink()
    try:
        run_command(
            [
                sys.executable, "-m", "adtof_pytorch.cli",
                "--audio", stems_dir / "drums.wav",
                "--out", temp_midi,
                "--device", device,
            ],
            "ADTOF drum transcription",
        )
    except PipelineError as error:
        print_warning(
            "Drum transcription failed, so drums.mid was not written.\n"
            "The rest of the output is complete. Details: %s" % error
        )
        return False
    if not temp_midi.exists():
        print_warning("ADTOF finished but did not write drums.mid. Skipping drum transcription.")
        return False
    set_midi_tempo(temp_midi, bpm)
    finish_file(temp_midi, target)
    print_info("drums.mid done in " + format_elapsed(started))
    return True


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def parse_arguments():
    parser = argparse.ArgumentParser(
        description="Split a song into stems and MIDI for FL Studio.",
    )
    parser.add_argument("source", help="YouTube URL or path to a local audio file")
    parser.add_argument(
        "--mode",
        choices=["fine", "fast"],
        default="fine",
        help="fine: Roformer vocals plus two Demucs passes (default). fast: one htdemucs_6s pass.",
    )
    parser.add_argument(
        "--shifts",
        type=int,
        default=1,
        help="Demucs random shifts. Higher is slower and slightly cleaner (default 1).",
    )
    parser.add_argument(
        "--no-drum-parts",
        action="store_true",
        help="Skip splitting the drums stem into kick, snare, toms, hihat, ride and crash",
    )
    parser.add_argument(
        "--cpu",
        action="store_true",
        help="Force CPU even when a CUDA GPU is available",
    )
    return parser.parse_args()


def run_pipeline(arguments):
    started = time.time()
    print("stem-pipeline")
    check_ffmpeg()
    basic_pitch_python = check_basic_pitch_env()
    if arguments.cpu:
        device = "cpu"
    else:
        device = detect_device()
    print_info("device: " + device)
    drum_parts_enabled = not arguments.no_drum_parts

    source_kind, source_value, song_title = resolve_source(arguments.source)
    song_name = sanitize_song_name(song_title)
    song_dir = choose_song_folder(song_name, source_value)
    stems_dir = song_dir / "stems"
    midi_dir = song_dir / "midi"
    work_dir = song_dir / ".work"
    work_dir.mkdir(exist_ok=True)
    print_info("song: " + song_title)
    print_info("output folder: " + str(song_dir))

    original = stage_prepare_audio(source_kind, source_value, song_dir, work_dir)
    stage_separate_stems(original, stems_dir, work_dir, device, arguments.mode, arguments.shifts)
    stage_drum_parts(stems_dir, work_dir, device, drum_parts_enabled)
    bpm = stage_info(
        original, song_dir, source_kind, source_value, song_title,
        device, arguments.mode, drum_parts_enabled,
    )
    stage_basic_pitch(stems_dir, midi_dir, work_dir, basic_pitch_python, bpm)
    drums_ok = stage_drums_midi(stems_dir, midi_dir, work_dir, device, bpm)

    shutil.rmtree(work_dir, ignore_errors=True)

    print("")
    print("Finished in " + format_elapsed(started))
    print("Output: " + str(song_dir))
    print("Set the FL Studio project tempo to %.1f BPM (see info.txt) so MIDI lines up with the stems." % bpm)
    if not drums_ok:
        print("Note: drums.mid is missing, see the warning above.")


def main():
    try:
        ensure_running_in_main_venv()
        arguments = parse_arguments()
        run_pipeline(arguments)
    except PipelineError as error:
        print("")
        print("ERROR: " + str(error), file=sys.stderr)
        sys.exit(1)
    except KeyboardInterrupt:
        print("")
        print("Interrupted. Run the same command again to resume.", file=sys.stderr)
        sys.exit(130)


if __name__ == "__main__":
    main()
