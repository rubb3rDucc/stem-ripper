# stem-pipeline

Turns a song into stems plus MIDI files you can drag into FL Studio.

Give it a YouTube URL or a local audio file and it produces:

- six stems (drums, bass, vocals, guitar, piano, other) from a chain of
  BS-Roformer and two Demucs models (see "How the separation works")
- the drums stem split again into kick, snare, toms, hi-hat, ride and crash
- MIDI for bass, guitar, piano and vocals from Basic Pitch
- drum MIDI from ADTOF (PyTorch port)
- BPM and key estimated with librosa, written to `info.txt`, and stamped into
  every MIDI file so the notes line up with the stems in FL Studio

## Requirements

- Windows 10 or 11 (the setup script is PowerShell; the pipeline itself is plain Python)
- Python 3.10. Basic Pitch 0.4.0 only supports Python 3.8 to 3.11 and only uses
  the light ONNX runtime on Windows below 3.11, so both environments use 3.10.
  Install with `winget install Python.Python.3.10`.
- ffmpeg on PATH. Install with `winget install Gyan.FFmpeg`, then open a new terminal.
- git on PATH (ADTOF-pytorch is installed from GitHub).
- Optional: an NVIDIA GPU. The CUDA 12.6 build of PyTorch is installed and used when
  a GPU is visible, otherwise everything runs on the CPU. The GPU build also works on
  machines without a GPU.

## Setup

```powershell
cd stem-pipeline
.\setup.ps1
```

This creates two virtual environments and installs the pinned packages:

| Environment        | Contents                                                             | Requirements file             |
| ------------------ | -------------------------------------------------------------------- | ----------------------------- |
| `.venv`            | yt-dlp, Demucs, audio-separator, PyTorch (CUDA 12.6), librosa, ADTOF | `requirements-main.txt`       |
| `.venv-basicpitch` | Basic Pitch with the ONNX runtime                                    | `requirements-basicpitch.txt` |

Two environments are needed because Basic Pitch 0.4.0 pins an old resampy and pulls
TensorFlow on Python 3.11 and newer, while Demucs 4.1 wants a current PyTorch. The
main script calls Basic Pitch through a subprocess in its own environment.

The first run downloads model weights: Demucs `htdemucs_ft` and `htdemucs_6s` from
Hugging Face, and the BS-Roformer and DrumSep checkpoints into the `models` folder.
About 1.5 GB in total, downloaded once.

If PowerShell refuses to run the script, use
`powershell -ExecutionPolicy Bypass -File .\setup.ps1`.

## Usage

```powershell
python pipeline.py "https://www.youtube.com/watch?v=..."
python pipeline.py "C:\Music\song.mp3"
python pipeline.py song.wav --mode fast
python pipeline.py song.wav --shifts 3
python pipeline.py song.wav --no-drum-parts
python pipeline.py song.wav --cpu
```

| Flag              | Effect                                                                 |
| ----------------- | ---------------------------------------------------------------------- |
| `--mode fine`     | Default. BS-Roformer vocals, then two Demucs passes. Best quality.     |
| `--mode fast`     | One `htdemucs_6s` pass. Roughly three times faster, more bleed.        |
| `--shifts N`      | Demucs averages N shifted passes. Slightly cleaner, N times slower.    |
| `--no-drum-parts` | Skip splitting drums into kick, snare, toms, hi-hat, ride and crash.   |
| `--cpu`           | Force CPU even when a GPU is available.                                |

`python pipeline.py` can be started with any Python. The script re-launches itself
inside `.venv` automatically.

Progress is printed per stage:

```
=== [1/6] Preparing original.wav ===
=== [2/6] Separating stems (fine mode, cuda) ===
=== [3/6] Splitting drums into parts (DrumSep) ===
=== [4/6] Estimating BPM and key, writing info.txt ===
=== [5/6] Transcribing melodic stems with Basic Pitch ===
=== [6/6] Transcribing drums with ADTOF (cuda) ===
```

### How the separation works

Fine mode runs three models in a chain, each on the job it scores best at:

1. **BS-Roformer** (`model_bs_roformer_ep_317_sdr_12.9755`) splits the mix into
   vocals and instrumental. It is the strongest open vocal model and keeps far
   more of the instruments out of the vocal stem than Demucs does.
2. **Demucs `htdemucs_ft`** takes drums and bass out of the instrumental. It scores
   about 1.5 dB better on both than the 6-stem model.
3. **Demucs `htdemucs_6s`** takes guitar and piano out of what remains. Anything
   it still labels drums, bass or vocals is folded back into `other.wav`, so the
   six stems always add up to the whole mix.

Then **DrumSep** (`MDX23C-DrumSep-aufr33-jarredou`) splits `drums.wav` into six parts.

Each step in the chain keeps its intermediate files, so a run that fails halfway
through the chain resumes from the last finished step.

What this does not do: separate instruments the models were never trained on.
Strings, brass, synth leads and pads end up in `other.wav`. Horns in particular can
still bleed into vocals when they double the vocal line, just less than before.

### Resuming

Every stage checks whether its output files already exist and skips the work when
they do. If a run fails or is interrupted, run the same command again and only the
missing parts are redone. To redo a stage on purpose, delete its files
(for example `midi\drums.mid` or the whole `stems` folder) and run again.

Files are written under temporary names and renamed when complete, so a crash never
leaves a half-written file that looks finished.

### Drum transcription failing

If ADTOF fails for any reason the run prints a warning, skips `drums.mid` and still
finishes everything else. The final summary says when `drums.mid` is missing.

## Output layout

```
output/<song-name>/
  original.wav        44.1 kHz stereo 16-bit WAV of the source
  stems/
    drums.wav  bass.wav  vocals.wav  guitar.wav  piano.wav  other.wav
    drums/
      kick.wav  snare.wav  toms.wav  hihat.wav  ride.wav  crash.wav
  midi/
    bass.mid  guitar.mid  piano.mid  vocals.mid   (Basic Pitch)
    drums.mid                                      (ADTOF)
  info.txt            title, source, date, BPM, key, separation mode, device
  .source             the URL or path this folder was made from (used for resume)
```

### Tempo in FL Studio

Set the project tempo to the BPM from `info.txt` before dragging MIDI in. The
MIDI files are written at that tempo, so the notes land on the same seconds as the
audio stems and on the bar grid at the same time. If the estimate is off by a few
BPM the MIDI still matches the audio; only the grid drifts.

`<song-name>` is the YouTube title or the local file name with characters that are
not allowed in folder names removed, trimmed to 80 characters. If two different
sources sanitize to the same name, the second one gets a short hash suffix so the
resume logic never mixes them up.

### Drum MIDI notes

ADTOF outputs five General MIDI drum notes on channel 10:

| Note | Drum          |
| ---- | ------------- |
| 35   | kick          |
| 38   | snare         |
| 42   | closed hi-hat |
| 47   | tom           |
| 49   | crash         |

FPC and most drum plugins follow this map. Note 35 is "Acoustic Bass Drum". If your
kit only responds to 36, move the kick notes up one semitone in the piano roll.

## Testing the setup

A synthetic 16 second song (120 BPM, A minor) is included for checking that
everything is installed:

```powershell
.\.venv\Scripts\python.exe tools\make_test_song.py test_song.wav
python pipeline.py test_song.wav
```

Expected result: `output\test_song` with all six stems, six drum parts, five MIDI
files, and `info.txt` reporting a key of A minor and a BPM near 120.

## Notes and limits

- BPM and key are estimates from librosa. Double-time or half-time tempo mistakes
  and relative major or minor confusion are the common failure modes. Treat them as
  a starting point.
- Basic Pitch is a general pitch tracker. Vocal MIDI in particular needs cleanup.
- yt-dlp breaks whenever YouTube changes. If downloads fail, update it inside the
  main environment with `.\.venv\Scripts\python.exe -m pip install -U yt-dlp`, then
  update the pin in `requirements-main.txt`.
- Python 3.10 reaches end of life in October 2026. It still works fine for this
  tool, and Basic Pitch 0.4.0 does not support anything newer than 3.11.

## Tools used

- [Demucs](https://github.com/adefossez/demucs) 4.1.0, models `htdemucs_ft` and `htdemucs_6s`
- [audio-separator](https://github.com/nomadkaraoke/python-audio-separator) 0.47.0
  running the BS-Roformer-Viperx-1297 vocal model and MDX23C DrumSep by aufr33 and jarredou
- [Basic Pitch](https://github.com/spotify/basic-pitch) 0.4.0 (ONNX runtime)
- [ADTOF-pytorch](https://github.com/xavriley/ADTOF-pytorch), a PyTorch port of
  [ADTOF](https://github.com/MZehren/ADTOF) that needs no TensorFlow or madmom
- [yt-dlp](https://github.com/yt-dlp/yt-dlp), [librosa](https://librosa.org), ffmpeg
