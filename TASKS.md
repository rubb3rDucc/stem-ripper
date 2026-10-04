# TASKS

Observations and follow-ups logged per file.

## pipeline.py

- `stage_info` skips when `info.txt` exists, so BPM and key are never re-estimated
  on resume. Delete `info.txt` to recompute. Cheap enough to always recompute if
  that turns out to be annoying.
- `stage_basic_pitch` starts one subprocess per stem (four model loads, about 2 to
  3 seconds each after the first). Basic Pitch accepts several inputs in one call;
  batching the missing stems would save a few seconds per song at the cost of
  less granular progress output.
- Basic Pitch prints three "not installed" warnings (CoreML, TFLite, TensorFlow)
  on every call. They are harmless. Could be hidden by filtering the subprocess
  output, but that would also hide real errors, so left as is.
- ADTOF runs as `python -m adtof_pytorch.cli` from the main env because its
  dependencies (torch, librosa, pretty_midi) fit there. If a future version adds
  conflicting pins, move it to its own venv like Basic Pitch.
- `detect_device` imports torch in the main process only to ask about CUDA. Takes
  about a second. Could be replaced by an `nvidia-smi` probe if startup time matters.
- librosa `beat_track` reported 117.5 BPM for the 120 BPM synthetic test song.
  Fine as a starting point, but rounding to the nearest common tempo or reporting
  a second candidate could be a later phase.
- Local inputs are always transcoded to 44.1 kHz stereo 16-bit PCM, even when the
  input is already a WAV. This keeps every downstream tool on one format. If
  bit-perfect originals matter, add a copy path for already-conforming WAVs.
- Output folder naming: a title that sanitizes to the empty string becomes
  `untitled`; a Windows reserved name gets `_song` appended; collisions between
  different sources get a 6-character hash suffix via the `.source` marker.
- Fine mode chain (Roformer, htdemucs_ft, htdemucs_6s): the residue folding in
  `separate_fine` keeps the six stems summing to the mix, but the 6s model is
  running on an already-reduced signal it was not trained on. Guitar and piano
  quality in fine mode should be compared against fast mode on a few real songs.
  If it is worse, an alternative is to run htdemucs_6s on the full instrumental
  and only take guitar and piano from it.
- `--mode` and `--shifts` are not part of the skip check. Stems made in one mode
  are reused by a later run in another mode. Delete the `stems` folder to
  re-separate.
- audio-separator picks the GPU on its own; `--cpu` hides it with
  `CUDA_VISIBLE_DEVICES=""` in the subprocess environment.
- audio-separator prints "onnxruntime is not built with CUDA 12.x support" and
  "CUDAExecutionProvider not available" on every call. Harmless: both models we
  use are PyTorch checkpoints and the log confirms "setting Torch device to
  CUDA". Installing onnxruntime-gpu would silence it at the cost of a 300 MB
  package nothing uses.
- Timing on the RTX 3070 for the 16 second test song: fine mode 52 s for the
  three-model chain plus 19 s for DrumSep, fast mode 4 s. Model loading is most
  of that, so real songs scale better than these numbers suggest.
- `set_midi_tempo` rebuilds the file with pretty_midi. Any tempo map, time
  signature or lyric events from the source file are dropped; neither Basic Pitch
  nor ADTOF writes any, so nothing is lost today.
- `stage_info` now runs before the MIDI stages because they need the BPM. On
  resume the BPM is parsed back out of `info.txt`.

## setup.ps1

- Only Windows is covered. A `setup.sh` for macOS or Linux would be the same two
  venv creations with `bin/python` paths; `pipeline.py` already handles both layouts.
- Re-running on existing environments re-runs pip against the pinned files, which
  is the intended upgrade path after editing a requirements file.

## requirements-main.txt

- `adtof-pytorch` is pinned to a git commit because it is not on PyPI. Installing
  it needs git on PATH.
- torch is pinned to the `+cu126` build. Driver 560.x supports CUDA 12.6; a newer
  driver can also use the `cu128` index if ever needed.

## requirements-basicpitch.txt

- Python 3.10 is required for this env so basic-pitch picks onnxruntime instead of
  TensorFlow. On 3.11 the same pin would pull TensorFlow 2.15 (about 500 MB).

## tools/make_test_song.py

- Mono content duplicated to both channels, so Demucs gets no stereo cues. Good
  enough for a smoke test, not for judging separation quality.

## requirements-main.txt (audio-separator)

- `audio-separator[cpu]` installs plain onnxruntime. The two models used are
  PyTorch checkpoints and run on CUDA through torch, so onnxruntime-gpu is not
  needed. Only matters if someone switches to an ONNX model (MDX-Net family).
- audio-separator resolved cleanly against torch 2.14.1+cu126, demucs 4.1.0 and
  numpy 2.2.6 (`pip check` clean), so it shares the main env.

## Later phases (not built)

- Batch mode: accept several URLs or files, or a text file of them.
- Optional `--keep-work` to keep the intermediate `.work` folder for debugging.
- Per-stem MIDI tuning flags for Basic Pitch (onset and frame thresholds,
  minimum note length) and ADTOF per-class thresholds.
- Vocal model choice flag. Candidates in audio-separator's list: MelBand Roformer
  Kim vocals (12.6 median SDR) and the "bleedless" Gabox variants, which may suit
  horn-heavy material better than the default BS-Roformer.
- Ensemble vocals (two Roformer models averaged) for another small gain at double
  the vocal-stage time.
