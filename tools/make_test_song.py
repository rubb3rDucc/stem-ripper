"""Generate a short synthetic test song for checking the pipeline end to end.

Usage:
    python tools/make_test_song.py [output.wav]

The song is 16 seconds at 120 BPM in A minor: kick, snare and hi-hat,
a bass line, a chord pad and a simple lead melody. It only needs numpy,
so it runs inside the main virtual environment.
"""

import math
import struct
import sys
import wave
from pathlib import Path

import numpy as np


SAMPLE_RATE = 44100
BPM = 120.0
BARS = 8
BEAT_SECONDS = 60.0 / BPM
BAR_SECONDS = BEAT_SECONDS * 4
TOTAL_SECONDS = BAR_SECONDS * BARS


def midi_to_hz(note):
    return 440.0 * (2.0 ** ((note - 69) / 12.0))


def envelope(length, attack, decay):
    """Simple attack then exponential decay envelope."""
    t = np.arange(length) / SAMPLE_RATE
    env = np.exp(-t / decay)
    attack_samples = int(attack * SAMPLE_RATE)
    if attack_samples > 0:
        env[:attack_samples] *= np.linspace(0.0, 1.0, attack_samples)
    return env


def add(buffer, start_seconds, signal):
    start = int(start_seconds * SAMPLE_RATE)
    end = min(start + len(signal), len(buffer))
    buffer[start:end] += signal[: end - start]


def kick(length_seconds=0.25):
    n = int(length_seconds * SAMPLE_RATE)
    t = np.arange(n) / SAMPLE_RATE
    freq = 50.0 + 100.0 * np.exp(-t * 30.0)
    phase = np.cumsum(2 * np.pi * freq / SAMPLE_RATE)
    return np.sin(phase) * envelope(n, 0.002, 0.08) * 0.9


def snare(length_seconds=0.2):
    n = int(length_seconds * SAMPLE_RATE)
    rng = np.random.default_rng(1)
    noise = rng.uniform(-1, 1, n) * envelope(n, 0.001, 0.05)
    t = np.arange(n) / SAMPLE_RATE
    tone = np.sin(2 * np.pi * 190.0 * t) * envelope(n, 0.001, 0.04)
    return (noise * 0.6 + tone * 0.5) * 0.7


def hihat(length_seconds=0.06):
    n = int(length_seconds * SAMPLE_RATE)
    rng = np.random.default_rng(2)
    noise = rng.uniform(-1, 1, n)
    # crude high-pass: difference of successive samples
    noise = np.diff(noise, prepend=0.0)
    return noise * envelope(n, 0.0005, 0.015) * 0.35


def saw(freq, length_seconds, gain):
    n = int(length_seconds * SAMPLE_RATE)
    t = np.arange(n) / SAMPLE_RATE
    signal = 2.0 * ((t * freq) % 1.0) - 1.0
    return signal * envelope(n, 0.005, length_seconds * 0.6) * gain


def triangle(freq, length_seconds, gain):
    n = int(length_seconds * SAMPLE_RATE)
    t = np.arange(n) / SAMPLE_RATE
    signal = 2.0 * np.abs(2.0 * ((t * freq) % 1.0) - 1.0) - 1.0
    return signal * envelope(n, 0.02, length_seconds * 0.9) * gain


def sine(freq, length_seconds, gain):
    n = int(length_seconds * SAMPLE_RATE)
    t = np.arange(n) / SAMPLE_RATE
    vibrato = 1.0 + 0.004 * np.sin(2 * np.pi * 5.5 * t)
    phase = np.cumsum(2 * np.pi * freq * vibrato / SAMPLE_RATE)
    return np.sin(phase) * envelope(n, 0.03, length_seconds * 0.8) * gain


def build_song():
    total = int(TOTAL_SECONDS * SAMPLE_RATE) + SAMPLE_RATE
    drums = np.zeros(total)
    bass = np.zeros(total)
    pad = np.zeros(total)
    lead = np.zeros(total)

    # Chord progression in A minor: Am, F, C, G (root MIDI notes)
    progression = [57, 53, 48, 55]
    chord_shapes = {57: [57, 60, 64], 53: [53, 57, 60], 48: [48, 52, 55], 55: [55, 59, 62]}
    bass_roots = {57: 33, 53: 29, 48: 36, 55: 31}
    melody = [69, 72, 76, 74, 72, 69, 67, 69]

    for bar in range(BARS):
        bar_start = bar * BAR_SECONDS
        root = progression[bar % len(progression)]

        for beat in range(4):
            beat_start = bar_start + beat * BEAT_SECONDS
            add(drums, beat_start, kick())
            if beat in (1, 3):
                add(drums, beat_start, snare())
            add(drums, beat_start, hihat())
            add(drums, beat_start + BEAT_SECONDS / 2, hihat())

            bass_note = bass_roots[root]
            if beat == 3:
                bass_note = bass_note + 7
            add(bass, beat_start, saw(midi_to_hz(bass_note), BEAT_SECONDS * 0.9, 0.35))

        for note in chord_shapes[root]:
            add(pad, bar_start, triangle(midi_to_hz(note), BAR_SECONDS, 0.12))

        for step in range(8):
            step_start = bar_start + step * (BEAT_SECONDS / 2)
            note = melody[(step + bar) % len(melody)]
            add(lead, step_start, sine(midi_to_hz(note), BEAT_SECONDS / 2, 0.25))

    mix = drums + bass + pad + lead
    peak = np.max(np.abs(mix))
    if peak > 0:
        mix = mix / peak * 0.9
    return mix


def write_wav(path, mono):
    stereo = np.stack([mono, mono], axis=1)
    samples = (stereo * 32767.0).astype(np.int16)
    with wave.open(str(path), "wb") as handle:
        handle.setnchannels(2)
        handle.setsampwidth(2)
        handle.setframerate(SAMPLE_RATE)
        handle.writeframes(samples.tobytes())


def main():
    if len(sys.argv) > 1:
        out_path = Path(sys.argv[1])
    else:
        out_path = Path("test_song.wav")
    write_wav(out_path, build_song())
    print("wrote %s (%.1f seconds, %d BPM, A minor)" % (out_path, TOTAL_SECONDS, int(BPM)))


if __name__ == "__main__":
    main()
