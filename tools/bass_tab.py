"""Turn the bass MIDI from pipeline.py into ASCII bass tab.

Usage:
    python tools/bass_tab.py output/<song>            (reads midi/bass.mid)
    python tools/bass_tab.py path/to/bass.mid
    python tools/bass_tab.py output/<song> --bpm 120 --shift 2

Writes bass_tab.txt into the song folder (or next to the MIDI file) and prints
it. Needs pretty_midi, numpy and librosa, so run it with the main environment:
    .venv/bin/python tools/bass_tab.py output/<song>

Steps:
1. Cleanup. Basic Pitch on bass leaves octave ghosts and short quiet notes. Notes
   shorter than --min-length or quieter than --min-velocity (a fraction of the
   median velocity) are dropped, then one note is kept per grid slot: the lowest
   of an octave pair, otherwise the loudest.
2. Beat grid. By default librosa's beat tracker runs on the song's audio (the
   drums stem, or original.wav without one) and notes snap to the beats it finds,
   so a tempo that drifts or pushes and pulls still lines up. Each beat is split
   into 16ths, or into triplets when those fit the notes clearly better. With
   --bpm, or when there is no audio, a fixed grid is used instead; without --bpm
   its tempo is refined to fit the note onsets.
3. Fretting. Each note gets a string and fret for the tuning. The search tracks
   where the fretting hand sits (a 4-fret window). Notes inside the window are
   free, open strings never move it, and every shift costs extra, so runs stay
   in one area of the neck.
4. Layout. 4/4, bar 1 starts on the beat of the first note (--shift moves it).
   16th grids are shown as 8ths when every note falls on an 8th.

Not handled yet: rhythm notation, chords, sections, alternate fingerings,
downbeat detection, time signatures other than 4/4.
"""

import argparse
import re
import sys
from pathlib import Path

import numpy as np
import pretty_midi


STANDARD_TUNING = "E1,A1,D2,G2"
MAX_FRET = 20
BEATS_PER_BAR = 4
TAB_WIDTH = 80

# Fretting costs. A shift is paid once plus per fret moved, so one long shift
# beats several short ones, and staying put beats both.
HAND_SPAN = 4
SHIFT_COST = 3.0
STRING_CHANGE_COST = 0.3
HIGH_POSITION_COST = 0.02


def parse_arguments():
    parser = argparse.ArgumentParser(description="Write ASCII bass tab from a bass MIDI file.")
    parser.add_argument("source", help="Song folder from pipeline.py, or a .mid file")
    parser.add_argument("--out", help="Output file (default: bass_tab.txt in the song folder)")
    parser.add_argument("--bpm", type=float, help="Use a fixed grid at this BPM instead of tracking beats")
    parser.add_argument(
        "--beats-from", choices=["drums", "mix"], default="drums",
        help="Audio for the beat tracker: the drums stem or original.wav (default %(default)s)",
    )
    parser.add_argument(
        "--subdivision", choices=["auto", "16th", "triplet"], default="auto",
        help="How each beat is split (default %(default)s: triplets only when they fit clearly better)",
    )
    parser.add_argument(
        "--tuning", default=STANDARD_TUNING,
        help="Open strings low to high, e.g. D1,A1,D2,G2 or B0,E1,A1,D2,G2 (default %(default)s)",
    )
    parser.add_argument(
        "--shift", type=int, default=0,
        help="Move every note this many grid slots later so bar lines fall in the right place",
    )
    parser.add_argument(
        "--min-length", type=float, default=0.06,
        help="Drop notes shorter than this many seconds (default %(default)s)",
    )
    parser.add_argument(
        "--min-velocity", type=float, default=0.6,
        help="Drop notes quieter than this fraction of the median velocity (default %(default)s)",
    )
    parser.add_argument("--bars-per-line", type=int, help="Default: as many as fit in %d columns" % TAB_WIDTH)
    return parser.parse_args()


def resolve_paths(source, out):
    source = Path(source).expanduser().resolve()
    if source.is_dir():
        midi_path = source / "midi" / "bass.mid"
        song_dir = source
    else:
        midi_path = source
        song_dir = source.parent.parent if source.parent.name == "midi" else source.parent
    if not midi_path.exists():
        sys.exit("No bass MIDI at %s" % midi_path)
    out_path = Path(out).expanduser().resolve() if out else song_dir / "bass_tab.txt"
    return midi_path, song_dir, out_path


def read_info(song_dir):
    """Song, key and BPM lines from pipeline.py's info.txt, if it is there."""
    info = {}
    info_path = song_dir / "info.txt"
    if info_path.exists():
        for line in info_path.read_text(encoding="utf-8").splitlines():
            key, _, value = line.partition(":")
            if key in ("Song", "Key", "BPM"):
                info[key] = value.strip()
    return info


def parse_tuning(text):
    try:
        strings = [pretty_midi.note_name_to_number(name.strip()) for name in text.split(",")]
    except ValueError as error:
        sys.exit("Bad --tuning %r: %s" % (text, error))
    if strings != sorted(strings):
        sys.exit("--tuning must list strings from lowest to highest")
    return strings


# ---------------------------------------------------------------------------
# Cleanup
# ---------------------------------------------------------------------------

def load_notes(midi_path):
    midi = pretty_midi.PrettyMIDI(str(midi_path))
    notes = [note for instrument in midi.instruments if not instrument.is_drum for note in instrument.notes]
    _, tempos = midi.get_tempo_changes()
    midi_bpm = float(tempos[0]) if len(tempos) else 120.0
    return sorted(notes, key=lambda note: note.start), midi_bpm


def filter_notes(notes, tuning, min_length, min_velocity):
    """Drop short, quiet and out-of-range notes. Returns kept notes and drop counts."""
    if not notes:
        return [], {}
    median_velocity = float(np.median([note.velocity for note in notes]))
    lowest, highest = tuning[0], tuning[-1] + MAX_FRET
    kept = []
    dropped = {"short": 0, "quiet": 0, "out of range": 0}
    for note in notes:
        if note.end - note.start < min_length:
            dropped["short"] += 1
        elif note.velocity < min_velocity * median_velocity:
            dropped["quiet"] += 1
        elif not lowest <= note.pitch <= highest:
            dropped["out of range"] += 1
        else:
            kept.append(note)
    return kept, dropped


# ---------------------------------------------------------------------------
# Beat grid
# ---------------------------------------------------------------------------
# A grid is a function from seconds to a position in beats (1.5 = halfway
# through the second beat). Quantizing multiplies by the slots per beat and
# rounds.

def tracked_grid(audio_path, start_bpm):
    """Grid from librosa's beat tracker. Returns (grid, beat times) or None."""
    import librosa

    # 44.1 kHz with a 256-sample hop: at librosa's defaults (22.05 kHz, 512) a
    # beat must be a whole number of 23 ms frames, so 120 BPM comes out as 117.5
    # and the beats drift.
    samples, sample_rate = librosa.load(str(audio_path), sr=44100, mono=True)
    _, beats = librosa.beat.beat_track(
        y=samples, sr=sample_rate, hop_length=256, start_bpm=start_bpm, units="time")
    if len(beats) < 8:
        return None
    # Extend past both ends at the median beat length so notes outside the
    # tracked range still get positions.
    interval = float(np.median(np.diff(beats)))
    padding = np.arange(1, 65) * interval
    times = np.concatenate([beats[0] - padding[::-1], beats, beats[-1] + padding])
    indexes = np.arange(len(times), dtype=float)
    return (lambda seconds: np.interp(seconds, times, indexes)), beats


def fixed_grid(onsets, bpm):
    """Grid at a constant BPM, phase-aligned to the onsets."""
    beat = 60.0 / bpm
    _, offset = grid_fit(onsets, bpm)
    return lambda seconds: (np.asarray(seconds) - offset) / beat


def refine_bpm(onsets, estimate):
    """BPM within 6% of estimate whose 16th grid best fits the onsets."""
    best_bpm, best_fit = estimate, -1.0
    for bpm in np.arange(estimate * 0.94, estimate * 1.06, 0.05):
        fit, _ = grid_fit(onsets, bpm)
        if fit > best_fit:
            best_bpm, best_fit = float(bpm), fit
    return best_bpm


def grid_fit(onsets, bpm):
    """How well onsets sit on a 16th grid at bpm, and the grid's offset.

    Fit is the length of the mean unit vector of each onset's phase on the grid
    (1.0 means every onset is exactly on a 16th)."""
    step = 60.0 / bpm / 4
    phases = np.exp(2j * np.pi * (np.asarray(onsets) / step))
    mean = phases.mean()
    offset = (np.angle(mean) / (2 * np.pi)) * step
    return float(abs(mean)), offset


def on_grid_share(positions, slots_per_beat):
    """Share of notes within a quarter slot of a grid line (0.5 is chance)."""
    scaled = np.asarray(positions) * slots_per_beat
    return float(np.mean(np.abs(scaled - np.round(scaled)) < 0.25))


def prefers_triplets(positions):
    """True when clearly more notes sit on triplet-only positions (1/3, 2/3 of a
    beat) than on 16th-only ones (1/4, 1/2, 3/4). Notes on the beat fit both
    grids and are ignored. One tolerance in beats for both, because a triplet
    slot is wider than a 16th and a per-slot tolerance would favour it."""
    fractions = np.asarray(positions) % 1.0
    tolerance = 1.0 / 24

    def near(points):
        distances = np.abs(fractions[:, None] - np.asarray(points)[None, :])
        return int(np.sum(distances.min(axis=1) < tolerance))

    triplet_only, sixteenth_only = near([1 / 3, 2 / 3]), near([1 / 4, 1 / 2, 3 / 4])
    return triplet_only >= 2 * sixteenth_only and triplet_only >= 0.1 * len(fractions)


def quantize(notes, positions, slots_per_beat, shift):
    """Map notes to grid slots with bar 1 starting on the beat of the first note.

    Keeps one note per slot: drops the upper note of an octave pair (a Basic
    Pitch harmonic ghost), then keeps the loudest."""
    by_slot = {}
    for note, position in zip(notes, positions):
        by_slot.setdefault(int(round(position * slots_per_beat)), []).append(note)
    first_beat = (min(by_slot) // slots_per_beat) * slots_per_beat
    merged = 0
    slots = {}
    for slot, group in by_slot.items():
        pitches = {note.pitch for note in group}
        candidates = [
            note for note in group
            if not any(note.pitch - interval in pitches for interval in (12, 24))
        ] or group
        slots[slot - first_beat + shift] = max(candidates, key=lambda note: note.velocity).pitch
        merged += len(group) - 1
    return dict(sorted(slots.items())), merged


# ---------------------------------------------------------------------------
# Fretting
# ---------------------------------------------------------------------------

def hand_states(pitch, tuning):
    """(string, fret, hand) options for pitch. hand is the lowest fret of the
    4-fret window the hand covers; an open string fits any window."""
    hands = range(1, MAX_FRET - HAND_SPAN + 2)
    states = []
    for string, open_pitch in enumerate(tuning):
        fret = pitch - open_pitch
        if fret == 0:
            states.extend((string, 0, hand) for hand in hands)
        elif 0 < fret <= MAX_FRET:
            states.extend((string, fret, hand) for hand in hands if hand <= fret < hand + HAND_SPAN)
    return states


def move_cost(previous, current):
    prev_string, _, prev_hand = previous
    string, _, hand = current
    cost = STRING_CHANGE_COST * abs(string - prev_string) + HIGH_POSITION_COST * hand
    if hand != prev_hand:
        cost += SHIFT_COST + abs(hand - prev_hand)
    return cost


def assign_frets(pitches, tuning):
    """Cheapest sequence of (string, fret, hand) for the pitches (dynamic programming)."""
    if not pitches:
        return []
    layers = [hand_states(pitch, tuning) for pitch in pitches]
    costs = [{state: HIGH_POSITION_COST * state[2] for state in layers[0]}]
    back = [{}]
    for layer in layers[1:]:
        previous_costs = costs[-1]
        layer_costs, layer_back = {}, {}
        for state in layer:
            best = min(previous_costs, key=lambda prev: previous_costs[prev] + move_cost(prev, state))
            layer_costs[state] = previous_costs[best] + move_cost(best, state)
            layer_back[state] = best
        costs.append(layer_costs)
        back.append(layer_back)
    state = min(costs[-1], key=costs[-1].get)
    path = [state]
    for layer_back in reversed(back[1:]):
        state = layer_back[state]
        path.append(state)
    return list(reversed(path))


# ---------------------------------------------------------------------------
# Layout
# ---------------------------------------------------------------------------

def render(slots, frets, tuning, header, bars_per_line, slots_per_beat):
    """ASCII tab: one row per string, highest string on top, two characters per column."""
    slots_per_bar = slots_per_beat * BEATS_PER_BAR
    if slots_per_beat == 3:
        columns_per_bar = slots_per_bar
        count = [part for beat in "1234" for part in (beat, "t", "l")]
    elif all(slot % 2 == 0 for slot in slots):
        columns_per_bar = slots_per_bar // 2
        count = [part for beat in "1234" for part in (beat, "&")]
    else:
        columns_per_bar = slots_per_bar
        count = [part for beat in "1234" for part in (beat, "e", "&", "a")]
    slots_per_column = slots_per_bar // columns_per_bar
    bar_width = columns_per_bar * 2
    if not bars_per_line:
        bars_per_line = max(1, (TAB_WIDTH - 2) // (bar_width + 1))

    names = [re.sub(r"-?\d+$", "", pretty_midi.note_number_to_name(pitch)) for pitch in tuning]
    label_width = max(len(name) for name in names)
    bar_count = max(slots) // slots_per_bar + 1 if slots else 0
    by_slot = dict(zip(slots, frets))

    lines = list(header)
    for first_bar in range(0, bar_count, bars_per_line):
        bars = range(first_bar, min(first_bar + bars_per_line, bar_count))
        lines.append("")
        lines.append(" " * (label_width + 1) + "".join(
            ("%d" % (bar + 1)).ljust(bar_width + 1) for bar in bars).rstrip())
        for string in reversed(range(len(tuning))):
            row = names[string].ljust(label_width) + "|"
            for bar in bars:
                cells = []
                for column in range(columns_per_bar):
                    slot = bar * slots_per_bar + column * slots_per_column
                    position = by_slot.get(slot)
                    if position and position[0] == string:
                        cells.append(str(position[1]).ljust(2, "-"))
                    else:
                        cells.append("--")
                row += "".join(cells) + "|"
            lines.append(row)
        lines.append(" " * (label_width + 1) + " ".join(
            "".join(part.ljust(2) for part in count) for _ in bars).rstrip())
    return "\n".join(lines) + "\n"


def build_grid(arguments, song_dir, onsets, midi_bpm):
    """Pick the grid. Returns (grid function, description for the header)."""
    if arguments.bpm:
        return fixed_grid(onsets, arguments.bpm), "fixed %.1f BPM (from --bpm)" % arguments.bpm
    audio = song_dir / "stems" / "drums.wav" if arguments.beats_from == "drums" else None
    if audio is None or not audio.exists():
        audio = song_dir / "original.wav"
    if audio.exists():
        tracked = tracked_grid(audio, midi_bpm)
        if tracked:
            grid, beats = tracked
            tempos = 60.0 / np.diff(beats)
            low, middle, high = np.percentile(tempos, [10, 50, 90])
            return grid, "beats tracked from %s, %.1f BPM (most beats %.0f-%.0f)" % (
                audio.name, middle, low, high)
    bpm = refine_bpm(onsets, midi_bpm)
    return fixed_grid(onsets, bpm), "fixed %.1f BPM (no audio to track; MIDI said %.1f)" % (bpm, midi_bpm)


def main():
    arguments = parse_arguments()
    midi_path, song_dir, out_path = resolve_paths(arguments.source, arguments.out)
    tuning = parse_tuning(arguments.tuning)
    info = read_info(song_dir)

    notes, midi_bpm = load_notes(midi_path)
    kept, dropped = filter_notes(notes, tuning, arguments.min_length, arguments.min_velocity)
    if not kept:
        sys.exit("No notes left in %s after cleanup (%d before)." % (midi_path, len(notes)))

    onsets = [note.start for note in kept]
    grid, grid_note = build_grid(arguments, song_dir, onsets, midi_bpm)
    positions = grid(onsets)
    if arguments.subdivision == "triplet" or (
            arguments.subdivision == "auto" and prefers_triplets(positions)):
        slots_per_beat = 3
    else:
        slots_per_beat = 4
    share = on_grid_share(positions, slots_per_beat)

    slots, merged = quantize(kept, positions, slots_per_beat, arguments.shift)
    if min(slots) < 0:
        sys.exit("--shift moved notes before bar 1; use a value of 0 or more.")
    frets = assign_frets(list(slots.values()), tuning)
    shifts = sum(1 for previous, current in zip(frets, frets[1:]) if previous[2] != current[2])

    header = [
        "Song: %s" % info.get("Song", midi_path.stem),
        "Key: %s    Time: 4/4    Tuning: %s (low to high)" % (info.get("Key", "unknown"), arguments.tuning),
        "Grid: %s, split into %s" % (grid_note, "triplets" if slots_per_beat == 3 else "16ths"),
        "Timing: %.0f%% of notes on the grid (50%% would be chance; under ~75%% expect misplaced notes)"
        % (100 * share),
        "Notes: %d from Basic Pitch, %d in tab, %d hand shifts. Dropped: %s, %d merged on the same slot."
        % (len(notes), len(slots), shifts,
           ", ".join("%d %s" % (count, reason) for reason, count in dropped.items()), merged),
        "Machine transcription: check it by ear.",
    ]
    text = render(list(slots), frets, tuning, header, arguments.bars_per_line, slots_per_beat)
    out_path.write_text(text, encoding="utf-8")
    print(text)
    print("Wrote " + str(out_path))


if __name__ == "__main__":
    main()
