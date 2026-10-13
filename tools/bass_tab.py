"""Turn the bass MIDI from pipeline.py into ASCII bass tab.

Usage:
    python tools/bass_tab.py output/<song>            (reads midi/bass.mid)
    python tools/bass_tab.py path/to/bass.mid
    python tools/bass_tab.py output/<song> --bpm 120 --shift 2

Writes bass_tab.txt into the song folder (or next to the MIDI file) and prints
it. Needs pretty_midi and numpy, so run it with the main environment:
    .venv/bin/python tools/bass_tab.py output/<song>

Steps:
1. Cleanup. Basic Pitch on bass leaves octave ghosts and short quiet notes. Notes
   shorter than --min-length or quieter than --min-velocity (a fraction of the
   median velocity) are dropped, then one note is kept per 16th: the lowest of an
   octave pair, otherwise the loudest.
2. Tempo. The BPM in the MIDI comes from librosa and can be a few percent off,
   which drifts notes off the grid within a few bars. Unless --bpm is given, the
   BPM is refined to the value within 6% that best fits the note onsets to a
   16th grid.
3. Fretting. Each note gets a string and fret for the tuning. A shortest-path
   search keeps hand movement small; open strings do not move the hand.
4. Layout. 4/4, bar 1 starts on the beat of the first note (--shift moves it).
   Shown on an 8th grid when every note falls on an 8th, otherwise 16ths.

Not handled yet: rhythm notation, chords, sections, alternate fingerings,
time signatures other than 4/4, tempo changes.
"""

import argparse
import re
import sys
from pathlib import Path

import numpy as np
import pretty_midi


STANDARD_TUNING = "E1,A1,D2,G2"
MAX_FRET = 20
SLOTS_PER_BAR = 16
TAB_WIDTH = 80


def parse_arguments():
    parser = argparse.ArgumentParser(description="Write ASCII bass tab from a bass MIDI file.")
    parser.add_argument("source", help="Song folder from pipeline.py, or a .mid file")
    parser.add_argument("--out", help="Output file (default: bass_tab.txt in the song folder)")
    parser.add_argument("--bpm", type=float, help="Use this BPM instead of refining the MIDI tempo")
    parser.add_argument(
        "--tuning", default=STANDARD_TUNING,
        help="Open strings low to high, e.g. D1,A1,D2,G2 or B0,E1,A1,D2,G2 (default %(default)s)",
    )
    parser.add_argument(
        "--shift", type=int, default=0,
        help="Move every note this many 16ths later so bar lines fall in the right place",
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
# Cleanup and tempo
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


def refine_bpm(onsets, estimate):
    """BPM within 6% of estimate whose 16th grid best fits the onsets.

    Fit is the length of the mean unit vector of each onset's phase on the grid
    (1.0 means every onset is exactly on a 16th). The phase also gives the grid
    offset, so a song that does not start at time zero still lines up."""
    best_bpm, best_fit = estimate, -1.0
    for bpm in np.arange(estimate * 0.94, estimate * 1.06, 0.05):
        fit, _ = grid_fit(onsets, bpm)
        if fit > best_fit:
            best_bpm, best_fit = float(bpm), fit
    return best_bpm


def grid_fit(onsets, bpm):
    step = 60.0 / bpm / 4
    phases = np.exp(2j * np.pi * (np.asarray(onsets) / step))
    mean = phases.mean()
    offset = (np.angle(mean) / (2 * np.pi)) * step
    return float(abs(mean)), offset


def quantize(notes, bpm, shift):
    """Map notes to 16th slots with bar 1 starting on the beat of the first note.

    Keeps one note per slot: drops the upper note of an octave pair (a Basic
    Pitch harmonic ghost), then keeps the loudest."""
    step = 60.0 / bpm / 4
    _, offset = grid_fit([note.start for note in notes], bpm)
    by_slot = {}
    for note in notes:
        slot = int(round((note.start - offset) / step))
        by_slot.setdefault(slot, []).append(note)
    first_beat = (min(by_slot) // 4) * 4
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

def positions(pitch, tuning):
    """(string index, fret) pairs that play pitch; string 0 is the lowest."""
    return [
        (index, pitch - open_pitch)
        for index, open_pitch in enumerate(tuning)
        if 0 <= pitch - open_pitch <= MAX_FRET
    ]


def move_cost(previous, current):
    (prev_string, prev_fret), (string, fret) = previous, current
    cost = 0.3 * abs(string - prev_string) + 0.05 * fret
    if prev_fret > 0 and fret > 0:
        cost += abs(fret - prev_fret)
    return cost


def assign_frets(pitches, tuning):
    """Cheapest sequence of positions for the pitches (dynamic programming)."""
    if not pitches:
        return []
    layers = [positions(pitch, tuning) for pitch in pitches]
    costs = [{position: 0.05 * position[1] for position in layers[0]}]
    back = [{}]
    for layer in layers[1:]:
        previous_costs = costs[-1]
        layer_costs, layer_back = {}, {}
        for position in layer:
            best = min(previous_costs, key=lambda prev: previous_costs[prev] + move_cost(prev, position))
            layer_costs[position] = previous_costs[best] + move_cost(best, position)
            layer_back[position] = best
        costs.append(layer_costs)
        back.append(layer_back)
    position = min(costs[-1], key=costs[-1].get)
    path = [position]
    for layer_back in reversed(back[1:]):
        position = layer_back[position]
        path.append(position)
    return list(reversed(path))


# ---------------------------------------------------------------------------
# Layout
# ---------------------------------------------------------------------------

def render(slots, frets, tuning, header, bars_per_line):
    """ASCII tab: one row per string, highest string on top, two characters per column."""
    eighths_only = all(slot % 2 == 0 for slot in slots)
    columns_per_bar = SLOTS_PER_BAR // 2 if eighths_only else SLOTS_PER_BAR
    slots_per_column = SLOTS_PER_BAR // columns_per_bar
    count = (["1", "&", "2", "&", "3", "&", "4", "&"] if eighths_only
             else [part for beat in "1234" for part in (beat, "e", "&", "a")])
    bar_width = columns_per_bar * 2
    if not bars_per_line:
        bars_per_line = max(1, (TAB_WIDTH - 2) // (bar_width + 1))

    names = [re.sub(r"-?\d+$", "", pretty_midi.note_number_to_name(pitch)) for pitch in tuning]
    label_width = max(len(name) for name in names)
    bar_count = max(slots) // SLOTS_PER_BAR + 1 if slots else 0
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
                    slot = bar * SLOTS_PER_BAR + column * slots_per_column
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


def main():
    arguments = parse_arguments()
    midi_path, song_dir, out_path = resolve_paths(arguments.source, arguments.out)
    tuning = parse_tuning(arguments.tuning)
    info = read_info(song_dir)

    notes, midi_bpm = load_notes(midi_path)
    kept, dropped = filter_notes(notes, tuning, arguments.min_length, arguments.min_velocity)
    if not kept:
        sys.exit("No notes left in %s after cleanup (%d before)." % (midi_path, len(notes)))

    if arguments.bpm:
        bpm = arguments.bpm
        bpm_note = "%.1f (from --bpm)" % bpm
    else:
        bpm = refine_bpm([note.start for note in kept], midi_bpm)
        bpm_note = "%.1f (MIDI said %.1f, refined to fit the notes)" % (bpm, midi_bpm)

    slots, merged = quantize(kept, bpm, arguments.shift)
    if min(slots) < 0:
        sys.exit("--shift moved notes before bar 1; use a value of 0 or more.")
    frets = assign_frets(list(slots.values()), tuning)

    header = [
        "Song: %s" % info.get("Song", midi_path.stem),
        "Key: %s    BPM: %s    Time: 4/4" % (info.get("Key", "unknown"), bpm_note),
        "Tuning: %s (low to high)" % arguments.tuning,
        "Notes: %d from Basic Pitch, %d in tab. Dropped: %s, %d merged on the same 16th."
        % (len(notes), len(slots),
           ", ".join("%d %s" % (count, reason) for reason, count in dropped.items()), merged),
        "Machine transcription: check it by ear.",
    ]
    text = render(list(slots), frets, tuning, header, arguments.bars_per_line)
    out_path.write_text(text, encoding="utf-8")
    print(text)
    print("Wrote " + str(out_path))


if __name__ == "__main__":
    main()
