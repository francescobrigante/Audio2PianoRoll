# ==============================================================================
# Signal-processing and windowing constants shared by the whole pipeline.
# The GuitarSet location is NOT hardcoded: it comes from the GUITARSET_DIR
# environment variable or from CLI arguments (default: ./guitarset).
# ==============================================================================

import os

import librosa

# audio / CQT
SR = 22050
N_NOTES = 52                          # 4 octaves (E2 to E6) + 4 notes or semitones (C2 to E2) = 4*12 + 4 = 52
                                      # 52 notes, however on MIDI there are 53 classes from C2 to E6
BINS_PER_SEMITONE = 3
BINS_PER_OCTAVE = 36                  # 3*12 = 36
N_BINS = BINS_PER_SEMITONE * N_NOTES  # 3*52 = 156
FMIN = librosa.midi_to_hz(36)         # MIDI 36 = C2

# piano roll (MIDI range C2–E6)
MIN_MIDI = 36
MAX_MIDI = 88
N_MIDI_CLASSES = MAX_MIDI - MIN_MIDI + 1   # 53

# windows and overlappings
WINDOW_SIZE = (156, 86)
WINDOW_FREQ, WINDOW_TIME = WINDOW_SIZE
OVERLAP_PERCENTAGE = 0.5
OVERLAP_FRAMES = int(WINDOW_TIME * OVERLAP_PERCENTAGE)

# GuitarSet layout: <root>/audio_*/ for the wav versions, <root>/annotation/ for .jams
AUDIO_SUBDIRS = [
    "audio_hex-pickup_debleeded",
    "audio_hex-pickup_original",
    "audio_mono-mic",
    "audio_mono-pickup_mix",
]
ANNOTATION_SUBDIR = "annotation"


def get_guitarset_dir(cli_value=None) -> str:
    """Resolves the GuitarSet root: CLI argument > GUITARSET_DIR env var > ./guitarset."""
    return cli_value or os.environ.get("GUITARSET_DIR", "guitarset")
