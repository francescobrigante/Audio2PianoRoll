# ==============================================================================
# CLI: transcribes a single guitar audio file into a piano roll.
# Usage: python scripts/transcribe.py path/to/audio.wav [--threshold 0.15]
# Outputs <stem>_pianoroll.npy (53, T) and <stem>_pianoroll.png next to it
# (or into --output-dir).
# ==============================================================================

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np
import torch

from src.config import HOP_LENGTH_SECONDS, MAX_MIDI, MIN_MIDI, OVERLAP_FRAMES, WINDOW_TIME
from src.data.guitarset import get_overlap_windows, load_cqt, patches2pianoroll
from src.data.midi_utils import plot_piano_roll
from src.model import UNetPianoRoll


def transcribe(audio_path, model, device, threshold=None):
    """Audio file → reconstructed piano roll (53, T) as a numpy array."""
    cqt = torch.tensor(load_cqt(audio_path)).unsqueeze(0)        # (1, 156, T)
    num_time_bins = cqt.shape[2]

    cqt_sections = get_overlap_windows(cqt, WINDOW_TIME, OVERLAP_FRAMES)  # (S, 1, 156, 86)

    predicted_sections = []
    with torch.no_grad():
        for section in cqt_sections:
            output = model(section.unsqueeze(0).to(device))      # (1, 53, 86)
            predicted_sections.append(output.squeeze(0).cpu())

    piano_roll = patches2pianoroll(torch.stack(predicted_sections), OVERLAP_FRAMES, num_time_bins)  # (53, T)
    piano_roll = piano_roll.numpy()

    if threshold is not None:
        piano_roll = (piano_roll >= threshold).astype(np.float32)
    return piano_roll


def main():
    parser = argparse.ArgumentParser(description="Transcribe a guitar audio file into a piano roll.")
    parser.add_argument("audio", help="path to the input audio file (wav)")
    parser.add_argument("--checkpoint", default="models/final144.pth")
    parser.add_argument("--threshold", type=float, default=0.15,
                        help="binarization threshold; pass -1 to keep raw activations")
    parser.add_argument("--output-dir", default=None, help="where to save outputs (default: next to the audio)")
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model = UNetPianoRoll().to(device)
    model.load_state_dict(torch.load(args.checkpoint, map_location=device))
    model.eval()

    threshold = None if args.threshold < 0 else args.threshold
    piano_roll = transcribe(args.audio, model, device, threshold=threshold)

    audio_path = Path(args.audio)
    out_dir = Path(args.output_dir) if args.output_dir else audio_path.parent
    out_dir.mkdir(parents=True, exist_ok=True)
    out_stem = out_dir / f"{audio_path.stem}_pianoroll"

    np.save(f"{out_stem}.npy", piano_roll)

    # plot on the real time/pitch axes and save alongside the .npy
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    time_axis = np.arange(piano_roll.shape[1]) * HOP_LENGTH_SECONDS
    midi_pitches = np.arange(MIN_MIDI, MAX_MIDI + 1)
    plot_piano_roll(piano_roll, time_axis, midi_pitches, show=False)
    plt.savefig(f"{out_stem}.png", dpi=150, bbox_inches="tight")

    active_notes = int((piano_roll > 0).any(axis=1).sum())
    print(f"Piano roll: {piano_roll.shape} | active pitches: {active_notes}")
    print(f"Saved: {out_stem}.npy, {out_stem}.png")


if __name__ == "__main__":
    main()
