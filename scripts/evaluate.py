# ==============================================================================
# CLI: evaluates a trained checkpoint on the persisted GuitarSet test split.
# Usage: python scripts/evaluate.py --checkpoint models/final144.pth [--full-roll]
# ==============================================================================

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import torch

from src.config import get_guitarset_dir
from src.data.guitarset import GuitarSetDataset, load_or_create_split
from src.evaluation import get_metrics, inference
from src.model import UNetPianoRoll


def main():
    parser = argparse.ArgumentParser(description="Evaluate a checkpoint on the GuitarSet test split.")
    parser.add_argument("--checkpoint", default="models/final144.pth")
    parser.add_argument("--data-dir", default=None, help="GuitarSet root (default: $GUITARSET_DIR or ./guitarset)")
    parser.add_argument("--split-path", default="test_split.json")
    parser.add_argument("--full-roll", action="store_true",
                        help="evaluate on reconstructed full piano rolls instead of individual sections")
    parser.add_argument("--threshold", type=float, default=None,
                        help="activation threshold (default: 0.11 sections, 0.15 full roll)")
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    data_dir = get_guitarset_dir(args.data_dir)
    apply_sections = not args.full_roll
    threshold = args.threshold if args.threshold is not None else (0.11 if apply_sections else 0.15)

    dataset = GuitarSetDataset(data_dir)
    _, test_samples = load_or_create_split(dataset.samples, split_path=args.split_path)

    test_dataset = GuitarSetDataset(data_dir, apply_nmf=False, apply_sections=apply_sections)
    test_dataset.samples = test_samples

    model = UNetPianoRoll().to(device)
    model.load_state_dict(torch.load(args.checkpoint, map_location=device))

    predictions, ground_truths = inference(model, test_dataset, apply_sections=apply_sections, device=device)
    avg_prec, avg_rec, avg_f1 = get_metrics(predictions, ground_truths, threshold=threshold)

    mode = "sections" if apply_sections else "full roll"
    print(f"[{mode}, threshold={threshold}]")
    print(f"Average Precision: {avg_prec}")
    print(f"Average Recall: {avg_rec}")
    print(f"Average F1 Score: {avg_f1}")


if __name__ == "__main__":
    main()
