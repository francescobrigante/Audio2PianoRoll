# ==============================================================================
# CLI: trains the CQT U-Net on GuitarSet.
# Usage: python scripts/train.py --data-dir /path/to/guitarset [--epochs 144]
# ==============================================================================

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import torch
from torch.utils.data import DataLoader

from src.config import get_guitarset_dir
from src.data.guitarset import GuitarSetDataset, load_or_create_split
from src.losses import CombinedLoss
from src.model import UNetPianoRoll
from src.training import train


def main():
    parser = argparse.ArgumentParser(description="Train the CQT U-Net on GuitarSet.")
    parser.add_argument("--data-dir", default=None, help="GuitarSet root (default: $GUITARSET_DIR or ./guitarset)")
    parser.add_argument("--epochs", type=int, default=144)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--dropout", type=float, default=0.5)
    parser.add_argument("--loss-alpha", type=float, default=0.5, help="BCE weight in the combined BCE+MSE loss")
    parser.add_argument("--split-path", default="test_split.json")
    parser.add_argument("--checkpoint-dir", default="models")
    parser.add_argument("--num-workers", type=int, default=8)
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    data_dir = get_guitarset_dir(args.data_dir)

    dataset = GuitarSetDataset(data_dir)
    train_samples, _ = load_or_create_split(dataset.samples, split_path=args.split_path)

    train_dataset = GuitarSetDataset(data_dir, apply_nmf=False, apply_sections=True)
    train_dataset.samples = train_samples

    train_loader = DataLoader(
        train_dataset, batch_size=1, shuffle=True, num_workers=args.num_workers,
        pin_memory=True, persistent_workers=args.num_workers > 0, prefetch_factor=8 if args.num_workers > 0 else None,
    )

    model = UNetPianoRoll(dropout_rate=args.dropout).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
    criterion = CombinedLoss(alpha=args.loss_alpha).to(device)

    train(model, train_loader, criterion, optimizer, num_epochs=args.epochs,
          checkpoint_dir=args.checkpoint_dir, device=device)


if __name__ == "__main__":
    main()
