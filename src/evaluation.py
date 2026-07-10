# ==============================================================================
# Inference (section-wise prediction + overlap-averaged reconstruction),
# frame-level precision/recall/F1, and predicted vs ground-truth plots.
# ==============================================================================

import matplotlib.pyplot as plt
import torch
from tqdm import tqdm

from src.config import OVERLAP_FRAMES, WINDOW_TIME
from src.data.guitarset import get_overlap_windows, patches2pianoroll


def inference(model, dataset, apply_sections, max_samples=None, device=None):
    ''' returns predictions and ground truth for the whole dataset or until max_samples is reached '''
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.eval()
    dataset.apply_sections = apply_sections

    predictions = []
    ground_truths = []

    with torch.no_grad():
        for idx in tqdm(range(len(dataset)), desc="Inference Progress"):
            if max_samples is not None and idx >= max_samples:
                break

            # full piano roll: predict on sections, then reconstruct
            if not apply_sections:
                cqt_tensor, piano_roll = dataset[idx]                      # (1, 156, T), (53, T)
                num_time_bins = cqt_tensor.shape[2]

                cqt_sections = get_overlap_windows(cqt_tensor, WINDOW_TIME, OVERLAP_FRAMES)  # (S, 1, 156, 86)

                predicted_sections = []
                for section in cqt_sections:
                    section = section.unsqueeze(0).to(device)              # (1, 1, 156, 86)
                    output = model(section)                                # (1, 53, 86)
                    predicted_sections.append(output.squeeze(0).cpu())

                predicted_sections = torch.stack(predicted_sections)       # (S, 53, 86)

                # reconstruct final piano roll averaging the overlaps
                final_piano_roll = patches2pianoroll(predicted_sections, OVERLAP_FRAMES, num_time_bins)  # (53, T)

                predictions.append(final_piano_roll)
                ground_truths.append(piano_roll)

            # section-wise: predict every patch independently
            else:
                cqt_tensor, piano_roll, _ = dataset[idx]                   # (S, 1, 156, 86), (S, 53, 86)

                if cqt_tensor.dim() == 3:                                  # (S, 156, 86) → add channel dim
                    cqt_tensor = cqt_tensor.unsqueeze(1)

                cqt_tensor = cqt_tensor.to(device)
                output = model(cqt_tensor)                                 # (S, 53, 86)

                predictions.append(output.cpu())
                ground_truths.append(piano_roll)

    return predictions, ground_truths


def get_metrics(predictions, ground_truths, threshold=0.5):
    """
    Computes average precision, recall, and F1-score across all samples.
    Handles both full piano rolls and batched patches.

    Args:
        predictions: List of tensors (each shape [53,86] or [num_patches,53,86])
        ground_truths: List of tensors (same shapes as predictions)
        threshold: Activation threshold for binary classification

    Returns:
        (avg_precision, avg_recall, avg_f1)
    """
    def compute_metrics(pred, gt):
        """Helper to compute metrics for a single piano roll"""
        pred_bin = (pred >= threshold).float()
        gt = gt.float()

        pred_flat = pred_bin.flatten()
        gt_flat = gt.flatten()

        tp = (pred_flat * gt_flat).sum().item()
        fp = (pred_flat * (1 - gt_flat)).sum().item()
        fn = ((1 - pred_flat) * gt_flat).sum().item()

        precision = tp / (tp + fp + 1e-12)
        recall = tp / (tp + fn + 1e-12)
        f1 = 2 * (precision * recall) / (precision + recall + 1e-12)

        return precision, recall, f1

    all_precision = []
    all_recall = []
    all_f1 = []

    for pred, gt in zip(predictions, ground_truths):
        if pred.dim() == 3:  # batched patches
            for p, g in zip(pred, gt):
                metrics = compute_metrics(p, g)
                all_precision.append(metrics[0])
                all_recall.append(metrics[1])
                all_f1.append(metrics[2])
        else:                # single piano roll
            metrics = compute_metrics(pred, gt)
            all_precision.append(metrics[0])
            all_recall.append(metrics[1])
            all_f1.append(metrics[2])

    avg_precision = sum(all_precision) / len(all_precision)
    avg_recall = sum(all_recall) / len(all_recall)
    avg_f1 = sum(all_f1) / len(all_f1)

    return avg_precision, avg_recall, avg_f1


def plot_pianoroll_comparison(prediction, ground_truth, apply_sections, threshold=None):
    """Plots the predicted vs ground truth piano roll for a given sample."""
    prediction = prediction.cpu().numpy()
    ground_truth = ground_truth.cpu().numpy()

    if threshold is not None:
        prediction = (prediction >= threshold).astype(int)

    # plot full piano roll
    if not apply_sections:
        fig, axes = plt.subplots(2, 1, figsize=(20, 6), sharex=True, sharey=True)

        axes[0].imshow(ground_truth, aspect='auto', cmap='gray_r', origin='lower')
        axes[0].set_title("Ground Truth Piano Roll")
        axes[0].set_ylabel("MIDI Notes")

        axes[1].imshow(prediction, aspect='auto', cmap='gray_r', origin='lower')
        axes[1].set_title("Predicted Piano Roll")
        axes[1].set_xlabel("Time Frames")
        axes[1].set_ylabel("MIDI Notes")

        plt.show()

    # plot individual patches
    else:
        num_patches = prediction.shape[0]
        fig, axes = plt.subplots(num_patches, 2, figsize=(10, 2 * num_patches), sharex=True, sharey=True)

        for i in range(num_patches):
            gt_patch = ground_truth[i]
            pred_patch = prediction[i]

            axes[i, 0].imshow(gt_patch, aspect='auto', cmap='gray_r', origin='lower')
            axes[i, 0].set_title(f"Ground Truth - Patch {i}")

            axes[i, 1].imshow(pred_patch, aspect='auto', cmap='gray_r', origin='lower')
            axes[i, 1].set_title(f"Predicted - Patch {i}")

        plt.xlabel("Time Frames")
        plt.tight_layout()
        plt.show()
