import torch
from fulldataloader import *
from cqtUnet import *
import matplotlib.pyplot as plt
from tqdm import tqdm

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def inference(model, dataset, apply_sections, max_samples=None, device=device):
    ''' returns predictions and ground truth for the whole dataset or until max_samples is reached '''
    model.eval()
    dataset.apply_sections = apply_sections
    
    predictions = []
    ground_truths = []
    
    with torch.no_grad():  # No gradients needed for inference
        for idx in tqdm(range(len(dataset)), desc="Inference Progress"):
            if max_samples is not None and idx >= max_samples:
                break  # Stop if max_samples is reached

            # Return whole piano roll
            if not apply_sections:
                cqt_tensor, piano_roll = dataset[idx]
                num_time_bins = cqt_tensor.shape[2]
                
                
                cqt_sections = get_overlap_windows(cqt_tensor, WINDOW_TIME, OVERLAP_FRAMES)
                
                # Predict on sections
                predicted_sections = []
                for section in cqt_sections:
                    section = section.unsqueeze(0).to(device)  # Add batch dimension
                    output = model(section)
                    predicted_sections.append(output.squeeze(0).cpu())  # Remove batch dim

                predicted_sections = torch.stack(predicted_sections)

                # Reconstruct final piano roll
                final_piano_roll = patches2pianoroll(predicted_sections, OVERLAP_FRAMES, num_time_bins)
                
                predictions.append(final_piano_roll)
                ground_truths.append(piano_roll)  # Store the ground truth

            else:
                cqt_tensor, piano_roll, _ = dataset[idx]  # piano_roll shape: (patches, channels, x, y)
                num_patches = cqt_tensor.shape[0]   
                
                # Ensure tensor has a channel dimension
                if cqt_tensor.dim() == 3:  # (patches, x, y)
                    cqt_tensor = cqt_tensor.unsqueeze(1)  # Add channel dim -> (patches, 1, x, y)
                
                cqt_tensor = cqt_tensor.to(device)  # Move to device
                output = model(cqt_tensor)  # Model output should match (patches, channels, x, y)
                
                predictions.append(output.cpu())  # Store predicted patches
                ground_truths.append(piano_roll)  # Store the ground truth patches
            
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
        
        # Flatten to 1D tensors
        pred_flat = pred_bin.flatten()
        gt_flat = gt.flatten()
        
        # Calculate TP/FP/FN
        tp = (pred_flat * gt_flat).sum().item()
        fp = (pred_flat * (1 - gt_flat)).sum().item()
        fn = ((1 - pred_flat) * gt_flat).sum().item()
        
        # Avoid division by zero
        precision = tp / (tp + fp + 1e-12)
        recall = tp / (tp + fn + 1e-12)
        f1 = 2 * (precision * recall) / (precision + recall + 1e-12)
        
        return precision, recall, f1

    all_precision = []
    all_recall = []
    all_f1 = []

    for pred, gt in zip(predictions, ground_truths):
        # Handle batched patches (3D tensor)
        if pred.dim() == 3:
            for p, g in zip(pred, gt):  # Iterate through patches
                metrics = compute_metrics(p, g)
                all_precision.append(metrics[0])
                all_recall.append(metrics[1])
                all_f1.append(metrics[2])
        # Handle single piano rolls (2D tensor)
        else:
            metrics = compute_metrics(pred, gt)
            all_precision.append(metrics[0])
            all_recall.append(metrics[1])
            all_f1.append(metrics[2])

    # Calculate averages
    avg_precision = sum(all_precision) / len(all_precision)
    avg_recall = sum(all_recall) / len(all_recall)
    avg_f1 = sum(all_f1) / len(all_f1)

    return avg_precision, avg_recall, avg_f1

def plot_pianoroll_comparison(prediction, ground_truth, apply_sections, threshold=None):
    """
    Plots the predicted vs ground truth piano roll for a given sample

    """
    prediction = prediction.cpu().numpy()
    ground_truth = ground_truth.cpu().numpy()

    if threshold is not None:
        prediction = (prediction >= threshold).astype(int)

    #plot full piano roll
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

    #plot individual patches
    else:
        num_patches = prediction.shape[0]
        fig, axes = plt.subplots(num_patches, 2, figsize=(10, 2 * num_patches), sharex=True, sharey=True)
        
        for i in range(num_patches):
            gt_patch = ground_truth[i]  # Shape (x, y)
            pred_patch = prediction[i]  # Shape (x, y)
            
            axes[i, 0].imshow(gt_patch, aspect='auto', cmap='gray_r', origin='lower')
            axes[i, 0].set_title(f"Ground Truth - Patch {i}")
            
            axes[i, 1].imshow(pred_patch, aspect='auto', cmap='gray_r', origin='lower')
            axes[i, 1].set_title(f"Predicted - Patch {i}")
        
        plt.xlabel("Time Frames")
        plt.tight_layout()
        plt.show()

