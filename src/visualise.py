import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap
from pathlib import Path
import json

from src.metrics import compute_metrics_per_image


def denormalise_image(image_tensor):
    """Denormalise image tensor for visualisation"""
    mean = np.array([0.485, 0.456, 0.406])
    std = np.array([0.229, 0.224, 0.225])

    image = image_tensor.numpy().transpose(1, 2, 0)
    image = image * std + mean
    image = np.clip(image, 0, 1)

    return image


def _select_informative_samples(dataset, n_samples):
    """Select samples that have roof positive labels, prioritising those with solar too.

    Strategy:
      1. Prefer images that have both roof AND solar positive labels
      2. Then images that have roof positive labels
      3. Fall back to random if not enough
    """
    both_positive = []
    roof_positive = []
    other = []

    for idx in range(len(dataset)):
        sample = dataset[idx]
        roof_mask = sample['masks'][0].numpy()
        solar_mask = sample['masks'][1].numpy()
        roof_valid = sample['valid_masks'][0].numpy()
        solar_valid = sample['valid_masks'][1].numpy()

        has_roof = (roof_mask[roof_valid == 1] >= 1).any() if roof_valid.sum() > 0 else False
        has_solar = (solar_mask[solar_valid == 1] >= 1).any() if solar_valid.sum() > 0 else False

        if has_roof and has_solar:
            both_positive.append(idx)
        elif has_roof:
            roof_positive.append(idx)
        else:
            other.append(idx)

    # Combine: both > roof-only > other, then take n_samples
    candidates = both_positive + roof_positive + other

    # Shuffle within each priority group for variety
    np.random.shuffle(both_positive)
    np.random.shuffle(roof_positive)
    np.random.shuffle(other)
    candidates = both_positive + roof_positive + other

    return candidates[:n_samples]


def _create_overlay(image_np, mask, color, alpha=0.4):
    """Create a colored semi-transparent overlay on the image.

    Args:
        image_np: Original image (H, W, 3)
        mask: Binary mask (H, W)
        color: RGB tuple e.g. (1, 0, 0) for red
        alpha: Transparency

    Returns:
        Blended image with overlay
    """
    overlay = image_np.copy()
    for c in range(3):
        overlay[:, :, c] = np.where(
            mask > 0,
            image_np[:, :, c] * (1 - alpha) + color[c] * alpha,
            image_np[:, :, c]
        )
    return overlay


def visualise_predictions(model, dataset, device, n_samples=5, save_dir='outputs/visualisations'):
    """Visualise predictions on sample images with informative sample selection."""
    import torch
    model.eval()

    save_dir = Path(save_dir)
    save_dir.mkdir(parents=True, exist_ok=True)

    # Select informative samples (prioritise images with roof/solar labels)
    indices = _select_informative_samples(dataset, n_samples)

    for idx_num, idx in enumerate(indices):
        sample = dataset[idx]

        # Prepare input
        image = sample['image'].unsqueeze(0).to(device)
        masks = sample['masks'].unsqueeze(0).to(device)
        valid_masks = sample['valid_masks'].unsqueeze(0).to(device)

        # Get prediction
        with torch.no_grad():
            logits = model(image)
            preds = torch.sigmoid(logits)

        # Convert to numpy
        image_np = denormalise_image(image[0].cpu())
        roof_mask = masks[0, 0].cpu().numpy()
        solar_mask = masks[0, 1].cpu().numpy()
        roof_valid = valid_masks[0, 0].cpu().numpy()
        solar_valid = valid_masks[0, 1].cpu().numpy()
        roof_pred = preds[0, 0].cpu().numpy()
        solar_pred = preds[0, 1].cpu().numpy()

        # Binary predictions
        roof_pred_bin = (roof_pred > 0.5).astype(np.float32)
        solar_pred_bin = (solar_pred > 0.5).astype(np.float32)

        # --- Create figure ---
        fig, axes = plt.subplots(2, 4, figsize=(20, 10))

        # Row 0: Roof
        # [0,0] Original image
        axes[0, 0].imshow(image_np)
        axes[0, 0].set_title('Original Image', fontsize=11, fontweight='bold')
        axes[0, 0].axis('off')

        # [0,1] Roof GT overlay (red)
        roof_gt_overlay = _create_overlay(image_np, roof_mask * roof_valid, (1, 0, 0), alpha=0.45)
        axes[0, 1].imshow(roof_gt_overlay)
        n_roof_pos = int((roof_mask * roof_valid).sum())
        axes[0, 1].set_title(f'Roof GT ({n_roof_pos} px)', fontsize=11, fontweight='bold')
        axes[0, 1].axis('off')

        # [0,2] Roof prediction overlay (red)
        roof_pred_overlay = _create_overlay(image_np, roof_pred_bin, (1, 0, 0), alpha=0.45)
        axes[0, 2].imshow(roof_pred_overlay)
        axes[0, 2].set_title('Roof Prediction', fontsize=11, fontweight='bold')
        axes[0, 2].axis('off')

        # [0,3] Roof probability heatmap (full model output, no valid_mask)
        axes[0, 3].imshow(image_np, alpha=0.3)
        im_roof = axes[0, 3].imshow(roof_pred, cmap='RdYlGn_r', vmin=0, vmax=1, alpha=0.7)
        axes[0, 3].set_title('Roof Probability', fontsize=11, fontweight='bold')
        axes[0, 3].axis('off')
        plt.colorbar(im_roof, ax=axes[0, 3], fraction=0.046, pad=0.04)

        # Row 1: Solar
        # [1,0] Combined prediction
        combined_overlay = image_np.copy()
        # Red for roof
        combined_overlay = _create_overlay(combined_overlay, roof_pred_bin, (1, 0, 0), alpha=0.35)
        # Blue for solar
        combined_overlay = _create_overlay(combined_overlay, solar_pred_bin, (0, 0.3, 1), alpha=0.5)
        axes[1, 0].imshow(combined_overlay)
        axes[1, 0].set_title('Combined (Red=Roof, Blue=Solar)', fontsize=11, fontweight='bold')
        axes[1, 0].axis('off')

        # [1,1] Solar GT overlay (blue)
        solar_gt_overlay = _create_overlay(image_np, solar_mask * solar_valid, (0, 0.3, 1), alpha=0.5)
        axes[1, 1].imshow(solar_gt_overlay)
        n_solar_pos = int((solar_mask * solar_valid).sum())
        axes[1, 1].set_title(f'Solar GT ({n_solar_pos} px)', fontsize=11, fontweight='bold')
        axes[1, 1].axis('off')

        # [1,2] Solar prediction overlay (blue)
        solar_pred_overlay = _create_overlay(image_np, solar_pred_bin, (0, 0.3, 1), alpha=0.5)
        axes[1, 2].imshow(solar_pred_overlay)
        axes[1, 2].set_title('Solar Prediction', fontsize=11, fontweight='bold')
        axes[1, 2].axis('off')

        # [1,3] Solar probability heatmap (full model output, no valid_mask)
        axes[1, 3].imshow(image_np, alpha=0.3)
        im_solar = axes[1, 3].imshow(solar_pred, cmap='RdYlGn_r', vmin=0, vmax=1, alpha=0.7)
        axes[1, 3].set_title('Solar Probability', fontsize=11, fontweight='bold')
        axes[1, 3].axis('off')
        plt.colorbar(im_solar, ax=axes[1, 3], fraction=0.046, pad=0.04)

        # Compute metrics
        roof_metrics = compute_metrics_per_image(roof_pred_bin, roof_mask, roof_valid)
        solar_metrics = compute_metrics_per_image(solar_pred_bin, solar_mask, solar_valid)

        # Title with metrics
        title = (
            f'Sample {idx_num + 1}\n'
            f'Roof — IoU: {roof_metrics["iou"]:.3f}, F1: {roof_metrics["f1"]:.3f}, '
            f'Prec: {roof_metrics["precision"]:.3f}, Rec: {roof_metrics["recall"]:.3f}\n'
            f'Solar — IoU: {solar_metrics["iou"]:.3f}, F1: {solar_metrics["f1"]:.3f}, '
            f'Prec: {solar_metrics["precision"]:.3f}, Rec: {solar_metrics["recall"]:.3f}'
        )
        fig.suptitle(title, fontsize=13, fontweight='bold', y=1.02)

        plt.tight_layout()

        # Save
        save_path = save_dir / f'prediction_sample_{idx_num + 1}.png'
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        plt.close()

        print(f"Saved visualisation to {save_path}")


def plot_training_history(history_path, save_dir='outputs/visualisations'):
    """Plot training history from JSON file with improved styling."""
    save_dir = Path(save_dir)
    save_dir.mkdir(parents=True, exist_ok=True)

    with open(history_path, 'r') as f:
        history = json.load(f)

    epochs = range(1, len(history['train_loss']) + 1)

    fig, axes = plt.subplots(2, 3, figsize=(16, 9))

    # --- Loss ---
    axes[0, 0].plot(epochs, history['train_loss'], 'b-o', markersize=4, label='Train', linewidth=2)
    axes[0, 0].plot(epochs, history['val_loss'], 'r-o', markersize=4, label='Val', linewidth=2)
    axes[0, 0].set_xlabel('Epoch')
    axes[0, 0].set_ylabel('Loss')
    axes[0, 0].set_title('Loss', fontweight='bold')
    axes[0, 0].legend()
    axes[0, 0].grid(True, alpha=0.3)

    # --- Accuracy ---
    roof_acc = [m['roof']['accuracy'] for m in history['val_metrics']]
    solar_acc = [m['solar']['accuracy'] for m in history['val_metrics']]
    axes[1, 0].plot(epochs, roof_acc, 'g-o', markersize=4, label='Roof', linewidth=2)
    axes[1, 0].plot(epochs, solar_acc, 'b-o', markersize=4, label='Solar', linewidth=2)
    axes[1, 0].set_xlabel('Epoch')
    axes[1, 0].set_ylabel('Accuracy')
    axes[1, 0].set_title('Pixel Accuracy', fontweight='bold')
    axes[1, 0].legend()
    axes[1, 0].grid(True, alpha=0.3)

    # --- F1 and IoU per channel ---
    colors = {'roof': '#2ecc71', 'solar': '#3498db'}
    for i, metric in enumerate(['f1', 'iou']):
        roof_values = [m['roof'][metric] for m in history['val_metrics']]
        solar_values = [m['solar'][metric] for m in history['val_metrics']]

        axes[0, 1 + i].plot(epochs, roof_values, '-o', color=colors['roof'],
                            markersize=4, linewidth=2)
        axes[0, 1 + i].set_xlabel('Epoch')
        axes[0, 1 + i].set_ylabel(metric.upper())
        axes[0, 1 + i].set_title(f'Roof {metric.upper()}', fontweight='bold')
        axes[0, 1 + i].grid(True, alpha=0.3)
        axes[0, 1 + i].set_ylim(bottom=0)

        axes[1, 1 + i].plot(epochs, solar_values, '-o', color=colors['solar'],
                            markersize=4, linewidth=2)
        axes[1, 1 + i].set_xlabel('Epoch')
        axes[1, 1 + i].set_ylabel(metric.upper())
        axes[1, 1 + i].set_title(f'Solar {metric.upper()}', fontweight='bold')
        axes[1, 1 + i].grid(True, alpha=0.3)
        axes[1, 1 + i].set_ylim(bottom=0)

    plt.suptitle('Training History', fontsize=14, fontweight='bold')
    plt.tight_layout()

    save_path = save_dir / 'training_history.png'
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()

    print(f"Saved training history plot to {save_path}")
