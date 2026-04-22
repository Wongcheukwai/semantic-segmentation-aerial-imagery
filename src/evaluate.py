import torch
from torch.utils.data import DataLoader
from pathlib import Path
import json
from tqdm import tqdm

from src.dataset import RoofSolarDataset
from src.model import TinyUNet, ResNetUNet
from src.metrics import compute_metrics
from src.visualise import visualise_predictions, plot_training_history


def load_model(checkpoint_path, device):
    """Load model from checkpoint"""
    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
    config = checkpoint['config']

    # Create model
    if config.get('model_type') == 'resnetunet':
        model = ResNetUNet(n_classes=2, pretrained=False)
    else:
        model = TinyUNet(n_channels=3, n_classes=2)

    # Load weights
    model.load_state_dict(checkpoint['model_state_dict'])
    model = model.to(device)
    model.eval()

    return model, config


def evaluate_model(model, dataloader, device):
    """Evaluate model on entire dataset using global TP/FP/FN/TN accumulation.

    This matches the same metric computation used during training validation,
    giving consistent and comparable results.
    """
    model.eval()

    # Global confusion matrix counters per channel
    global_counts = {
        'roof': {'tp': 0, 'fp': 0, 'fn': 0, 'tn': 0},
        'solar': {'tp': 0, 'fp': 0, 'fn': 0, 'tn': 0}
    }

    with torch.no_grad():
        pbar = tqdm(dataloader, desc="Evaluating")
        for batch in pbar:
            images = batch['image'].to(device)
            masks = batch['masks'].to(device)
            valid_masks = batch['valid_masks'].to(device)

            # Forward pass
            logits = model(images)
            preds = (torch.sigmoid(logits) > 0.5)

            # Accumulate TP/FP/FN/TN for each channel
            for c, channel_name in enumerate(['roof', 'solar']):
                pred_channel = preds[:, c].cpu().numpy()
                mask_channel = masks[:, c].cpu().numpy()
                valid_channel = valid_masks[:, c].cpu().numpy()

                # Only consider valid pixels
                pred_valid = pred_channel[valid_channel == 1]
                mask_valid = mask_channel[valid_channel == 1]

                if len(pred_valid) > 0:
                    global_counts[channel_name]['tp'] += int(((pred_valid == 1) & (mask_valid == 1)).sum())
                    global_counts[channel_name]['fp'] += int(((pred_valid == 1) & (mask_valid == 0)).sum())
                    global_counts[channel_name]['fn'] += int(((pred_valid == 0) & (mask_valid == 1)).sum())
                    global_counts[channel_name]['tn'] += int(((pred_valid == 0) & (mask_valid == 0)).sum())

    # Compute final metrics from global counts
    final_metrics = {}
    for channel_name, counts in global_counts.items():
        final_metrics[channel_name] = compute_metrics(
            counts['tp'], counts['fp'], counts['fn'], counts['tn']
        )

    return final_metrics


def main(checkpoint_path='outputs/checkpoints/best_model.pth', n_samples=5):
    checkpoint_path = Path(checkpoint_path)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    print(f"Using device: {device}")

    if not checkpoint_path.exists():
        print(f"Checkpoint not found at {checkpoint_path}")
        print("Please run training first.")
        return

    # Load model
    print(f"\nLoading model from {checkpoint_path}")
    model, config = load_model(checkpoint_path, device)

    # Create validation dataset
    val_dataset = RoofSolarDataset(
        data_dir=config['data_dir'],
        split='val',
        transform=None,
        img_size=config['img_size']
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=config['batch_size'],
        shuffle=False,
        num_workers=2
    )

    print(f"Validation dataset: {len(val_dataset)} samples")

    # Evaluate model
    print("\nEvaluating model on validation set...")
    metrics = evaluate_model(model, val_loader, device)

    # Print metrics
    print("\n" + "=" * 50)
    print("EVALUATION RESULTS")
    print("=" * 50)

    for channel_name, channel_metrics in metrics.items():
        print(f"\n{channel_name.upper()} Metrics:")
        for metric_name, value in channel_metrics.items():
            print(f"  {metric_name.capitalize()}: {value:.4f}")

    # Save metrics (convert numpy floats to Python floats for JSON)
    metrics_path = Path('outputs/logs/evaluation_metrics.json')
    metrics_path.parent.mkdir(parents=True, exist_ok=True)
    metrics_serialisable = {
        ch: {k: float(v) for k, v in ch_metrics.items()}
        for ch, ch_metrics in metrics.items()
    }
    with open(metrics_path, 'w') as f:
        json.dump(metrics_serialisable, f, indent=2)
    print(f"\nSaved metrics to {metrics_path}")

    # Visualise predictions
    print("\nGenerating visualisation samples...")
    visualise_predictions(model, val_dataset, device, n_samples=n_samples)

    # Plot training history if available
    history_path = Path('outputs/logs/training_history.json')
    if history_path.exists():
        print("\nPlotting training history...")
        plot_training_history(history_path)

    print("\nEvaluation completed!")
