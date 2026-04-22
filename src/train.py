import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from tqdm import tqdm
from pathlib import Path
import json

from src.dataset import RoofSolarDataset, get_augmentation
from src.model import TinyUNet, ResNetUNet
from src.metrics import compute_metrics


class MaskedBCEWithLogitsLoss(nn.Module):
    """BCE loss that handles invalid pixels (-1 in original mask)"""

    def __init__(self):
        super().__init__()

    def forward(self, logits, targets, valid_masks):
        loss_unreduced = nn.functional.binary_cross_entropy_with_logits(
            logits, targets, reduction='none'
        )

        # Apply valid mask and compute mean
        loss_masked = loss_unreduced * valid_masks
        n_valid = valid_masks.sum() + 1e-8  # avoid division by zero

        return loss_masked.sum() / n_valid


class BCEDiceLoss(nn.Module):
    """Combined BCE + Dice loss. BCE for pixel stability, Dice for region overlap."""

    def __init__(self, bce_weight=0.5, dice_weight=0.5):
        super().__init__()
        self.bce_weight = bce_weight
        self.dice_weight = dice_weight

    def forward(self, logits, targets, valid_masks):
        n_valid = valid_masks.sum() + 1e-8

        # Masked BCE
        bce = nn.functional.binary_cross_entropy_with_logits(
            logits, targets, reduction='none'
        )
        bce = (bce * valid_masks).sum() / n_valid

        # Masked Dice
        probs = torch.sigmoid(logits)
        probs = probs * valid_masks
        targets = targets * valid_masks

        intersection = (probs * targets).sum()
        union = probs.sum() + targets.sum()
        dice = 1.0 - (2.0 * intersection + 1e-8) / (union + 1e-8)

        return self.bce_weight * bce + self.dice_weight * dice


def train_epoch(model, dataloader, criterion, optimiser, device, channel_weights=None):
    """Train for one epoch"""
    model.train()
    running_loss = 0.0
    n_batches = 0

    pbar = tqdm(dataloader, desc="Training")
    for batch in pbar:
        images = batch['image'].to(device)
        masks = batch['masks'].to(device)
        valid_masks = batch['valid_masks'].to(device)

        # Forward pass
        logits = model(images)

        # Compute loss for each channel
        losses = []
        weights = channel_weights if channel_weights is not None else [1.0, 1.0]

        for i in range(2):  # 2 channels: roof and solar
            channel_loss = criterion(
                logits[:, i:i+1, :, :],
                masks[:, i:i+1, :, :],
                valid_masks[:, i:i+1, :, :]
            )
            losses.append(channel_loss * weights[i])

        # Total loss
        loss = sum(losses)

        # Backward pass
        optimiser.zero_grad()
        loss.backward()
        optimiser.step()

        # Update statistics
        running_loss += loss.item()
        n_batches += 1

        # Update progress bar
        pbar.set_postfix({'loss': f'{loss.item():.4f}'})

    return running_loss / max(n_batches, 1)


def validate_epoch(model, dataloader, criterion, device, channel_weights=None):
    """Validate for one epoch"""
    model.eval()
    running_loss = 0.0
    n_batches = 0

    # Metrics storage
    all_metrics = {
        'roof': {'tp': 0, 'fp': 0, 'fn': 0, 'tn': 0},
        'solar': {'tp': 0, 'fp': 0, 'fn': 0, 'tn': 0}
    }

    with torch.no_grad():
        pbar = tqdm(dataloader, desc="Validation")
        for batch in pbar:
            images = batch['image'].to(device)
            masks = batch['masks'].to(device)
            valid_masks = batch['valid_masks'].to(device)

            # Forward pass
            logits = model(images)

            # Compute loss
            losses = []
            weights = channel_weights if channel_weights is not None else [1.0, 1.0]

            for i in range(2):
                channel_loss = criterion(
                    logits[:, i:i+1, :, :],
                    masks[:, i:i+1, :, :],
                    valid_masks[:, i:i+1, :, :]
                )
                losses.append(channel_loss * weights[i])

            loss = sum(losses)
            running_loss += loss.item()
            n_batches += 1

            # Compute predictions
            preds = torch.sigmoid(logits) > 0.5

            # Update metrics for each channel
            for i, channel_name in enumerate(['roof', 'solar']):
                pred_channel = preds[:, i].cpu().numpy()
                mask_channel = masks[:, i].cpu().numpy()
                valid_channel = valid_masks[:, i].cpu().numpy()

                # Only consider valid pixels
                pred_valid = pred_channel[valid_channel == 1]
                mask_valid = mask_channel[valid_channel == 1]

                if len(pred_valid) > 0:
                    tp = ((pred_valid == 1) & (mask_valid == 1)).sum()
                    fp = ((pred_valid == 1) & (mask_valid == 0)).sum()
                    fn = ((pred_valid == 0) & (mask_valid == 1)).sum()
                    tn = ((pred_valid == 0) & (mask_valid == 0)).sum()

                    all_metrics[channel_name]['tp'] += tp
                    all_metrics[channel_name]['fp'] += fp
                    all_metrics[channel_name]['fn'] += fn
                    all_metrics[channel_name]['tn'] += tn

            pbar.set_postfix({'loss': f'{loss.item():.4f}'})

    # Compute final metrics
    final_metrics = {}
    for channel_name, counts in all_metrics.items():
        metrics = compute_metrics(
            counts['tp'], counts['fp'], counts['fn'], counts['tn']
        )
        final_metrics[channel_name] = metrics

    return running_loss / max(n_batches, 1), final_metrics


def main(config=None):
    if config is None:
        config = {}

    config = {
        'data_dir': config.get('data_dir', 'MLtestdata'),
        'batch_size': config.get('batch_size', 4),
        'num_epochs': config.get('num_epochs', 10),
        'learning_rate': config.get('learning_rate', 1e-3),
        'weight_decay': config.get('weight_decay', 1e-4),
        'img_size': config.get('img_size', 896),
        'model_type': config.get('model_type', 'tinyunet'),
        'loss_type': config.get('loss_type', 'bce'),
        'channel_weights': config.get('channel_weights', [1.0, 2.0]),
        'device': config.get('device', 'cuda' if torch.cuda.is_available() else 'cpu'),
    }

    print("Configuration:")
    print(json.dumps(config, indent=2))

    device = torch.device(config['device'])
    print(f"\nUsing device: {device}")

    # Create datasets
    train_dataset = RoofSolarDataset(
        data_dir=config['data_dir'],
        split='train',
        transform=get_augmentation(),
        img_size=config['img_size']
    )

    val_dataset = RoofSolarDataset(
        data_dir=config['data_dir'],
        split='val',
        transform=None,
        img_size=config['img_size']
    )

    # Create dataloaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=config['batch_size'],
        shuffle=True,
        num_workers=2,
        pin_memory=True if config['device'] == 'cuda' else False
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=config['batch_size'],
        shuffle=False,
        num_workers=2,
        pin_memory=True if config['device'] == 'cuda' else False
    )

    print(f"\nDataset sizes:")
    print(f"  Training: {len(train_dataset)} samples")
    print(f"  Validation: {len(val_dataset)} samples")

    # Create model
    if config['model_type'] == 'resnetunet':
        model = ResNetUNet(n_classes=2, pretrained=True).to(device)
    else:
        model = TinyUNet(n_channels=3, n_classes=2).to(device)

    n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"\nModel: {config['model_type']}")
    print(f"Number of parameters: {n_params:,}")

    # Loss
    if config['loss_type'] == 'bce_dice':
        criterion = BCEDiceLoss()
    else:
        criterion = MaskedBCEWithLogitsLoss()
    optimiser = optim.AdamW(
        model.parameters(),
        lr=config['learning_rate'],
        weight_decay=config['weight_decay']
    )

    # Learning rate scheduler
    scheduler = optim.lr_scheduler.CosineAnnealingLR(
        optimiser, T_max=config['num_epochs']
    )

    # Training history
    history = {
        'train_loss': [],
        'val_loss': [],
        'val_metrics': []
    }

    # Create output directories
    checkpoint_dir = Path('outputs/checkpoints')
    logs_dir = Path('outputs/logs')
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    logs_dir.mkdir(parents=True, exist_ok=True)

    # Training loop
    print("\nStarting training...")
    print("-" * 50)

    best_val_loss = float('inf')

    for epoch in range(config['num_epochs']):
        print(f"\nEpoch {epoch+1}/{config['num_epochs']}")
        print(f"Learning rate: {scheduler.get_last_lr()[0]:.6f}")

        # Train
        train_loss = train_epoch(
            model, train_loader, criterion, optimiser, device,
            channel_weights=torch.tensor(config['channel_weights']).to(device)
        )

        # Validate
        val_loss, val_metrics = validate_epoch(
            model, val_loader, criterion, device,
            channel_weights=torch.tensor(config['channel_weights']).to(device)
        )

        # Update learning rate
        scheduler.step()

        # Store history
        history['train_loss'].append(train_loss)
        history['val_loss'].append(val_loss)
        history['val_metrics'].append(val_metrics)

        # Print metrics
        print(f"\nEpoch {epoch+1} Results:")
        print(f"  Train Loss: {train_loss:.4f}")
        print(f"  Val Loss: {val_loss:.4f}")

        print("\n  Validation Metrics:")
        for channel_name, metrics in val_metrics.items():
            print(f"    {channel_name.upper()}:")
            print(f"      Accuracy: {metrics['accuracy']:.4f}")
            print(f"      Precision: {metrics['precision']:.4f}")
            print(f"      Recall: {metrics['recall']:.4f}")
            print(f"      F1/Dice: {metrics['f1']:.4f}")
            print(f"      IoU: {metrics['iou']:.4f}")

        # Save checkpoint if best
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            checkpoint_path = checkpoint_dir / 'best_model.pth'
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimiser.state_dict(),
                'train_loss': train_loss,
                'val_loss': val_loss,
                'val_metrics': val_metrics,
                'config': config
            }, checkpoint_path)
            print(f"\n  Saved best model (val_loss: {val_loss:.4f})")

    # Save final model
    final_checkpoint_path = checkpoint_dir / 'final_model.pth'
    torch.save({
        'epoch': config['num_epochs'],
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimiser.state_dict(),
        'history': history,
        'config': config
    }, final_checkpoint_path)

    print("\n" + "=" * 50)
    print("Training completed!")
    print(f"Best validation loss: {best_val_loss:.4f}")
    print(f"Checkpoints saved in: {checkpoint_dir}")

    # Save training history
    history_path = logs_dir / 'training_history.json'
    with open(history_path, 'w') as f:
        # Convert numpy values to float for JSON serialisation
        history_serialisable = {
            'train_loss': history['train_loss'],
            'val_loss': history['val_loss'],
            'val_metrics': [
                {
                    channel: {
                        metric: float(value) for metric, value in ch_metrics.items()
                    }
                    for channel, ch_metrics in epoch_metrics.items()
                }
                for epoch_metrics in history['val_metrics']
            ]
        }
        json.dump(history_serialisable, f, indent=2)

    print(f"Training history saved to: {history_path}")


if __name__ == "__main__":
    main()
