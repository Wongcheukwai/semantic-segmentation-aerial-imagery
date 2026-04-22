#!/usr/bin/env python
"""
Nearmap Roof and Solar Panel Detection Pipeline

Usage:
    python run.py
    python run.py --model resnetunet --loss bce_dice
    python run.py --model resnetunet --loss bce_dice --epochs 20 --batch-size 8
"""

import argparse
import sys
from pathlib import Path


def main():
    parser = argparse.ArgumentParser(
        description='Nearmap Roof and Solar Panel Detection',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )

    # Model
    parser.add_argument('--model', choices=['tinyunet', 'resnetunet'], default='tinyunet',
                        help='Model architecture')
    parser.add_argument('--loss', choices=['bce', 'bce_dice'], default='bce',
                        help='Loss function')

    # Training
    parser.add_argument('--epochs', type=int, default=10, help='Number of training epochs')
    parser.add_argument('--batch-size', type=int, default=4, help='Batch size')
    parser.add_argument('--lr', type=float, default=1e-3, help='Learning rate')
    parser.add_argument('--img-size', type=int, default=896, help='Input image size')

    # Data
    parser.add_argument('--data-dir', type=str, default='MLtestdata', help='Path to dataset')

    # Evaluation
    parser.add_argument('--n-samples', type=int, default=5, help='Number of visualisation samples')

    args = parser.parse_args()

    # Check data exists
    data_dir = Path(args.data_dir)
    if not data_dir.exists():
        print(f"Error: Data directory '{data_dir}' not found!")
        sys.exit(1)

    n_images = len(list(data_dir.glob("*.jpg")))
    n_roof_masks = len(list(data_dir.glob("*_2.npz")))
    n_solar_masks = len(list(data_dir.glob("*_3.npz")))

    print("Nearmap Roof and Solar Panel Detection")
    print("=" * 50)
    print(f"\nDataset: {n_images} images, {n_roof_masks} roof masks, {n_solar_masks} solar masks")
    print(f"Model: {args.model} | Loss: {args.loss} | Epochs: {args.epochs}")
    print(f"Batch size: {args.batch_size} | LR: {args.lr} | Image size: {args.img_size}")

    # Step 1: Train
    print("\n" + "=" * 50)
    print("STEP 1: Training")
    print("=" * 50)

    from src.train import main as train_main
    train_main(config={
        'data_dir': args.data_dir,
        'batch_size': args.batch_size,
        'num_epochs': args.epochs,
        'learning_rate': args.lr,
        'img_size': args.img_size,
        'model_type': args.model,
        'loss_type': args.loss,
    })

    # Step 2: Evaluate
    checkpoint_path = Path("outputs/checkpoints/best_model.pth")
    if checkpoint_path.exists():
        print("\n" + "=" * 50)
        print("STEP 2: Evaluation")
        print("=" * 50)

        from src.evaluate import main as evaluate_main
        evaluate_main(
            checkpoint_path=str(checkpoint_path),
            n_samples=args.n_samples,
        )

    # Summary
    print("\n" + "=" * 50)
    print("Done!")
    print("=" * 50)
    print("\nOutputs:")
    print("  checkpoints:     outputs/checkpoints/")
    print("  training logs:   outputs/logs/")
    print("  visualisations:  outputs/visualisations/")


if __name__ == "__main__":
    main()
