import numpy as np


def compute_metrics(tp, fp, fn, tn):
    """
    Compute segmentation metrics from confusion matrix components

    Args:
        tp: True positives
        fp: False positives
        fn: False negatives
        tn: True negatives

    Returns:
        Dictionary with various metrics
    """
    eps = 1e-8  # Small epsilon to avoid division by zero

    # Accuracy
    accuracy = (tp + tn) / (tp + fp + fn + tn + eps)

    # Precision
    precision = tp / (tp + fp + eps)

    # Recall (Sensitivity)
    recall = tp / (tp + fn + eps)

    # F1 Score (Dice coefficient)
    f1 = 2 * tp / (2 * tp + fp + fn + eps)

    # IoU (Intersection over Union / Jaccard index)
    iou = tp / (tp + fp + fn + eps)

    # Specificity
    specificity = tn / (tn + fp + eps)

    return {
        'accuracy': accuracy,
        'precision': precision,
        'recall': recall,
        'f1': f1,
        'iou': iou,
        'specificity': specificity
    }


def compute_metrics_per_image(pred, target, valid_mask):
    """
    Compute metrics for a single image/channel

    Args:
        pred: Binary predictions (H, W)
        target: Binary ground truth (H, W)
        valid_mask: Valid pixel mask (H, W)

    Returns:
        Dictionary with metrics
    """
    # Only consider valid pixels
    pred_valid = pred[valid_mask == 1]
    target_valid = target[valid_mask == 1]

    if len(pred_valid) == 0:
        return {
            'accuracy': 0.0,
            'precision': 0.0,
            'recall': 0.0,
            'f1': 0.0,
            'iou': 0.0,
            'specificity': 0.0
        }

    # Compute confusion matrix components
    tp = ((pred_valid == 1) & (target_valid == 1)).sum()
    fp = ((pred_valid == 1) & (target_valid == 0)).sum()
    fn = ((pred_valid == 0) & (target_valid == 1)).sum()
    tn = ((pred_valid == 0) & (target_valid == 0)).sum()

    return compute_metrics(tp, fp, fn, tn)


def aggregate_metrics(metrics_list):
    """
    Aggregate metrics from multiple batches

    Args:
        metrics_list: List of metric dictionaries

    Returns:
        Dictionary with aggregated metrics
    """
    if not metrics_list:
        return {}

    # Get all metric names
    metric_names = list(metrics_list[0].keys())

    # Compute mean for each metric
    aggregated = {}
    for name in metric_names:
        values = [m[name] for m in metrics_list if name in m]
        aggregated[name] = np.mean(values) if values else 0.0

    return aggregated


if __name__ == "__main__":
    # Test metrics computation
    tp, fp, fn, tn = 100, 20, 30, 850

    metrics = compute_metrics(tp, fp, fn, tn)

    print("Test metrics computation:")
    print("-" * 30)
    print(f"True Positives: {tp}")
    print(f"False Positives: {fp}")
    print(f"False Negatives: {fn}")
    print(f"True Negatives: {tn}")
    print("\nMetrics:")
    for name, value in metrics.items():
        print(f"  {name}: {value:.4f}")