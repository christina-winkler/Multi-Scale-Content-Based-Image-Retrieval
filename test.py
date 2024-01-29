import torch
from sklearn.metrics import precision_recall_fscore_support, accuracy_score

def calculate_multiclass_metrics(predictions, targets, average='weighted'):
    """
    Calculate precision, recall, and accuracy for multiclass classification.

    Parameters:
        predictions (torch.Tensor): Tensor containing predicted labels (integer values).
        targets (torch.Tensor): Tensor containing true labels (integer values).
        average (str): Type of averaging for precision, recall, and F1. Options: 'micro', 'macro', 'weighted'.

    Returns:
        float: Precision value.
        float: Recall value.
        float: Accuracy value.
    """
    # Convert predictions and targets to numpy arrays
    predictions_np = predictions.cpu().numpy()
    targets_np = targets.cpu().numpy()

    # Calculate precision, recall, and F1 score using scikit-learn functions
    precision, recall, _, _ = precision_recall_fscore_support(targets_np, predictions_np, average=average)
    accuracy = accuracy_score(targets_np, predictions_np)

    return precision, recall, accuracy


# TODO: store features over test set (output before classifier layer) in dictionary
# TODO: then cluster them (KDTree ?)
