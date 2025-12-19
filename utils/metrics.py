"""
Metrics utilities
"""
import numpy as np
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix,
    classification_report
)


def calculate_metrics(y_true, y_pred):
    """
    Calculate classification metrics
    
    Args:
        y_true: True labels
        y_pred: Predicted labels
    
    Returns:
        Dictionary of metrics
    """
    metrics = {
        'accuracy': accuracy_score(y_true, y_pred),
        'precision_macro': precision_score(y_true, y_pred, average='macro'),
        'recall_macro': recall_score(y_true, y_pred, average='macro'),
        'f1_macro': f1_score(y_true, y_pred, average='macro'),
        'precision_weighted': precision_score(y_true, y_pred, average='weighted'),
        'recall_weighted': recall_score(y_true, y_pred, average='weighted'),
        'f1_weighted': f1_score(y_true, y_pred, average='weighted'),
    }
    
    return metrics


def print_metrics(metrics):
    """Print metrics in a formatted way"""
    print("\n" + "="*50)
    print("METRICS")
    print("="*50)
    for key, value in metrics.items():
        print(f"{key:25s}: {value:.4f}")
    print("="*50)


def get_confusion_matrix(y_true, y_pred):
    """Get confusion matrix"""
    return confusion_matrix(y_true, y_pred)


def get_classification_report(y_true, y_pred, target_names=None):
    """Get detailed classification report"""
    if target_names is None:
        target_names = ['no_pain', 'low_pain', 'high_pain']
    
    return classification_report(y_true, y_pred, target_names=target_names)
